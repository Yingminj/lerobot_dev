# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from dataclasses import dataclass
from types import SimpleNamespace

import draccus
import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.datasets.dataset_reader import DatasetReader
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.act.modeling_act import ACT
from lerobot.policies.act_eef.configuration_act_eef import ACTEEFConfig
from lerobot.policies.act_eef.modeling_act_eef import ACTEEFPolicy
from lerobot.policies.act_eef_cvae import processor_act_eef_cvae
from lerobot.policies.act_eef_cvae.configuration_act_eef_cvae import ACTEEFCVAEConfig
from lerobot.policies.act_eef_cvae.modeling_act_eef_cvae import (
    CVAE_ACTION,
    CVAE_ACTION_IS_PAD,
    ACTEEFCVAEPolicy,
)
from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


@dataclass
class PolicyArgs:
    policy: PreTrainedConfig


def make_config(config_class=ACTEEFCVAEConfig, **overrides):
    kwargs = {
        "device": "cpu",
        "input_features": {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(14,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(4,)),
        },
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
        "chunk_size": 3,
        "n_action_steps": 3,
        "dim_model": 32,
        "n_heads": 4,
        "dim_feedforward": 64,
        "n_encoder_layers": 1,
        "n_decoder_layers": 1,
        "n_vae_encoder_layers": 1,
        "latent_dim": 8,
        "pretrained_backbone_weights": None,
    }
    kwargs.update(overrides)
    return config_class(**kwargs)


def make_batch():
    return {
        OBS_STATE: torch.randn(2, 14),
        OBS_ENV_STATE: torch.randn(2, 4),
        ACTION: torch.arange(4, dtype=torch.float32).view(1, 4, 1).expand(2, 4, 14).clone(),
        "action_is_pad": torch.tensor([[True, False, False, False], [False, False, True, True]]),
    }


def test_registration_and_cli_config():
    cfg = make_policy_config("act_eef_cvae", device="cpu")
    assert isinstance(cfg, ACTEEFCVAEConfig)
    assert isinstance(cfg, ACTEEFConfig)
    assert get_policy_class("act_eef_cvae") is ACTEEFCVAEPolicy
    parsed = draccus.parse(PolicyArgs, args=["--policy.type=act_eef_cvae", "--policy.device=cpu"])
    assert isinstance(parsed.policy, ACTEEFCVAEConfig)
    assert cfg.action_delta_indices == list(range(-1, cfg.chunk_size))
    meta = SimpleNamespace(features={ACTION: {}, OBS_STATE: {}}, fps=50)
    assert resolve_delta_timestamps(cfg, meta) == {ACTION: [i / 50 for i in cfg.action_delta_indices]}
    assert cfg.observation_delta_indices is None
    assert cfg.get_optimizer_preset() == ACTEEFConfig(device="cpu").get_optimizer_preset()
    assert cfg.kl_weight == ACTEEFConfig(device="cpu").kl_weight


@pytest.mark.parametrize(
    ("frame", "end", "indices", "mask"),
    [
        (10, 13, [10, 10, 11, 12], [True, False, False, False]),
        (11, 13, [10, 11, 12, 12], [False, False, False, True]),
        (12, 13, [11, 12, 12, 12], [False, False, True, True]),
        (10, 11, [10, 10, 10, 10], [True, False, True, True]),
    ],
)
def test_episode_boundaries_and_window_masks(frame, end, indices, mask):
    reader = DatasetReader.__new__(DatasetReader)
    reader._meta = SimpleNamespace(episodes={7: {"dataset_from_index": 10, "dataset_to_index": end}})
    reader.delta_indices = {ACTION: make_config().action_delta_indices}
    actual_indices, padding = reader._get_query_indices(frame, 7)
    assert actual_indices[ACTION] == indices
    assert padding["action_is_pad"].tolist() == mask

    policy = ACTEEFCVAEPolicy(make_config())
    batch = make_batch()
    batch[ACTION] = torch.tensor(indices, dtype=torch.float32).view(1, 4, 1).expand(2, 4, 14)
    batch["action_is_pad"] = padding["action_is_pad"].expand(2, 4)
    captured = {}

    def capture_batch(_module, args):
        captured.update(args[0])

    handle = policy.model.register_forward_pre_hook(capture_batch)
    loss, _ = policy(batch)
    handle.remove()
    assert torch.isfinite(loss)
    assert captured[CVAE_ACTION][0, :, 0].tolist() == indices[:-1]
    assert captured[ACTION][0, :, 0].tolist() == indices[1:]
    assert captured[CVAE_ACTION_IS_PAD][0].tolist() == mask[:-1]
    assert captured["action_is_pad"][0].tolist() == mask[1:]


def test_actual_cvae_inputs_target_loss_and_gradients():
    policy = ACTEEFCVAEPolicy(make_config())
    batch = make_batch()
    original = {key: value.clone() for key, value in batch.items()}
    captured = {}

    def capture_actions(_module, args):
        captured["history"] = args[0].clone()

    def capture_mask(_module, _args, kwargs):
        captured["mask"] = kwargs["key_padding_mask"].clone()

    def capture_result(_module, _args, output):
        captured["result"] = output

    handles = [
        policy.model.vae_encoder_action_input_proj.register_forward_pre_hook(capture_actions),
        policy.model.vae_encoder.register_forward_pre_hook(capture_mask, with_kwargs=True),
        policy.model.register_forward_hook(capture_result),
    ]
    loss, metrics = policy(batch)
    for handle in handles:
        handle.remove()

    torch.testing.assert_close(captured["history"], batch[ACTION][1:, :-1])
    torch.testing.assert_close(captured["mask"][:, 2:], batch["action_is_pad"][1:, :-1])
    assert not captured["mask"][:, :2].any()
    prediction, (mu, log_var) = captured["result"]
    assert torch.count_nonzero(mu[0]) == torch.count_nonzero(log_var[0]) == 0
    valid = (~batch["action_is_pad"][:, 1:]).unsqueeze(-1)
    expected_l1 = ((prediction - batch[ACTION][:, 1:]).abs() * valid).sum() / (valid.sum() * 14)
    expected_kl = (-0.5 * (1 + log_var - mu.square() - log_var.exp())).sum(-1).mean()
    noninitial_kl = (-0.5 * (1 + log_var[1] - mu[1].square() - log_var[1].exp())).sum()
    torch.testing.assert_close(expected_kl, noninitial_kl / 2)
    torch.testing.assert_close(loss, expected_l1 + policy.config.kl_weight * expected_kl)
    assert metrics["l1_loss"] == expected_l1.item()
    assert metrics["kld_loss"] == expected_kl.item()
    loss.backward()
    latent_grad = policy.model.vae_encoder_latent_output_proj.weight.grad
    for grad in latent_grad.chunk(2):
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert policy.model.action_head.weight.grad.abs().sum() > 0
    for key in original:
        torch.testing.assert_close(batch[key], original[key])
    assert batch.keys() == original.keys()


def test_original_computation_is_exact_when_cvae_inputs_match():
    torch.manual_seed(31)
    original = ACTEEFPolicy(make_config(ACTEEFConfig))
    torch.manual_seed(31)
    policy = ACTEEFCVAEPolicy(make_config())
    assert original.state_dict().keys() == policy.state_dict().keys()
    for key, value in original.state_dict().items():
        torch.testing.assert_close(policy.state_dict()[key], value, rtol=0, atol=0)

    batch = make_batch()
    batch[ACTION] = batch[ACTION][:, 1:]
    batch["action_is_pad"] = batch["action_is_pad"][:, 1:]
    prepared = {**batch, CVAE_ACTION: batch[ACTION], CVAE_ACTION_IS_PAD: batch["action_is_pad"]}
    torch.manual_seed(7)
    expected_loss, expected_metrics = original(batch)
    torch.manual_seed(7)
    # Bypass only the K+1 splitter to supply identical CVAE and target windows.
    actual_loss, actual_metrics = ACTEEFPolicy.forward(policy, prepared)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    assert actual_metrics == expected_metrics
    expected_loss.backward()
    actual_loss.backward()
    for name, parameter in original.named_parameters():
        actual_grad = dict(policy.named_parameters())[name].grad
        if parameter.grad is None:
            assert actual_grad is None
        else:
            torch.testing.assert_close(actual_grad, parameter.grad, rtol=0, atol=0)


@pytest.mark.parametrize(("n_action_steps", "ensemble"), [(3, None), (1, None), (1, 0.01)])
def test_inference_is_unchanged_without_history(n_action_steps, ensemble):
    kwargs = {"n_action_steps": n_action_steps, "temporal_ensemble_coeff": ensemble}
    original = ACTEEFPolicy(make_config(ACTEEFConfig, **kwargs))
    policy = ACTEEFCVAEPolicy(make_config(**kwargs))
    policy.load_state_dict(original.state_dict(), strict=True)
    batch = {key: value for key, value in make_batch().items() if key.startswith("observation.")}
    for _ in range(5):
        torch.testing.assert_close(policy.select_action(batch), original.select_action(batch), rtol=0, atol=0)
    policy.reset()
    original.reset()
    torch.testing.assert_close(policy.select_action(batch), original.select_action(batch), rtol=0, atol=0)


@pytest.mark.parametrize("training", [True, False])
def test_no_vae_and_validation_keep_current_target(training):
    policy = ACTEEFCVAEPolicy(make_config(use_vae=False))
    original = ACTEEFPolicy(make_config(ACTEEFConfig, use_vae=False))
    original.load_state_dict(policy.state_dict(), strict=True)
    policy.train(training)
    original.train(training)
    batch = make_batch()
    target_batch = {**batch, ACTION: batch[ACTION][:, 1:], "action_is_pad": batch["action_is_pad"][:, 1:]}
    torch.manual_seed(8)
    loss, metrics = policy(batch)
    torch.manual_seed(8)
    expected, expected_metrics = original(target_batch)
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    assert metrics == expected_metrics


def test_eval_loss_ignores_history_and_uses_zero_latent():
    policy = ACTEEFCVAEPolicy(make_config()).eval()
    batch = make_batch()
    loss, metrics = policy(batch)
    batch[ACTION][:, 0] = 1000
    changed_loss, changed_metrics = policy(batch)
    torch.testing.assert_close(changed_loss, loss, rtol=0, atol=0)
    assert metrics == changed_metrics
    assert "kld_loss" not in metrics


def test_processors_and_checkpoint_round_trip(tmp_path, monkeypatch):
    cfg = make_config()
    stats = {
        key: {"mean": torch.ones(feature.shape), "std": torch.full(feature.shape, 2.0)}
        for key, feature in {**cfg.input_features, **cfg.output_features}.items()
    }
    calls = []
    actual_factory = processor_act_eef_cvae.make_act_eef_cvae_pre_post_processors

    def spy_factory(**kwargs):
        calls.append(kwargs["config"])
        return actual_factory(**kwargs)

    monkeypatch.setattr(processor_act_eef_cvae, "make_act_eef_cvae_pre_post_processors", spy_factory)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=stats)
    assert calls == [cfg]
    batch = make_batch()
    normalized = preprocessor(batch)
    assert normalized[ACTION].shape == (2, 4, 14)
    torch.testing.assert_close(normalized[ACTION], (batch[ACTION] - 1) / 2)
    torch.testing.assert_close(postprocessor(normalized[ACTION]), batch[ACTION])
    torch.testing.assert_close(normalized["action_is_pad"], batch["action_is_pad"])

    policy = ACTEEFCVAEPolicy(cfg)
    policy.save_pretrained(tmp_path)
    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)
    restored = ACTEEFCVAEPolicy.from_pretrained(tmp_path, local_files_only=True, strict=True)
    assert isinstance(restored.config, ACTEEFCVAEConfig)
    assert restored.config.action_delta_indices == [-1, 0, 1, 2]
    obs = {key: value for key, value in normalized.items() if key.startswith("observation.")}
    torch.testing.assert_close(restored.predict_action_chunk(obs), policy.predict_action_chunk(obs))
    restored_pre, restored_post = make_pre_post_processors(restored.config, pretrained_path=str(tmp_path))
    torch.testing.assert_close(restored_pre(batch)[ACTION], normalized[ACTION])
    torch.testing.assert_close(restored_post(normalized[ACTION]), batch[ACTION])


def test_rejects_unshifted_training_batch():
    batch = make_batch()
    batch[ACTION] = batch[ACTION][:, 1:]
    with pytest.raises(ValueError, match="offsets"):
        ACTEEFCVAEPolicy(make_config())(batch)


@pytest.mark.parametrize("mode", ["sample", "zero"])
def test_initial_mode_cli_save_load_and_old_config_default(mode, tmp_path):
    parsed = draccus.parse(
        PolicyArgs,
        args=["--policy.type=act_eef_cvae", "--policy.device=cpu", f"--policy.initial_z_mode={mode}"],
    )
    assert parsed.policy.initial_z_mode == mode
    parsed.policy.save_pretrained(tmp_path)
    assert PreTrainedConfig.from_pretrained(tmp_path).initial_z_mode == mode
    config_file = tmp_path / "config.json"
    payload = json.loads(config_file.read_text())
    del payload["initial_z_mode"]
    config_file.write_text(json.dumps(payload))
    assert PreTrainedConfig.from_pretrained(tmp_path).initial_z_mode == "sample"


def test_rejects_invalid_initial_mode():
    with pytest.raises(ValueError, match="initial_z_mode"):
        make_config(initial_z_mode="invalid")
    with pytest.raises(draccus.utils.DecodingError, match="initial_z_mode"):
        draccus.parse(
            PolicyArgs,
            args=["--policy.type=act_eef_cvae", "--policy.device=cpu", "--policy.initial_z_mode=invalid"],
        )


@pytest.mark.parametrize("mode", ["sample", "zero"])
def test_all_first_frames_skip_cvae_and_do_not_read_action_truth(mode):
    policy = ACTEEFCVAEPolicy(make_config(initial_z_mode=mode))
    batch = make_batch()
    batch["action_is_pad"][:, 0] = True
    latents, predictions = [], []

    def forbid_cvae(_module, _args):
        pytest.fail("First-frame samples must not enter any CVAE module")

    def capture_latent(_module, args):
        latents.append(args[0].detach().clone())

    def capture_prediction(_module, _args, output):
        predictions.append(output[0].detach().clone())

    handles = [
        module.register_forward_pre_hook(forbid_cvae)
        for name, module in policy.model.named_children()
        if name.startswith("vae_encoder")
    ]
    handles += [
        policy.model.encoder_latent_input_proj.register_forward_pre_hook(capture_latent),
        policy.model.register_forward_hook(capture_prediction),
    ]
    torch.manual_seed(91)
    loss, metrics = policy(batch)
    assert torch.isfinite(loss)
    assert metrics["kld_loss"] == 0.0
    assert loss.item() == metrics["l1_loss"]
    loss.backward()
    for name, parameter in policy.model.named_parameters():
        if name.startswith("vae_encoder"):
            assert parameter.grad is None
    assert policy.model.action_head.weight.grad.abs().sum() > 0

    changed = {**batch, ACTION: batch[ACTION] + 100}
    torch.manual_seed(91)
    changed_loss, _ = policy(changed)
    torch.testing.assert_close(latents[1], latents[0], rtol=0, atol=0)
    torch.testing.assert_close(predictions[1], predictions[0], rtol=0, atol=0)
    assert changed_loss.item() != loss.item()
    torch.manual_seed(92)
    policy(batch)
    if mode == "sample":
        torch.manual_seed(91)
        torch.testing.assert_close(latents[0], torch.randn(2, policy.config.latent_dim), rtol=0, atol=0)
        assert not torch.equal(latents[0][0], latents[0][1])
        assert not torch.equal(latents[0], latents[2])
    else:
        assert all(torch.count_nonzero(z) == 0 for z in latents)
    for handle in handles:
        handle.remove()
    assert policy.training and policy.model.training and policy.model.vae_encoder.training


@pytest.mark.parametrize("mode", ["sample", "zero"])
def test_mixed_batch_preserves_sample_order_and_zero_kl_for_initial_frames(mode):
    policy = ACTEEFCVAEPolicy(make_config(initial_z_mode=mode, dropout=0.0))
    batch = {key: value.repeat_interleave(2, dim=0) for key, value in make_batch().items()}
    batch["action_is_pad"][:, 0] = torch.tensor([False, True, False, True])
    batch[OBS_STATE] = torch.arange(4, dtype=torch.float32).unsqueeze(1).expand(4, 14)
    batch[ACTION] = batch[ACTION] + torch.arange(4).view(4, 1, 1) * 10
    captured = {}

    def capture_state(_module, args):
        captured["state"] = args[0].detach().clone()

    def capture_latent(_module, args):
        captured["latent"] = args[0].detach().clone()

    def capture_params(_module, _args, output):
        captured["params"] = output.detach().clone()

    def capture_result(_module, _args, output):
        captured["result"] = output

    handles = [
        policy.model.vae_encoder_robot_state_input_proj.register_forward_pre_hook(capture_state),
        policy.model.vae_encoder_latent_output_proj.register_forward_hook(capture_params),
        policy.model.encoder_latent_input_proj.register_forward_pre_hook(capture_latent),
        policy.model.register_forward_hook(capture_result),
    ]
    _, metrics = policy(batch)
    for handle in handles:
        handle.remove()
    torch.testing.assert_close(captured["state"], batch[OBS_STATE][[0, 2]])
    _, (mu, log_var) = captured["result"]
    torch.testing.assert_close(torch.cat([mu[[0, 2]], log_var[[0, 2]]], dim=-1), captured["params"])
    assert mu[[1, 3]].count_nonzero() == log_var[[1, 3]].count_nonzero() == 0
    per_history_kl = (-0.5 * (1 + log_var[[0, 2]] - mu[[0, 2]].square() - log_var[[0, 2]].exp())).sum()
    assert metrics["kld_loss"] == pytest.approx(per_history_kl.item() / 4)
    if mode == "zero":
        assert captured["latent"][[1, 3]].count_nonzero() == 0
    else:
        assert not torch.equal(captured["latent"][1], captured["latent"][3])


def test_no_initial_frames_match_previous_shifted_training_path():
    policy = ACTEEFCVAEPolicy(make_config())
    batch = make_batch()
    batch["action_is_pad"][:, 0] = False
    torch.manual_seed(51)
    loss, metrics = policy(batch)
    loss.backward()
    grads = {name: p.grad.clone() for name, p in policy.named_parameters() if p.grad is not None}
    policy.zero_grad(set_to_none=True)
    history_batch = {**batch, ACTION: batch[ACTION][:, :-1], "action_is_pad": batch["action_is_pad"][:, :-1]}
    torch.manual_seed(51)
    prediction, (mu, log_var) = ACT.forward(policy.model, history_batch)
    valid = (~batch["action_is_pad"][:, 1:]).unsqueeze(-1)
    l1 = ((prediction - batch[ACTION][:, 1:]).abs() * valid).sum() / (valid.sum() * 14)
    kl = (-0.5 * (1 + log_var - mu.square() - log_var.exp())).sum(-1).mean()
    expected = l1 + policy.config.kl_weight * kl
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    assert metrics == {"l1_loss": l1.item(), "kld_loss": kl.item()}
    expected.backward()
    for name, p in policy.named_parameters():
        if name in grads:
            torch.testing.assert_close(p.grad, grads[name], rtol=0, atol=0)


@pytest.mark.parametrize("with_image", [False, True])
def test_explicit_latent_decoder_matches_original_act(with_image):
    cfg = make_config(use_vae=False)
    batch = make_batch()
    if with_image:
        cfg.input_features["observation.images.camera"] = PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 64, 64)
        )
        batch[OBS_IMAGES] = [torch.rand(2, 3, 64, 64)]
    policy = ACTEEFCVAEPolicy(cfg)
    torch.manual_seed(3)
    expected, _ = ACT.forward(policy.model, batch)
    torch.manual_seed(3)
    actual = policy.model._decode_with_latent(batch, torch.zeros(2, cfg.latent_dim))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_mixed_batch_with_cpu_autocast():
    policy = ACTEEFCVAEPolicy(make_config())
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss, _ = policy(make_batch())
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(policy.model.vae_encoder_latent_output_proj.weight.grad).all()
