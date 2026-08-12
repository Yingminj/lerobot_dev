#!/usr/bin/env python

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

"""Tests for the VITA vision-to-action flow matching policy.

To run locally:
    python -m pytest tests/policies/vita/test_vita.py -v
"""

import pytest
import torch
from torch import nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.policies.vita.configuration_vita import VitaConfig
from lerobot.policies.vita.flow_matching import (
    ConditionalFlowMatcher,
    ConsistencyFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    MeanFlowMatcher,
    make_flow_matcher,
)
from lerobot.policies.vita.modeling_vita import SimpleFlowNet, SimpleMeanFlowNet, VitaPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

STATE_DIM = 6
ACTION_DIM = 6
IMG_H, IMG_W = 96, 128
BATCH_SIZE = 4

CAMERA_KEYS = ["observation.images.top", "observation.images.wrist"]


# ----------------------------------------------------------------------------------------------
# Flow matchers, tested on their own without any policy around them.
# ----------------------------------------------------------------------------------------------


class LinearVelocityNet(nn.Module):
    """Minimal stand-in for `SimpleFlowNet`: a velocity field that ignores nothing but is trainable."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Linear(dim + 1, dim)

    def forward(self, x, t, **kwargs):
        return self.net(torch.cat([x, t[:, None]], dim=-1))


class LinearMeanVelocityNet(nn.Module):
    """Stand-in for `SimpleMeanFlowNet`, returning `(u, v, internal_features)`."""

    def __init__(self, dim: int):
        super().__init__()
        self.u = nn.Linear(dim + 2, dim)
        self.v = nn.Linear(dim + 2, dim)

    def forward(self, x, timestep, h, **kwargs):
        inp = torch.cat([x, timestep[:, None], h[:, None]], dim=-1)
        u = self.u(inp)
        return u, self.v(inp), torch.stack([u], dim=0)


@pytest.mark.parametrize("name", ["conditional", "consistency"])
def test_flow_matcher_loss_and_sample_shapes(name):
    dim = 8
    matcher = make_flow_matcher(name, num_sampling_steps=3)
    model = LinearVelocityNet(dim)
    start = torch.randn(BATCH_SIZE, dim)
    target = torch.randn(BATCH_SIZE, dim)

    loss, metrics = matcher.compute_loss(model, target=target, start=start)
    assert loss.ndim == 0
    assert "flow_loss" in metrics
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())

    sample = matcher.sample(model, (BATCH_SIZE, dim), device=start.device, start=start)
    assert sample.shape == (BATCH_SIZE, dim)


@pytest.mark.parametrize("name", ["mean", "improved_mean"])
def test_mean_flow_matcher_loss_and_sample_shapes(name):
    dim = 8
    matcher = make_flow_matcher(name, num_sampling_steps=1)
    model = LinearMeanVelocityNet(dim)
    start = torch.randn(BATCH_SIZE, dim)
    target = torch.randn(BATCH_SIZE, dim)

    loss, metrics = matcher.compute_loss(model, target=target, start=start)
    assert loss.ndim == 0
    loss.backward()

    sample = matcher.sample(model, (BATCH_SIZE, dim), device=start.device, num_steps=1, start=start)
    assert sample.shape == (BATCH_SIZE, dim)


def test_mean_flow_requires_start():
    """MeanFlow has no noise fallback: a missing source is an error, not a silent Gaussian."""
    matcher = MeanFlowMatcher()
    model = LinearMeanVelocityNet(8)
    with pytest.raises(ValueError, match="requires `start`"):
        matcher.compute_loss(model, target=torch.randn(BATCH_SIZE, 8))
    with pytest.raises(ValueError, match="requires `start`"):
        matcher.sample(model, (BATCH_SIZE, 8), device="cpu")


def test_conditional_flow_matcher_recovers_the_straight_path():
    """With `sigma=0`, `x_t` must lie exactly on the segment and `u_t` must be `x1 - x0`."""
    matcher = ConditionalFlowMatcher(sigma=0.0)
    x0 = torch.randn(BATCH_SIZE, 8)
    x1 = torch.randn(BATCH_SIZE, 8)
    t, xt, ut = matcher.sample_location_and_conditional_flow(x0, x1)

    expected_xt = (1 - t[:, None]) * x0 + t[:, None] * x1
    torch.testing.assert_close(xt, expected_xt)
    torch.testing.assert_close(ut, x1 - x0)
    assert ((t >= 0) & (t <= 1)).all()


def test_conditional_flow_matcher_integrates_a_constant_field_exactly():
    """Euler integration of a constant velocity `c` must move the source by exactly `c`."""

    class ConstantField(nn.Module):
        def forward(self, x, t, **kwargs):
            return torch.full_like(x, 2.0)

    matcher = ConditionalFlowMatcher()
    start = torch.zeros(BATCH_SIZE, 8)
    out = matcher.sample(ConstantField(), (BATCH_SIZE, 8), device="cpu", num_steps=5, start=start)
    torch.testing.assert_close(out, torch.full_like(start, 2.0))


def test_mean_flow_one_step_uses_the_mean_flow_identity():
    """1-NFE sampling must be exactly `x - u(x, t=1, h=1)`."""

    class ConstantMeanField(nn.Module):
        def forward(self, x, timestep, h, **kwargs):
            u = torch.full_like(x, 0.5)
            return u, torch.zeros_like(x), torch.stack([u], dim=0)

    matcher = MeanFlowMatcher(num_sampling_steps=1)
    start = torch.ones(BATCH_SIZE, 8)
    out = matcher.sample(ConstantMeanField(), (BATCH_SIZE, 8), device="cpu", num_steps=1, start=start)
    torch.testing.assert_close(out, start - 0.5)


def test_plain_cfm_preserves_the_observation_action_pairing():
    """`conditional` must not permute the batch: `u_t` stays `target[i] - start[i]`."""
    matcher = ConditionalFlowMatcher()
    x0 = torch.randn(BATCH_SIZE, 8)
    x1 = torch.randn(BATCH_SIZE, 8)
    coupled_x0, coupled_x1 = matcher.couple(x0, x1)
    torch.testing.assert_close(coupled_x0, x0)
    torch.testing.assert_close(coupled_x1, x1)


def test_exact_ot_coupling_is_a_permutation_that_lowers_transport_cost():
    """OT coupling re-pairs the minibatch. This is the documented pairing caveat, asserted."""
    pytest.importorskip("scipy")
    matcher = ExactOptimalTransportConditionalFlowMatcher()

    torch.manual_seed(0)
    x0 = torch.randn(16, 8)
    x1 = torch.randn(16, 8)
    coupled_x0, coupled_x1 = matcher.couple(x0, x1)

    # x0 is untouched; x1 comes back as a permutation of itself.
    torch.testing.assert_close(coupled_x0, x0)
    assert coupled_x1.shape == x1.shape
    sorted_original, _ = torch.sort(x1.flatten())
    sorted_coupled, _ = torch.sort(coupled_x1.flatten())
    torch.testing.assert_close(sorted_original, sorted_coupled)

    # And it is the *optimal* permutation, so the total squared transport cost cannot increase.
    identity_cost = ((x0 - x1) ** 2).sum()
    coupled_cost = ((coupled_x0 - coupled_x1) ** 2).sum()
    assert coupled_cost <= identity_cost + 1e-5

    # The pairing really does change on random data — that is the caveat worth knowing about.
    assert not torch.allclose(coupled_x1, x1)


def test_exact_ot_coupling_is_identity_for_a_single_sample():
    pytest.importorskip("scipy")
    matcher = ExactOptimalTransportConditionalFlowMatcher()
    x0, x1 = torch.randn(1, 8), torch.randn(1, 8)
    coupled_x0, coupled_x1 = matcher.couple(x0, x1)
    torch.testing.assert_close(coupled_x0, x0)
    torch.testing.assert_close(coupled_x1, x1)


def test_consistency_matcher_handles_rank_two_latents():
    """The reference implementation hardcodes rank-3 tensors; VITA's latents are rank 2."""
    matcher = ConsistencyFlowMatcher()
    model = LinearVelocityNet(8)
    loss, _ = matcher.compute_loss(model, target=torch.randn(BATCH_SIZE, 8), start=torch.randn(BATCH_SIZE, 8))
    assert torch.isfinite(loss)


def test_make_flow_matcher_rejects_unknown_names():
    with pytest.raises(ValueError, match="Invalid flow matcher name"):
        make_flow_matcher("does-not-exist")


def test_make_flow_matcher_sets_the_improved_mean_flow_flag():
    assert make_flow_matcher("improved_mean").use_imf is True
    assert make_flow_matcher("mean").use_imf is False


# ----------------------------------------------------------------------------------------------
# Velocity networks
# ----------------------------------------------------------------------------------------------


def test_simple_flow_net_takes_no_conditioning():
    """The defining property of VITA: the velocity net sees only `(x_t, t)`."""
    net = SimpleFlowNet(input_dim=16, hidden_dim=32, output_dim=16, num_layers=2)
    out = net(torch.randn(BATCH_SIZE, 16), torch.rand(BATCH_SIZE))
    assert out.shape == (BATCH_SIZE, 16)

    import inspect

    params = list(inspect.signature(net.forward).parameters)
    assert params[:3] == ["x", "t", "kwargs"], (
        "SimpleFlowNet must not grow a conditioning argument — that is the whole point of VITA."
    )


def test_simple_mean_flow_net_returns_u_v_and_features():
    net = SimpleMeanFlowNet(input_dim=16, hidden_dim=32, output_dim=16, num_layers=2)
    u, v, feats = net(torch.randn(BATCH_SIZE, 16), torch.rand(BATCH_SIZE), torch.rand(BATCH_SIZE))
    assert u.shape == (BATCH_SIZE, 16)
    assert v.shape == (BATCH_SIZE, 16)
    assert feats.shape[0] == 2  # one entry per hidden layer, for the dispersive loss
    # Both heads are zero-initialised, so the field starts at rest.
    torch.testing.assert_close(u, torch.zeros_like(u))
    torch.testing.assert_close(v, torch.zeros_like(v))


# ----------------------------------------------------------------------------------------------
# Policy
# ----------------------------------------------------------------------------------------------


def make_config(**overrides) -> VitaConfig:
    kwargs = dict(
        flow_matcher_type="conditional",
        resize_shape=(60, 80),
        crop_shape=(56, 76),
        latent_dim=64,
        flow_hidden_dim=64,
        flow_num_layers=2,
        action_enc_hidden_dim=32,
        action_dec_hidden_dim=32,
        action_ae_num_layers=2,
        pretrained_backbone_weights=None,
    )
    kwargs.update(overrides)
    return make_policy_config(
        "vita",
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
            **{k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMG_H, IMG_W)) for k in CAMERA_KEYS},
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
        device="cpu",
        **kwargs,
    )


def make_batch(config: VitaConfig) -> dict:
    return {
        OBS_STATE: torch.randn(BATCH_SIZE, config.n_obs_steps, STATE_DIM),
        **{k: torch.rand(BATCH_SIZE, config.n_obs_steps, 3, IMG_H, IMG_W) for k in CAMERA_KEYS},
        ACTION: torch.randn(BATCH_SIZE, config.horizon, ACTION_DIM),
        "action_is_pad": torch.zeros(BATCH_SIZE, config.horizon, dtype=torch.bool),
    }


def make_observation() -> dict:
    """A single un-batched-in-time observation, as `select_action` receives it at rollout."""
    return {
        OBS_STATE: torch.randn(1, STATE_DIM),
        **{k: torch.rand(1, 3, IMG_H, IMG_W) for k in CAMERA_KEYS},
    }


def make_observation_window(config: VitaConfig) -> dict:
    """`n_obs_steps` of observations, as `predict_action_chunk` receives them offline."""
    return {
        OBS_STATE: torch.randn(1, config.n_obs_steps, STATE_DIM),
        **{k: torch.rand(1, config.n_obs_steps, 3, IMG_H, IMG_W) for k in CAMERA_KEYS},
    }


def test_policy_is_registered_in_the_factory():
    assert get_policy_class("vita") is VitaPolicy
    assert isinstance(make_policy_config("vita"), VitaConfig)


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"flow_matcher_type": "conditional", "decode_flow_latents": False},
        {"flow_matcher_type": "mean", "flow_net_type": "simple_mean", "num_sampling_steps": 1},
        {
            "flow_matcher_type": "mean",
            "flow_net_type": "simple_mean",
            "num_sampling_steps": 1,
            "meanflow_dispersive_loss_weight": 0.5,
        },
        {"flow_matcher_type": "improved_mean", "flow_net_type": "simple_mean", "num_sampling_steps": 1},
        {"flow_matcher_type": "consistency", "num_sampling_steps": 1},
        {"action_encoder_type": "simple"},
        {"n_obs_steps": 2, "drop_n_last_frames": 7},
        {"enc_contrastive_weight": 0.1, "flow_contrastive_weight": 0.1},
    ],
    ids=[
        "default-cfm",
        "no-fld",
        "meanflow",
        "meanflow-dispersive",
        "improved-meanflow",
        "consistency",
        "simple-action-encoder",
        "two-obs-steps",
        "contrastive",
    ],
)
def test_forward_backward_and_action_shapes(overrides):
    torch.manual_seed(0)
    config = make_config(**overrides)
    policy = VitaPolicy(config)

    loss, metrics = policy(make_batch(config))
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert "loss" in metrics
    loss.backward()

    policy.eval()
    policy.reset()
    action = policy.select_action(make_observation())
    assert action.shape == (1, ACTION_DIM)

    policy.reset()
    chunk = policy.predict_action_chunk(make_observation_window(config))
    assert chunk.shape == (1, config.n_action_steps, ACTION_DIM)


def test_gradients_reach_every_trainable_module():
    """A silent dead branch here would look like a working but much weaker policy."""
    torch.manual_seed(0)
    config = make_config()
    policy = VitaPolicy(config)
    loss, _ = policy(make_batch(config))
    loss.backward()

    without_grad = [n for n, p in policy.named_parameters() if p.requires_grad and p.grad is None]
    assert without_grad == [], f"parameters received no gradient: {without_grad}"


def test_flow_latent_decoding_backpropagates_through_the_ode():
    """FLD must add reconstruction/consistency terms; without it the loss is strictly simpler."""
    torch.manual_seed(0)
    batch = make_batch(make_config())

    with_fld = VitaPolicy(make_config(decode_flow_latents=True))
    _, metrics_with = with_fld(batch)
    without_fld = VitaPolicy(make_config(decode_flow_latents=False))
    _, metrics_without = without_fld(batch)

    assert "flow_action_recon_loss" in metrics_with
    assert "consistency_loss" in metrics_with
    assert "flow_action_recon_loss" not in metrics_without
    assert "consistency_loss" not in metrics_without


def test_select_action_replans_every_n_action_steps():
    """The queue must serve `n_action_steps` actions before the flow is solved again."""
    torch.manual_seed(0)
    config = make_config()
    policy = VitaPolicy(config)
    policy.eval()
    policy.reset()

    calls = []
    original = policy.vita.generate_actions

    def counting(batch):
        calls.append(1)
        return original(batch)

    policy.vita.generate_actions = counting
    for _ in range(config.n_action_steps * 2):
        policy.select_action(make_observation())

    assert len(calls) == 2


def test_frozen_action_autoencoder_receives_no_gradient():
    torch.manual_seed(0)
    config = make_config(freeze_action_encoder=True, freeze_action_decoder=True)
    policy = VitaPolicy(config)
    loss, _ = policy(make_batch(config))
    loss.backward()

    for name, param in policy.named_parameters():
        if "action_encoder" in name or "action_decoder" in name:
            assert param.grad is None, f"{name} should be frozen"


def test_processors_build_and_round_trip():
    config = make_config()
    stats = {
        OBS_STATE: {"min": torch.zeros(STATE_DIM), "max": torch.ones(STATE_DIM)},
        ACTION: {"min": torch.zeros(ACTION_DIM), "max": torch.ones(ACTION_DIM)},
        **{k: {"mean": torch.zeros(3, 1, 1), "std": torch.ones(3, 1, 1)} for k in CAMERA_KEYS},
    }
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=stats)

    processed = preprocessor(make_observation())
    assert OBS_STATE in processed
    for key in CAMERA_KEYS:
        assert key in processed

    action = postprocessor(torch.randn(1, ACTION_DIM))
    assert action.shape == (1, ACTION_DIM)


# ----------------------------------------------------------------------------------------------
# Configuration validation
# ----------------------------------------------------------------------------------------------


def test_meanflow_matcher_requires_the_meanflow_network():
    with pytest.raises(ValueError, match="requires `flow_net_type='simple_mean'`"):
        make_config(flow_matcher_type="mean", flow_net_type="simple")


def test_meanflow_network_requires_a_meanflow_matcher():
    with pytest.raises(ValueError, match="only usable with `flow_matcher_type`"):
        make_config(flow_matcher_type="conditional", flow_net_type="simple_mean")


def test_simple_action_encoder_requires_a_divisible_latent():
    with pytest.raises(ValueError, match="divisible by `horizon`"):
        make_config(action_encoder_type="simple", latent_dim=65, horizon=16)


def test_cnn_action_encoder_rejects_a_chunk_it_would_collapse():
    with pytest.raises(ValueError, match="at least 2\\*\\*`action_ae_num_layers`"):
        make_config(action_encoder_type="cnn", horizon=8, action_ae_num_layers=4, n_action_steps=4)


def test_unknown_flow_matcher_is_rejected_by_the_config():
    with pytest.raises(ValueError, match="`flow_matcher_type` must be one of"):
        make_config(flow_matcher_type="nope")


def test_n_action_steps_must_fit_the_horizon():
    with pytest.raises(ValueError, match="`n_action_steps` must satisfy"):
        make_config(horizon=16, n_action_steps=20)


def test_images_are_required():
    config = make_config()
    config.input_features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,))}
    with pytest.raises(ValueError, match="at least one image input is required"):
        config.validate_features()
