#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""End-to-end smoke test for the `act_delta` policy, on synthetic data.

Run with::

    python -m lerobot.policies.act_delta.selftest

It checks, on CPU and in a few seconds:

1. `act_delta` is registered and resolvable through the standard policy factories.
2. A training step runs (forward + backward) for R0 (absolute), R1 (all-relative) and
   R2 (relative except gripper).
3. The preprocessor really converts to relative actions, in physical units, *before*
   normalization — checked against a hand-computed target.
4. preprocessor → model → postprocessor round-trips back to absolute actions.
5. `ChunkFIFOActionServer` writes the chunk anchor exactly once per chunk: the actions it
   serves are element-wise equal to postprocessing the whole chunk in one call, even when
   the observation state changes wildly between ticks (the drift bug of plan §2.3).
6. `select_action` refuses the drifting queued path in relative mode.

Deliberately not a pytest module: `lerobot.configs.parser.load_plugin` imports every
submodule of a plugin package, so nothing here may pull in test-only dependencies.
"""

from __future__ import annotations

import logging

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE

from .inference_act_delta import ChunkFIFOActionServer, predict_absolute_chunk
from .processor_act_delta import make_act_delta_pre_post_processors

STATE_DIM = 7
ACTION_DIM = 7
CHUNK = 8
IMG_SHAPE = (3, 64, 64)
ACTION_NAMES = [f"joint_{i}" for i in range(ACTION_DIM - 1)] + ["gripper"]


def _make_config(**kwargs):
    cfg = make_policy_config(
        "act_delta",
        chunk_size=CHUNK,
        n_action_steps=CHUNK,
        dim_model=64,
        dim_feedforward=128,
        n_heads=4,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        latent_dim=8,
        pretrained_backbone_weights=None,
        device="cpu",
        action_feature_names=list(ACTION_NAMES),
        **kwargs,
    )
    cfg.input_features = {
        OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=IMG_SHAPE),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
    }
    cfg.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))}
    return cfg


def _make_stats(relative: bool):
    """Absolute-ish stats, or relative-ish stats (zero-mean) for the relative arms."""
    return {
        OBS_IMAGE: {
            "mean": torch.full((3, 1, 1), 0.5),
            "std": torch.full((3, 1, 1), 0.25),
        },
        OBS_STATE: {
            "mean": torch.full((STATE_DIM,), 1.0),
            "std": torch.full((STATE_DIM,), 0.5),
        },
        ACTION: {
            "mean": torch.zeros(ACTION_DIM) if relative else torch.full((ACTION_DIM,), 1.0),
            "std": torch.full((ACTION_DIM,), 0.1 if relative else 0.5),
        },
    }


def _make_batch(batch_size: int = 2, state_value: float = 1.0):
    torch.manual_seed(0)
    return {
        OBS_IMAGE: torch.rand(batch_size, *IMG_SHAPE),
        OBS_STATE: torch.full((batch_size, STATE_DIM), state_value),
        ACTION: torch.randn(batch_size, CHUNK, ACTION_DIM) * 0.05 + state_value,
        "action_is_pad": torch.zeros(batch_size, CHUNK, dtype=torch.bool),
    }


def _build(use_relative: bool, exclude_joints: list[str]):
    cfg = _make_config(
        use_relative_actions=use_relative,
        relative_exclude_joints=exclude_joints,
        relative_consistency_check="error",
    )
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls(cfg)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg, dataset_stats=_make_stats(relative=use_relative)
    )
    return cfg, policy, preprocessor, postprocessor


def check_registration() -> None:
    cfg = _make_config()
    assert cfg.type == "act_delta", cfg.type
    policy_cls = get_policy_class("act_delta")
    assert policy_cls.__name__ == "ACTDeltaPolicy", policy_cls
    # The processors must come from act_delta, not from the absolute ACT pipeline.
    pre, post = make_pre_post_processors(policy_cfg=cfg, dataset_stats=_make_stats(relative=False))
    step_names = [type(s).__name__ for s in pre.steps]
    assert "RelativeActionsProcessorStep" in step_names, step_names
    assert "AbsoluteActionsProcessorStep" in [type(s).__name__ for s in post.steps]
    # Order matters: relative before normalize, absolute after unnormalize.
    assert step_names.index("RelativeActionsProcessorStep") < step_names.index(
        "NormalizerProcessorStep"
    ), step_names
    out_names = [type(s).__name__ for s in post.steps]
    assert out_names.index("UnnormalizerProcessorStep") < out_names.index(
        "AbsoluteActionsProcessorStep"
    ), out_names
    print("  [ok] act_delta registered; pipelines assembled in OpenPI order")


def check_relative_mask() -> None:
    cfg = _make_config(use_relative_actions=True, relative_exclude_joints=["gripper"])
    mask = cfg.build_relative_mask(ACTION_DIM)
    assert mask == [True] * (ACTION_DIM - 1) + [False], mask
    cfg_r1 = _make_config(use_relative_actions=True, relative_exclude_joints=[])
    assert cfg_r1.build_relative_mask(ACTION_DIM) == [True] * ACTION_DIM
    # A mismatching exclude list must be caught, not silently ignored.
    bad = _make_config(
        use_relative_actions=True,
        relative_exclude_joints=["nonexistent_joint"],
        relative_consistency_check="error",
    )
    try:
        make_act_delta_pre_post_processors(bad, dataset_stats=_make_stats(relative=True))
    except ValueError as e:
        assert "matched no action dimension" in str(e), e
    else:
        raise AssertionError("an exclude list matching nothing should have been reported")
    # Absolute dataset stats under a relative policy must be caught too.
    stale = _make_config(use_relative_actions=True, relative_consistency_check="error")
    try:
        make_act_delta_pre_post_processors(stale, dataset_stats=_make_stats(relative=False))
    except ValueError as e:
        assert "ABSOLUTE action statistics" in str(e), e
    else:
        raise AssertionError("absolute dataset stats under a relative policy should have been reported")
    print("  [ok] exclude mask, mask/stats consistency guards")


def check_preprocessor_math() -> None:
    cfg, _, preprocessor, _ = _build(use_relative=True, exclude_joints=["gripper"])
    batch = _make_batch(state_value=1.3)
    processed = preprocessor({k: v.clone() for k, v in batch.items()})

    stats = _make_stats(relative=True)
    mask = torch.tensor(cfg.build_relative_mask(ACTION_DIM), dtype=torch.float32)
    expected_relative = batch[ACTION] - batch[OBS_STATE].unsqueeze(1) * mask
    expected = (expected_relative - stats[ACTION]["mean"]) / stats[ACTION]["std"]
    torch.testing.assert_close(processed[ACTION], expected, rtol=1e-5, atol=1e-6)
    # The gripper dim must be untouched by the relative conversion.
    torch.testing.assert_close(
        processed[ACTION][..., -1],
        (batch[ACTION][..., -1] - stats[ACTION]["mean"][-1]) / stats[ACTION]["std"][-1],
    )
    print("  [ok] relative conversion applied in physical units, before normalization")


def check_training_step(use_relative: bool, exclude_joints: list[str], label: str) -> None:
    _, policy, preprocessor, _ = _build(use_relative, exclude_joints)
    policy.train()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    batch = preprocessor(_make_batch())
    loss, loss_dict = policy.forward(batch)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
    assert torch.isfinite(loss), loss
    assert torch.isfinite(grad_norm), grad_norm
    assert "l1_loss" in loss_dict and "kld_loss" in loss_dict, loss_dict
    print(f"  [ok] {label}: train step ran (loss={loss.item():.4f}, |grad|={grad_norm:.3f})")


def check_roundtrip() -> None:
    _, policy, preprocessor, postprocessor = _build(use_relative=True, exclude_joints=["gripper"])
    policy.eval()
    batch = _make_batch(batch_size=1, state_value=2.0)
    absolute_chunk = predict_absolute_chunk(policy, preprocessor, postprocessor, batch)
    assert absolute_chunk.shape == (1, CHUNK, ACTION_DIM), absolute_chunk.shape

    # Reproduce by hand: unnormalize the raw model output and add the anchor state back.
    processed = preprocessor({k: v.clone() for k, v in batch.items()})
    raw = policy.predict_action_chunk(processed)
    stats = _make_stats(relative=True)
    unnormalized = raw * stats[ACTION]["std"] + stats[ACTION]["mean"]
    mask = torch.tensor(_make_config().build_relative_mask(ACTION_DIM), dtype=torch.float32)
    expected = unnormalized + batch[OBS_STATE].unsqueeze(1) * mask
    torch.testing.assert_close(absolute_chunk, expected, rtol=1e-4, atol=1e-5)
    print("  [ok] postprocessor round-trips relative predictions back to absolute actions")


def check_chunk_fifo_no_drift() -> None:
    """The anchor state must be written exactly once per chunk (plan §2.3 invariant)."""
    _, policy, preprocessor, postprocessor = _build(use_relative=True, exclude_joints=["gripper"])
    policy.eval()
    obs_a = _make_batch(batch_size=1, state_value=0.5)
    obs_b = _make_batch(batch_size=1, state_value=9.0)  # wildly different state
    for key in (ACTION, "action_is_pad"):
        obs_a.pop(key)
        obs_b.pop(key)

    reference = predict_absolute_chunk(
        policy, preprocessor, postprocessor, {k: v.clone() for k, v in obs_a.items()}, CHUNK
    )

    server = ChunkFIFOActionServer(policy, preprocessor, postprocessor)
    server.reset()
    served = []
    for tick in range(CHUNK):
        # Feed a *different* observation on every tick after the first. A drifting
        # implementation would re-anchor the queued actions on these states.
        obs = obs_a if tick == 0 else obs_b
        served.append(server.get_action({k: v.clone() for k, v in obs.items()}))
    served_chunk = torch.stack(served, dim=1)

    assert server.n_chunks == 1, server.n_chunks
    torch.testing.assert_close(served_chunk, reference, rtol=0, atol=0)
    # The next tick after the chunk is exhausted must re-anchor on the current state.
    server.get_action({k: v.clone() for k, v in obs_b.items()})
    assert server.n_chunks == 2, server.n_chunks
    print("  [ok] ChunkFIFOActionServer: one anchor per chunk, no drift across ticks")


def check_select_action_guard() -> None:
    _, policy, preprocessor, _ = _build(use_relative=True, exclude_joints=["gripper"])
    batch = _make_batch(batch_size=1)
    batch.pop(ACTION)
    batch.pop("action_is_pad")
    processed = preprocessor(batch)
    try:
        policy.select_action(processed)
    except NotImplementedError as e:
        assert "drift" in str(e), e
    else:
        raise AssertionError("select_action should refuse the queued path in relative mode")

    # The absolute arm (R0) keeps the normal queued behaviour.
    _, policy_abs, pre_abs, _ = _build(use_relative=False, exclude_joints=["gripper"])
    batch = _make_batch(batch_size=1)
    batch.pop(ACTION)
    batch.pop("action_is_pad")
    action = policy_abs.select_action(pre_abs(batch))
    assert action.shape == (1, ACTION_DIM), action.shape
    print("  [ok] select_action guard on relative arms; R0 unchanged")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    torch.manual_seed(0)
    print("act_delta selftest")
    check_registration()
    check_relative_mask()
    check_preprocessor_math()
    check_training_step(False, ["gripper"], "R0 absolute")
    check_training_step(True, [], "R1 all-relative")
    check_training_step(True, ["gripper"], "R2 relative-except-gripper")
    check_roundtrip()
    check_chunk_fifo_no_drift()
    check_select_action_guard()
    print("all checks passed")


if __name__ == "__main__":
    main()
