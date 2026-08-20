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

"""Patch Policy tests.

Run:
    python -m pytest tests/policies/patch_policy/test_patch_policy.py -v
    python tests/policies/patch_policy/test_patch_policy.py     # same checks, no pytest

The real backbones are 100M-1B parameter downloads, so every test here swaps in a random-weight
stub encoder with the same interface. What is under test is the part this port actually wrote:
the block-causal masks, the token layout, and the three heads' shapes and gradients.
"""

import pytest
import torch
from torch import nn

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.policies.patch_policy.configuration_patch_policy import (
    PATCH_ENCODER_PRESETS,
    PatchPolicyConfig,
)
from lerobot.policies.patch_policy.modeling_patch_policy import (
    PatchPolicy,
    PatchPolicyModel,
    block_causal_memory_mask,
    generate_mask_matrix,
)
from lerobot.policies.patch_policy.patch_encoders import PatchEncoder, make_patch_encoder
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE

N_PATCHES = 4
FEATURE_DIM = 16


class StubEncoder(PatchEncoder):
    """Same contract as the real encoders: (..., C, H, W) in [0, 1] -> (..., P, E)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.proj = nn.Linear(3 * 8 * 8, N_PATCHES * FEATURE_DIM)

    @property
    def output_dim(self) -> int:
        return FEATURE_DIM

    def encode(self, x):
        x = nn.functional.adaptive_avg_pool2d(x, (8, 8)).flatten(1)
        return self.proj(x).view(-1, N_PATCHES, FEATURE_DIM)


def make_config(action_head: str, **overrides) -> PatchPolicyConfig:
    config = PatchPolicyConfig(
        n_obs_steps=3,
        action_chunk_size=2,
        n_action_steps=2,
        gpt_n_layer=2,
        gpt_n_head=2,
        gpt_hidden_dim=32,
        gpt_output_dim=32,
        diffusion_n_layer=1,
        diffusion_n_head=2,
        diffusion_hidden_dim=32,
        num_train_timesteps=4,
        n_decoder_layers=1,
        dim_model=32,
        n_heads=2,
        dim_feedforward=64,
        n_vqvae_training_steps=2,
        vqvae_embedding_dim=32,
        action_head=action_head,
        resize_shape=(32, 32),
        **overrides,
    )
    config.input_features = {
        OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
    }
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))}
    config.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.MIN_MAX,
        "ACTION": NormalizationMode.MIN_MAX,
    }
    return config


def make_policy(config: PatchPolicyConfig) -> PatchPolicy:
    policy = PatchPolicy.__new__(PatchPolicy)
    from lerobot.policies.patch_policy import modeling_patch_policy as mod

    original = mod.make_patch_encoder
    mod.make_patch_encoder = lambda preset, resize_shape: StubEncoder(resize_shape=resize_shape)
    try:
        PatchPolicy.__init__(policy, config)
    finally:
        mod.make_patch_encoder = original
    return policy


def make_batch(config: PatchPolicyConfig, batch_size: int = 2) -> dict:
    horizon = config.n_obs_steps + config.action_chunk_size - 1
    return {
        OBS_IMAGE: torch.rand(batch_size, config.n_obs_steps, 3, 32, 32),
        OBS_STATE: torch.randn(batch_size, config.n_obs_steps, 4),
        ACTION: torch.randn(batch_size, horizon, 2),
    }


# --------------------------------------------------------------------------------------------
# The mask. This is the paper's entire contribution, so it gets checked directly.
# --------------------------------------------------------------------------------------------
def test_block_causal_mask_is_bidirectional_within_a_frame_and_causal_across_frames():
    p, t = 3, 4
    mask = generate_mask_matrix(p, t)[0, 0]
    assert mask.shape == (p * t, p * t)

    for qi in range(t):
        for ki in range(t):
            block = mask[qi * p : (qi + 1) * p, ki * p : (ki + 1) * p]
            if ki <= qi:
                # Every patch of frame ki is visible to every patch of frame qi, in both
                # directions when ki == qi. This is what a token-causal mask would forbid.
                assert block.all(), f"frame {qi} cannot see frame {ki}"
            else:
                assert not block.any(), f"frame {qi} leaks future frame {ki}"

    # Sanity: a token-causal mask would fail the intra-frame half of the check above.
    token_causal = torch.tril(torch.ones(p * t, p * t))
    assert not token_causal[: p, : p].all()


def test_memory_mask_gives_each_decoder_step_exactly_its_own_past():
    p, t, horizon = 3, 4, 6
    mask = block_causal_memory_mask(p, t, horizon, n_leading_tokens=1)
    assert mask.shape == (horizon, 1 + t * p)

    allowed = torch.isfinite(mask)
    assert allowed[:, 0].all(), "the leading (timestep) token must always be visible"
    for step in range(horizon):
        # Decoder position i reads frames 0..i, clamped to the last observation.
        expected = (min(step, t - 1) + 1) * p
        assert allowed[step, 1:].sum() == expected
        assert allowed[step, 1 : 1 + expected].all()
        assert not allowed[step, 1 + expected :].any()


# --------------------------------------------------------------------------------------------
# Token layout.
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("use_robot_state", [False, True])
def test_token_layout_stacks_cameras_along_the_patch_dim(use_robot_state):
    config = make_config("act", use_robot_state=use_robot_state)
    policy = make_policy(config)
    expected = N_PATCHES + int(use_robot_state)
    assert policy.model.tokens_per_frame == expected

    batch = make_batch(config)
    batch["observation.images"] = batch[OBS_IMAGE].unsqueeze(2)
    tokens = policy.model.encode_observations(batch)
    assert tokens.shape == (2, config.n_obs_steps, expected, FEATURE_DIM)


def test_unpack_actions_builds_one_chunk_per_observation_step():
    actions = torch.arange(6).float().view(1, 6, 1)
    unpacked = PatchPolicyModel.unpack_actions(actions, action_chunk_size=3)
    assert unpacked.shape == (1, 4, 3, 1)
    assert torch.equal(unpacked[0, 0, :, 0], torch.tensor([0.0, 1.0, 2.0]))
    assert torch.equal(unpacked[0, 3, :, 0], torch.tensor([3.0, 4.0, 5.0]))


def test_encoder_factory_builds_every_preset_shape_correctly():
    """The pooled presets must still emit a (dummy) patch dim, so downstream code never branches."""
    encoder = make_patch_encoder(
        PATCH_ENCODER_PRESETS["resnet18_random"], resize_shape=(64, 64)
    )  # random weights: no download
    out = encoder(torch.rand(2, 5, 3, 64, 64))
    assert out.shape == (2, 5, 1, 512), "pooled encoders must keep a patch dim of 1"


def test_policy_is_registered_with_the_factory():
    assert isinstance(make_policy_config("patch_policy"), PatchPolicyConfig)
    assert get_policy_class("patch_policy") is PatchPolicy


# --------------------------------------------------------------------------------------------
# The three heads.
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("action_head", ["vqbet", "diffusion", "act"])
def test_forward_produces_a_finite_loss_with_gradients(action_head):
    torch.manual_seed(0)
    config = make_config(action_head)
    policy = make_policy(config)
    batch = make_batch(config)

    if action_head == "vqbet":
        # Phase 1: the VQ-VAE trains alone until `n_vqvae_training_steps` is reached.
        for _ in range(config.n_vqvae_training_steps):
            loss, out = policy.forward(batch)
            assert "recon_l1_error" in out
        assert policy.model.head.vqvae_model.discretized.item()

    loss, _ = policy.forward(batch)
    assert loss.isfinite(), f"{action_head} produced a non-finite loss"

    loss.backward()
    trained = [p for p in policy.parameters() if p.requires_grad and p.grad is not None]
    assert trained, f"{action_head} produced no gradients"
    assert all(p.grad.isfinite().all() for p in trained)


@pytest.mark.parametrize("action_head", ["vqbet", "diffusion", "act"])
def test_select_action_returns_one_action_per_call(action_head):
    torch.manual_seed(0)
    config = make_config(action_head)
    policy = make_policy(config)
    if action_head == "vqbet":
        policy.model.head.vqvae_model.discretized.fill_(True)
    policy.eval()

    obs = {
        OBS_IMAGE: torch.rand(1, 3, 32, 32),
        OBS_STATE: torch.randn(1, 4),
    }
    for _ in range(config.n_action_steps + 1):  # forces one queue refill
        action = policy.select_action(dict(obs))
        assert action.shape == (1, 2)
        assert action.isfinite().all()


def test_vision_encoder_is_frozen_by_default():
    config = make_config("act")
    policy = make_policy(config)
    assert all(not p.requires_grad for p in policy.model.encoder.parameters())
    # ...and the frozen parameters stay out of the optimizer.
    encoder_ids = {id(p) for p in policy.model.encoder.parameters()}
    for group in policy.get_optim_params():
        assert not any(id(p) in encoder_ids for p in group["params"])


def test_frozen_encoder_stays_in_eval_mode_after_train():
    policy = make_policy(make_config("act"))
    policy.train()
    assert not policy.model.encoder.training, "policy.train() must not un-freeze the encoder"
    assert policy.model.head.training, "the trainable head must follow policy.train()"


def test_optimizer_groups_contain_no_duplicate_parameters():
    for action_head in ("vqbet", "diffusion", "act"):
        policy = make_policy(make_config(action_head))
        seen = [id(p) for group in policy.get_optim_params() for p in group["params"]]
        assert len(seen) == len(set(seen)), f"{action_head}: a parameter is in two groups"
        expected = {id(p) for p in policy.parameters() if p.requires_grad}
        assert set(seen) == expected, f"{action_head}: optimizer groups miss trainable parameters"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
