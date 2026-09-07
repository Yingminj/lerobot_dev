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

"""patch_new_policy tests.

Run:
    python -m pytest tests/policies/patch_new_policy/test_patch_new_policy.py -v
    python tests/policies/patch_new_policy/test_patch_new_policy.py     # same checks, no pytest

The real backbones are 100M-1B parameter downloads, so a random-weight stub encoder with the same
interface stands in. What is under test is what this policy adds over `patch_policy`: the masks
once the decoder sequence holds state tokens, the 3 x 5 x 4 injection grid, and the one silent
failure mode in the taxonomy -- base-10000 sinusoids under a [0, 1] flow-matching time.
"""

import itertools

import pytest
import torch
from torch import nn

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.policies.patch_new_policy.configuration_patch_new_policy import (
    ACTION_HEADS,
    STATE_INJECTIONS,
    TIME_INJECTIONS,
    PatchNewPolicyConfig,
)
from lerobot.policies.patch_new_policy.modeling_patch_new_policy import (
    PatchNewPolicy,
    TimeEmbedding,
    decoder_masks,
)
from lerobot.policies.patch_policy.patch_encoders import PatchEncoder
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


def make_config(action_head: str = "flow", **overrides) -> PatchNewPolicyConfig:
    config = PatchNewPolicyConfig(
        n_obs_steps=3,
        action_chunk_size=2,
        n_action_steps=2,
        action_head=action_head,
        dim_model=32,
        n_heads=2,
        dim_feedforward=64,
        n_decoder_layers=2,
        num_train_timesteps=4,
        num_flow_steps=2,
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


def make_policy(config: PatchNewPolicyConfig) -> PatchNewPolicy:
    policy = PatchNewPolicy.__new__(PatchNewPolicy)
    from lerobot.policies.patch_new_policy import modeling_patch_new_policy as mod

    original = mod.make_patch_encoder
    mod.make_patch_encoder = lambda preset, resize_shape: StubEncoder(resize_shape=resize_shape)
    try:
        PatchNewPolicy.__init__(policy, config)
    finally:
        mod.make_patch_encoder = original
    return policy


def make_batch(config: PatchNewPolicyConfig, batch_size: int = 2) -> dict:
    return {
        OBS_IMAGE: torch.rand(batch_size, config.n_obs_steps, 3, 32, 32),
        OBS_STATE: torch.randn(batch_size, config.n_obs_steps, 4),
        ACTION: torch.randn(batch_size, config.horizon, 2),
    }


# --------------------------------------------------------------------------------------------
# The masks. Block causality is the whole point of the architecture, and adding state tokens to
# the decoder sequence is the one change here that can break it.
# --------------------------------------------------------------------------------------------
def test_memory_mask_gives_each_decoder_step_exactly_its_own_past():
    p, s, horizon = 3, 4, 6
    _, memory = decoder_masks(p, s, horizon, n_state_tokens=0, n_leading_memory_tokens=1)
    assert memory.shape == (horizon, 1 + s * p)

    allowed = torch.isfinite(memory)
    assert allowed[:, 0].all(), "the leading (timestep) token must always be visible"
    for step in range(horizon):
        expected = (min(step, s - 1) + 1) * p
        assert allowed[step, 1 : 1 + expected].all()
        assert not allowed[step, 1 + expected :].any()


def test_state_tokens_never_leak_a_future_observation():
    p, s, horizon = 3, 4, 6
    self_mask, memory = decoder_masks(p, s, horizon, n_state_tokens=s, n_leading_memory_tokens=0)
    assert self_mask.shape == (s + horizon, s + horizon)

    allowed = torch.isfinite(self_mask)
    for t in range(horizon):
        row = s + t
        step = min(t, s - 1)
        # This is the trap: one shared "latest state" token would be visible to action token 0,
        # which is aligned with the oldest frame, laundering a future observation into it.
        assert allowed[row, : step + 1].all(), "action token cannot see its own past states"
        assert not allowed[row, step + 1 : s].any(), f"action {t} leaks the state of a future frame"
        assert allowed[row, s : s + t + 1].all()
        assert not allowed[row, s + t + 1 :].any(), "action tokens must stay causal"
        # ...and its memory access is unchanged by the presence of the state tokens.
        assert torch.isfinite(memory[row]).sum() == (step + 1) * p

    for i in range(s):
        assert not allowed[i, s:].any(), "a state token must not read action tokens"
        assert not allowed[i, i + 1 : s].any(), "a state token must not read a later state"

    # No row may be fully masked: nn.MultiheadAttention returns NaN for that, it does not raise.
    assert allowed.any(dim=1).all()
    assert torch.isfinite(memory).any(dim=1).all()


# --------------------------------------------------------------------------------------------
# The silent bug of the taxonomy: base-10000 sinusoids under flow matching's t in [0, 1].
# --------------------------------------------------------------------------------------------
def test_ddpm_sinusoid_collapses_on_flow_time_and_the_openpi_one_does_not():
    t = torch.linspace(0.01, 0.99, 16)
    ddpm = TimeEmbedding(64, "ddpm", 4e-3, 4.0)
    openpi = TimeEmbedding(64, "openpi", 4e-3, 4.0)

    with torch.no_grad():
        d = ddpm(t)
        o = openpi(t)
    # Both produce finite output; the difference is how much of it varies with t.
    assert d.isfinite().all() and o.isfinite().all()
    assert o.std(dim=0).mean() > d.std(dim=0).mean(), (
        "openpi's period range must carry more of the [0, 1] time signal than base 10000"
    )


def test_sincos_variant_defaults_follow_the_head():
    assert make_config("flow").sincos_variant == "openpi"
    assert make_config("diffusion").sincos_variant == "ddpm"
    assert make_config("flow", time_sincos="ddpm").sincos_variant == "ddpm"


# --------------------------------------------------------------------------------------------
# The ablation grid.
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("action_head", "state_injection", "time_injection"),
    list(itertools.product(ACTION_HEADS, STATE_INJECTIONS, TIME_INJECTIONS)),
)
def test_every_injection_combination_trains(action_head, state_injection, time_injection):
    torch.manual_seed(0)
    config = make_config(action_head, state_injection=state_injection, time_injection=time_injection)
    policy = make_policy(config)
    batch = make_batch(config)

    loss, out = policy.forward(batch)
    assert loss.isfinite(), f"{action_head}/{state_injection}/{time_injection}: non-finite loss"
    assert out

    loss.backward()
    trained = [p for p in policy.parameters() if p.requires_grad and p.grad is not None]
    assert trained, "no gradients"
    assert all(p.grad.isfinite().all() for p in trained)


@pytest.mark.parametrize("state_injection", STATE_INJECTIONS)
def test_the_state_actually_reaches_the_output(state_injection):
    """A knob that changes nothing is the failure mode an ablation cannot detect on its own."""
    torch.manual_seed(0)
    # AdaLN-Zero starts as an identity block on purpose, so the P5 arm has no effect at step 0;
    # `test_adaln_zero_starts_as_an_identity_block` covers that case and its gradient path.
    config = make_config("act", state_injection=state_injection, adaln_zero_init=False)
    policy = make_policy(config).eval()
    batch = make_batch(config)
    batch["observation.images"] = batch[OBS_IMAGE].unsqueeze(2)

    with torch.no_grad():
        a = policy.model.forward({**batch, ACTION: batch[ACTION]})[0]
        b = policy.model.forward({**batch, OBS_STATE: batch[OBS_STATE] + 5.0})[0]
    if state_injection == "none":
        assert torch.equal(a, b), "P0 must ignore the state"
    else:
        assert not torch.allclose(a, b), f"{state_injection}: the state is not reaching the loss"


@pytest.mark.parametrize("action_head", ACTION_HEADS)
def test_select_action_returns_one_action_per_call(action_head):
    torch.manual_seed(0)
    config = make_config(action_head)
    policy = make_policy(config).eval()

    obs = {OBS_IMAGE: torch.rand(1, 3, 32, 32), OBS_STATE: torch.randn(1, 4)}
    for _ in range(config.n_action_steps + 1):  # forces one queue refill
        action = policy.select_action(dict(obs))
        assert action.shape == (1, 2)
        assert action.isfinite().all()


def test_adaln_zero_starts_as_an_identity_block_but_still_gets_gradients():
    """The gates are 0 at init, so the block is the identity; the modulation must still train,
    otherwise the conditioning pathway can never grow in and the arm is silently a no-op."""
    torch.manual_seed(0)
    config = make_config("flow", time_injection="adaln")
    policy = make_policy(config)
    for layer in policy.model.trunk.layers:
        assert layer.modulation is not None
        assert torch.count_nonzero(layer.modulation.weight) == 0
        assert torch.count_nonzero(layer.modulation.bias) == 0

    policy.forward(make_batch(config))[0].backward()
    for layer in policy.model.trunk.layers:
        assert layer.modulation.weight.grad is not None
        assert torch.count_nonzero(layer.modulation.weight.grad) > 0


@pytest.mark.parametrize("action_head", ["diffusion", "flow"])
def test_a_scalar_cpu_timestep_reaches_the_model(action_head):
    """A DDPM scheduler hands back one 0-d timestep for the whole batch, and keeps it on the CPU
    even when the model is on the GPU. Both have to survive the time embedding."""
    torch.manual_seed(0)
    config = make_config(action_head)
    policy = make_policy(config).eval()
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        model = policy.model.to(device)
        patch_tokens = torch.randn(
            2, config.n_obs_steps, model.tokens_per_frame, model.feature_dim, device=device
        )
        state = torch.randn(2, config.n_obs_steps, 4, device=device)
        sample = torch.randn(2, config.horizon, 2, device=device)
        with torch.no_grad():
            out = model.trunk(patch_tokens, state=state, timestep=torch.tensor(3), sample=sample)
        assert out.shape == (2, config.horizon, 2)
        assert out.isfinite().all()
    policy.model.to("cpu")


def test_vision_encoder_is_frozen_and_out_of_the_optimizer():
    policy = make_policy(make_config("flow"))
    assert all(not p.requires_grad for p in policy.model.encoder.parameters())
    encoder_ids = {id(p) for p in policy.model.encoder.parameters()}
    for group in policy.get_optim_params():
        assert not any(id(p) in encoder_ids for p in group["params"])

    policy.train()
    assert not policy.model.encoder.training, "policy.train() must not un-freeze the encoder"
    assert policy.model.trunk.training


def test_optimizer_groups_are_a_partition_of_the_trainable_parameters():
    for action_head in ACTION_HEADS:
        policy = make_policy(make_config(action_head))
        seen = [id(p) for group in policy.get_optim_params() for p in group["params"]]
        assert len(seen) == len(set(seen)), f"{action_head}: a parameter is in two groups"
        expected = {id(p) for p in policy.parameters() if p.requires_grad}
        assert set(seen) == expected, f"{action_head}: optimizer groups miss trainable parameters"


def test_policy_is_registered_with_the_factory():
    assert isinstance(make_policy_config("patch_new_policy"), PatchNewPolicyConfig)
    assert get_policy_class("patch_new_policy") is PatchNewPolicy


def test_vqbet_head_is_gone():
    with pytest.raises(ValueError):
        make_config("vqbet")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
