#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Derived from Patch Policy (https://github.com/gaoyuezhou/patch_policy, MIT License) and from
# openpi / lerobot's pi0 and pi0.5 ports.
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
"""Patch Policy's block-causal patch memory with pi0's state and time injection, made ablatable.

Architecture, top to bottom:

    frozen ViT  ->  (B, S, V*P, E) patch tokens          [reused from patch_policy]
                    (+ one state token per frame, if state_injection == "obs_token")
                 ->  cross-attention memory, block-causally masked
    decoder     ->  [state tokens?] + [action tokens] with a causal self-attention mask
                 ->  (B, horizon, action_dim)

`horizon = n_obs_steps + action_chunk_size - 1`, and decoder position `t` predicts the action at
observation step `min(t, n_obs_steps - 1)`: exactly Patch Policy's alignment, which is what makes
the block-causal memory mask mean anything.

Two axes are configuration rather than code:

  time_injection   concat_mlp (pi0, L1/M1) | adaln (pi0.5, L3/M2) | additive (Evo-1, L4/M4)
                   | memory_token (patch_policy, L2/M5) | none
  state_injection  suffix_token (pi0, P1) | obs_token (patch_policy, P2)
                   | action_concat (GR00T, P4) | adaln (Diffusion Policy, P5) | none (P0)

plus `use_relative_actions` (P6), which lives in the processor pipeline, not here.

Module provenance:

  REUSED
    `PatchEncoder` / `make_patch_encoder`  <- policies/patch_policy/patch_encoders.py
    `generate_mask_matrix`                 <- policies/patch_policy/modeling_patch_policy.py
    `_make_noise_scheduler`                <- policies/diffusion/modeling_diffusion.py
    `create_sinusoidal_pos_embedding`, `sample_beta` <- policies/pi0/modeling_pi0.py

  NEW
    `decoder_masks`        the block-causal masks generalised over a mixed state/action sequence
    `TimeEmbedding`        one sinusoid+MLP encoder with both frequency conventions
    `CondDecoderLayer`     pre-norm decoder block with optional adaLN-Zero / additive conditioning
    `PatchNewTransformer`  the shared trunk behind all three heads
"""

from collections import deque

import einops
import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from ..diffusion.modeling_diffusion import _make_noise_scheduler
from ..patch_policy.modeling_patch_policy import generate_mask_matrix
from ..patch_policy.patch_encoders import make_patch_encoder
from ..pi0.modeling_pi0 import create_sinusoidal_pos_embedding, sample_beta
from ..pretrained import PreTrainedPolicy
from ..utils import populate_queues
from .configuration_patch_new_policy import PatchNewPolicyConfig

# ruff: noqa: N806


# ---------------------------------------------------------------------------------------------
# Masks. `generate_mask_matrix` (the block-lower-triangular patch mask) is reused from
# patch_policy; what is new here is that the decoder sequence may hold state tokens as well as
# action tokens, and a state token must not be readable by a decoder position that predates it.
# ---------------------------------------------------------------------------------------------
def decoder_masks(
    tokens_per_frame: int,
    n_obs_steps: int,
    horizon: int,
    n_state_tokens: int,
    n_leading_memory_tokens: int,
) -> tuple[Tensor, Tensor]:
    """Self-attention and cross-attention masks, `0.0` allowed / `-inf` blocked.

    The decoder sequence is `[n_state_tokens state tokens] + [horizon action tokens]`. Each row is
    tagged with the observation step it belongs to -- state token `i` to frame `i`, action token
    `t` to frame `min(t, n_obs_steps - 1)` -- and may attend only to columns tagged with an equal
    or earlier frame. Same rule on both sides, so:

      * action token 0 sees state token 0 but not state token 1, which is one frame in its future;
      * every row keeps at least frame 0's memory block, so no row is fully masked (a fully masked
        row makes `nn.MultiheadAttention` return NaN, not an error).

    Returns:
        `(self_mask, memory_mask)` of shapes `(R, R)` and
        `(R, n_leading_memory_tokens + n_obs_steps * tokens_per_frame)` with
        `R = n_state_tokens + horizon`.
    """
    state_steps = torch.arange(n_state_tokens)
    action_steps = torch.arange(horizon).clamp(max=n_obs_steps - 1)
    row_step = torch.cat([state_steps, action_steps])
    # 0 = state token, 1 = action token. State tokens never read action tokens.
    row_group = torch.cat([torch.zeros(n_state_tokens), torch.ones(horizon)])
    R = row_step.shape[0]

    index = torch.arange(R)
    allowed = (
        (row_group[None, :] <= row_group[:, None])
        & (row_step[None, :] <= row_step[:, None])
        # Within one group the frame tag saturates (all t >= n_obs_steps-1 share a step), so
        # plain causality is still needed to stop a token reading its own future.
        & ((row_group[None, :] < row_group[:, None]) | (index[None, :] <= index[:, None]))
    )
    self_mask = torch.zeros(R, R).masked_fill(~allowed, float("-inf"))

    n_patch_tokens = n_obs_steps * tokens_per_frame
    mem_allowed = torch.zeros((R, n_leading_memory_tokens + n_patch_tokens), dtype=torch.bool)
    mem_allowed[:, :n_leading_memory_tokens] = True  # the timestep token, always visible
    for r in range(R):
        visible = (int(row_step[r]) + 1) * tokens_per_frame
        mem_allowed[r, n_leading_memory_tokens : n_leading_memory_tokens + visible] = True
    memory_mask = torch.zeros(mem_allowed.shape).masked_fill(~mem_allowed, float("-inf"))

    return self_mask, memory_mask


# ---------------------------------------------------------------------------------------------
# NEW: time embedding. Every model in the taxonomy encodes the scalar the same way -- sinusoid ->
# 2-layer MLP -- and differs only in where the result is injected, so the encoder is shared across
# all arms and only the frequency convention is switched.
# ---------------------------------------------------------------------------------------------
class TimeEmbedding(nn.Module):
    """Scalar timestep -> `(B, dim)` conditioning vector.

    `variant="ddpm"` is the standard base-10000 embedding, which needs the integer timesteps of a
    DDPM schedule. `variant="openpi"` is pi0's `min_period`/`max_period` pair, which is what
    `t in [0, 1]` flow matching needs: under base 10000 all but the top channel of a `[0, 1]` input
    is constant, so the time signal silently disappears and the loss just stops moving.
    """

    def __init__(self, dim: int, variant: str, min_period: float, max_period: float):
        super().__init__()
        self.dim = dim
        self.variant = variant
        self.min_period = min_period
        self.max_period = max_period
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.SiLU(), nn.Linear(4 * dim, dim))

    def forward(self, t: Tensor) -> Tensor:
        t = t.to(dtype=torch.float32).reshape(-1)
        if self.variant == "openpi":
            emb = create_sinusoidal_pos_embedding(
                t, self.dim, self.min_period, self.max_period, device=t.device
            )
        else:
            half = self.dim // 2
            freqs = torch.exp(
                torch.arange(half, device=t.device, dtype=torch.float32)
                * (-torch.log(torch.tensor(10000.0, device=t.device)) / (half - 1))
            )
            arg = t[:, None] * freqs[None, :]
            emb = torch.cat([arg.sin(), arg.cos()], dim=-1)
        return self.mlp(emb.to(dtype=self.mlp[0].weight.dtype))


# ---------------------------------------------------------------------------------------------
# NEW: conditioned decoder block. lerobot's `nn.TransformerDecoderLayer` (patch_policy) and
# `ACTDecoderLayer` both hardcode "no conditioning", which is precisely the axis under ablation.
# One pre-norm block covers all four time-injection variants: `adaln` drives the modulation,
# `additive` drives the per-block bias, the rest leave both off.
# ---------------------------------------------------------------------------------------------
def _modulate(x: Tensor, scale: Tensor | None, shift: Tensor | None) -> Tensor:
    return x if scale is None else x * (1 + scale) + shift


def _gate(y: Tensor, gate: Tensor | None) -> Tensor:
    return y if gate is None else y * gate


class CondDecoderLayer(nn.Module):
    """Pre-norm self-attn / cross-attn / FFN block with optional adaLN-Zero modulation.

    The 9-way modulation split (scale, shift, gate per sub-block) follows MolmoAct2; pi0.5 and
    multi_task_dit use 3- and 6-way splits of the same construction. Zero-initialising the
    modulation projection makes every gate 0, so the block starts as the identity and the
    conditioning pathway grows in during training (AdaLN-Zero) instead of scrambling the
    initialisation -- the same trick openpi, multi_task_dit and MolmoAct2 each arrived at
    independently.
    """

    def __init__(self, config: PatchNewPolicyConfig, use_adaln: bool):
        super().__init__()
        d = config.dim_model
        self.self_attn = nn.MultiheadAttention(d, config.n_heads, dropout=config.dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, config.n_heads, dropout=config.dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d, config.dim_feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, d),
        )
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.dropout = nn.Dropout(config.dropout)

        self.modulation = None
        if use_adaln:
            self.modulation = nn.Linear(d, 9 * d)
            if config.adaln_zero_init:
                nn.init.zeros_(self.modulation.weight)
                nn.init.zeros_(self.modulation.bias)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        self_mask: Tensor,
        memory_mask: Tensor,
        cond: Tensor | None = None,
        additive: Tensor | None = None,
    ) -> Tensor:
        if additive is not None:
            x = x + additive

        mods: list[Tensor | None] = [None] * 9
        if self.modulation is not None and cond is not None:
            mods = list(self.modulation(cond).chunk(9, dim=-1))
        s1, b1, g1, s2, b2, g2, s3, b3, g3 = mods

        h = _modulate(self.norm1(x), s1, b1)
        h = self.self_attn(h, h, h, attn_mask=self_mask, need_weights=False)[0]
        x = x + _gate(self.dropout(h), g1)

        h = _modulate(self.norm2(x), s2, b2)
        h = self.cross_attn(h, memory, memory, attn_mask=memory_mask, need_weights=False)[0]
        x = x + _gate(self.dropout(h), g2)

        h = _modulate(self.norm3(x), s3, b3)
        x = x + _gate(self.dropout(self.ff(h)), g3)
        return x


# ---------------------------------------------------------------------------------------------
# NEW: the shared trunk. All three heads run the same decoder over the same masked memory; they
# differ only in what fills the action slots (noisy actions vs. learned queries) and in the loss.
# ---------------------------------------------------------------------------------------------
class PatchNewTransformer(nn.Module):
    """`(patch tokens, state, timestep, x_t) -> (B, horizon, action_dim)`."""

    def __init__(self, config: PatchNewPolicyConfig, cond_dim: int, tokens_per_frame: int):
        super().__init__()
        self.config = config
        d = config.dim_model
        A = config.action_feature.shape[0]
        H = config.horizon
        S = config.n_obs_steps
        self.needs_time = config.action_head in ("diffusion", "flow")

        # --- decoder inputs -----------------------------------------------------------------
        if config.action_head == "act":
            # ACT's learned action queries: no noisy sample to embed.
            self.query_emb = nn.Embedding(H, d)
        else:
            self.input_emb = nn.Linear(A, d)
        self.pos_emb = nn.Parameter(torch.zeros(1, H, d))
        self.drop = nn.Dropout(config.p_drop_emb)

        # --- state pathway ------------------------------------------------------------------
        self.n_state_tokens = 0
        if config.state_injection in ("suffix_token", "action_concat", "adaln"):
            state_dim = config.robot_state_feature.shape[0]
            self.state_proj = nn.Linear(state_dim, d)
            if config.state_injection == "suffix_token":
                self.n_state_tokens = S
                self.state_pos_emb = nn.Parameter(torch.zeros(1, S, d))
            elif config.state_injection == "action_concat":
                self.state_concat_proj = nn.Linear(2 * d, d)
        # `obs_token` (P2) and `none` (P0) need nothing here: P2 is handled in the encoder, where
        # the state token joins its frame's block and inherits the block-causal mask unchanged.

        # --- time pathway -------------------------------------------------------------------
        self.time_emb = None
        if self.needs_time and config.time_injection != "none":
            self.time_emb = TimeEmbedding(
                d, config.sincos_variant, config.time_min_period, config.time_max_period
            )
            if config.time_injection == "concat_mlp":
                # pi0's `action_time_mlp`: cat -> Linear -> swish -> Linear.
                self.action_time_mlp_in = nn.Linear(2 * d, d)
                self.action_time_mlp_out = nn.Linear(d, d)

        use_adaln = config.state_injection == "adaln" or (
            self.time_emb is not None and config.time_injection == "adaln"
        )

        # --- memory -------------------------------------------------------------------------
        n_leading = 1 if (self.time_emb is not None and config.time_injection == "memory_token") else 0
        self.n_leading = n_leading
        self.cond_obs_emb = nn.Linear(cond_dim, d)
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, n_leading + S * tokens_per_frame, d))
        # Patch Policy runs `n_cond_layers=0`, i.e. an MLP, so the patch tokens never self-attend.
        # That is load-bearing, not a default: a self-attending memory encoder would mix frame S-1
        # into frame 0's tokens and the block causality below would read future observations.
        self.memory_encoder = nn.Sequential(nn.Linear(d, 4 * d), nn.Mish(), nn.Linear(4 * d, d))

        # --- decoder ------------------------------------------------------------------------
        self.layers = nn.ModuleList(
            [CondDecoderLayer(config, use_adaln) for _ in range(config.n_decoder_layers)]
        )
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, A)

        self_mask, memory_mask = decoder_masks(
            tokens_per_frame, S, H, self.n_state_tokens, n_leading
        )
        self.register_buffer("self_mask", self_mask, persistent=False)
        self.register_buffer("memory_mask", memory_mask, persistent=False)
        # Decoder position t is aligned with observation step min(t, S-1); the P4/P5 arms read the
        # state of that step, so they never see a frame the memory mask hides.
        self.register_buffer(
            "obs_index", torch.arange(H).clamp(max=S - 1), persistent=False
        )

        self.apply(self._init_weights)
        if use_adaln and config.adaln_zero_init:
            for layer in self.layers:  # `apply` above overwrote the zero init
                nn.init.zeros_(layer.modulation.weight)
                nn.init.zeros_(layer.modulation.bias)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(
        self,
        patch_tokens: Tensor,
        state: Tensor | None = None,
        timestep: Tensor | None = None,
        sample: Tensor | None = None,
    ) -> Tensor:
        """`patch_tokens`: `(B, S, tokens_per_frame, E)`; `state`: `(B, S, state_dim)`;
        `sample`: `(B, horizon, A)` for the denoising heads. -> `(B, horizon, A)`."""
        cfg = self.config
        B = patch_tokens.shape[0]

        time_vec = None
        if self.time_emb is not None:
            # A DDPM scheduler yields one scalar timestep for the whole batch, and on CPU even when
            # the model is on the GPU; flow matching passes a per-sample (B,) tensor.
            t = timestep if torch.is_tensor(timestep) else torch.tensor(timestep)
            t = t.to(device=patch_tokens.device).reshape(-1)
            if t.numel() == 1:
                t = t.expand(B)
            time_vec = self.time_emb(t)[:, None, :]  # (B, 1, d)

        # --- memory -------------------------------------------------------------------------
        memory = self.cond_obs_emb(einops.rearrange(patch_tokens, "b s p e -> b (s p) e"))
        if self.n_leading:
            memory = torch.cat([time_vec, memory], dim=1)
        memory = self.memory_encoder(self.drop(memory + self.cond_pos_emb))

        # --- decoder tokens -----------------------------------------------------------------
        if cfg.action_head == "act":
            x = self.query_emb.weight[None].expand(B, -1, -1)
        else:
            x = self.input_emb(sample)
        x = x + self.pos_emb

        cond = None
        if cfg.state_injection in ("action_concat", "adaln"):
            # One state per decoder position, taken from that position's own observation step.
            state_seq = self.state_proj(state[:, self.obs_index])  # (B, H, d)
            if cfg.state_injection == "action_concat":
                x = self.state_concat_proj(torch.cat([x, state_seq], dim=-1))
            else:
                cond = state_seq

        if time_vec is not None:
            if cfg.time_injection == "concat_mlp":
                x = self.action_time_mlp_out(
                    F.silu(self.action_time_mlp_in(torch.cat([x, time_vec.expand_as(x)], dim=-1)))
                )
            elif cfg.time_injection == "adaln":
                cond = time_vec if cond is None else cond + time_vec

        x = self.drop(x)
        if self.n_state_tokens:
            state_tokens = self.state_proj(state) + self.state_pos_emb
            x = torch.cat([state_tokens, x], dim=1)
            if cond is not None:  # only reachable via time-only adaLN, which broadcasts
                cond = cond.expand(-1, x.shape[1], -1) if cond.shape[1] != 1 else cond

        additive = time_vec if (time_vec is not None and cfg.time_injection == "additive") else None
        for layer in self.layers:
            x = layer(x, memory, self.self_mask, self.memory_mask, cond=cond, additive=additive)

        return self.head(self.ln_f(x))[:, self.n_state_tokens :]


# ---------------------------------------------------------------------------------------------
# The policy.
# ---------------------------------------------------------------------------------------------
class PatchNewPolicyModel(nn.Module):
    """Frozen patch encoder -> block-causal memory -> one conditioned decoder -> one of three heads."""

    def __init__(self, config: PatchNewPolicyConfig):
        super().__init__()
        self.config = config
        self.num_images = len(config.image_features)

        self.encoder = make_patch_encoder(config.encoder_preset, config.resize_shape)
        if config.freeze_vision_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.n_patches_per_camera = self._measure_n_patches()
        self.feature_dim = self.encoder.output_dim
        self.tokens_per_frame = self.n_patches_per_camera * self.num_images + int(
            config.state_injection == "obs_token"
        )

        if config.state_injection == "obs_token":
            self.obs_state_proj = nn.Linear(config.robot_state_feature.shape[0], self.feature_dim)

        self.trunk = PatchNewTransformer(config, self.feature_dim, self.tokens_per_frame)

        if config.action_head == "diffusion":
            self.noise_scheduler = _make_noise_scheduler(
                config.noise_scheduler_type,
                num_train_timesteps=config.num_train_timesteps,
                beta_start=config.beta_start,
                beta_end=config.beta_end,
                beta_schedule=config.beta_schedule,
                clip_sample=config.clip_sample,
                clip_sample_range=config.clip_sample_range,
                prediction_type=config.prediction_type,
            )
            self.num_inference_steps = config.num_inference_steps or config.num_train_timesteps

    def train(self, mode: bool = True):
        super().train(mode)
        # Without this the frozen encoder returns to training mode on every epoch: ResNet-18's
        # BatchNorm would update its running statistics and the ViTs would apply dropout.
        if self.config.freeze_vision_encoder:
            self.encoder.eval()
        return self

    @torch.no_grad()
    def _measure_n_patches(self) -> int:
        """Count the encoder's patch tokens with a dry run, so a wrong constant cannot misalign
        the block-causal mask with the token stream."""
        if self.config.n_patches_override is not None:
            return self.config.n_patches_override
        c = next(iter(self.config.image_features.values())).shape[0]
        dummy = torch.zeros(1, c, *self.config.resize_shape)
        return self.encoder(dummy).shape[-2]

    def encode_observations(self, batch: dict[str, Tensor]) -> Tensor:
        """-> `(B, n_obs_steps, tokens_per_frame, feature_dim)`."""
        batch_size, n_obs_steps = batch[OBS_IMAGES].shape[:2]
        images = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")

        with torch.set_grad_enabled(not self.config.freeze_vision_encoder):
            patch_tokens = self.encoder(images)  # ((b s n), P, E)
        patch_tokens = einops.rearrange(
            patch_tokens, "(b s n) p e -> b s (n p) e", b=batch_size, s=n_obs_steps, n=self.num_images
        )

        if self.config.state_injection == "obs_token":
            # P2: intra-frame, so the block-causal mask needs no change.
            patch_tokens = torch.cat(
                [patch_tokens, self.obs_state_proj(batch[OBS_STATE]).unsqueeze(2)], dim=2
            )
        return patch_tokens

    def _sample_flow_time(self, bsize: int, device) -> Tensor:
        cfg = self.config
        t = sample_beta(cfg.time_sampling_beta_alpha, cfg.time_sampling_beta_beta, bsize, device)
        return t * cfg.time_sampling_scale + cfg.time_sampling_offset

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        cfg = self.config
        patch_tokens = self.encode_observations(batch)
        state = batch.get(OBS_STATE)
        actions = batch[ACTION]

        if cfg.action_head == "act":
            pred = self.trunk(patch_tokens, state=state)
            loss = F.l1_loss(pred, actions)
            return loss, {"l1_loss": loss.item()}

        noise = torch.randn_like(actions)

        if cfg.action_head == "diffusion":
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, (actions.shape[0],), device=actions.device
            ).long()
            x_t = self.noise_scheduler.add_noise(actions, noise, timesteps)
            pred = self.trunk(patch_tokens, state=state, timestep=timesteps, sample=x_t)
            target = noise if cfg.prediction_type == "epsilon" else actions
            loss = F.mse_loss(pred, target)
            return loss, {"mse_loss": loss.item()}

        # Flow matching, openpi's parameterisation: x_t is a linear interpolation between noise
        # and data (not a variance-preserving "added noise"), and the regression target is the
        # constant velocity of that path. One t per sample, one forward pass -- no iteration.
        t = self._sample_flow_time(actions.shape[0], actions.device)
        t_exp = t[:, None, None]
        x_t = t_exp * noise + (1 - t_exp) * actions
        u_t = noise - actions
        pred = self.trunk(patch_tokens, state=state, timestep=t, sample=x_t)
        loss = F.mse_loss(pred, u_t)
        return loss, {"flow_loss": loss.item()}

    @torch.no_grad()
    def predict(self, batch: dict[str, Tensor]) -> Tensor:
        """-> `(B, action_chunk_size, A)`, the chunk anchored at the newest observation."""
        cfg = self.config
        patch_tokens = self.encode_observations(batch)
        state = batch.get(OBS_STATE)
        B = patch_tokens.shape[0]
        start = cfg.n_obs_steps - 1

        if cfg.action_head == "act":
            pred = self.trunk(patch_tokens, state=state)
            return pred[:, start : start + cfg.action_chunk_size]

        shape = (B, cfg.horizon, cfg.action_feature.shape[0])
        x = torch.randn(shape, device=patch_tokens.device, dtype=patch_tokens.dtype)

        if cfg.action_head == "diffusion":
            self.noise_scheduler.set_timesteps(self.num_inference_steps)
            for t in self.noise_scheduler.timesteps:
                model_output = self.trunk(patch_tokens, state=state, timestep=t, sample=x)
                x = self.noise_scheduler.step(model_output, t, x).prev_sample
        else:
            dt = -1.0 / cfg.num_flow_steps
            for step in range(cfg.num_flow_steps):
                t = torch.full((B,), 1.0 + step * dt, device=x.device)
                x = x + dt * self.trunk(patch_tokens, state=state, timestep=t, sample=x)

        return x[:, start : start + cfg.action_chunk_size]


class PatchNewPolicy(PreTrainedPolicy):
    """Patch Policy's block-causal patch memory with configurable pi0-style state and time injection."""

    config_class = PatchNewPolicyConfig
    name = "patch_new_policy"

    def __init__(self, config: PatchNewPolicyConfig | None = None, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = PatchNewPolicyModel(config)
        self.reset()

    def get_optim_params(self) -> dict:
        """nanoGPT's rule, as in Patch Policy's `configure_optimizers`: decay the weights of
        linear/conv layers only. Norm weights, biases, embedding tables and bare position
        `nn.Parameter`s land in the no-decay group."""
        whitelist = (nn.Linear, nn.Conv1d, nn.Conv2d)
        decay, no_decay = [], []
        # `recurse=False` visits each parameter exactly once, so none can land in two groups.
        for module in self.model.modules():
            for name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                if name.endswith("weight") and isinstance(module, whitelist):
                    decay.append(param)
                else:
                    no_decay.append(param)
        return [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}]

    def reset(self):
        self._queues = {
            OBS_IMAGES: deque(maxlen=self.config.n_obs_steps),
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        return self.model.predict(batch)[:, : self.config.n_action_steps]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        batch = dict(batch)
        batch.pop(ACTION, None)
        batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)

        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1))
        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        batch = dict(batch)
        batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        return self.model(batch)
