#!/usr/bin/env python

# Copyright 2026 Gaoyue Zhou and Zichen Jeff Cui and Ada Langford and Bowen Tan
# and Yann LeCun and Lerrel Pinto and The HuggingFace Inc. team. All rights reserved.
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
"""Patch Policy: efficient embodied control via dense visual representations.

Paper: https://huggingface.co/papers/2607.18236
Reference implementation: https://github.com/gaoyuezhou/patch_policy

The whole idea is one attention mask. A frozen ViT emits P patch tokens per frame; the policy
consumes all of them instead of a pooled summary, and a *block-causal* mask makes attention
bidirectional inside a frame (there is no temporal order among the patches of one image) while
staying strictly causal across frames (no peeking at future observations).

Module provenance:

  REUSED from lerobot
    `GPT`, `ResidualVQ`               <- policies/vqbet/vqbet_utils.py
    `VqVae`, `VQBeTHead`, `MLP`       <- policies/vqbet/modeling_vqbet.py
    `ACTDecoder`, `ACTDecoderLayer`   <- policies/act/modeling_act.py
    `_make_noise_scheduler`           <- policies/diffusion/modeling_diffusion.py

  NEW (no lerobot equivalent; see the banner comment on each)
    `generate_mask_matrix`            <- reference models/vq_behavior_transformer/gpt.py
    `block_causal_memory_mask`        <- reference models/diffusion_policy/diffusion_policy.py
    `BlockCausalGPT`                  <- reference models/vq_behavior_transformer/gpt.py
    `TransformerForDiffusion`         <- reference models/diffusion_policy/diffusion_policy.py
    `PatchACTHead`                    <- no counterpart in the reference
    `PatchEncoder` and subclasses     <- patch_encoders.py
"""

from collections import deque

import einops
import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from ..act.modeling_act import ACTDecoder
from ..diffusion.modeling_diffusion import _make_noise_scheduler
from ..pretrained import PreTrainedPolicy
from ..utils import populate_queues
from ..vqbet.modeling_vqbet import MLP, VQBeTHead
from ..vqbet.vqbet_utils import GPT
from .configuration_patch_policy import PatchPolicyConfig
from .patch_encoders import make_patch_encoder

# ruff: noqa: N806


# ---------------------------------------------------------------------------------------------
# NEW: block-causal attention masks. Verbatim from `models/vq_behavior_transformer/gpt.py` and
# `models/diffusion_policy/diffusion_policy.py` of the reference. lerobot has no equivalent —
# every transformer policy in the tree is either fully causal per token or fully bidirectional.
# ---------------------------------------------------------------------------------------------
def generate_mask_matrix(npatch: int, nwindow: int) -> Tensor:
    """Block-lower-triangular mask of shape `(1, 1, npatch*nwindow, npatch*nwindow)`.

    `1` means "may attend". Within a frame's `npatch x npatch` block the mask is all ones (full
    bidirectional spatial attention); across frames it is lower block-triangular (temporal
    causality). This is the paper's sole architectural contribution.

    Reference: `generate_mask_matrix`, models/vq_behavior_transformer/gpt.py.
    """
    zeros = torch.zeros(npatch, npatch)
    ones = torch.ones(npatch, npatch)
    rows = []
    for i in range(nwindow):
        row = torch.cat([ones] * (i + 1) + [zeros] * (nwindow - i - 1), dim=1)
        rows.append(row)
    mask = torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)
    return mask


def block_causal_memory_mask(
    npatch: int, n_obs_steps: int, horizon: int, n_leading_tokens: int = 0
) -> Tensor:
    """Additive cross-attention mask, `0.0` where allowed and `-inf` where blocked.

    Shape `(horizon, n_leading_tokens + n_obs_steps * npatch)`. Decoder position `t` predicts the
    action at observation step `t` (clamped to the last one), so it may read the patch tokens of
    frames `0..t` and no further. `n_leading_tokens` accounts for memory tokens that sit in front
    of the patches and are always visible — the diffusion head's timestep token.

    Reference: `TransformerForDiffusion.__init__`, models/diffusion_policy/diffusion_policy.py.
    """
    n_patch_tokens = n_obs_steps * npatch
    patch_block = generate_mask_matrix(npatch, n_obs_steps).squeeze(0).squeeze(0).bool()

    allowed = torch.zeros((horizon, n_leading_tokens + n_patch_tokens), dtype=torch.bool)
    allowed[:, :n_leading_tokens] = True
    for t_idx in range(horizon):
        obs_step = min(t_idx, n_obs_steps - 1)
        rows_to_use = (obs_step + 1) * npatch
        allowed[t_idx, n_leading_tokens:] = patch_block[:rows_to_use, :].any(dim=0)

    return torch.zeros_like(allowed, dtype=torch.float32).masked_fill(~allowed, float("-inf"))


def causal_mask(size: int) -> Tensor:
    """Additive `(size, size)` causal mask, `0.0` allowed / `-inf` blocked."""
    blocked = torch.triu(torch.ones(size, size, dtype=torch.bool), diagonal=1)
    return torch.zeros(size, size).masked_fill(blocked, float("-inf"))


class _MaskedAttention(nn.Module):
    """Wraps an `nn.MultiheadAttention` so it always applies a fixed `attn_mask`.

    `ACTDecoderLayer` calls its attention modules without a mask. Rather than fork its `forward`,
    the two attention submodules are swapped for this wrapper, which keeps `ACTDecoder` and
    `ACTDecoderLayer` reused verbatim.
    """

    def __init__(self, attn: nn.MultiheadAttention, mask: Tensor):
        super().__init__()
        self.attn = attn
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, query, key, value, **kwargs):
        return self.attn(query, key, value, attn_mask=self.mask, **kwargs)


# ---------------------------------------------------------------------------------------------
# NEW: block-causal GPT trunk. Subclasses lerobot's `GPT` (policies/vqbet/vqbet_utils.py), which
# is already the same nanoGPT the reference forked; only the mask, the position table and the
# (B, T, P, D) forward differ.
# ---------------------------------------------------------------------------------------------
class BlockCausalGPT(GPT):
    """nanoGPT trunk over `(B, T, P, D)` spatio-temporal patch tokens.

    Two changes to the parent:
      1. the token-causal `tril` mask becomes the block-causal mask, shared by every layer;
      2. positions are learned over `block_size * n_patches` slots rather than `block_size`.

    Reference: `GPT`, models/vq_behavior_transformer/gpt.py.
    """

    def __init__(self, config: PatchPolicyConfig, n_patches: int):
        super().__init__(config)
        self.n_patches = n_patches

        # One learned embedding per (frame, patch) slot, as in the reference.
        self.transformer.wpe = nn.Embedding(config.gpt_block_size * n_patches, config.gpt_hidden_dim)
        self._init_weights(self.transformer.wpe)

        # Bool rather than the reference's float: identical under `bias == 0`, and 4x smaller,
        # which matters because this is a (T*P)^2 matrix — 2560^2 at T=10, P=256.
        # One tensor shared by every layer, and non-persistent so it stays out of checkpoints.
        mask = generate_mask_matrix(n_patches, config.gpt_block_size).bool()
        for block in self.transformer.h:
            block.attn.register_buffer("bias", mask, persistent=False)

    def forward(self, input: Tensor, targets: Tensor | None = None) -> Tensor:
        """`(B, T, P, D)` -> `(B, T, gpt_output_dim)`, reading out the last token of each frame."""
        b, t, p, d = input.size()
        assert t <= self.config.gpt_block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.gpt_block_size}"
        )
        assert p == self.n_patches, f"Expected {self.n_patches} tokens per frame, got {p}"

        pos = torch.arange(0, t * p, dtype=torch.long, device=input.device).unsqueeze(0)
        tok_emb = self.transformer.wte(input)
        pos_emb = einops.rearrange(self.transformer.wpe(pos), "b (t p) d -> b t p d", t=t)
        x = self.transformer.drop(tok_emb + pos_emb)

        x = einops.rearrange(x, "b t p d -> b (t p) d")
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        logits = einops.rearrange(logits, "b (t p) d -> b t p d", t=t)
        # The last token of a frame's block has attended to every token in that frame and to all
        # previous frames, so it is the per-frame summary the action head reads.
        return logits[:, :, -1]


# ---------------------------------------------------------------------------------------------
# NEW: transformer denoiser. lerobot's diffusion policy is a 1D UNet with FiLM conditioning on a
# single pooled vector, which cannot take a token sequence as memory, let alone mask it.
# Ported from `TransformerForDiffusion`, models/diffusion_policy/diffusion_policy.py — only the
# branch the reference actually instantiates (time_as_cond, obs_as_cond, causal_attn,
# n_cond_layers=0) is kept.
# ---------------------------------------------------------------------------------------------
class DiffusionSinusoidalPosEmb(nn.Module):
    """Sinusoidal embedding of the diffusion timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0, device=x.device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class TransformerForDiffusion(nn.Module):
    """Denoiser whose cross-attention memory is the block-causally masked patch sequence.

    Memory layout is `[timestep token] + [T * P patch tokens]`; the decoder runs `horizon` action
    tokens with a causal self-attention mask and the block-causal memory mask.
    """

    def __init__(self, config: PatchPolicyConfig, cond_dim: int, n_patches: int):
        super().__init__()
        action_dim = config.action_feature.shape[0]
        n_emb = config.diffusion_hidden_dim
        horizon = config.horizon

        self.input_emb = nn.Linear(action_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, horizon, n_emb))
        self.drop = nn.Dropout(config.diffusion_p_drop_emb)

        self.time_emb = DiffusionSinusoidalPosEmb(n_emb)
        self.cond_obs_emb = nn.Linear(cond_dim, n_emb)
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, 1 + config.n_obs_steps * n_patches, n_emb))
        # `n_cond_layers=0` in every reference config: the memory encoder is a plain MLP, so the
        # patch tokens never self-attend. Block-causality enters only through `memory_mask`.
        self.encoder = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb), nn.Mish(), nn.Linear(4 * n_emb, n_emb)
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_emb,
            nhead=config.diffusion_n_head,
            dim_feedforward=4 * n_emb,
            dropout=config.diffusion_p_drop_attn,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # important for stability
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.diffusion_n_layer)

        self.register_buffer("mask", causal_mask(horizon), persistent=False)
        self.register_buffer(
            "memory_mask",
            block_causal_memory_mask(n_patches, config.n_obs_steps, horizon, n_leading_tokens=1),
            persistent=False,
        )

        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, action_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, sample: Tensor, timestep: Tensor | int, cond: Tensor) -> Tensor:
        """`sample`: `(B, horizon, A)`, `cond`: `(B, T*P, cond_dim)` -> `(B, horizon, A)`."""
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)  # (B, 1, n_emb)

        cond_embeddings = torch.cat([time_emb, self.cond_obs_emb(cond)], dim=1)
        memory = self.encoder(self.drop(cond_embeddings + self.cond_pos_emb))

        x = self.drop(self.input_emb(sample) + self.pos_emb)
        x = self.decoder(tgt=x, memory=memory, tgt_mask=self.mask, memory_mask=self.memory_mask)
        return self.head(self.ln_f(x))


# ---------------------------------------------------------------------------------------------
# NEW: ACT-style head. No counterpart in the reference repo — Patch Policy ships VQ-BeT and
# Diffusion Policy heads only. Structurally it is the diffusion head with the noisy-action input
# replaced by learned query embeddings and the noise-prediction loss replaced by L1, i.e. exactly
# what ACT's decoder does, reading the same block-causally masked patch memory.
# `ACTDecoder`/`ACTDecoderLayer` are reused verbatim from policies/act/modeling_act.py.
# ---------------------------------------------------------------------------------------------
class PatchACTHead(nn.Module):
    """`horizon` learned action queries cross-attending to the masked patch memory."""

    def __init__(self, config: PatchPolicyConfig, cond_dim: int, n_patches: int):
        super().__init__()
        self.config = config
        action_dim = config.action_feature.shape[0]
        horizon = config.horizon
        n_memory = config.n_obs_steps * n_patches

        self.memory_proj = nn.Linear(cond_dim, config.dim_model)
        self.memory_pos_embed = nn.Embedding(n_memory, config.dim_model)
        self.decoder_pos_embed = nn.Embedding(horizon, config.dim_model)

        self.decoder = ACTDecoder(config)
        # Same two masks as the diffusion head. Without the causal self-attention mask, query i
        # could read query j > i, which has seen frame j — laundering a future observation into an
        # earlier prediction and defeating the block-causal memory mask.
        tgt_mask = causal_mask(horizon)
        memory_mask = block_causal_memory_mask(n_patches, config.n_obs_steps, horizon)
        for layer in self.decoder.layers:
            layer.self_attn = _MaskedAttention(layer.self_attn, tgt_mask)
            layer.multihead_attn = _MaskedAttention(layer.multihead_attn, memory_mask)

        self.action_head = nn.Linear(config.dim_model, action_dim)

    def forward(self, patch_tokens: Tensor) -> Tensor:
        """`(B, T, P, D)` -> `(B, horizon, A)`."""
        B = patch_tokens.shape[0]
        memory = self.memory_proj(einops.rearrange(patch_tokens, "b t p d -> b (t p) d"))
        # ACTDecoder works in (sequence, batch, channel).
        memory = memory.transpose(0, 1)
        memory_pos = self.memory_pos_embed.weight.unsqueeze(1)
        queries_pos = self.decoder_pos_embed.weight.unsqueeze(1)

        x = torch.zeros(
            queries_pos.shape[0], B, self.config.dim_model, device=memory.device, dtype=memory.dtype
        )
        out = self.decoder(x, memory, decoder_pos_embed=queries_pos, encoder_pos_embed=memory_pos)
        return self.action_head(out.transpose(0, 1))


# ---------------------------------------------------------------------------------------------
# The policy.
# ---------------------------------------------------------------------------------------------
class PatchPolicyModel(nn.Module):
    """Frozen patch encoder -> observation token sequence -> one of three action heads.

    Token layout per observation step, following the reference (`train_policy.py`, which flattens
    `N T V P E -> N T (V P) E`): the P patch tokens of every camera are concatenated along the
    patch dim, so a frame's block is `V * P` tokens wide and the block-causal mask lets a patch of
    the wrist camera attend to a patch of the head camera at the same instant, but not to either
    one step later. With `use_robot_state`, a projected state token is appended to the block.
    """

    def __init__(self, config: PatchPolicyConfig):
        super().__init__()
        self.config = config
        self.num_images = len(config.image_features)

        self.encoder = make_patch_encoder(config.encoder_preset, config.resize_shape)
        if config.freeze_vision_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.n_patches_per_camera = self._measure_n_patches()
        self.tokens_per_frame = self.n_patches_per_camera * self.num_images + int(config.use_robot_state)
        self.feature_dim = self.encoder.output_dim

        if config.use_robot_state:
            # The reference has no state pathway at all; this is a lerobot-side addition.
            self.state_projector = MLP(
                config.robot_state_feature.shape[0], hidden_channels=[self.feature_dim]
            )

        if config.action_head == "vqbet":
            # `gpt_input_dim` is set by the encoder, so it is derived rather than configured.
            config.gpt_input_dim = self.feature_dim
            self.trunk = BlockCausalGPT(config, n_patches=self.tokens_per_frame)
            self.head = VQBeTHead(config)
        elif config.action_head == "diffusion":
            self.head = TransformerForDiffusion(
                config, cond_dim=self.feature_dim, n_patches=self.tokens_per_frame
            )
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
        else:
            self.head = PatchACTHead(
                config, cond_dim=self.feature_dim, n_patches=self.tokens_per_frame
            )

    def train(self, mode: bool = True):
        super().train(mode)
        # `nn.Module.train()` walks every submodule, so without this the frozen encoder would go
        # back into training mode on each epoch: ResNet-18's BatchNorm would update its running
        # statistics and the ViTs would apply dropout, quietly breaking "the encoder is frozen".
        if self.config.freeze_vision_encoder:
            self.encoder.eval()
        return self

    @torch.no_grad()
    def _measure_n_patches(self) -> int:
        """Count the encoder's patch tokens with a dry run.

        The reference hardcodes `n_patches` per encoder YAML, where a wrong value silently
        misaligns the block-causal mask with the token stream. Measuring removes that failure mode
        (`n_patches_override` is there if instantiating the encoder here is not possible).
        """
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

        if self.config.use_robot_state:
            state_token = self.state_projector(batch[OBS_STATE]).unsqueeze(2)  # (b, s, 1, e)
            patch_tokens = torch.cat([patch_tokens, state_token], dim=2)
        return patch_tokens

    @staticmethod
    def unpack_actions(action_seq: Tensor, action_chunk_size: int) -> Tensor:
        """`(N, T+W-1, A)` -> `(N, T, W, A)`, one action chunk per observation step.

        Reference: `BehaviorTransformer._unpack_actions`.
        """
        n_obs_steps = action_seq.shape[1] + 1 - action_chunk_size
        return torch.stack(
            [action_seq[:, i : i + action_chunk_size] for i in range(n_obs_steps)], dim=1
        )

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        patch_tokens = self.encode_observations(batch)
        actions = batch[ACTION]

        if self.config.action_head == "vqbet":
            features = self.trunk(patch_tokens)  # (B, T, gpt_output_dim)
            pred = self.head(features)
            loss_dict = self.head.loss_fn(
                pred, self.unpack_actions(actions, self.config.action_chunk_size)
            )
            return loss_dict.pop("loss"), loss_dict

        if self.config.action_head == "diffusion":
            noise = torch.randn_like(actions)
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (actions.shape[0],),
                device=actions.device,
            ).long()
            noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
            cond = einops.rearrange(patch_tokens, "b t p d -> b (t p) d")
            noise_pred = self.head(noisy_actions, timesteps, cond=cond)

            target = noise if self.config.prediction_type == "epsilon" else actions
            loss = F.mse_loss(noise_pred, target)
            return loss, {}

        pred = self.head(patch_tokens)
        loss = F.l1_loss(pred, actions)
        return loss, {"l1_loss": loss.item()}

    @torch.no_grad()
    def predict(self, batch: dict[str, Tensor]) -> Tensor:
        """-> `(B, action_chunk_size, A)`, the chunk that starts at the latest observation."""
        patch_tokens = self.encode_observations(batch)
        batch_size = patch_tokens.shape[0]
        start = self.config.n_obs_steps - 1

        if self.config.action_head == "vqbet":
            features = self.trunk(patch_tokens)
            predicted = self.head(features)["predicted_action"]
            # Only the chunk anchored at the newest frame is rolled out.
            return predicted[:, start].reshape(batch_size, self.config.action_chunk_size, -1)

        if self.config.action_head == "diffusion":
            cond = einops.rearrange(patch_tokens, "b t p d -> b (t p) d")
            sample = torch.randn(
                (batch_size, self.config.horizon, self.config.action_feature.shape[0]),
                device=cond.device,
                dtype=cond.dtype,
            )
            self.noise_scheduler.set_timesteps(self.num_inference_steps)
            for t in self.noise_scheduler.timesteps:
                model_output = self.head(sample, t, cond=cond)
                sample = self.noise_scheduler.step(model_output, t, sample).prev_sample
            return sample[:, start : start + self.config.action_chunk_size]

        pred = self.head(patch_tokens)
        return pred[:, start : start + self.config.action_chunk_size]


class PatchPolicy(PreTrainedPolicy):
    """Patch Policy as per "Patch Policy: Efficient Embodied Control via Dense Visual
    Representations" (https://huggingface.co/papers/2607.18236).
    """

    config_class = PatchPolicyConfig
    name = "patch_policy"

    def __init__(self, config: PatchPolicyConfig | None = None, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = PatchPolicyModel(config)
        self.reset()

    def get_optim_params(self) -> dict:
        """AdamW groups, following the reference's `configure_optimizers`.

        Norms, biases and embeddings are excluded from weight decay; the VQ-VAE keeps its own
        Adam(lr=1e-3, weight_decay=1e-4) hyperparameters, as in `BehaviorTransformer.__init__`.
        """
        # nanoGPT's rule, as in the reference's `GPT.configure_optimizers`: decay the weights of
        # linear/conv layers, nothing else. Norm weights, biases, embedding tables and bare
        # position `nn.Parameter`s all land in the no-decay group.
        whitelist = (nn.Linear, nn.Conv1d, nn.Conv2d)
        vqvae_params, decay, no_decay = [], [], []

        vqvae = getattr(self.model.head, "vqvae_model", None)
        vqvae_param_ids = {id(p) for p in vqvae.parameters()} if vqvae is not None else set()

        # `recurse=False` over every module visits each parameter exactly once, so no parameter
        # can end up in two optimizer groups.
        for module in self.model.modules():
            for name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                if id(param) in vqvae_param_ids:
                    vqvae_params.append(param)
                elif name.endswith("weight") and isinstance(module, whitelist):
                    decay.append(param)
                else:
                    no_decay.append(param)

        groups = [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}]
        if vqvae_params:
            groups.append(
                {
                    "params": vqvae_params,
                    "lr": self.config.optimizer_vqvae_lr,
                    "weight_decay": self.config.optimizer_vqvae_weight_decay,
                }
            )
        return groups

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

        # Phase 1 for the VQ-BeT head: train the residual VQ on actions alone before the trunk sees
        # anything. lerobot's convention (the first `n_vqvae_training_steps` forward passes) rather
        # than the reference's separate pre-training loop over the whole action set.
        if self.config.action_head == "vqbet" and not self.model.head.vqvae_model.discretized.item():
            loss, n_different_codes, n_different_combinations, recon_l1_error = self.model.head.discretize(
                self.config.n_vqvae_training_steps, batch[ACTION]
            )
            return loss, {
                "n_different_codes": n_different_codes,
                "n_different_combinations": n_different_combinations,
                "recon_l1_error": recon_l1_error,
            }

        return self.model(batch)
