#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
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
"""ACT-DiT: ACT's decoder turned into a DiT denoiser (scheme S1).

The observation path is ACT's, untouched: ResNet feature maps flattened per camera into
~900 tokens, a 4-layer transformer encoder, and token-level cross-attention from every
decoder layer. Only the decoder's *job* changes:

    ACT      decoder_in = zeros                     -> action, trained with L1
    ACT-DiT  decoder_in = proj(noisy_action) + t     -> velocity (or noise), trained with MSE

so the distribution modelling that ACT throws away at inference (its CVAE latent is zeroed
in eval mode) is moved onto a path that survives deployment.

The encoder runs **once** per chunk; only the decoder is iterated by the ODE/denoising loop
(same trick as `modeling_diffusion.py`, which hoists `global_cond` out of the loop). At
`num_integration_steps=10` and one decoder layer of 100 queries, the added cost is a
fraction of the encoder pass over ~900 tokens.

Conditioning follows DiT-X / Tenma rather than `multi_task_dit`: the low-dimensional,
chunk-constant signals (diffusion timestep, robot state) go through adaLN-Zero, while the
high-dimensional, spatially structured visual tokens stay on cross-attention. Ablation D
(`use_cross_attention=False`) collapses that to adaLN-only for measurement purposes.
"""

import einops
import torch
from torch import Tensor, nn

from lerobot.policies.act.modeling_act import ACT, ACTDecoder, ACTDecoderLayer, ACTPolicy
from lerobot.policies.multi_task_dit.modeling_multi_task_dit import (
    DiffusionObjective,
    FlowMatchingObjective,
    SinusoidalPosEmb,
    modulate,
)
from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

from .configuration_act_dit import ACTDiTConfig


class ACTDiTWeightEMA(nn.Module):
    """Shadow copy of a module's trainable weights, swapped in whenever the policy is in eval mode.

    Lives inside the policy rather than in `lerobot_train.py` for three reasons: the trainer
    already calls `policy.update()` after every optimizer step (that hook is unused by
    `ACTPolicy`), the shadow rides in the policy's own `state_dict` so save and resume need no
    plumbing, and the scope stays the policy's business - a trainer-level averager would have to
    guess which parameters are frozen.

    The shadow is registered as buffers, so it moves with `.to(device)`, is written into
    `model.safetensors`, and comes back on resume. Two consequences: the checkpoint is ~2x, and
    flipping `use_ema` on mid-run makes an existing checkpoint fail a strict load.

    ponytail: dense single-device / DDP only. Under FSDP the parameters are shards, so the swap
    would splice shards together; guard `use_ema` off there rather than trusting this.
    """

    def __init__(self, model: nn.Module, decay: float):
        super().__init__()
        self.decay = decay
        # Frozen parameters are excluded at construction: averaging a constant is a copy, and it
        # would be paid for in checkpoint size. A parameter frozen *later* keeps being averaged.
        self._names = [n for n, p in model.named_parameters() if p.requires_grad]
        for name in self._names:
            # `register_buffer` rejects "." in a name; the module tree is flattened here anyway.
            self.register_buffer(name.replace(".", "/"), model.get_parameter(name).detach().clone())
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))
        self._live: list[Tensor] | None = None  # live weights parked while the shadow is active

    def _pairs(self, model: nn.Module) -> tuple[list[Tensor], list[Tensor]]:
        """(shadow, live) tensors, re-resolved every call so a `.to()` between steps is followed."""
        params = dict(model.named_parameters())
        return (
            [self.get_buffer(n.replace(".", "/")) for n in self._names],
            [params[n] for n in self._names],
        )

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        if self._live is not None:
            raise RuntimeError("EMA update while the shadow is swapped in - the policy is in eval mode.")
        self.num_updates += 1
        n = int(self.num_updates)
        # Warmup ramp: a cold shadow averaged at 0.9999 would trail the weights for the first
        # ~10k steps. `(1+n)/(10+n)` starts near 0.2 and reaches the configured decay quickly.
        decay = min(self.decay, (1.0 + n) / (10.0 + n))
        shadow, live = self._pairs(model)
        torch._foreach_lerp_(shadow, live, 1.0 - decay)

    @torch.no_grad()
    def set_active(self, model: nn.Module, active: bool) -> None:
        """Swap the shadow in (`active`) or restore the live weights. Idempotent: the training
        loop calls `policy.train()` every step, and `select_action` calls `policy.eval()` on
        every call, so this has to be free when the state already matches."""
        if active == (self._live is not None):
            return
        shadow, live = self._pairs(model)
        if active:
            self._live = [p.detach().clone() for p in live]
            for p, s in zip(live, shadow, strict=True):
                p.data.copy_(s)
        else:
            for p, saved in zip(live, self._live, strict=True):
                p.data.copy_(saved)
            self._live = None


class ACTDiTPolicy(ACTPolicy):
    """ACT with a flow-matching / diffusion decoder.

    Inherits `select_action`, `reset`, the temporal ensembler and `get_optim_params` from
    `ACTPolicy` unchanged - the action queue and chunk semantics are identical, only the way
    a chunk is produced differs.
    """

    config_class = ACTDiTConfig
    name = "act_dit"

    def __init__(self, config: ACTDiTConfig, **kwargs):
        super().__init__(config, **kwargs)
        # `ACTPolicy.__init__` hard-codes `self.model = ACT(config)`, so the denoiser replaces it
        # here rather than in `act/modeling_act.py`, which stays untouched. Deferring to super()
        # first (instead of copying its body) keeps this correct if ACTPolicy.__init__ grows a
        # line; the cost is one discarded ACT, a transient allocation at construction time.
        self.model = ACTDiT(config)

        objective_cls = FlowMatchingObjective if config.objective == "flow_matching" else DiffusionObjective
        self.objective = objective_cls(
            config,
            action_dim=config.action_feature.shape[0],
            horizon=config.chunk_size,
            do_mask_loss_for_padding=True,
        )

        # Submodules live in `nn.Module._modules` and are reached through `__getattr__`, which
        # only fires when normal lookup fails - so this must not be shadowed by a class attribute.
        self.ema = ACTDiTWeightEMA(self.model, config.ema_decay) if config.use_ema else None

    def update(self) -> None:
        """EMA step. Called by `lerobot_train.py` after every optimizer step, via the
        `has_method(policy, "update")` hook - `ACTPolicy` defines no `update`, so this is the
        whole contract. Note `rl/learner.py` never calls it; the RL path gets no EMA."""
        if self.ema is not None:
            self.ema.update(self.model)

    def train(self, mode: bool = True):
        """Sample from the averaged weights whenever the policy is not training.

        This covers eval loss (the loop toggles `eval()`/`train()` around it) and deployment
        (`ACTPolicy.select_action` calls `self.eval()` on every call, and `from_pretrained`
        ends with one) without either caller knowing about EMA. Checkpoints are written in
        train mode, so `model.safetensors` holds the live weights; a resumed policy arrives in
        eval mode and its first `update_policy` step swaps them back in."""
        # `getattr`: `nn.Module.__init__` machinery may toggle training mode before the
        # assignment in `__init__` above has run.
        ema = getattr(self, "ema", None)
        if ema is not None:
            ema.set_active(self.model, active=not mode)
        return super().train(mode)

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        return batch

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions by integrating the decoder against one encoder pass."""
        self.eval()
        batch = self._prepare_batch(batch)
        conditioning = self.model.encode_conditioning(batch)
        batch_size = conditioning[0].shape[1]
        return self.objective.conditional_sample(self.model, batch_size, conditioning)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self._prepare_batch(batch)
        conditioning = self.model.encode_conditioning(batch)
        loss = self.objective.compute_loss(self.model, batch, conditioning)
        return loss, {f"{self.config.objective}_loss": loss.item()}


class ACTDiT(ACT):
    """ACT's module tree with a denoising decoder.

    `forward` is the *denoiser* signature - `(noisy_actions, timestep, conditioning_vec)` -
    which is what `multi_task_dit`'s objectives call. The observation encoder is reached
    separately through `encode_conditioning`, so it can be hoisted out of the sampling loop.
    """

    def __init__(self, config: ACTDiTConfig):
        super().__init__(config)

        # Same story as the policy: `ACT.__init__` hard-codes `self.decoder = ACTDecoder(config)`.
        # Replace it, then redo the xavier init `ACT._reset_parameters` applied to the decoder we
        # just dropped.
        self.decoder = ACTDiTDecoder(config)
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # The decoder input is a noisy action chunk instead of ACT's zeros.
        self.action_in_proj = nn.Linear(config.action_feature.shape[0], config.dim_model)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(config.timestep_embed_dim),
            nn.Linear(config.timestep_embed_dim, config.timestep_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(config.timestep_embed_dim * 4, config.timestep_embed_dim),
        )

        # adaLN-Zero: gates start closed so the block starts as the identity, as in DiT.
        # Applied after the xavier init above, so the zeros win.
        for layer in self.decoder.layers:
            nn.init.zeros_(layer.adaln[-1].weight)
            nn.init.zeros_(layer.adaln[-1].bias)

    def encode_observations(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Run ACT's transformer encoder over the observations, once per chunk.

        Mirrors the encoder half of `ACT.forward`, minus the CVAE branch: `ACTDiTConfig`
        refuses `use_vae=True`, so ACT's latent is unconditionally the zero vector and the
        VAE encoder is never built. Kept here rather than factored out of `act/` so that
        `act/modeling_act.py` stays untouched; every module it touches is ACT's own, so the
        only thing that can drift is the *order* of the encoder tokens, which shows up
        immediately as a shape or quality regression, not silently.

        Returns:
            (ES, B, C) encoder output tokens.
            (ES, 1, C) encoder positional embeddings (needed again as cross-attention keys).
        """
        batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]

        # ACT's zeroed latent token: the shape the encoder expects, carrying no information.
        latent_sample = torch.zeros(
            [batch_size, self.config.latent_dim], dtype=torch.float32, device=batch[OBS_STATE].device
        )
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        if self.config.image_features:
            # For a list of images, the H and W may vary but H*W is constant.
            for img in batch[OBS_IMAGES]:
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)
                encoder_in_tokens.extend(list(einops.rearrange(cam_features, "b c h w -> (h w) b c")))
                encoder_in_pos_embed.extend(list(einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")))

        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)
        return self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed), encoder_in_pos_embed

    def encode_conditioning(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor | None]:
        """Encode the observations once; returns everything the denoiser needs per step.

        Returns:
            (ES, B, C) encoder tokens (cross-attention keys/values),
            (ES, 1, C) their positional embeddings,
            (B, static_cond_dim) the chunk-constant part of the adaLN conditioning, or None.
        """
        encoder_out, encoder_pos_embed = self.encode_observations(batch)

        static_cond = []
        if self.config.robot_state_feature and self.config.state_in_adaln:
            # Off by default: this road starves the encoder within a few hundred steps.
            # Reuses the encoder's state projection: the same 512-d embedding already exists
            # as an encoder token, no reason to learn a second one.
            static_cond.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        if not self.config.use_cross_attention:
            # Ablation D: with cross-attention gone this pooled vector is the *only* route
            # from the observation to the decoder - deliberately the low-bandwidth path.
            static_cond.append(encoder_out.mean(dim=0))

        return encoder_out, encoder_pos_embed, torch.cat(static_cond, dim=-1) if static_cond else None

    def forward(
        self,
        noisy_actions: Tensor,
        timestep: Tensor,
        conditioning_vec: tuple[Tensor, Tensor, Tensor | None],
    ) -> Tensor:
        """One denoiser evaluation.

        Args:
            noisy_actions: (B, chunk_size, action_dim) noisy/interpolated action chunk.
            timestep: (B,) flow-matching time in [0, 1], or integer diffusion timestep.
            conditioning_vec: the tuple returned by `encode_conditioning`.
        Returns:
            (B, chunk_size, action_dim) predicted velocity (flow matching) or noise/sample
            (diffusion), per `config.prediction_type`.
        """
        encoder_out, encoder_pos_embed, static_cond = conditioning_vec

        cond = self.time_mlp(timestep)
        if static_cond is not None:
            cond = torch.cat([cond, static_cond], dim=-1)

        # (B, S, A) -> (S, B, C), matching ACT's sequence-first decoder convention.
        decoder_in = self.action_in_proj(noisy_actions).transpose(0, 1)
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            cond=cond,
            encoder_pos_embed=encoder_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )
        return self.action_head(decoder_out.transpose(0, 1))


def _static_cond_dim(config: ACTDiTConfig) -> int:
    """Width of the chunk-constant part of the adaLN conditioning vector."""
    dim = config.dim_model if (config.robot_state_feature and config.state_in_adaln) else 0
    if not config.use_cross_attention:
        dim += config.dim_model  # mean-pooled encoder output
    return dim


class ACTDiTDecoder(ACTDecoder):
    def __init__(self, config: ACTDiTConfig):
        nn.Module.__init__(self)
        self.layers = nn.ModuleList([ACTDiTDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        cond: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x,
                encoder_out,
                cond=cond,
                decoder_pos_embed=decoder_pos_embed,
                encoder_pos_embed=encoder_pos_embed,
            )
        return self.norm(x)


class ACTDiTDecoderLayer(ACTDecoderLayer):
    """ACT's decoder layer with adaLN-Zero modulation on the self-attention and FFN branches.

    Pre-norm throughout, regardless of `config.pre_norm`: adaLN modulates the *input* of a
    branch, which only means anything in a pre-norm block, and post-norm DiT blocks are not
    a thing anyone reports training stably. The cross-attention branch is left unmodulated -
    it carries the high-bandwidth visual conditioning and needs no scalar gating.
    """

    def __init__(self, config: ACTDiTConfig):
        super().__init__(config)
        self.use_cross_attention = config.use_cross_attention
        if not self.use_cross_attention:
            # Don't leave unused parameters around: DDP errors on them, and they would
            # inflate the parameter count of the ablation arm.
            del self.multihead_attn, self.norm2, self.dropout2
        self.adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.timestep_embed_dim + _static_cond_dim(config), 6 * config.dim_model),
        )

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        cond: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: (DS, B, C) noisy action tokens.
            encoder_out: (ES, B, C) observation tokens to cross-attend.
            cond: (B, cond_dim) timestep (+ robot state (+ pooled observation)) conditioning.
        Returns:
            (DS, B, C)
        """
        # (B, 6C) -> six (1, B, C) tensors broadcasting over the chunk.
        shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff = (
            self.adaln(cond).unsqueeze(0).chunk(6, dim=-1)
        )

        h = modulate(self.norm1(x), shift_sa, scale_sa)
        q = k = self.maybe_add_pos_embed(h, decoder_pos_embed)
        x = x + gate_sa * self.dropout1(self.self_attn(q, k, value=h)[0])

        if self.use_cross_attention:
            h = self.norm2(x)
            x = x + self.dropout2(
                self.multihead_attn(
                    query=self.maybe_add_pos_embed(h, decoder_pos_embed),
                    key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
                    value=encoder_out,
                )[0]
            )

        h = modulate(self.norm3(x), shift_ff, scale_ff)
        x = x + gate_ff * self.dropout3(self.linear2(self.dropout(self.activation(self.linear1(h)))))
        return x
