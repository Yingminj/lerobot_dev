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

import einops
import torch
from torch import Tensor

from lerobot.policies.act.modeling_act import ACT, ACTTemporalEnsembler
from lerobot.policies.act_eef.modeling_act_eef import ACTEEFPolicy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

from .configuration_act_eef_cvae import ACTEEFCVAEConfig

CVAE_ACTION = "cvae_action"
CVAE_ACTION_IS_PAD = "cvae_action_is_pad"


class ACTEEFCVAE(ACT):
    """Reuse ACT with a separate, training-only action input to its CVAE."""

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        if self.training and self.config.use_vae:
            # The first history position is offset -1. Its padding flag means
            # this sample has no previous frame, even if later tokens are valid.
            first_frame = batch[CVAE_ACTION_IS_PAD][:, 0]
            if first_frame.any():
                latent, mu, log_var = self._latents_with_first_frames(batch, first_frame)
                return self._decode_with_latent(batch, latent), (mu, log_var)
            # ACT only consumes actions in the CVAE. Keep the caller's target
            # and mask intact for ACTPolicy.forward's original L1 + KL loss.
            batch = dict(batch)
            batch[ACTION] = batch[CVAE_ACTION]
            batch["action_is_pad"] = batch[CVAE_ACTION_IS_PAD]
        return super().forward(batch)

    def _latents_with_first_frames(
        self, batch: dict[str, Tensor], first_frame: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Encode only noninitial samples and scatter latents into batch order."""
        history_indices = (~first_frame).nonzero(as_tuple=True)[0]
        shape = (first_frame.shape[0], self.config.latent_dim)
        if history_indices.numel():
            state = batch[OBS_STATE][history_indices]
            actions = batch[CVAE_ACTION][history_indices]
            history_pad = batch[CVAE_ACTION_IS_PAD][history_indices]
            cls = einops.repeat(self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=state.shape[0])
            tokens = torch.cat(
                [
                    cls,
                    self.vae_encoder_robot_state_input_proj(state).unsqueeze(1),
                    self.vae_encoder_action_input_proj(actions),
                ],
                dim=1,
            )
            prefix_pad = torch.zeros((state.shape[0], 2), dtype=torch.bool, device=state.device)
            encoded = self.vae_encoder(
                tokens.permute(1, 0, 2),
                pos_embed=self.vae_encoder_pos_enc.clone().detach().permute(1, 0, 2),
                key_padding_mask=torch.cat([prefix_pad, history_pad], dim=1),
            )[0]
            params = self.vae_encoder_latent_output_proj(encoded)
            history_mu, history_log_var = params.split(self.config.latent_dim, dim=-1)
            history_z = history_mu + history_log_var.div(2).exp() * torch.randn_like(history_mu)
            mu = params.new_zeros(shape).index_copy(0, history_indices, history_mu)
            log_var = params.new_zeros(shape).index_copy(0, history_indices, history_log_var)
            latent = params.new_zeros(shape).index_copy(0, history_indices, history_z)
        else:
            # Do not execute any CVAE module for an all-first-frame batch.
            mu = batch[OBS_STATE].new_zeros(shape, dtype=torch.float32)
            log_var = torch.zeros_like(mu)
            latent = torch.zeros_like(mu)

        if self.config.initial_z_mode == "sample":
            initial_indices = first_frame.nonzero(as_tuple=True)[0]
            prior_z = torch.randn(
                (initial_indices.numel(), self.config.latent_dim), device=latent.device, dtype=latent.dtype
            )
            latent = latent.index_copy(0, initial_indices, prior_z)

        # Zero mean/log-variance placeholders give exactly zero KL for first
        # frames in the inherited loss, which averages over the complete batch.
        # They are constants, not posterior estimates produced by the CVAE.
        return latent, mu, log_var

    def _decode_with_latent(self, batch: dict[str, Tensor], latent: Tensor) -> Tensor:
        """ACT's unchanged encoder/decoder computation with an explicit latent.

        Kept local because ACT.forward does not expose a latent override. All
        modules are inherited; no mode switches, hooks or new parameters occur.
        """
        encoder_in_tokens = [self.encoder_latent_input_proj(latent)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))
        if self.config.image_features:
            for img in batch[OBS_IMAGES]:
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")
                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))
        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        decoder_in = torch.zeros(
            (self.config.chunk_size, latent.shape[0], self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )
        return self.action_head(decoder_out.transpose(0, 1))


class ACTEEFCVAEPolicy(ACTEEFPolicy):
    """Train on previous-frame truth; inherit ACT-EEF inference unchanged."""

    config_class = ACTEEFCVAEConfig
    name = "act_eef_cvae"

    def __init__(self, config: ACTEEFCVAEConfig, **kwargs):
        # Initialize once, as ACTPolicy does, but instantiate our model subclass.
        # Constructing and replacing a base model would consume extra RNG draws.
        del kwargs
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.model = ACTEEFCVAE(config)
        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)
        self.reset()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Split K+1 normalized actions for training or loss-based validation.

        In eval mode the target is still sliced, but the inherited ACT model
        ignores CVAE inputs and uses z=0, just as it does for ordinary ACT-EEF.
        """
        actions = batch[ACTION]
        is_pad = batch["action_is_pad"]
        expected_steps = self.config.chunk_size + 1
        if actions.ndim != 3 or actions.shape[1] != expected_steps:
            raise ValueError(
                f"ACT-EEF-CVAE requires action shape (B, {expected_steps}, action_dim) "
                f"for offsets [-1, ..., chunk_size-1], got {tuple(actions.shape)}."
            )
        if is_pad.shape != actions.shape[:2]:
            raise ValueError("action_is_pad must match the action batch and time dimensions.")

        batch = dict(batch)
        batch[CVAE_ACTION] = actions[:, :-1]
        batch[CVAE_ACTION_IS_PAD] = is_pad[:, :-1]
        batch[ACTION] = actions[:, 1:]
        batch["action_is_pad"] = is_pad[:, 1:]
        return super().forward(batch)
