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

"""ACT policy that cannot imitate actions labeled as low quality."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES

from .configuration_act_quality import ACTQualityConfig
from .quality_index import QualityIndex, activate_quality_index, install_training_sampler_hook

if TYPE_CHECKING:
    from lerobot.datasets import LeRobotDatasetMetadata


class ACTQualityPolicy(ACTPolicy):
    """Upstream ACT architecture with chunk-, anchor-, and CVAE-aware quality masks."""

    config_class = ACTQualityConfig
    name = "act_quality"

    def __init__(
        self,
        config: ACTQualityConfig,
        dataset_meta: LeRobotDatasetMetadata | None = None,
        **kwargs,
    ) -> None:
        super().__init__(config, **kwargs)
        self.config = config
        self.register_buffer(
            "_quality_by_index",
            torch.empty(0, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "_recovery_anchor_by_index",
            torch.empty(0, dtype=torch.bool),
            persistent=False,
        )
        self._quality_index_cpu: QualityIndex | None = None

        if dataset_meta is not None and config.quality_label_key in dataset_meta.features:
            quality_index = QualityIndex.from_dataset_meta(
                dataset_meta,
                config.quality_label_key,
                require_monotonic=config.quality_require_monotonic,
            )
            self.set_quality_index(quality_index)
        elif dataset_meta is not None and config.quality_require_labels:
            raise ValueError(
                f"Dataset is missing required quality feature {config.quality_label_key!r}."
            )

    def set_quality_index(self, quality_index: QualityIndex) -> None:
        """Attach a dataset index without saving it in model checkpoints."""
        self._quality_index_cpu = quality_index
        self._quality_by_index = quality_index.labels.to(device=self.config.device)
        self._recovery_anchor_by_index = quality_index.recovery_anchor_mask().to(
            device=self.config.device
        )
        activate_quality_index(
            quality_index,
            balance_anchor_pools=self.config.quality_balance_anchor_pools,
            recovery_anchor_fraction=self.config.quality_recovery_anchor_fraction,
            balanced_epoch_size=self.config.quality_balanced_epoch_size,
        )
        if self.config.quality_filter_invalid_anchors:
            installed = install_training_sampler_hook()
            if not installed:
                logging.debug(
                    "[act_quality] sampler hook was not installed because lerobot-train is not active"
                )

    def _quality_mask(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Return target validity, anchor validity, and recovery-anchor membership."""
        action = batch[ACTION]
        batch_size, horizon = action.shape[:2]
        if horizon != self.config.chunk_size:
            raise ValueError(
                f"Action horizon {horizon} does not match chunk_size={self.config.chunk_size}."
            )
        action_is_pad = batch.get("action_is_pad")
        if action_is_pad is None or tuple(action_is_pad.shape) != (batch_size, horizon):
            raise ValueError(
                f"action_is_pad must have shape {(batch_size, horizon)}, got "
                f"{None if action_is_pad is None else tuple(action_is_pad.shape)}."
            )

        if self._quality_by_index.numel() == 0:
            if self.config.quality_require_labels:
                raise RuntimeError(
                    "No quality index is attached. Construct ACTQualityPolicy through make_policy "
                    "with a labeled dataset_meta."
                )
            valid = ~action_is_pad.bool()
            recovery_anchor = torch.zeros(batch_size, dtype=torch.bool, device=valid.device)
            return valid, valid[:, 0], recovery_anchor

        absolute_index = batch.get("index")
        if absolute_index is None:
            raise KeyError(
                "Training batch has no global `index`; act_quality needs it to align quality labels."
            )
        absolute_index = absolute_index.to(
            device=self._quality_by_index.device, dtype=torch.long
        ).reshape(-1)
        if absolute_index.numel() != batch_size:
            raise ValueError(
                f"Batch index has {absolute_index.numel()} elements but batch size is {batch_size}."
            )
        if absolute_index.numel() and (
            int(absolute_index.min()) < 0
            or int(absolute_index.max()) >= self._quality_by_index.numel()
        ):
            raise IndexError(
                f"Batch index range [{int(absolute_index.min())}, {int(absolute_index.max())}] "
                f"is outside quality lookup [0, {self._quality_by_index.numel() - 1}]."
            )

        offsets = torch.arange(horizon, device=absolute_index.device)
        target_indices = absolute_index.unsqueeze(1) + offsets.unsqueeze(0)
        target_indices = target_indices.clamp_max(self._quality_by_index.numel() - 1)
        target_quality = self._quality_by_index[target_indices]
        anchor_quality = self._quality_by_index[absolute_index]
        recovery_anchor = self._recovery_anchor_by_index[absolute_index]

        # An invalid current observation must never be paired with a distant valid
        # suffix in the same action chunk.  The sampler normally removes these
        # anchors; this guard makes the loss safe even in eval or custom loaders.
        valid = target_quality & anchor_quality.unsqueeze(1) & ~action_is_pad.bool()
        return valid, anchor_quality, recovery_anchor

    def forward(
        self,
        batch: dict[str, Tensor],
        reduction: str = "mean",
    ) -> tuple[Tensor, dict]:
        """Compute quality-masked reconstruction and KL losses.

        ``reduction='none'`` is supported for the repository's generic sample
        weighting interface.  The default ``mean`` preserves upstream global
        valid-action normalization and separately normalizes KL by valid samples.
        """
        if reduction not in {"mean", "none"}:
            raise ValueError(f"Unsupported reduction {reduction!r}; expected 'mean' or 'none'.")
        if ACTION not in batch:
            raise KeyError("ACTQualityPolicy.forward requires an action target.")

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        quality_mask, anchor_quality, recovery_anchor = self._quality_mask(batch)
        model_batch = dict(batch)
        model_batch["action_is_pad"] = ~quality_mask
        if self.config.quality_zero_masked_vae_actions:
            model_batch[ACTION] = batch[ACTION].masked_fill(
                ~quality_mask.unsqueeze(-1), 0.0
            )

        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(model_batch)
        abs_err = F.l1_loss(batch[ACTION], actions_hat, reduction="none")
        action_dim = abs_err.shape[-1]
        mask_f = quality_mask.to(dtype=abs_err.dtype)
        masked_error = abs_err * mask_f.unsqueeze(-1)
        valid_steps_per_sample = mask_f.sum(dim=1)
        valid_samples = valid_steps_per_sample > 0

        per_sample_l1 = masked_error.sum(dim=(1, 2)) / (
            valid_steps_per_sample * action_dim
        ).clamp_min(1.0)

        if self.config.use_vae and log_sigma_x2_hat is not None:
            per_sample_kld = -0.5 * (
                1 + log_sigma_x2_hat - mu_hat.pow(2) - log_sigma_x2_hat.exp()
            ).sum(dim=-1)
            per_sample_kld = per_sample_kld * valid_samples.to(per_sample_kld.dtype)
        else:
            per_sample_kld = torch.zeros_like(per_sample_l1)

        per_sample_loss = per_sample_l1 + self.config.kl_weight * per_sample_kld
        if reduction == "none":
            loss = per_sample_loss
            l1_loss = per_sample_l1.sum() / valid_samples.sum().clamp_min(1)
            mean_kld = per_sample_kld.sum() / valid_samples.sum().clamp_min(1)
        else:
            total_valid_elements = (mask_f.sum() * action_dim).clamp_min(1.0)
            l1_loss = masked_error.sum() / total_valid_elements
            mean_kld = per_sample_kld.sum() / valid_samples.sum().clamp_min(1)
            loss = l1_loss + self.config.kl_weight * mean_kld

        loss_dict = {
            "l1_loss": float(l1_loss.detach()),
            "quality_valid_action_fraction": float(quality_mask.float().mean().detach()),
            "quality_invalid_anchor_fraction": float((~anchor_quality).float().mean().detach()),
            "quality_recovery_anchor_fraction": float(recovery_anchor.float().mean().detach()),
            "quality_normal_anchor_fraction": float(
                (anchor_quality & ~recovery_anchor).float().mean().detach()
            ),
            "quality_valid_samples": int(valid_samples.sum().detach()),
        }
        if self.config.use_vae and log_sigma_x2_hat is not None:
            loss_dict["kld_loss"] = float(mean_kld.detach())
        return loss, loss_dict
