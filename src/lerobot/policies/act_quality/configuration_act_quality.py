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

"""Configuration for ACT with hard masks over low-quality actions.

``ACTQualityConfig`` deliberately subclasses the upstream ``ACTConfig``.  All
standard ACT fields (chunk size, architecture, VAE, inference, normalization,
and optimizer fields) are therefore constructor arguments and CLI options of
this config as well.  Only the quality-specific additions are declared below.
"""

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("act_quality")
@dataclass
class ACTQualityConfig(ACTConfig):
    """ACT configuration extended with a per-frame action-quality label.

    The network architecture is identical to upstream ACT.  Inherited fields
    include ``chunk_size``, ``n_action_steps``, ``vision_backbone``,
    ``dim_model``, ``n_heads``, all encoder/decoder and CVAE settings,
    ``temporal_ensemble_coeff``, ``dropout``, ``kl_weight``, normalization, and
    optimizer settings.  The fields declared here only control target
    selection and loss masking, so ACT checkpoints remain weight-compatible
    with this policy.
    """

    quality_label_key: str = "action_quality"
    quality_require_labels: bool = True
    quality_require_monotonic: bool = True
    quality_filter_invalid_anchors: bool = True
    quality_zero_masked_vae_actions: bool = True
    quality_balance_anchor_pools: bool = True
    # A warm-started ACT expects observations and actions in the normalization
    # coordinates stored with the pretrained checkpoint. Keep those processor
    # statistics by default so loading recovery data does not change the policy
    # before the first optimizer step. From-scratch training still uses the
    # current dataset statistics, and resume always keeps checkpoint statistics.
    quality_keep_pretrained_normalization: bool = True
    # `merged` is the conservative warm-start baseline: label 2/3 share one
    # recovery pool. `semantic` gives rollback and reentry independent quotas.
    quality_pool_mode: str = "merged"
    quality_recovery_anchor_fraction: float = 0.10
    # Semantic S/M/E sampling: label 2 is rollback (S->M), label 3 is reentry
    # (M->E). The reentry quota is part of the total recovery quota; rollback
    # receives the difference. Fractions are measured over the whole epoch.
    quality_reentry_anchor_fraction: float = 0.05
    # Deprecated checkpoint-compatibility fields. They are deliberately not
    # used by the semantic sampler: no phase boundary is inferred from a fixed
    # frame count anymore. Keep them so older checkpoints remain loadable.
    quality_recovery_onset_steps: int = 30
    quality_recovery_onset_fraction: float = 0.10
    quality_balanced_epoch_size: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.quality_label_key:
            raise ValueError("`quality_label_key` must not be empty.")
        if self.quality_pool_mode not in {"merged", "semantic"}:
            raise ValueError(
                "`quality_pool_mode` must be 'merged' or 'semantic', got "
                f"{self.quality_pool_mode!r}."
            )
        if not 0.0 <= self.quality_recovery_anchor_fraction <= 1.0:
            raise ValueError(
                "`quality_recovery_anchor_fraction` must be in [0, 1], got "
                f"{self.quality_recovery_anchor_fraction}."
            )
        if not 0.0 <= self.quality_reentry_anchor_fraction <= 1.0:
            raise ValueError(
                "`quality_reentry_anchor_fraction` must be in [0, 1], got "
                f"{self.quality_reentry_anchor_fraction}."
            )
        if (
            self.quality_pool_mode == "semantic"
            and self.quality_reentry_anchor_fraction > self.quality_recovery_anchor_fraction
        ):
            raise ValueError(
                "`quality_reentry_anchor_fraction` is part of the total recovery quota and "
                "must be <= `quality_recovery_anchor_fraction`, got "
                f"{self.quality_reentry_anchor_fraction} > "
                f"{self.quality_recovery_anchor_fraction}."
            )
        if self.quality_balanced_epoch_size < 0:
            raise ValueError(
                "`quality_balanced_epoch_size` must be >= 0, got "
                f"{self.quality_balanced_epoch_size}."
            )
        if self.quality_balance_anchor_pools and not self.quality_filter_invalid_anchors:
            raise ValueError(
                "Balanced anchor sampling requires `quality_filter_invalid_anchors=true`."
            )

    def set_dataset_feature_metadata(self, dataset_features: dict[str, dict]) -> None:
        """Validate the label and keep it out of ACT input/output feature sets.

        A key named ``action_quality`` is classified as an ACTION feature by the
        generic LeRobot feature converter because it starts with ``action``.
        It is metadata for the loss, not a model output, so remove it after the
        factory has populated the feature dictionaries.
        """
        feature = dataset_features.get(self.quality_label_key)
        if feature is None:
            if self.quality_require_labels:
                raise ValueError(
                    f"Dataset is missing required quality feature {self.quality_label_key!r}. "
                    "Run the recovery quality-label tool before training."
                )
        else:
            dtype = str(feature.get("dtype"))
            shape = tuple(feature.get("shape", ()))
            if dtype not in {"bool", "int64"} or shape != (1,):
                raise ValueError(
                    f"Quality feature {self.quality_label_key!r} must have dtype=bool or "
                    "dtype=int64 and "
                    f"shape=(1,), got dtype={dtype!r}, shape={shape!r}."
                )

        self.input_features.pop(self.quality_label_key, None)
        self.output_features.pop(self.quality_label_key, None)
