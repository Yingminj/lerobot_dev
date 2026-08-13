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
"""Configuration for ACT with relative (state-anchored) action representation.

This is a *copy* of `ACTConfig` (not a subclass) plus the three relative-action
fields that pi0/pi05/pi0_fast already carry. It is deliberately not a subclass:
`lerobot.policies.factory.make_pre_post_processors` dispatches on
`isinstance(policy_cfg, ACTConfig)`, so a subclass would silently be handed the
*absolute* ACT processor pipeline and the relative conversion would never run.

Naming: `use_relative_actions` is the action *representation* (a_k - state).
It has nothing to do with `action_delta_indices` / `delta_timestamps`, which are
time offsets. See the experiment plan, §0.1.
"""

import logging
from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig

CHECK_LEVELS = ("error", "warn", "off")


@PreTrainedConfig.register_subclass("act_delta")
@dataclass
class ACTDeltaConfig(PreTrainedConfig):
    """Configuration class for ACT with an optional relative action representation.

    All architecture fields are identical to `ACTConfig` so that an `act_delta` run
    with `use_relative_actions=False` is numerically the same experiment as an `act`
    run (same modules, same parameter names, interchangeable `model.safetensors`).

    Relative-action arguments:
        use_relative_actions: If True, actions are converted to `action - observation.state`
            (for the non-excluded dimensions) *before* normalization, and converted back
            *after* unnormalization at inference time.
        relative_exclude_joints: Substrings of action dimension names that stay absolute.
            Matched case-insensitively against `action_feature_names`. Upstream pi0/pi05
            default to `["gripper"]` (experiment arm R2); pass `[]` for full-dim relative
            actions (arm R1).
        action_feature_names: Action dimension names, filled in automatically from dataset
            metadata by `lerobot.policies.factory.make_policy`. Without it the exclude mask
            cannot be built and *every* dimension becomes relative (R2 silently degrades
            to R1) - which is what `relative_consistency_check` guards against.
        relative_consistency_check: What to do when the relative setup looks inconsistent
            (empty exclude mask, or `dataset_stats` that were computed in absolute space).
            One of "error", "warn", "off".
        allow_unsafe_relative_select_action: Escape hatch for the chunk-anchor drift
            described in the experiment plan §2.3. Leave False; use
            `ChunkFIFOActionServer` from `inference_act_delta.py` for deployment.
    """

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Architecture.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False
    # Transformer layers.
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    # Note: Although the original ACT implementation has 7 for `n_decoder_layers`, there is a bug in the code
    # that means only the first layer is used. Here we match the original implementation by setting this to 1.
    # See this issue https://github.com/tonyzhaozh/act/issues/25#issue-2258740521.
    n_decoder_layers: int = 1
    # VAE.
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # Inference.
    # Note: the value used in ACT when temporal ensembling is enabled is 0.01.
    temporal_ensemble_coeff: float | None = None

    # Training and loss computation.
    dropout: float = 0.1
    kl_weight: float = 10.0

    # Training preset
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    # --- Relative action representation (the only functional difference vs `act`) ---
    use_relative_actions: bool = False
    relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    action_feature_names: list[str] | None = None
    relative_consistency_check: str = "warn"
    allow_unsafe_relative_select_action: bool = False

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling. This is "
                "because the policy needs to be queried every step to compute the ensembled action."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )
        if self.relative_consistency_check not in CHECK_LEVELS:
            raise ValueError(
                f"`relative_consistency_check` must be one of {CHECK_LEVELS}. "
                f"Got {self.relative_consistency_check}."
            )
        if self.use_relative_actions and self.temporal_ensemble_coeff is not None:
            # ACTTemporalEnsembler averages the *raw model outputs*, i.e. relative offsets that are
            # anchored on different states. That bias is a known open issue (plan §2.3); Phase 2 runs
            # with `temporal_ensemble_coeff=None`.
            raise NotImplementedError(
                "Temporal ensembling is not supported together with `use_relative_actions=True`: the "
                "ensembler averages relative offsets anchored on different observation states. Set "
                "`temporal_ensemble_coeff=None`."
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")
        if self.use_relative_actions and self.robot_state_feature is None:
            raise ValueError(
                "`use_relative_actions=True` requires an `observation.state` input feature to anchor "
                "the relative actions on."
            )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    # --- Relative-action helpers -------------------------------------------------

    def build_relative_mask(self, action_dim: int) -> list[bool]:
        """Return the per-dimension mask of which action dims are converted to relative.

        Uses the exact same code path as `RelativeActionsProcessorStep` and
        `compute_relative_action_stats`, so the mask here is the mask used everywhere else.
        """
        from lerobot.processor import RelativeActionsProcessorStep

        step = RelativeActionsProcessorStep(
            enabled=True,
            exclude_joints=list(self.relative_exclude_joints or []),
            action_names=self.action_feature_names,
        )
        return step._build_mask(action_dim)

    def report_check(self, message: str) -> None:
        """Raise / log / ignore a consistency problem according to `relative_consistency_check`."""
        if self.relative_consistency_check == "error":
            raise ValueError(message)
        if self.relative_consistency_check == "warn":
            logging.warning(message)
