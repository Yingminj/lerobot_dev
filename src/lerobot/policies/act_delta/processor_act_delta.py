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
"""Pre/post-processing pipelines for `act_delta`.

Difference vs `policies/act/processor_act.py`: the pipelines are assembled in the
OpenPI order used by pi0/pi05, so the relative conversion happens in *physical*
units and the inverse happens after unnormalization:

    raw → relative → normalize → model → unnormalize → absolute → cpu

Getting that order wrong (e.g. relative after normalize) subtracts a raw state from
a normalized action and is dimensionally meaningless while raising no error, which is
why the order is spelled out here instead of reusing the default ACT pipeline.
"""

import logging
from typing import Any

import torch

from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import ACTION, POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_act_delta import ACTDeltaConfig


def validate_relative_setup(
    config: ACTDeltaConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> None:
    """Check the three ways a relative-action run fails silently (plan §2.2).

    1. The exclude mask ends up all-True (`relative_exclude_joints=["gripper"]` but no
       action dimension name contains "gripper", or no names at all) → R2 silently becomes R1.
    2. `dataset_stats` were computed on *absolute* actions → the normalizer divides relative
       offsets by an absolute std ~3x too large and the loss scale collapses.
    3. Relative actions requested without an `observation.state` feature → nothing to anchor on
       (already rejected in `ACTDeltaConfig.validate_features`).

    Severity is controlled by `config.relative_consistency_check` ("error" / "warn" / "off").
    """
    if not config.use_relative_actions:
        return

    action_dim = config.action_feature.shape[0] if config.output_features else None
    if action_dim is None:
        return

    mask = config.build_relative_mask(action_dim)
    n_relative = int(sum(mask))
    excluded = [
        name
        for name, keep in zip(config.action_feature_names or [], mask, strict=False)
        if not keep
    ]
    logging.info(
        "[act_delta] relative action mask: %d/%d dims relative, excluded=%s",
        n_relative,
        action_dim,
        excluded or "none",
    )

    if config.relative_exclude_joints:
        if config.action_feature_names is None:
            config.report_check(
                "[act_delta] `relative_exclude_joints`="
                f"{config.relative_exclude_joints} was requested but `action_feature_names` is None, so "
                "the exclude mask cannot be built and ALL action dims will be relative. Build the policy "
                "through `lerobot.policies.factory.make_policy` (it fills the names from dataset metadata) "
                "or set the names explicitly."
            )
        elif n_relative == action_dim:
            config.report_check(
                "[act_delta] `relative_exclude_joints`="
                f"{config.relative_exclude_joints} matched no action dimension name "
                f"({config.action_feature_names}), so every dim is relative. This silently turns arm R2 "
                "into arm R1. Fix the exclude substrings to match your dataset's action names."
            )

    if dataset_stats and ACTION in dataset_stats:
        action_stats = dataset_stats[ACTION]
        mean = action_stats.get("mean")
        std = action_stats.get("std")
        if mean is not None and std is not None:
            mean_t = torch.as_tensor(mean, dtype=torch.float32).flatten()
            std_t = torch.as_tensor(std, dtype=torch.float32).flatten()
            mask_t = torch.tensor(mask[: mean_t.numel()], dtype=torch.bool)
            if mask_t.any():
                # Relative actions are centred near zero; absolute joint targets are not. A |mean|
                # that is large compared to the spread is the signature of absolute stats.
                ratio = (
                    mean_t[mask_t].abs().mean() / std_t[mask_t].abs().mean().clamp_min(1e-8)
                ).item()
                logging.info("[act_delta] action stats |mean|/std over relative dims: %.3f", ratio)
                if ratio > 1.0:
                    exclude = " ".join(config.relative_exclude_joints) or ""
                    config.report_check(
                        "[act_delta] `dataset_stats[action]` look like ABSOLUTE action statistics "
                        f"(mean/std ratio {ratio:.2f} over the relative dims). Recompute them in relative "
                        "space before training, with the *same* exclude list and chunk_size as the policy:\n"
                        "  python -m lerobot.policies.act_delta.prepare_relative_stats \\\n"
                        "      --root <dataset folder> \\\n"
                        f"      --chunk-size {config.chunk_size} --exclude-joints {exclude}\n"
                        "then train against the stats view it prints (`--dataset.root=<...>`). Set "
                        "`--policy.relative_consistency_check=off` to silence this check."
                    )


def make_act_delta_pre_post_processors(
    config: ACTDeltaConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Creates the pre- and post-processing pipelines for the `act_delta` policy.

    Args:
        config: The `act_delta` policy configuration object.
        dataset_stats: Dataset statistics used for normalization. When
            `config.use_relative_actions` is True these must have been computed in relative
            space (`recompute_stats(..., relative_action=True, chunk_size=config.chunk_size)`).

    Returns:
        A tuple `(preprocessor, postprocessor)`.
    """
    validate_relative_setup(config, dataset_stats)

    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=list(config.relative_exclude_joints or []),
        action_names=config.action_feature_names,
    )

    # OpenPI order: raw → relative → normalize → model → unnormalize → absolute
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        relative_step,
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            device=config.device,
        ),
    ]
    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        AbsoluteActionsProcessorStep(enabled=config.use_relative_actions, relative_step=relative_step),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
