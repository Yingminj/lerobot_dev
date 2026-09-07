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
from dataclasses import dataclass
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
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.processor.relative_action_processor import to_relative_actions
from lerobot.types import TransitionKey
from lerobot.utils.constants import (
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_patch_new_policy import PatchNewPolicyConfig


@ProcessorStepRegistry.register("patch_new_relative_actions_processor")
@dataclass
class ChunkAnchoredRelativeActionsStep(RelativeActionsProcessorStep):
    """`RelativeActionsProcessorStep` with the anchor pinned to the newest observation step.

    The base step assumes `observation.state` is `(B, state_dim)`. This policy observes a window,
    so its state is `(B, n_obs_steps, state_dim)` and the anchor has to be chosen: the newest step
    is the one the predicted chunk starts from, matching pi0's single state token.

    Only the anchor selection differs; the mask (`relative_exclude_joints`) and the arithmetic are
    the base class's.
    """

    def __call__(self, transition):
        observation = transition.get(TransitionKey.OBSERVATION, {})
        state = observation.get(OBS_STATE) if observation else None
        if state is not None:
            # Cached for the paired AbsoluteActionsProcessorStep, which reverses this on the output.
            self._last_state = state[:, -1] if state.ndim == 3 else state

        if not self.enabled:
            return transition

        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None or self._last_state is None:
            return new_transition

        mask = self._build_mask(action.shape[-1])
        new_transition[TransitionKey.ACTION] = to_relative_actions(action, self._last_state, mask)
        return new_transition


def make_patch_new_policy_pre_post_processors(
    config: PatchNewPolicyConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build the pre- and post-processing pipelines.

    `VISUAL` normalization is `IDENTITY`: the frozen ViTs apply their own mean/std to pixels in
    [0, 1], as the reference encoders do, so normalizing here would double-normalize.

    With `use_relative_actions` (P6) the order is openpi's -- raw -> relative -> normalize ->
    model -> unnormalize -> absolute -- so the anchor is subtracted in the raw action units.
    Two caveats worth knowing before reading an ablation off this switch:

      * the normalization statistics are still the absolute action distribution's, so `MIN_MAX`
        squeezes a small delta range into a ruler sized by absolute joint extremes; prefer
        `MEAN_STD` for `ACTION` on this arm;
      * at execution time the post-step re-anchors on whatever state was last seen, so with
        `n_action_steps > 1` the chunk's later steps are added back onto a newer anchor than the
        one training subtracted. That is the same behaviour as lerobot's pi0.5, and it is what a
        deployment-side state bridge does anyway -- but it is a difference from training, not an
        identity.
    """
    relative_step = ChunkAnchoredRelativeActionsStep(
        enabled=config.use_relative_actions,
        exclude_joints=config.relative_exclude_joints,
        action_names=config.action_feature_names,
    )

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        relative_step,
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device=config.device),
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
