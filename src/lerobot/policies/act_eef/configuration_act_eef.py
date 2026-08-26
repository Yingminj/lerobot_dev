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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lerobot.configs import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.utils.constants import ACTION, OBS_STATE

EEF_FEATURE_NAMES = (
    "eef_l_x",
    "eef_l_y",
    "eef_l_z",
    "eef_l_roll",
    "eef_l_pitch",
    "eef_l_yaw",
    "eef_r_x",
    "eef_r_y",
    "eef_r_z",
    "eef_r_roll",
    "eef_r_pitch",
    "eef_r_yaw",
    "gripper_L",
    "gripper_R",
)
EEF_FEATURE_DIM = len(EEF_FEATURE_NAMES)


@PreTrainedConfig.register_subclass("act_eef")
@dataclass
class ACTEEFConfig(ACTConfig):
    """ACT configuration for the 14-D dual-arm EEF representation.

    Both ``observation.state`` and ``action`` use the same ordering:
    ``left xyz+rpy, right xyz+rpy, left gripper, right gripper``. Positions and
    Euler angles retain the units stored in the dataset (metres and radians for
    the converted dataset). Camera and all non-state features are unchanged.

    ACT already sizes its input and output projections from dataset feature
    metadata. This subclass exists to give EEF training a separate policy type
    and to fail early when a joint-space dataset is selected accidentally.
    """

    def validate_features(self) -> None:
        super().validate_features()

        state_feature = self.robot_state_feature
        if state_feature is None:
            raise ValueError("ACT-EEF requires an 'observation.state' input feature.")
        if state_feature.shape != (EEF_FEATURE_DIM,):
            raise ValueError(
                "ACT-EEF requires observation.state shape "
                f"({EEF_FEATURE_DIM},), got {state_feature.shape}."
            )

        action_feature = self.action_feature
        if action_feature is None:
            raise ValueError("ACT-EEF requires an 'action' output feature.")
        if action_feature.shape != (EEF_FEATURE_DIM,):
            raise ValueError(
                f"ACT-EEF requires action shape ({EEF_FEATURE_DIM},), got {action_feature.shape}."
            )

    def set_dataset_feature_metadata(self, dataset_features: dict[str, dict[str, Any]]) -> None:
        """Validate the semantic ordering when LeRobot metadata provides names."""
        for feature_key in (OBS_STATE, ACTION):
            feature = dataset_features.get(feature_key)
            if feature is None:
                raise ValueError(f"ACT-EEF dataset is missing required feature {feature_key!r}.")

            names = feature.get("names")
            if names is not None and tuple(names) != EEF_FEATURE_NAMES:
                raise ValueError(
                    f"ACT-EEF requires {feature_key} names in this order: "
                    f"{list(EEF_FEATURE_NAMES)}; got {list(names)}."
                )
