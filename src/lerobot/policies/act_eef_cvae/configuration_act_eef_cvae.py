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
from typing import Literal

import draccus

from lerobot.configs import PreTrainedConfig
from lerobot.policies.act_eef.configuration_act_eef import ACTEEFConfig


def _decode_initial_z_mode(value: object) -> Literal["sample", "zero"]:
    """Support this exact Literal in Draccus versions without Literal decoding."""
    if value == "sample":
        return "sample"
    if value == "zero":
        return "zero"
    raise ValueError("initial_z_mode must be 'sample' or 'zero'.")


# The decorator form in older Draccus requires a class, while the explicit
# registration form also accepts this exact typing.Literal key.
draccus.decode.register(Literal["sample", "zero"], _decode_initial_z_mode)


@PreTrainedConfig.register_subclass("act_eef_cvae")
@dataclass
class ACTEEFCVAEConfig(ACTEEFConfig):
    """ACT-EEF whose training CVAE encodes the previous frame's ground-truth chunk.

    The K+1 requested actions are split into a K-action history window and a
    K-action target window. First frames bypass the CVAE during training.
    Inference still uses the original zero latent.
    """

    initial_z_mode: Literal["sample", "zero"] = "sample"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.initial_z_mode not in ("sample", "zero"):
            raise ValueError("initial_z_mode must be 'sample' or 'zero'.")

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(-1, self.chunk_size))
