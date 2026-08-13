# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""ACT with a relative (state-anchored) action representation — experiment plan §2.

`lerobot/policies/act/` is left untouched; this package is the R1/R2 arm.
"""

from .configuration_act_delta import ACTDeltaConfig
from .inference_act_delta import (
    ChunkFIFOActionServer,
    ChunkFIFOInferenceEngine,
    predict_absolute_chunk,
)
from .modeling_act_delta import ACTDeltaPolicy
from .processor_act_delta import make_act_delta_pre_post_processors, validate_relative_setup

__all__ = [
    "ACTDeltaConfig",
    "ACTDeltaPolicy",
    "ChunkFIFOActionServer",
    "ChunkFIFOInferenceEngine",
    "make_act_delta_pre_post_processors",
    "predict_absolute_chunk",
    "validate_relative_setup",
]
