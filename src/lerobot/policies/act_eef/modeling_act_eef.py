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

"""ACT policy specialized for 14-D dual-arm EEF state and actions."""

from lerobot.policies.act.modeling_act import ACT, ACTPolicy, ACTTemporalEnsembler  # noqa: F401

from .configuration_act_eef import ACTEEFConfig


class ACTEEFPolicy(ACTPolicy):
    """ACT with strict EEF feature validation and a distinct checkpoint type."""

    config_class = ACTEEFConfig
    name = "act_eef"

    def __init__(self, config: ACTEEFConfig, **kwargs):
        super().__init__(config, **kwargs)
