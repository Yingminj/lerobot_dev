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

from typing import Any

import torch

from lerobot.policies.act_eef.processor_act_eef import make_act_eef_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline

from .configuration_act_eef_cvae import ACTEEFCVAEConfig


def make_act_eef_cvae_pre_post_processors(
    config: ACTEEFCVAEConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Normalize the full action window before the policy splits it."""
    return make_act_eef_pre_post_processors(config=config, dataset_stats=dataset_stats)
