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
"""ACT policy for the relative-action experiment (plan §2).

The network is *unchanged*: `ACTDeltaPolicy` subclasses the upstream `ACTPolicy`, so the
module tree, parameter names and forward maths are identical and `model.safetensors` is
interchangeable between `act` and `act_delta`. The action representation lives entirely in
the processor pipeline (`processor_act_delta.py`), which is exactly the variable isolation
the experiment needs: R0 vs R1/R2 differ only in what the model is asked to regress.

The one behavioural override is `select_action`, which refuses to serve a per-tick action
queue in relative mode - see the chunk-anchor drift discussion below.
"""

import logging

from torch import Tensor

from lerobot.policies.act.modeling_act import ACT, ACTPolicy, ACTTemporalEnsembler  # noqa: F401

from .configuration_act_delta import ACTDeltaConfig


class ACTDeltaPolicy(ACTPolicy):
    """ACT with an optional relative (state-anchored) action representation.

    See `ACTDeltaConfig`. With `use_relative_actions=False` this is bit-for-bit the
    upstream ACT policy (arm R0 of the experiment plan).
    """

    config_class = ACTDeltaConfig
    name = "act_delta"

    def __init__(self, config: ACTDeltaConfig, **kwargs):
        super().__init__(config, **kwargs)
        if config.use_relative_actions:
            logging.info(
                "[act_delta] relative actions ON (exclude_joints=%s, chunk_size=%d). Dataset stats must "
                "have been recomputed in relative space with the same exclude list and chunk size.",
                list(config.relative_exclude_joints or []),
                config.chunk_size,
            )

    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        Refuses the queued path in relative mode. Reason (plan §2.3, mirrored by
        `lerobot/rollout/inference/sync.py`): the caller runs the preprocessor on every tick,
        which refreshes `RelativeActionsProcessorStep._last_state`. Actions popped from the
        queue on later ticks were predicted relative to the state at the *start* of the chunk
        but the postprocessor re-anchors them on the *current* state, so the absolute target
        drifts across the chunk. Nothing raises; the robot just goes to the wrong place.

        Use `ChunkFIFOActionServer` (`inference_act_delta.py`), which converts a whole chunk to
        absolute actions once and then pops from a local FIFO, or set `n_action_steps=1`.
        """
        if (
            self.config.use_relative_actions
            and self.config.n_action_steps > 1
            and not self.config.allow_unsafe_relative_select_action
        ):
            raise NotImplementedError(
                "`select_action` with `use_relative_actions=True` and `n_action_steps>1` re-anchors queued "
                "actions on the current state, so absolute targets drift through the chunk. Use "
                "`lerobot.policies.act_delta.ChunkFIFOActionServer` (postprocesses the whole chunk once), "
                "or set `n_action_steps=1`, or set `allow_unsafe_relative_select_action=True` if you "
                "handle the absolute conversion yourself."
            )
        return super().select_action(batch)
