#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Relative-action-safe inference for `act_delta` (plan §2.3, option P-a).

Why this exists
---------------
`RelativeActionsProcessorStep` caches the observation state each time the preprocessor
runs, and `AbsoluteActionsProcessorStep` adds that cached state back after unnormalization.
The usual control loop is::

    per tick:  preprocessor(obs) → policy.select_action(...) → postprocessor(action)

With `n_action_steps > 1`, `select_action` pops actions that were predicted relative to the
state at the start of the chunk, while the preprocessor has meanwhile re-cached the *current*
state — so the absolute target drifts through the chunk. Upstream rejects that combination
outright (`lerobot/rollout/context.py`, `lerobot/rollout/inference/sync.py`).

The fix here is structural: predict the chunk once, convert the *whole* chunk to absolute
actions in a single postprocessor call (so every action in it shares one anchor state), and
serve the chunk from a local FIFO. The invariant is that the anchor state is written exactly
once per chunk. As a side effect it also removes the per-tick pre/post-processing cost.
"""

from __future__ import annotations

import logging
from collections import deque
from copy import copy

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyProcessorPipeline

logger = logging.getLogger(__name__)


@torch.no_grad()
def predict_absolute_chunk(
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    batch: dict,
    n_action_steps: int | None = None,
) -> torch.Tensor:
    """Predict one action chunk and return it in absolute, unnormalized units.

    This is the evaluation-time counterpart of the FIFO server and the primitive the offline
    MAE metrics of plan §0.3 should be built on: it uses `predict_action_chunk` (latent = 0,
    same as deployment) and runs the full postprocessor, so relative and absolute arms are
    compared in the same physical space.

    Args:
        policy: An `ACTDeltaPolicy` (or any policy exposing `predict_action_chunk`).
        preprocessor: The policy preprocessor pipeline.
        postprocessor: The policy postprocessor pipeline.
        batch: A raw (unprocessed) observation batch.
        n_action_steps: Truncate the chunk to this many steps. Defaults to the full chunk.

    Returns:
        A `(batch, steps, action_dim)` tensor of absolute actions.
    """
    processed = preprocessor(batch)
    chunk = policy.predict_action_chunk(processed)
    if n_action_steps is not None:
        chunk = chunk[:, :n_action_steps]
    return postprocessor(chunk)


class ChunkFIFOActionServer:
    """Serve actions one tick at a time from a chunk that was converted to absolute units once.

    Framework-agnostic: it only needs a policy and its two pipelines, so it can drive a custom
    deployment script as easily as a LeRobot rollout. For the LeRobot rollout stack, use
    `ChunkFIFOInferenceEngine` below, which wraps this with the engine interface.

    Example::

        server = ChunkFIFOActionServer(policy, preprocessor, postprocessor)
        server.reset()
        while running:
            action = server.get_action(observation)  # absolute, unnormalized, (action_dim,)
    """

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        n_action_steps: int | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._n_action_steps = int(n_action_steps or policy.config.n_action_steps)
        chunk_size = getattr(policy.config, "chunk_size", self._n_action_steps)
        if self._n_action_steps > chunk_size:
            raise ValueError(
                f"n_action_steps ({self._n_action_steps}) cannot exceed chunk_size ({chunk_size})."
            )
        self._device = torch.device(device or getattr(policy.config, "device", None) or "cpu")
        self._fifo: deque[torch.Tensor] = deque()
        # Number of chunks predicted since the last reset; useful for the boundary-jump metric.
        self.n_chunks = 0

    @property
    def n_action_steps(self) -> int:
        return self._n_action_steps

    def reset(self) -> None:
        """Clear the FIFO and the policy's episode state. Call this on every episode reset."""
        self._fifo.clear()
        self.n_chunks = 0
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()

    @torch.no_grad()
    def get_action(self, observation: dict) -> torch.Tensor:
        """Return the next absolute action for a single (unbatched or batch-1) observation."""
        if not self._fifo:
            absolute_chunk = predict_absolute_chunk(
                self._policy,
                self._preprocessor,
                self._postprocessor,
                copy(observation),
                self._n_action_steps,
            )
            # (B, T, D) → T entries of (B, D)
            self._fifo.extend(absolute_chunk.transpose(0, 1))
            self.n_chunks += 1
        return self._fifo.popleft()


try:  # pragma: no cover - the rollout stack pulls optional extras
    from lerobot.rollout.inference.base import InferenceEngine as _InferenceEngineBase
except Exception:  # noqa: BLE001 - rollout extras unavailable; duck-typing still works
    _InferenceEngineBase = object  # type: ignore[assignment, misc]


class ChunkFIFOInferenceEngine(_InferenceEngineBase):  # type: ignore[misc, valid-type]
    """`InferenceEngine` drop-in for relative-action policies.

    Same constructor signature and same return contract as `SyncInferenceEngine` (a CPU tensor
    ordered by `ordered_action_keys`), so it can replace it wherever an engine is built by hand::

        engine = ChunkFIFOInferenceEngine(
            policy=policy, preprocessor=pre, postprocessor=post,
            dataset_features=dataset_features, ordered_action_keys=ordered_action_keys,
            task=task, device=cfg.device, robot_type=robot.robot_type,
        )

    Note that `lerobot.rollout.context.build_rollout_context` still rejects relative-action
    policies when `--inference.type=sync`; this engine has to be wired in explicitly (or use
    `--inference.type=rtc`, which postprocesses whole chunks and is unaffected by the drift).
    """

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        dataset_features: dict,
        ordered_action_keys: list[str],
        task: str,
        device: str | None,
        robot_type: str,
        n_action_steps: int | None = None,
    ) -> None:
        self._server = ChunkFIFOActionServer(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_action_steps=n_action_steps,
            device=device,
        )
        self._policy = policy
        self._dataset_features = dataset_features
        self._ordered_action_keys = ordered_action_keys
        self._task = task
        self._device = torch.device(device or "cpu")
        self._robot_type = robot_type
        logger.info(
            "ChunkFIFOInferenceEngine initialized (device=%s, n_action_steps=%d, action_keys=%d)",
            self._device,
            self._server.n_action_steps,
            len(ordered_action_keys),
        )

    def start(self) -> None:
        """No background resources to start."""
        logger.info("ChunkFIFOInferenceEngine started (inline mode — no background thread)")

    def stop(self) -> None:
        """No background resources to stop."""
        logger.info("ChunkFIFOInferenceEngine stopped")

    def reset(self) -> None:
        """Drop the pending chunk and reset the policy and processors."""
        logger.info("Resetting chunk-FIFO inference state (FIFO + policy + processors)")
        self._server.reset()

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        """Return the next action tensor, predicting a new chunk only when the FIFO is empty."""
        from contextlib import nullcontext

        from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference

        if obs_frame is None:
            return None
        observation = copy(obs_frame)
        autocast_ctx = (
            torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and self._policy.config.use_amp
            else nullcontext()
        )
        with torch.inference_mode(), autocast_ctx:
            observation = prepare_observation_for_inference(
                observation, self._device, self._task, self._robot_type
            )
            action = self._server.get_action(observation)
        action_tensor = action.squeeze(0).cpu()

        action_dict = make_robot_action(action_tensor, self._dataset_features)
        return torch.tensor([action_dict[k] for k in self._ordered_action_keys])
