#!/usr/bin/env python

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

"""Dataset quality-index loading and invalid-anchor sampling for ``act_quality``."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow.parquet as pq
import torch

from lerobot.datasets import EpisodeAwareSampler

if TYPE_CHECKING:
    from lerobot.datasets import LeRobotDatasetMetadata

QUALITY_INVALID = 0
QUALITY_NORMAL = 1
QUALITY_RECOVERY = 2


@dataclass(frozen=True)
class QualitySamplingConfig:
    """Runtime-only controls for normal/onset/remainder anchor sampling."""

    balance_anchor_pools: bool = False
    recovery_anchor_fraction: float = 0.25
    recovery_onset_steps: int = 30
    recovery_onset_fraction: float = 0.10
    balanced_epoch_size: int = 0


@dataclass(frozen=True)
class _AnchorPool:
    """Compact mapping from a pool position to an absolute dataset index."""

    starts: np.ndarray
    cumulative_lengths: np.ndarray

    @property
    def size(self) -> int:
        if self.cumulative_lengths.size == 0:
            return 0
        return int(self.cumulative_lengths[-1])

    def absolute_indices(self, positions: torch.Tensor) -> torch.Tensor:
        if positions.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        positions_np = positions.numpy().astype(np.int64, copy=False)
        episodes = np.searchsorted(self.cumulative_lengths, positions_np, side="right")
        previous = np.zeros_like(positions_np)
        has_previous = episodes > 0
        previous[has_previous] = self.cumulative_lengths[episodes[has_previous] - 1]
        absolute = self.starts[episodes] + positions_np - previous
        return torch.from_numpy(absolute.astype(np.int64, copy=False))


@dataclass(frozen=True)
class QualityIndex:
    """Dense canonical 0/1/2 labels indexed by absolute dataset frame index.

    Bool datasets are upgraded in memory using manifest episode provenance:
    False becomes 0, normal True becomes 1, and recovery True becomes 2.
    Ternary datasets are already stored in this canonical representation.
    """

    labels: torch.Tensor
    episode_from_indices: torch.Tensor
    episode_to_indices: torch.Tensor
    first_valid_indices: torch.Tensor
    label_key: str
    recovery_episode_flags: torch.Tensor | None = None

    def valid_frame_mask(self) -> torch.Tensor:
        return self.labels != QUALITY_INVALID

    def normal_frame_mask(self) -> torch.Tensor:
        return self.labels == QUALITY_NORMAL

    def recovery_frame_mask(self) -> torch.Tensor:
        return self.labels == QUALITY_RECOVERY

    def recovery_episode_mask(self) -> torch.Tensor:
        """Return episode provenance, falling back to canonical label value 2."""
        if self.recovery_episode_flags is not None:
            if self.recovery_episode_flags.shape != self.episode_from_indices.shape:
                raise ValueError(
                    "recovery_episode_flags must align with episode metadata, got "
                    f"{tuple(self.recovery_episode_flags.shape)} and "
                    f"{tuple(self.episode_from_indices.shape)}."
                )
            return self.recovery_episode_flags.bool()
        flags = torch.zeros_like(self.episode_from_indices, dtype=torch.bool)
        for position, (start, stop) in enumerate(
            zip(
                self.episode_from_indices.tolist(),
                self.episode_to_indices.tolist(),
                strict=True,
            )
        ):
            flags[position] = bool((self.labels[start:stop] == QUALITY_RECOVERY).any())
        return flags

    @property
    def total_frames(self) -> int:
        return int(self.labels.numel())

    @property
    def total_episodes(self) -> int:
        return int(self.episode_from_indices.numel())

    @property
    def invalid_frames(self) -> int:
        return int((self.labels == QUALITY_INVALID).sum().item())

    @property
    def invalid_anchor_frames(self) -> int:
        return int((self.first_valid_indices - self.episode_from_indices).sum().item())

    @property
    def normal_anchor_frames(self) -> int:
        normal = ~self.recovery_episode_mask() & (
            self.first_valid_indices < self.episode_to_indices
        )
        return int(
            (self.episode_to_indices[normal] - self.first_valid_indices[normal]).sum().item()
        )

    @property
    def recovery_anchor_frames(self) -> int:
        recovery = self.recovery_episode_mask() & (
            self.first_valid_indices < self.episode_to_indices
        )
        return int(
            (self.episode_to_indices[recovery] - self.first_valid_indices[recovery]).sum().item()
        )

    @property
    def normal_episodes(self) -> int:
        normal = ~self.recovery_episode_mask() & (
            self.first_valid_indices < self.episode_to_indices
        )
        return int(normal.sum().item())

    @property
    def recovery_episodes(self) -> int:
        recovery = self.recovery_episode_mask()
        return int(recovery.sum().item())

    @property
    def episodes_without_valid_anchors(self) -> int:
        return int((self.first_valid_indices == self.episode_to_indices).sum().item())

    def recovery_anchor_mask(self) -> torch.Tensor:
        """Return a dense mask that is True only on valid recovery-suffix anchors."""
        return self.recovery_frame_mask()

    def recovery_onset_anchor_mask(self, onset_steps: int) -> torch.Tensor:
        """Return the first ``onset_steps`` valid anchors of each recovery episode."""
        if onset_steps <= 0:
            raise ValueError(f"onset_steps must be > 0, got {onset_steps}.")
        mask = torch.zeros(self.total_frames, dtype=torch.bool)
        for is_recovery, first_valid, stop in zip(
            self.recovery_episode_mask().tolist(),
            self.first_valid_indices.tolist(),
            self.episode_to_indices.tolist(),
            strict=True,
        ):
            if is_recovery and first_valid < stop:
                mask[first_valid : min(first_valid + onset_steps, stop)] = True
        return mask

    @classmethod
    def from_dataset_meta(
        cls,
        dataset_meta: LeRobotDatasetMetadata,
        label_key: str,
        *,
        require_monotonic: bool = True,
    ) -> QualityIndex:
        """Read bool or ternary labels, canonicalize to 0/1/2, and validate."""
        feature = dataset_meta.features.get(label_key)
        if feature is None:
            raise ValueError(f"Dataset is missing required quality feature {label_key!r}.")
        label_dtype = str(feature.get("dtype"))
        if label_dtype not in {"bool", "int64"} or tuple(feature.get("shape", ())) != (1,):
            raise ValueError(
                f"Quality feature {label_key!r} must be scalar bool or int64, got {feature!r}."
            )
        is_legacy_bool = label_dtype == "bool"

        total_frames = int(dataset_meta.total_frames)
        raw_labels = torch.empty(total_frames, dtype=torch.long)
        seen = torch.zeros(total_frames, dtype=torch.bool)
        parquet_paths = sorted((Path(dataset_meta.root) / "data").rglob("*.parquet"))
        if not parquet_paths:
            raise FileNotFoundError(f"No data parquet found below {Path(dataset_meta.root) / 'data'}.")

        for parquet_path in parquet_paths:
            parquet = pq.ParquetFile(parquet_path)
            schema_names = parquet.schema_arrow.names
            if label_key not in schema_names:
                raise ValueError(f"Parquet is missing {label_key!r}: {parquet_path}")
            table = parquet.read(columns=["index", label_key])
            if table.column(label_key).null_count:
                raise ValueError(f"Quality label contains null values: {parquet_path}")
            indices_np = table.column("index").to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            labels_np = table.column(label_key).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            indices = torch.from_numpy(np.array(indices_np, copy=True))
            shard_labels = torch.from_numpy(np.array(labels_np, copy=True)).long()

            if indices.numel() != torch.unique(indices).numel():
                raise ValueError(f"Duplicate index values inside parquet: {parquet_path}")
            if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= total_frames):
                raise ValueError(
                    f"Parquet index is outside [0, {total_frames - 1}]: {parquet_path}"
                )
            if seen[indices].any():
                duplicate = int(indices[seen[indices]][0])
                raise ValueError(f"Duplicate dataset index {duplicate} across parquet files.")
            raw_labels[indices] = shard_labels
            seen[indices] = True

        if not seen.all():
            missing = torch.nonzero(~seen, as_tuple=False).flatten()
            raise ValueError(
                f"Quality index is incomplete: {missing.numel()} frame(s) are missing; "
                f"first missing index={int(missing[0])}."
            )
        allowed_values = {QUALITY_INVALID, QUALITY_NORMAL} if is_legacy_bool else {
            QUALITY_INVALID,
            QUALITY_NORMAL,
            QUALITY_RECOVERY,
        }
        actual_values = set(raw_labels.unique().tolist())
        if not actual_values <= allowed_values:
            raise ValueError(
                f"Quality labels contain unsupported values {sorted(actual_values)}; "
                f"expected {sorted(allowed_values)} for dtype={label_dtype}."
            )

        episode_from = torch.as_tensor(
            np.asarray(dataset_meta.episodes["dataset_from_index"], dtype=np.int64)
        )
        episode_to = torch.as_tensor(
            np.asarray(dataset_meta.episodes["dataset_to_index"], dtype=np.int64)
        )
        if episode_from.numel() != episode_to.numel():
            raise ValueError("Episode metadata has mismatched from/to index arrays.")

        manifest_path = Path(dataset_meta.root) / "meta" / "quality_label_manifest.json"
        if manifest_path.is_file():
            with manifest_path.open(encoding="utf-8") as file:
                manifest = json.load(file)
            if str(manifest.get("label_key")) != label_key:
                raise ValueError(
                    f"Quality manifest label_key={manifest.get('label_key')!r} does not match "
                    f"the configured key {label_key!r}: {manifest_path}"
                )
            selections = manifest.get("selections")
            if not isinstance(selections, dict):
                raise ValueError(f"Quality manifest has no selections object: {manifest_path}")
            recovery_episode_flags = torch.zeros(episode_from.numel(), dtype=torch.bool)
            manifest_boundaries: dict[int, int] = {}
            for raw_position, selection in selections.items():
                position = int(raw_position)
                if not 0 <= position < episode_from.numel():
                    raise ValueError(
                        f"Quality manifest recovery episode position {position} is outside "
                        f"[0, {episode_from.numel() - 1}]."
                    )
                if not isinstance(selection, dict) or selection.get("kind") not in {
                    "recovery",
                    "all_valid",
                }:
                    raise ValueError(
                        f"Quality manifest selection {raw_position!r} has invalid kind: {selection!r}"
                    )
                recovery_episode_flags[position] = True
                start = int(episode_from[position])
                stop = int(episode_to[position])
                if selection["kind"] == "recovery":
                    manifest_boundaries[position] = start + int(selection["recovery_start_frame"])
                else:
                    manifest_boundaries[position] = start
            expected_recovery_count = manifest.get("recovery_candidate_count")
            if expected_recovery_count is not None and int(expected_recovery_count) != int(
                recovery_episode_flags.sum()
            ):
                raise ValueError(
                    "Quality manifest recovery_candidate_count does not match selections: "
                    f"{expected_recovery_count} != {int(recovery_episode_flags.sum())}."
                )
            logging.info(
                "[act_quality] loaded recovery episode provenance for %d episodes from %s",
                int(recovery_episode_flags.sum()),
                manifest_path,
            )
        else:
            manifest_boundaries = {}
            if is_legacy_bool:
                recovery_episode_flags = torch.zeros(episode_from.numel(), dtype=torch.bool)
                for position, (start, stop) in enumerate(
                    zip(episode_from.tolist(), episode_to.tolist(), strict=True)
                ):
                    segment = raw_labels[start:stop]
                    recovery_episode_flags[position] = bool(
                        (segment == QUALITY_INVALID).any() and (segment == QUALITY_NORMAL).any()
                    )
                logging.warning(
                    "[act_quality] %s is absent; bool recovery episodes are inferred only from "
                    "0-prefix/1-suffix patterns. All-valid recovery recordings are "
                    "indistinguishable from normal success.",
                    manifest_path,
                )
            else:
                recovery_episode_flags = torch.zeros(episode_from.numel(), dtype=torch.bool)
                for position, (start, stop) in enumerate(
                    zip(episode_from.tolist(), episode_to.tolist(), strict=True)
                ):
                    recovery_episode_flags[position] = bool(
                        (raw_labels[start:stop] == QUALITY_RECOVERY).any()
                    )
                logging.info(
                    "[act_quality] %s is absent; ternary recovery provenance is inferred from "
                    "label value 2.",
                    manifest_path,
                )

        # Canonical in-memory representation is always 0/1/2. This lets old
        # bool datasets use the same three-pool sampler without rewriting them.
        if is_legacy_bool:
            labels = torch.zeros_like(raw_labels)
            for is_recovery, start, stop in zip(
                recovery_episode_flags.tolist(),
                episode_from.tolist(),
                episode_to.tolist(),
                strict=True,
            ):
                valid = raw_labels[start:stop] != QUALITY_INVALID
                segment = labels[start:stop]
                segment[valid] = QUALITY_RECOVERY if is_recovery else QUALITY_NORMAL
        else:
            labels = raw_labels.clone()

        first_valid = episode_to.clone()
        for episode_pos, (start, stop, is_recovery) in enumerate(
            zip(
                episode_from.tolist(),
                episode_to.tolist(),
                recovery_episode_flags.tolist(),
                strict=True,
            )
        ):
            segment = labels[start:stop]
            if segment.numel() == 0:
                raise ValueError(f"Episode position {episode_pos} has no frames.")
            valid = segment != QUALITY_INVALID
            valid_offsets = torch.nonzero(valid, as_tuple=False).flatten()
            if valid_offsets.numel():
                first_valid[episode_pos] = start + valid_offsets[0]
            if not require_monotonic:
                continue
            if valid.numel() > 1 and (valid[:-1] & ~valid[1:]).any():
                raise ValueError(
                    f"Episode position {episode_pos} has a valid->invalid transition; expected "
                    "one invalid prefix followed by one valid suffix."
                )
            expected_value = QUALITY_RECOVERY if is_recovery else QUALITY_NORMAL
            unexpected = segment[valid] != expected_value
            if unexpected.any():
                raise ValueError(
                    f"Episode position {episode_pos} is classified as "
                    f"{'recovery' if is_recovery else 'normal'} but its valid suffix contains "
                    f"labels {sorted(set(segment[valid].tolist()))}; expected only "
                    f"{expected_value}."
                )
            if not is_recovery and not valid.all():
                raise ValueError(
                    f"Normal episode position {episode_pos} contains label 0; normal episodes "
                    "must be all 1."
                )

        for position, expected_first_valid in manifest_boundaries.items():
            if int(first_valid[position]) != expected_first_valid:
                raise ValueError(
                    f"Quality manifest recovery boundary for episode position {position} is "
                    f"{expected_first_valid}, but parquet labels start at "
                    f"{int(first_valid[position])}."
                )

        result = cls(
            labels=labels,
            episode_from_indices=episode_from,
            episode_to_indices=episode_to,
            first_valid_indices=first_valid,
            label_key=label_key,
            recovery_episode_flags=recovery_episode_flags,
        )
        logging.info(
            "[act_quality] loaded %s (%s -> canonical int64 0/1/2): %d frames, "
            "%d invalid target frames, "
            "%d invalid anchors, %d normal anchors, %d recovery anchors, "
            "%d normal episodes, %d recovery episodes, %d episodes without valid anchors",
            label_key,
            label_dtype,
            result.total_frames,
            result.invalid_frames,
            result.invalid_anchor_frames,
            result.normal_anchor_frames,
            result.recovery_anchor_frames,
            result.normal_episodes,
            result.recovery_episodes,
            result.episodes_without_valid_anchors,
        )
        return result


_ACTIVE_QUALITY_INDEX: QualityIndex | None = None
_ACTIVE_SAMPLING_CONFIG = QualitySamplingConfig()


def activate_quality_index(
    quality_index: QualityIndex,
    *,
    balance_anchor_pools: bool = False,
    recovery_anchor_fraction: float = 0.25,
    recovery_onset_steps: int = 30,
    recovery_onset_fraction: float = 0.10,
    balanced_epoch_size: int = 0,
) -> None:
    """Make the index and sampling policy available to the train-process hook."""
    if not 0.0 <= recovery_anchor_fraction <= 1.0:
        raise ValueError(
            f"recovery_anchor_fraction must be in [0, 1], got {recovery_anchor_fraction}."
        )
    if recovery_onset_steps <= 0:
        raise ValueError(f"recovery_onset_steps must be > 0, got {recovery_onset_steps}.")
    if not 0.0 <= recovery_onset_fraction <= recovery_anchor_fraction:
        raise ValueError(
            "recovery_onset_fraction must be in [0, recovery_anchor_fraction], got "
            f"{recovery_onset_fraction} and recovery_anchor_fraction="
            f"{recovery_anchor_fraction}."
        )
    if balanced_epoch_size < 0:
        raise ValueError(f"balanced_epoch_size must be >= 0, got {balanced_epoch_size}.")
    global _ACTIVE_QUALITY_INDEX, _ACTIVE_SAMPLING_CONFIG
    _ACTIVE_QUALITY_INDEX = quality_index
    _ACTIVE_SAMPLING_CONFIG = QualitySamplingConfig(
        balance_anchor_pools=balance_anchor_pools,
        recovery_anchor_fraction=recovery_anchor_fraction,
        recovery_onset_steps=recovery_onset_steps,
        recovery_onset_fraction=recovery_onset_fraction,
        balanced_epoch_size=balanced_epoch_size,
    )


class QualityAwareEpisodeSampler(EpisodeAwareSampler):
    """Filter invalid anchors and optionally balance three semantic anchor pools.

    Episode provenance comes from ``quality_label_manifest.json`` when present,
    so an all-valid recovery episode stays in the recovery pool. Without that
    manifest, bool labels only identify episodes with a detectable False prefix;
    ternary labels identify recovery episodes directly through value 2. The
    balanced epoch draws exact normal, recovery-onset, and recovery-remainder
    quotas, repeating shuffled passes only when a pool is oversampled.
    """

    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
        seed: int = 0,
        absolute_to_relative_idx: dict[int, int] | None = None,
    ) -> None:
        if _ACTIVE_QUALITY_INDEX is None:
            raise RuntimeError(
                "QualityAwareEpisodeSampler was installed before a QualityIndex was activated."
            )
        quality = _ACTIVE_QUALITY_INDEX
        original_from = np.asarray(dataset_from_indices, dtype=np.int64)
        original_to = np.asarray(dataset_to_indices, dtype=np.int64)
        if len(original_from) != quality.total_episodes:
            raise ValueError(
                f"Sampler received {len(original_from)} episodes but quality index has "
                f"{quality.total_episodes}."
            )
        expected_from = quality.episode_from_indices.numpy()
        expected_to = quality.episode_to_indices.numpy()
        if not np.array_equal(original_from, expected_from) or not np.array_equal(
            original_to, expected_to
        ):
            raise ValueError("Sampler episode boundaries do not match the active quality index.")

        adjusted_from = np.maximum(original_from, quality.first_valid_indices.numpy())
        removed = int((adjusted_from - original_from).sum())
        logging.info("[act_quality] sampler excludes %d invalid anchor frames per epoch", removed)
        super().__init__(
            adjusted_from.tolist(),
            original_to.tolist(),
            episode_indices_to_use=episode_indices_to_use,
            drop_n_first_frames=drop_n_first_frames,
            drop_n_last_frames=drop_n_last_frames,
            shuffle=shuffle,
            seed=seed,
            absolute_to_relative_idx=absolute_to_relative_idx,
        )
        self._balance_anchor_pools = _ACTIVE_SAMPLING_CONFIG.balance_anchor_pools
        self._normal_pool = _AnchorPool(
            starts=np.empty(0, dtype=np.int64),
            cumulative_lengths=np.empty(0, dtype=np.int64),
        )
        self._recovery_onset_pool = self._normal_pool
        self._recovery_remainder_pool = self._normal_pool
        self._normal_samples_per_epoch = 0
        self._recovery_onset_samples_per_epoch = 0
        self._recovery_remainder_samples_per_epoch = 0
        self._absolute_to_relative_balanced = absolute_to_relative_idx

        if not self._balance_anchor_pools:
            logging.info(
                "[act_quality] anchor-pool balancing disabled; using %d naturally sampled valid anchors",
                len(self),
            )
            return

        used = np.ones(len(original_from), dtype=bool)
        if episode_indices_to_use is not None:
            used = np.zeros(len(original_from), dtype=bool)
            used[np.asarray(episode_indices_to_use, dtype=np.int64)] = True

        starts = adjusted_from + drop_n_first_frames
        stops = original_to - drop_n_last_frames
        usable = used & (stops > starts)
        recovery_provenance = quality.recovery_episode_mask().numpy()
        normal_episodes = usable & ~recovery_provenance
        recovery_episodes = usable & recovery_provenance
        onset_stops = np.minimum(
            stops,
            quality.first_valid_indices.numpy()
            + _ACTIVE_SAMPLING_CONFIG.recovery_onset_steps,
        )
        remainder_starts = np.maximum(starts, onset_stops)
        self._normal_pool = self._make_interval_pool(starts, stops, normal_episodes)
        self._recovery_onset_pool = self._make_interval_pool(
            starts,
            onset_stops,
            recovery_episodes & (onset_stops > starts),
        )
        self._recovery_remainder_pool = self._make_interval_pool(
            remainder_starts,
            stops,
            recovery_episodes & (stops > remainder_starts),
        )

        natural_epoch_size = len(self)
        epoch_size = _ACTIVE_SAMPLING_CONFIG.balanced_epoch_size or natural_epoch_size
        recovery_samples = int(
            np.floor(epoch_size * _ACTIVE_SAMPLING_CONFIG.recovery_anchor_fraction + 0.5)
        )
        recovery_onset_samples = int(
            np.floor(epoch_size * _ACTIVE_SAMPLING_CONFIG.recovery_onset_fraction + 0.5)
        )
        recovery_onset_samples = min(recovery_onset_samples, recovery_samples)
        recovery_remainder_samples = recovery_samples - recovery_onset_samples
        normal_samples = epoch_size - recovery_samples
        if recovery_onset_samples and self._recovery_onset_pool.size == 0:
            raise ValueError(
                "Balanced sampling requested recovery-onset anchors, but no onset anchor remains "
                "after episode selection and frame dropping. Set "
                "`quality_recovery_onset_fraction=0` or fix the labels."
            )
        if recovery_remainder_samples and self._recovery_remainder_pool.size == 0:
            raise ValueError(
                "Balanced sampling requested recovery-remainder anchors, but no remainder anchor "
                "remains after episode selection and frame dropping. Set "
                "`quality_recovery_onset_fraction=quality_recovery_anchor_fraction`, reduce "
                "`quality_recovery_onset_steps`, or add longer recovery suffixes."
            )
        if normal_samples and self._normal_pool.size == 0:
            raise ValueError(
                "Balanced sampling requested normal-success anchors, but no normal anchor remains "
                "after episode selection and frame dropping. Set "
                "`quality_recovery_anchor_fraction=1` or include normal demonstrations."
            )

        self._normal_samples_per_epoch = normal_samples
        self._recovery_onset_samples_per_epoch = recovery_onset_samples
        self._recovery_remainder_samples_per_epoch = recovery_remainder_samples
        self._num_frames = epoch_size
        normal_repeat = normal_samples / self._normal_pool.size if self._normal_pool.size else 0.0
        onset_repeat = (
            recovery_onset_samples / self._recovery_onset_pool.size
            if self._recovery_onset_pool.size
            else 0.0
        )
        remainder_repeat = (
            recovery_remainder_samples / self._recovery_remainder_pool.size
            if self._recovery_remainder_pool.size
            else 0.0
        )
        logging.info(
            "[act_quality] balanced anchor pools: normal_unique=%d, onset_unique=%d, "
            "remainder_unique=%d, epoch_size=%d, normal_samples=%d (%.2f%%), "
            "onset_samples=%d (%.2f%%), remainder_samples=%d (%.2f%%), "
            "sampling_factors=normal %.3fx/onset %.3fx/remainder %.3fx",
            self._normal_pool.size,
            self._recovery_onset_pool.size,
            self._recovery_remainder_pool.size,
            epoch_size,
            normal_samples,
            100.0 * normal_samples / epoch_size,
            recovery_onset_samples,
            100.0 * recovery_onset_samples / epoch_size,
            recovery_remainder_samples,
            100.0 * recovery_remainder_samples / epoch_size,
            normal_repeat,
            onset_repeat,
            remainder_repeat,
        )

    @staticmethod
    def _make_interval_pool(
        starts: np.ndarray, stops: np.ndarray, selected: np.ndarray
    ) -> _AnchorPool:
        selected_starts = starts[selected].astype(np.int64, copy=False)
        selected_lengths = (stops[selected] - starts[selected]).astype(np.int64, copy=False)
        return _AnchorPool(
            starts=selected_starts,
            cumulative_lengths=np.cumsum(selected_lengths, dtype=np.int64),
        )

    @property
    def normal_pool_size(self) -> int:
        return self._normal_pool.size

    @property
    def recovery_pool_size(self) -> int:
        return self.recovery_onset_pool_size + self.recovery_remainder_pool_size

    @property
    def recovery_onset_pool_size(self) -> int:
        return self._recovery_onset_pool.size

    @property
    def recovery_remainder_pool_size(self) -> int:
        return self._recovery_remainder_pool.size

    @property
    def normal_samples_per_epoch(self) -> int:
        return self._normal_samples_per_epoch

    @property
    def recovery_samples_per_epoch(self) -> int:
        return (
            self._recovery_onset_samples_per_epoch
            + self._recovery_remainder_samples_per_epoch
        )

    @property
    def recovery_onset_samples_per_epoch(self) -> int:
        return self._recovery_onset_samples_per_epoch

    @property
    def recovery_remainder_samples_per_epoch(self) -> int:
        return self._recovery_remainder_samples_per_epoch

    @property
    def indices(self) -> list[int]:
        if not self._balance_anchor_pools:
            return super().indices
        return list(self._iter_epoch(epoch=0, start=0))

    @staticmethod
    def _sample_pool_positions(
        pool_size: int,
        sample_count: int,
        *,
        generator: torch.Generator,
        shuffle: bool,
    ) -> torch.Tensor:
        if sample_count == 0:
            return torch.empty(0, dtype=torch.long)
        if not shuffle:
            return torch.arange(sample_count, dtype=torch.long).remainder(pool_size)

        complete_passes, remainder = divmod(sample_count, pool_size)
        parts = [torch.randperm(pool_size, generator=generator) for _ in range(complete_passes)]
        if remainder:
            parts.append(torch.randperm(pool_size, generator=generator)[:remainder])
        return torch.cat(parts)

    @staticmethod
    def _interleave_without_shuffle(*groups: torch.Tensor) -> torch.Tensor:
        """Distribute any number of ordered groups approximately uniformly."""
        nonempty = [group for group in groups if group.numel()]
        if not nonempty:
            return torch.empty(0, dtype=torch.long)
        priorities = [
            (torch.arange(group.numel(), dtype=torch.float64) + 0.5) / group.numel()
            for group in nonempty
        ]
        combined = torch.cat(nonempty)
        order = torch.argsort(torch.cat(priorities), stable=True)
        return combined[order]

    def _balanced_epoch_indices(self, epoch: int) -> torch.Tensor:
        generator = self._epoch_generator(epoch)
        normal_positions = self._sample_pool_positions(
            self._normal_pool.size,
            self._normal_samples_per_epoch,
            generator=generator,
            shuffle=self.shuffle,
        )
        onset_positions = self._sample_pool_positions(
            self._recovery_onset_pool.size,
            self._recovery_onset_samples_per_epoch,
            generator=generator,
            shuffle=self.shuffle,
        )
        remainder_positions = self._sample_pool_positions(
            self._recovery_remainder_pool.size,
            self._recovery_remainder_samples_per_epoch,
            generator=generator,
            shuffle=self.shuffle,
        )
        normal_indices = self._normal_pool.absolute_indices(normal_positions)
        onset_indices = self._recovery_onset_pool.absolute_indices(onset_positions)
        remainder_indices = self._recovery_remainder_pool.absolute_indices(
            remainder_positions
        )
        if self.shuffle:
            combined = torch.cat((normal_indices, onset_indices, remainder_indices))
            return combined[torch.randperm(combined.numel(), generator=generator)]
        return self._interleave_without_shuffle(
            normal_indices,
            onset_indices,
            remainder_indices,
        )

    def _iter_epoch(self, epoch: int, start: int) -> Iterator[int]:
        if not self._balance_anchor_pools:
            yield from super()._iter_epoch(epoch, start)
            return

        order = self._balanced_epoch_indices(epoch)
        for position in range(start, order.numel()):
            absolute_index = int(order[position])
            if self._absolute_to_relative_balanced is not None:
                yield self._absolute_to_relative_balanced[absolute_index]
            else:
                yield absolute_index


def install_training_sampler_hook() -> bool:
    """Replace only the active train module's sampler symbol at runtime.

    No file outside this package is edited.  ``lerobot-train`` imports its sampler
    before policy construction and constructs it afterward, so installing here is
    early enough for the current run while leaving all other processes untouched.
    """
    candidates = []
    named_module = sys.modules.get("lerobot.scripts.lerobot_train")
    if named_module is not None:
        candidates.append(named_module)
    main_module = sys.modules.get("__main__")
    main_file = str(getattr(main_module, "__file__", ""))
    if main_module is not None and main_file.endswith("lerobot_train.py"):
        candidates.append(main_module)

    if not candidates:
        logging.debug("[act_quality] lerobot_train is not active; sampler hook not needed")
        return False

    installed = False
    for train_module in candidates:
        current = getattr(train_module, "EpisodeAwareSampler", None)
        if current is QualityAwareEpisodeSampler:
            installed = True
            continue
        if current is not None:
            setattr(train_module, "EpisodeAwareSampler", QualityAwareEpisodeSampler)
            installed = True
    if not installed:
        return False
    logging.info("[act_quality] installed quality-aware EpisodeAwareSampler for this train process")
    return True
