#!/usr/bin/env python

"""Build one S/M/E semantic quality dataset from two existing manual label passes.

The reentry dataset supplies S and M (return to the correct ready pose). The
finish dataset supplies the same S and the later E (realignment/insertion
finished). Videos and unchanged metadata are hard-linked when possible; only
the parquet quality column and semantic manifest are rewritten.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .quality_index import QUALITY_INVALID, QUALITY_NORMAL, QUALITY_REENTRY, QUALITY_ROLLBACK


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary, path)


def _link_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def _semantic_selections(reentry: dict, finish: dict) -> dict[str, dict]:
    left = reentry.get("selections")
    right = finish.get("selections")
    if not isinstance(left, dict) or not isinstance(right, dict) or set(left) != set(right):
        raise ValueError("The reentry and finish manifests must contain the same selections.")

    result: dict[str, dict] = {}
    for raw_position in sorted(left, key=int):
        first = left[raw_position]
        second = right[raw_position]
        if first.get("kind") != second.get("kind"):
            raise ValueError(f"Selection {raw_position} changes kind between manifests.")
        if first.get("kind") == "all_valid":
            result[raw_position] = dict(first)
            continue
        if first.get("kind") != "recovery":
            raise ValueError(f"Selection {raw_position} has unsupported kind {first.get('kind')!r}.")
        start_a = int(first["recovery_start_frame"])
        start_b = int(second["recovery_start_frame"])
        mid = int(first["recovery_end_frame"])
        end = int(second["recovery_end_frame"])
        if start_a != start_b or not start_a < mid < end:
            raise ValueError(
                f"Selection {raw_position} must satisfy common S < M < E, got "
                f"S={start_a}/{start_b}, M={mid}, E={end}."
            )
        length = int(first.get("source_length", second.get("source_length", end + 1)))
        if end >= length:
            raise ValueError(f"Selection {raw_position} has E={end} outside length={length}.")
        result[raw_position] = {
            "kind": "recovery",
            "source_episode_index": int(first.get("source_episode_index", raw_position)),
            "source_length": length,
            "recovery_start_frame": start_a,
            "recovery_mid_frame": mid,
            "recovery_end_frame": end,
            "invalid_prefix_frames": start_a,
            "rollback_frames": mid - start_a,
            "reentry_frames": end - mid,
            "post_recovery_normal_frames": length - end,
            "valid_frames": length - start_a,
            "confirmed_at": first.get("confirmed_at"),
        }
    return result


def _rewrite_parquets(root: Path, selections: dict[str, dict], label_key: str) -> Counter:
    by_episode = {
        int(selection.get("source_episode_index", raw_position)): selection
        for raw_position, selection in selections.items()
    }
    counts: Counter = Counter()
    seen_recovery: set[int] = set()
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        for required in ("episode_index", "frame_index", label_key):
            if required not in table.schema.names:
                raise ValueError(f"Parquet is missing {required!r}: {path}")
        episode_indices = table["episode_index"].combine_chunks().to_numpy().astype(np.int64)
        frame_indices = table["frame_index"].combine_chunks().to_numpy().astype(np.int64)
        labels = np.full(len(table), QUALITY_NORMAL, dtype=np.int64)
        for episode_index in np.unique(episode_indices):
            selection = by_episode.get(int(episode_index))
            if selection is None or selection["kind"] == "all_valid":
                continue
            seen_recovery.add(int(episode_index))
            rows = episode_indices == episode_index
            frames = frame_indices[rows]
            start = int(selection["recovery_start_frame"])
            mid = int(selection["recovery_mid_frame"])
            end = int(selection["recovery_end_frame"])
            episode_labels = np.full(frames.shape, QUALITY_NORMAL, dtype=np.int64)
            episode_labels[frames < start] = QUALITY_INVALID
            episode_labels[(frames >= start) & (frames < mid)] = QUALITY_ROLLBACK
            episode_labels[(frames >= mid) & (frames < end)] = QUALITY_REENTRY
            labels[rows] = episode_labels

        counts.update(dict(zip(*np.unique(labels, return_counts=True), strict=True)))
        quality_column = pa.array(labels, type=pa.int64())
        table = table.set_column(table.schema.get_field_index(label_key), label_key, quality_column)
        temporary = path.with_name(f".{path.name}.semantic-{os.getpid()}")
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, path)

    expected_recovery = {
        int(selection.get("source_episode_index", raw_position))
        for raw_position, selection in selections.items()
        if selection["kind"] == "recovery"
    }
    if seen_recovery != expected_recovery:
        missing = sorted(expected_recovery - seen_recovery)
        raise ValueError(f"Recovery episodes were not found in parquet data: {missing[:12]}")
    return counts


def build_dataset(reentry_root: Path, finish_root: Path, output_root: Path) -> dict:
    reentry_root = reentry_root.resolve()
    finish_root = finish_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output_root}")
    for root in (reentry_root, finish_root):
        if not (root / "meta" / "quality_label_manifest.json").is_file():
            raise FileNotFoundError(f"Missing quality manifest below {root}")

    reentry_info = _read_json(reentry_root / "meta" / "info.json")
    finish_info = _read_json(finish_root / "meta" / "info.json")
    for key in ("total_episodes", "total_frames", "fps", "data_path", "video_path"):
        if reentry_info.get(key) != finish_info.get(key):
            raise ValueError(f"Dataset metadata differs for {key!r}.")
    label_key = "action_quality"
    reentry_manifest = _read_json(reentry_root / "meta" / "quality_label_manifest.json")
    finish_manifest = _read_json(finish_root / "meta" / "quality_label_manifest.json")
    if reentry_manifest.get("label_key") != label_key or finish_manifest.get("label_key") != label_key:
        raise ValueError("Both input manifests must use label_key='action_quality'.")
    selections = _semantic_selections(reentry_manifest, finish_manifest)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent))
    try:
        shutil.rmtree(staging)
        shutil.copytree(reentry_root, staging, copy_function=_link_or_copy, symlinks=True)
        counts = _rewrite_parquets(staging, selections, label_key)
        total_frames = int(reentry_info["total_frames"])
        if sum(counts.values()) != total_frames:
            raise ValueError(f"Rewritten labels cover {sum(counts.values())} != {total_frames} frames.")

        manifest = {
            "format_version": 4,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_reentry_root": str(reentry_root),
            "source_finish_root": str(finish_root),
            "output_root": str(output_root),
            "label_key": label_key,
            "label_semantics": (
                "0 before S (invalid error-producing prefix), 2 from S to M "
                "(rollback to the correct ready pose), 3 from M to E (realign and insert), "
                "and 1 from E onward (normal suffix)."
            ),
            "label_dtype": "int64",
            "label_values": {
                "0": "invalid_error_prefix",
                "1": "normal",
                "2": "rollback_S_to_M",
                "3": "reentry_M_to_E",
            },
            "recovery_candidate_count": len(selections),
            "recovery_episode_count": sum(s["kind"] == "recovery" for s in selections.values()),
            "all_valid_recovery_candidates": sum(
                s["kind"] == "all_valid" for s in selections.values()
            ),
            "total_frames": total_frames,
            "invalid_frames": int(counts[QUALITY_INVALID]),
            "normal_frames": int(counts[QUALITY_NORMAL]),
            "rollback_frames": int(counts[QUALITY_ROLLBACK]),
            "reentry_frames": int(counts[QUALITY_REENTRY]),
            "valid_frames": total_frames - int(counts[QUALITY_INVALID]),
            "selections": selections,
        }
        _write_json(staging / "meta" / "quality_label_manifest.json", manifest)
        os.rename(staging, output_root)
        return manifest
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reentry-root", type=Path, required=True)
    parser.add_argument("--finish-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_dataset(args.reentry_root, args.finish_root, args.output)
    print(json.dumps({key: manifest[key] for key in (
        "output_root",
        "total_frames",
        "invalid_frames",
        "normal_frames",
        "rollback_frames",
        "reentry_frames",
    )}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
