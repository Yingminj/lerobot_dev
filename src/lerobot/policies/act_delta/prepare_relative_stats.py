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
"""Build a relative-action stats view of a dataset, and report the Phase 2.0 gate.

Why a separate view instead of `lerobot-edit-dataset --operation.type=recompute_stats`:

* in-place recomputation *destroys* the absolute stats that arm R0 needs, and
* the out-of-place mode `shutil.copytree`s the whole dataset, videos included.

This creates a light-weight clone: `data/`, `videos/` and `images/` are symlinked, only
`meta/` is copied, and `meta/stats.json` gets its `action` entry replaced by statistics
computed in relative space (`action - observation.state` over `chunk_size`-long chunks,
excluding the named joints). Train R0 against the original root and R1/R2 against the
clones; the two normalizations then never overwrite each other.

It also prints the plan's §2.0 pre-gate: per-dimension `std(relative)/std(absolute)`.
Median well below 0.5 means the relative representation really does shrink the target
distribution; above 0.8 means the mechanism does not apply to this dataset.

Usage::

    python -m lerobot.policies.act_delta.prepare_relative_stats \
        --root /mnt/robot_platform/datasets/express \
        --chunk-size 100 --exclude-joints gripper

    # gate only, writes nothing
    python -m lerobot.policies.act_delta.prepare_relative_stats --root ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np

from lerobot.datasets.compute_stats import compute_relative_action_stats
from lerobot.datasets.io_utils import write_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION

LINKED_DIRS = ("data", "videos", "images")


def report_gate(
    absolute_std: np.ndarray,
    relative_std: np.ndarray,
    mask: list[bool],
    action_names: list[str] | None,
) -> float:
    """Print per-dimension std(relative)/std(absolute) and return the median over relative dims."""
    ratios = relative_std / np.maximum(absolute_std, 1e-8)
    names = action_names or [f"dim_{i}" for i in range(len(ratios))]
    print("\nper-dimension std(relative) / std(absolute):")
    for i, (name, ratio) in enumerate(zip(names, ratios, strict=False)):
        flag = "" if mask[i] else "   (kept absolute)"
        print(f"  {name:<28} {absolute_std[i]:>10.5f} → {relative_std[i]:>10.5f}   ratio {ratio:>6.3f}{flag}")
    relative_ratios = ratios[np.array(mask, dtype=bool)]
    median = float(np.median(relative_ratios)) if relative_ratios.size else float("nan")
    print(f"\nmedian ratio over relative dims: {median:.3f}")
    if median < 0.5:
        print("  → gate PASSED (<0.5): the relative representation shrinks the target distribution.")
    elif median > 0.8:
        print("  → gate FAILED (>0.8): little to gain here; consider running R2 only, or skipping Phase 2.")
    else:
        print("  → gate INCONCLUSIVE (0.5–0.8): proceed, but expect a small effect.")
    return median


def make_stats_view(
    root: Path,
    output_root: Path,
    stats: dict,
    metadata: dict,
) -> None:
    """Create a symlink clone of `root` at `output_root` carrying the new stats."""
    if output_root.exists():
        raise FileExistsError(f"{output_root} already exists. Remove it or pass a different --output-root.")
    output_root.mkdir(parents=True)
    shutil.copytree(root / "meta", output_root / "meta")
    for name in LINKED_DIRS:
        source = root / name
        if source.exists():
            (output_root / name).symlink_to(source.resolve(), target_is_directory=True)
    write_stats(stats, output_root)
    (output_root / "meta" / "relative_action_stats.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nwrote relative-stats dataset view: {output_root}")
    print(f"  provenance: {output_root / 'meta' / 'relative_action_stats.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, type=Path, help="Dataset folder (contains meta/, data/).")
    parser.add_argument("--repo-id", default=None, help="Defaults to the dataset folder name.")
    parser.add_argument(
        "--output-root",
        default=None,
        type=Path,
        help="Where to write the stats view. Defaults to <root>_rel<chunk-size>[_<excluded>].",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="MUST equal the policy's chunk_size (ACT default 100, not the lerobot default of 50).",
    )
    parser.add_argument(
        "--exclude-joints",
        nargs="*",
        default=["gripper"],
        help="Substrings of action dim names kept absolute. Pass with no values for arm R1.",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="Only report the gate; write nothing.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    root = args.root.expanduser().resolve()
    repo_id = args.repo_id or root.name
    dataset = LeRobotDataset(repo_id, root=root)
    features = dataset.meta.features
    action_names = features[ACTION].get("names")
    action_dim = features[ACTION]["shape"][0]

    from lerobot.processor import RelativeActionsProcessorStep

    mask = RelativeActionsProcessorStep(
        enabled=True, exclude_joints=list(args.exclude_joints), action_names=action_names
    )._build_mask(action_dim)
    n_relative = int(sum(mask))
    print(
        f"dataset {repo_id} ({root})\n"
        f"  action_dim={action_dim}, relative dims={n_relative}/{action_dim}, "
        f"exclude={args.exclude_joints or 'none'}, chunk_size={args.chunk_size}"
    )
    if args.exclude_joints and n_relative == action_dim:
        print(
            "  WARNING: the exclude list matched no action dimension name, so every dim will be "
            "relative (arm R2 collapses into arm R1). Action names are:\n"
            f"  {action_names}"
        )

    relative_stats = compute_relative_action_stats(
        hf_dataset=dataset.hf_dataset,
        features=features,
        chunk_size=args.chunk_size,
        exclude_joints=list(args.exclude_joints),
        num_workers=args.num_workers,
    )

    absolute_std = np.asarray(dataset.meta.stats[ACTION]["std"], dtype=np.float64).reshape(-1)
    relative_std = np.asarray(relative_stats["std"], dtype=np.float64).reshape(-1)
    median_ratio = report_gate(absolute_std, relative_std, mask, action_names)

    if args.dry_run:
        print("\n--dry-run: no files written.")
        return

    suffix = f"_rel{args.chunk_size}"
    if args.exclude_joints:
        suffix += "_keep-" + "-".join(args.exclude_joints)
    output_root = (args.output_root or root.with_name(root.name + suffix)).expanduser().resolve()

    stats = dict(dataset.meta.stats)
    stats[ACTION] = relative_stats
    metadata = {
        "source_root": str(root),
        "source_repo_id": repo_id,
        "relative_action": True,
        "relative_exclude_joints": list(args.exclude_joints),
        "chunk_size": args.chunk_size,
        "relative_dims": n_relative,
        "action_dim": action_dim,
        "median_std_ratio": median_ratio,
    }
    make_stats_view(root, output_root, stats, metadata)
    print(
        "\nTrain against it with:\n"
        f"  --dataset.repo_id={repo_id} --dataset.root={output_root} \\\n"
        "  --policy.type=act_delta --policy.use_relative_actions=true \\\n"
        f"  --policy.relative_exclude_joints=\"{list(args.exclude_joints)}\" "
        f"--policy.chunk_size={args.chunk_size}"
    )


if __name__ == "__main__":
    main()
