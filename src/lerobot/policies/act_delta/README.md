# `act_delta` — ACT with a relative (state-anchored) action representation

Phase 2 of `ACT-experiment-plan-2026-08.md`. `lerobot/policies/act/` is **not modified**;
this package is a parallel policy type registered as `act_delta`.

The network is the upstream one: `ACTDeltaPolicy` subclasses `ACTPolicy`, so module tree,
parameter names and forward maths are identical and `model.safetensors` is interchangeable
between the two. The only functional difference lives in the processor pipeline.

| file | what it is |
|---|---|
| `configuration_act_delta.py` | `ACTDeltaConfig` = `ACTConfig` fields + `use_relative_actions` / `relative_exclude_joints` / `action_feature_names` (same names as pi0/pi05) + consistency-check knobs |
| `processor_act_delta.py` | pre/post pipelines in OpenPI order, plus the silent-failure guards |
| `modeling_act_delta.py` | `ACTDeltaPolicy`; overrides only `select_action` (chunk-anchor guard) |
| `inference_act_delta.py` | `ChunkFIFOActionServer` / `ChunkFIFOInferenceEngine` (plan §2.3, option P-a) and `predict_absolute_chunk` for offline eval |
| `prepare_relative_stats.py` | builds the relative-space stats view of a dataset and prints the §2.0 pre-gate |
| `selftest.py` | CPU smoke test of all of the above |

`ACTDeltaConfig` is deliberately **not** a subclass of `ACTConfig`:
`policies/factory.py` dispatches processors on `isinstance(policy_cfg, ACTConfig)`, so a
subclass would silently be handed the absolute ACT pipeline. The only edit outside this
folder is one import line in `policies/__init__.py` that registers the config so
`--policy.type=act_delta` resolves.

## Pipeline order

```
raw → rename → batch → device → relative → normalize → MODEL → unnormalize → absolute → cpu
```

`relative` must come **before** `normalize` and `absolute` **after** `unnormalize`: the
conversion `action -= observation.state` is only meaningful in physical units. Both halves
share one `RelativeActionsProcessorStep` instance, which caches the anchor state.

## The three arms

| arm | flags |
|---|---|
| R0 | `--policy.use_relative_actions=false` |
| R1 | `--policy.use_relative_actions=true --policy.relative_exclude_joints="[]"` |
| R2 | `--policy.use_relative_actions=true --policy.relative_exclude_joints="['gripper']"` |

R2 is the pi0/pi05 default. `relative_exclude_joints` is matched as a **case-insensitive
substring** against the dataset's action dimension names, which
`lerobot.policies.factory.make_policy` fills into `action_feature_names` automatically.

## Workflow

### 1. Pre-gate + relative stats (once per dataset per chunk size)

Relative actions must be normalized by relative statistics. Recomputing them in place would
destroy the absolute statistics R0 needs, and lerobot's out-of-place mode copies the videos,
so this builds a light-weight *view*: `data/`, `videos/`, `images/` symlinked, `meta/` copied,
`meta/stats.json` with a relative `action` entry.

```bash
python -m lerobot.policies.act_delta.prepare_relative_stats \
    --root /mnt/robot_platform/datasets/express \
    --chunk-size 100 --exclude-joints gripper          # omit values entirely for R1
# --dry-run reports the gate without writing anything
```

It prints per-dimension `std(relative)/std(absolute)` and its median over the relative dims —
the plan's §2.0 gate (< 0.5 mechanism holds, > 0.8 it does not). Measured on `express`
(16-dim, chunk 100): **median 0.661** — inconclusive band, so expect a small effect.

`--chunk-size` **must** equal `policy.chunk_size` (100 for ACT; lerobot's own stats helpers
default to 50, which silently normalizes a 100-step target with a 50-step distribution).

### 2. Train

```bash
lerobot-train \
    --policy.type=act_delta \
    --policy.use_relative_actions=true \
    --policy.relative_exclude_joints="['gripper']" \
    --policy.chunk_size=100 --policy.n_action_steps=100 \
    --policy.device=cuda \
    --dataset.repo_id=express \
    --dataset.root=<the stats view printed in step 1> \
    --dataset.eval_split=0.2 --eval_steps=500 \
    --dataset.image_transforms.enable=false --ema.enable=false \
    --optim.grad_clip_norm=1.0 \
    --steps=100000 --batch_size=8 --seed=0 \
    --output_dir=outputs/R2_s0
```

R0 uses the same command with `--policy.use_relative_actions=false` and the **original**
dataset root.

Do not compare `eval_loss` across arms — it is computed in normalized space and the relative
arms have a smaller std, so their numbers are smaller for free (plan §0.2). Use absolute-space
MAE built on `predict_absolute_chunk`.

### 3. Guards (they fire at policy-construction time)

`--policy.relative_consistency_check` = `warn` (default) / `error` / `off`:

* the exclude list matches no action dimension name → R2 would silently become R1
  (this *will* happen on a 54-dim dexhand dataset, whose names contain no `gripper`);
* `action_feature_names` missing entirely → same failure mode;
* `dataset_stats[action]` still look absolute (`|mean|/std` over relative dims > 1) → you
  forgot step 1.

The mask is always logged: `[act_delta] relative action mask: 14/16 dims relative, excluded=[...]`.

## Deployment

Relative actions + an action queue drift: the caller re-runs the preprocessor every tick,
which re-caches the anchor state, so queued actions get re-anchored and the absolute target
walks away mid-chunk. Upstream rejects that combination outright
(`rollout/context.py`, `rollout/inference/sync.py`), and `ACTDeltaPolicy.select_action`
raises rather than drift silently.

Use the chunk-FIFO server, which converts a whole chunk to absolute actions in one
postprocessor call (one anchor per chunk, by construction):

```python
from lerobot.policies.act_delta import ChunkFIFOActionServer

server = ChunkFIFOActionServer(policy, preprocessor, postprocessor)
server.reset()                       # per episode
action = server.get_action(obs)      # absolute, unnormalized
```

For the LeRobot rollout stack, `ChunkFIFOInferenceEngine` has `SyncInferenceEngine`'s exact
constructor and return contract, but `build_rollout_context` only knows `sync` and `rtc`, so
either wire it in by hand or use `--inference.type=rtc` (RTC postprocesses whole chunks and is
unaffected). `n_action_steps=1` also avoids the problem, at maximum replanning cost.

Temporal ensembling is **rejected** with `use_relative_actions=True`: `ACTTemporalEnsembler`
averages raw model outputs, i.e. offsets anchored on different states. Phase 2 runs with
`temporal_ensemble_coeff=None`.

## Verify

```bash
python -m lerobot.policies.act_delta.selftest
```

Checks registration and factory wiring, pipeline order, a real train step for R0/R1/R2, the
relative conversion against a hand-computed target, the absolute round-trip, the
one-anchor-per-chunk invariant under a deliberately drifting state, and the `select_action`
guard.
