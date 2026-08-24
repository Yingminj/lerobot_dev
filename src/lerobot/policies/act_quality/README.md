# `act_quality` — ACT with hard recovery-quality masks

This package implements recovery-data filtering without editing any file outside
`lerobot/policies/act_quality/`.

The ACT network architecture and state-dict names are unchanged.  The package adds:

1. a scalar ternary dataset feature (`action_quality` by default), with legacy
   boolean-label compatibility;
2. a dense read-only lookup from global frame `index` to quality;
3. a sampler hook that starts every episode at its first non-zero frame;
4. a `[B,H]` target mask constructed from the lookup;
5. the same mask in reconstruction loss and the CVAE key-padding mask;
6. zeroed invalid action tokens before the CVAE action projection;
7. valid-element L1 and valid-sample KL renormalization;
8. separate normal-action, recovery-onset, and recovery-remainder anchor pools
   with configurable epoch quotas.

## ACT configuration inheritance

`ACTQualityConfig` subclasses the upstream `ACTConfig`; it does not replace it.
Consequently every standard ACT option remains available under `--policy.*`,
including `chunk_size`, `n_action_steps`, backbone and transformer dimensions,
CVAE settings, `temporal_ensemble_coeff`, `dropout`, `kl_weight`, normalization,
and optimizer settings. `configuration_act_quality.py` only declares the eight
additional quality options. Keeping one upstream definition prevents the two
ACT configurations from silently drifting apart.

## Required label semantics

The preferred dataset dtype is scalar `int64`. For every episode, labels must
have one of these forms:

```text
1 1 1 1 ...             # normal-success or explicitly all-valid episode
0 0 0 2 2 2 1 1 ...     # mistake prefix, active recovery, normal continuation
```

The meanings are:

- `0`: invalid target/anchor; excluded from reconstruction loss, CVAE action
  encoding, and the main sampler;
- `1`: valid normal-execution target/anchor, including frames after recovery E;
- `2`: valid active-recovery target/anchor in `[S,E)`.

Labels `1` and `2` use the same reconstruction and KL objectives. Their numeric
values are never used as loss weights. Value `2` exists so the sampler can
split the active-recovery interval into its first
`quality_recovery_onset_steps` anchors and the remaining anchors. Label `1`
after E returns to the normal pool.

Normal/all-valid episodes must be entirely `1`. Recovery episodes must follow
`0* -> 2+ -> 1*`; with the semantic labeling tool this is `[0,S)=0`,
`[S,E)=2`, and `[E,end)=1`. Other transitions are rejected by default. This
validation allows the sampler to skip invalid prefixes and stop recovery pools
exactly at E.

Legacy scalar `bool` datasets remain supported. They are canonicalized in
memory to `0/1/2`: manifest-marked recovery `True` frames become `2`, normal or
`all_valid` `True` frames become `1`, and `False` remains `0`. Legacy bool data
has no E boundary, so a recovery's valid suffix remains all `2`. New datasets
should use ternary `int64` to represent `0→2→1` directly.

When `meta/quality_label_manifest.json` is present, its `selections` identify
true recovery episodes and explicitly reclassify `all_valid` candidates as
normal. Manifest S/E boundaries are checked against parquet labels. Without the
manifest, ternary data identifies recovery episodes directly from value `2`.
Legacy bool data falls back to a non-empty `False` prefix plus a `True` suffix.
Only non-zero frames enter the three valid anchor pools.

## Train

Load the package through LeRobot's existing plugin interface:

```bash
PYTHONPATH=/home/snorlax/repo/robot_data_platform/lerobot/src \
/home/snorlax/.conda/envs/lerobot/bin/lerobot-train \
  --policy.discover_packages_path=lerobot.policies.act_quality \
  --policy.type=act_quality \
  --policy.chunk_size=100 \
  --policy.n_action_steps=100 \
  --policy.quality_label_key=action_quality \
  --policy.quality_balance_anchor_pools=true \
  --policy.quality_recovery_anchor_fraction=0.25 \
  --policy.quality_recovery_onset_steps=30 \
  --policy.quality_recovery_onset_fraction=0.10 \
  --dataset.repo_id=local/my_quality_dataset \
  --dataset.root=/path/to/my_quality_dataset \
  --output_dir=/path/to/output
```

No edit to `policies/__init__.py`, `policies/factory.py`, dataset factory, sampler,
or upstream ACT is needed.  Plugin discovery registers `ACTQualityConfig`; the
generic factory derives `ACTQualityPolicy` from the class name and module path.
Because the config subclasses `ACTConfig`, the existing factory deliberately routes
normalization through the upstream ACT processor pipeline.

## Three-pool anchor sampling

Balanced sampling is enabled by default. `quality_recovery_anchor_fraction`
controls the exact total fraction of recovery anchors in each sampler epoch.
`quality_recovery_onset_steps` defines the first `K` label-2 frames of every
active-recovery interval as onset anchors. `quality_recovery_onset_fraction` controls
their fraction of the complete epoch and is included inside—not added on top
of—the total recovery fraction.

With the defaults, each epoch is exactly:

```text
normal label 1       75%   (success data plus post-E normal continuation)
recovery onset       10%   (first 30 label-2 anchors per recovery episode)
recovery remainder   15%
```

`quality_balanced_epoch_size=0` preserves the natural total number of valid
anchors as the epoch length. A positive value sets an explicit epoch length.
When a requested pool quota is larger than the number of unique anchors in that
pool, the sampler repeats independently shuffled full passes through that pool.
It therefore provides exact epoch-level three-pool ratios while maximizing
unique-anchor coverage before duplication. Individual minibatches fluctuate
around the target ratio because the combined epoch is shuffled.

Useful ablations are:

```bash
# Natural post-mask distribution (previous behavior)
--policy.quality_balance_anchor_pools=false

# Recovery anchors are 25% of each epoch
--policy.quality_balance_anchor_pools=true \
--policy.quality_recovery_anchor_fraction=0.25 \
--policy.quality_recovery_onset_steps=30 \
--policy.quality_recovery_onset_fraction=0.10

# 50% total recovery, of which onset is 20% of the whole epoch
--policy.quality_balance_anchor_pools=true \
--policy.quality_recovery_anchor_fraction=0.5 \
--policy.quality_recovery_onset_steps=30 \
--policy.quality_recovery_onset_fraction=0.2
```

The training logs report unique pool sizes, per-epoch quotas, and the effective
oversampling factor for each pool. Policy metrics also report normal, total
recovery, recovery-onset, and recovery-remainder fractions for each minibatch.

## Initialize from an ACT checkpoint

The model module tree is identical to ACT.  A fresh `act_quality` config may set
`--policy.pretrained_path=/path/to/act/pretrained_model` to reuse its
`model.safetensors`.  `action_quality` is a non-persistent dataset buffer and is
not saved in the checkpoint.

## Inference

Quality labels are training-only.  `select_action` and `predict_action_chunk` are
inherited from ACT and do not read the quality index.  When loading a saved
`act_quality` checkpoint through plugin discovery for deployment, no labeled
dataset is required.

## Safety guards

- Missing labels or labels that are neither scalar bool nor scalar int64 fail
  before training.
- Ternary values outside `{0,1,2}` fail before training.
- Missing, duplicate, out-of-range, or null frame indices fail during policy creation.
- Dataset/parquet frame alignment is checked exhaustively.
- Recovery provenance and manifest boundaries are checked against parquet labels.
- Non-monotonic episode labels fail by default.
- Invalid anchors are removed by the sampler and independently forced to zero loss.
- Normal, recovery-onset, and recovery-remainder anchors are sampled from
  separate deterministic pools.
- Masked actions are zeroed before entering the CVAE encoder.
- The label lookup is not serialized in model checkpoints.

## Verify

```bash
python -m lerobot.policies.act_quality.selftest
```
