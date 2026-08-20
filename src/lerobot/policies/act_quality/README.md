# `act_quality` — ACT with hard recovery-quality masks

This package implements recovery-data filtering without editing any file outside
`lerobot/policies/act_quality/`.

The ACT network architecture and state-dict names are unchanged.  The package adds:

1. a scalar boolean dataset feature (`action_quality` by default);
2. a dense read-only lookup from global frame `index` to quality;
3. a sampler hook that starts every episode at its first `True` frame;
4. a `[B,H]` target mask constructed from the lookup;
5. the same mask in reconstruction loss and the CVAE key-padding mask;
6. zeroed invalid action tokens before the CVAE action projection;
7. valid-element L1 and valid-sample KL renormalization;
8. separate normal-success and recovery anchor pools with a configurable
   recovery sampling fraction.

## ACT configuration inheritance

`ACTQualityConfig` subclasses the upstream `ACTConfig`; it does not replace it.
Consequently every standard ACT option remains available under `--policy.*`,
including `chunk_size`, `n_action_steps`, backbone and transformer dimensions,
CVAE settings, `temporal_ensemble_coeff`, `dropout`, `kl_weight`, normalization,
and optimizer settings. `configuration_act_quality.py` only declares the eight
additional quality options. Keeping one upstream definition prevents the two
ACT configurations from silently drifting apart.

## Required label semantics

For every episode, labels must have one of these forms:

```text
1 1 1 1 ...       # normal demonstration
0 0 0 1 1 1 ...   # invalid prefix, valid recovery suffix
```

`True -> False` transitions are rejected by default.  This validation is what
allows the sampler to skip an entire invalid prefix efficiently.

When `meta/quality_label_manifest.json` is present, its `selections` identify
recovery episodes, including recovery recordings intentionally labeled
`all_valid`. Without the manifest, the sampler falls back to treating an
episode with a non-empty `False` prefix plus a `True` suffix as recovery; in
that fallback mode an all-valid recovery recording is indistinguishable from a
normal success. Only valid frames enter either anchor pool.

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
  --dataset.repo_id=local/my_quality_dataset \
  --dataset.root=/path/to/my_quality_dataset \
  --output_dir=/path/to/output
```

No edit to `policies/__init__.py`, `policies/factory.py`, dataset factory, sampler,
or upstream ACT is needed.  Plugin discovery registers `ACTQualityConfig`; the
generic factory derives `ACTQualityPolicy` from the class name and module path.
Because the config subclasses `ACTConfig`, the existing factory deliberately routes
normalization through the upstream ACT processor pipeline.

## Balanced anchor-pool sampling

Balanced sampling is enabled by default. `quality_recovery_anchor_fraction`
controls the exact fraction of recovery anchors in each sampler epoch rather
than relying on the post-mask natural ratio. The conservative default is
`0.25`; use `0.5` as a stronger equal-pool ablation.

`quality_balanced_epoch_size=0` preserves the natural total number of valid
anchors as the epoch length. A positive value sets an explicit epoch length.
When a requested pool quota is larger than the number of unique anchors in that
pool, the sampler repeats independently shuffled full passes through that pool.
It therefore provides an exact epoch-level ratio while maximizing unique-anchor
coverage before duplication. Individual minibatches fluctuate around the target
ratio because the combined epoch is shuffled.

Useful ablations are:

```bash
# Natural post-mask distribution (previous behavior)
--policy.quality_balance_anchor_pools=false

# Recovery anchors are 25% of each epoch
--policy.quality_balance_anchor_pools=true \
--policy.quality_recovery_anchor_fraction=0.25

# Equal normal/recovery anchor quotas
--policy.quality_balance_anchor_pools=true \
--policy.quality_recovery_anchor_fraction=0.5
```

The training logs report unique pool sizes, per-epoch quotas, and the effective
oversampling factor for each pool. Policy metrics also report
`quality_recovery_anchor_fraction` and `quality_normal_anchor_fraction` for the
actual minibatch.

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

- Missing/non-boolean labels fail before training.
- Missing, duplicate, out-of-range, or null frame indices fail during policy creation.
- Dataset/parquet frame alignment is checked exhaustively.
- Recovery provenance and manifest boundaries are checked against parquet labels.
- Non-monotonic episode labels fail by default.
- Invalid anchors are removed by the sampler and independently forced to zero loss.
- Normal-success and recovery anchors are sampled from separate deterministic pools.
- Masked actions are zeroed before entering the CVAE encoder.
- The label lookup is not serialized in model checkpoints.

## Verify

```bash
python -m lerobot.policies.act_quality.selftest
```
