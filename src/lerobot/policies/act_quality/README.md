# `act_quality` — semantic S/M/E recovery training

This package keeps the upstream ACT network unchanged while adding training-only
quality masks and semantic anchor sampling. The generic training entry point has
one narrow integration hook so an `act_quality` warm-start can retain the
pretrained checkpoint's normalization coordinates.

## Semantic labels

New datasets use scalar `int64` `action_quality` labels:

```text
normal episode:   1 1 1 1 1 ...
recovery episode: 0 ... 0 | 2 ... 2 | 3 ... 3 | 1 ... 1
                            S         M         E
```

- `0`: invalid error-producing prefix before S; excluded from the sampler,
  reconstruction loss, and CVAE action encoder.
- `1`: normal execution, including the suffix after E.
- `2`: manually labeled rollback from the wrong pose to the correct ready pose,
  `[S,M)`.
- `3`: manually labeled realignment and insertion, `[M,E)`.

Labels 1/2/3 are all valid reconstruction targets and use the same L1/KL
objective. Their numeric values are not loss weights. Labels 2 and 3 exist to
give the two recovery phases independent anchor-sampling budgets. Quality labels
are removed from model inputs and outputs and are never required at inference.

The old fixed-frame rule has been removed. No phase boundary is inferred from
"the first 30 frames" or any other hard-coded duration.

Old bool and 0/1/2 datasets can still be loaded for checkpoint compatibility or
unbalanced ablations. Semantic balanced sampling fails fast if a non-zero
reentry quota is requested but label 3 is absent, preventing accidental training
with the old fixed-frame behavior.

## Build the S/M/E dataset from existing labels

The two completed manual passes already contain every boundary:

- reentry manifest: common S and M=`recovery_end_frame`;
- finish manifest: common S and E=`recovery_end_frame`.

Combine them without decoding or re-encoding videos:

```bash
cd /home/snorlax/repo/robot_data_platform/lerobot
PYTHONPATH=src /opt/robot-platform/train-venv/bin/python -m \
  lerobot.policies.act_quality.build_semantic_dataset \
  --reentry-root /path/to/batch_quality_labeled_reentry \
  --finish-root /path/to/batch_quality_labeled_reentry_finish \
  --output /path/to/batch_quality_labeled_sme
```

The builder hard-links unchanged assets when possible and rewrites only the
quality column and manifest. It refuses to overwrite an existing output.

## Merged baseline and semantic sampling

`quality_pool_mode=merged` is the default warm-start baseline. It uses two
mutually exclusive pools:

```text
normal       label 1                         90%
recovery     label 2 + label 3 as one pool  10%
```

Recovery anchors are sampled uniformly from the union, so rollback/reentry
frequency follows their number of unique frames. No fixed-frame split exists.

`quality_pool_mode=semantic` instead uses three pools:

```text
normal       label 1
rollback     label 2, manual S->M
reentry      label 3, manual M->E
```

`quality_recovery_anchor_fraction` is the total rollback+reentry share of an
epoch. In semantic mode, `quality_reentry_anchor_fraction` is the label-3 share
of the whole epoch and rollback receives the difference. A 90/5/5 experiment is:

```text
normal       90%
rollback      5%
reentry       5%
```

`quality_balanced_epoch_size=0` keeps the natural valid-anchor count as epoch
length. A positive value sets an explicit length. Each pool is shuffled and
repeated independently when its requested quota exceeds its number of unique
anchors. Training logs report pool sizes, quotas, and repeat factors.

`quality_recovery_onset_steps` and `quality_recovery_onset_fraction` remain in
the config only so older checkpoints can be deserialized. The semantic sampler
does not read them.

## Train

```bash
PYTHONPATH=/home/snorlax/repo/robot_data_platform/lerobot/src \
/opt/robot-platform/train-venv/bin/lerobot-train \
  --policy.type=act_quality \
  --policy.chunk_size=100 \
  --policy.n_action_steps=100 \
  --policy.quality_label_key=action_quality \
  --policy.quality_balance_anchor_pools=true \
  --policy.quality_pool_mode=merged \
  --policy.quality_recovery_anchor_fraction=0.03 \
  --policy.quality_keep_pretrained_normalization=true \
  --policy.pretrained_path=/path/to/success_only/pretrained_model \
  --dataset.repo_id=local/my_semantic_quality_dataset \
  --dataset.root=/path/to/my_semantic_quality_dataset \
  --output_dir=/path/to/output
```

All upstream ACT fields remain available because `ACTQualityConfig` subclasses
`ACTConfig`. The policy model weights and state-dict names are unchanged.

### Behavior-preserving warm-start normalization

`quality_keep_pretrained_normalization=true` is the default. When
`pretrained_path` is set, both the model weights and the saved normalizer /
unnormalizer statistics are loaded from that checkpoint. Recovery samples are
therefore mapped into the same state/action coordinates used to train the base
policy; adding the recovery dataset does not silently change the policy before
the first optimizer step. New checkpoints save these retained processors, so
deployment automatically uses the matching statistics.

From-scratch training still computes statistics from the current dataset.
Resume always retains the processor stored with the resumed checkpoint. Set
`quality_keep_pretrained_normalization=false` only for an intentional
normalization rebase; that changes the behavior at optimizer step zero and is
not the conservative fine-tuning mode.

For the matched semantic experiment, keep the same initialization and total
recovery fraction, then change only:

```bash
--policy.quality_pool_mode=semantic \
--policy.quality_recovery_anchor_fraction=0.10 \
--policy.quality_reentry_anchor_fraction=0.05
```

## Chunk loss

For anchor `t`, a `[B,H]` valid mask is built from labels at
`t ... t+H-1`. Label 0 and padding positions are masked without compacting or
reordering time. Labels 1/2/3 remain in their original positions and contribute
equally. The same mask is used for reconstruction and the CVAE action encoder;
invalid CVAE action tokens are zeroed. L1 is normalized by valid action elements
and KL by valid samples.

## Inference

`select_action` and `predict_action_chunk` are inherited from ACT. The dataset
index and semantic labels are non-persistent training-only state, so deployment
does not need a labeled dataset. Older and newer `act_quality` checkpoints retain
the same network architecture.

## Guards

- Semantic recovery episodes must follow `0* -> 2+ -> 3+ -> 1*`.
- Manifest S/M/E boundaries are checked against parquet labels.
- Normal/all-valid episodes must be entirely label 1.
- Missing, duplicate, null, or out-of-range frame indices fail early.
- A requested rollback or reentry quota with an empty pool fails early.
- No fallback to first-K-frame onset splitting exists.

## Verify

```bash
PYTHONPATH=src /opt/robot-platform/train-venv/bin/python -m \
  lerobot.policies.act_quality.selftest
```
