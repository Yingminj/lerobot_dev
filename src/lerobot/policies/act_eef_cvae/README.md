# ACT-EEF CVAE previous-frame experiment

Use `--policy.type=act_eef_cvae` in the existing LeRobot training command;
use `--policy.type=act_eef` for the baseline. Keep the dataset, seed, training
steps and other hyperparameters identical for comparison.

This policy inherits ACT-EEF's 14-dimensional state/action representation and
normalization. For chunk length K, the dataset supplies K+1 actions at offsets
`[-1, 0, ..., K-1]`. Training uses:

- CVAE for T>0: current state and ground-truth actions at `[-1, ..., K-2]`.
- Reconstruction target: ground-truth actions at `[0, ..., K-1]`.

Each window uses its own padding mask. Episode boundaries follow the dataset's
existing clamping and padding behavior; no action is read from another episode.
The architecture, mean/variance heads, reparameterization, standard-normal KL,
L1 loss and optimizer defaults are unchanged. No model parameters are added.

At T=0, the offset -1 padding flag identifies a missing previous frame. These
samples bypass every CVAE module; even their valid future actions are not encoded.
Choose the training-only initialization with:

```bash
--policy.type=act_eef_cvae --policy.initial_z_mode=sample
--policy.type=act_eef_cvae --policy.initial_z_mode=zero
```

`sample` (the default, including configs without this field) draws one independent
standard-normal latent vector per first-frame sample per forward pass for the
entire chunk. `zero` supplies an all-zero vector. Both compute only action L1 for
first frames. Noninitial samples retain CVAE sampling and KL. KL is summed over
noninitial samples and divided by the full batch size; it is zero for an entirely
initial batch. `use_vae=False` retains ACT's all-zero latent regardless of this setting.

Only training inputs change. Inference and eval-mode loss computation retain
ACT's zero latent; no previous-prediction cache or inference sampling is added.
Loss-based validation still slices the current target out of the K+1 window.

The training history overlaps the target by K-1 actions. A lower training loss
alone does not establish better closed-loop performance. Feedback inference
using previous predictions is a separate experiment.
