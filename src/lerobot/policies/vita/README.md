# VITA — Vision-to-Action Flow Matching Policy

Port of [ucd-dare/VITA](https://github.com/ucd-dare/VITA) (Gao et al., ICLR 2026,
[paper](https://huggingface.co/papers/2507.13231)) onto LeRobot's `PreTrainedPolicy` API.

## The idea

Every other flow-matching policy here starts from Gaussian noise and injects the observation into
the velocity network at every denoising step (cross-attention in `pi0`, AdaLN in `groot`, FiLM in
`diffusion`). VITA notes that a flow's source distribution does not have to be noise, and uses the
**visual latent itself** as the source:

```
z_img = obs_encoder(resnet(images), state)   # source of the probability path, at t=0
z_act = action_encoder(action_chunk)         # target, at t=1
v     = flow_net(z_t, t)                     # no conditioning input — only the timestep
```

The conditioning modules disappear, which is where the reported 1.5–2× inference speedup comes from.
Two things pay for it:

* the source and target must be comparable, hence the **action autoencoder** mapping a chunk to a
  vector of the same width as the visual latent;
* the flow-matching loss alone no longer ties a specific observation to a specific action, hence
  **flow latent decoding (FLD)** — the sampler runs inside the training graph and the action
  reconstruction loss is backpropagated through every ODE step.

## Train

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/my_task \
  --policy.type=vita \
  --policy.device=cuda \
  --output_dir=outputs/train/vita_my_task \
  --job_name=vita_my_task \
  --batch_size=64
```

Useful overrides:

```bash
--policy.flow_matcher_type=conditional   # plain CFM instead of minibatch-OT CFM (see caveat below)
--policy.num_sampling_steps=4            # Euler steps at inference (6 by default)
--policy.horizon=32 --policy.n_action_steps=16
--policy.latent_dim=512                  # width of the shared vision/action latent
--policy.decode_flow_latents=false       # disable FLD (read the caveat first)
```

One-step generation with MeanFlow (`num_sampling_steps=1`, a single network evaluation per chunk):

```bash
--policy.flow_matcher_type=mean --policy.flow_net_type=simple_mean --policy.num_sampling_steps=1
```

## Configuration reference

Every field of `VitaConfig` (`configuration_vita.py`) is a draccus CLI override: `--policy.<field>=<value>`.
Defaults reproduce `flare/configs/policy/vita.yaml` + `flare/configs/default_policy.yaml` upstream, except
for the renames noted in [Deviations](#deviations-from-the-reference-implementation). Anything checked in
`__post_init__` fails fast at startup; anything not checked there fails later, usually as a shape error deep
in a forward pass, so read [Validation rules](#validation-rules-all-of-these-raise-at-startup) before
changing shapes.

Where a field feeds the network, the symbols used below are the ones from the top of this README:
`z_img` = visual latent (flow source, `t=0`), `z_act` = encoded ground-truth action latent (flow target,
`t=1`), `ẑ_act` = latent produced by *sampling* the flow from `z_img`.

### 1. Chunking and I/O shape

| field | default | meaning |
|---|---|---|
| `n_obs_steps` | `1` | Observation steps fed to the policy, counting backwards from now. |
| `horizon` | `16` | Length of the predicted action chunk. |
| `n_action_steps` | `8` | Actions executed from each chunk before re-planning. |
| `drop_n_last_frames` | `8` | Frames dropped from the end of every episode by the dataset sampler. |

**`n_obs_steps`** — VITA uses 1, unlike the diffusion policy's 2. Larger values do *not* add a temporal
module: `VitaObservationEncoder.forward` concatenates the per-step features and state and flattens them,
so the single projection layer grows to `n_obs_steps × (num_cameras × 512 + state_dim) → latent_dim`. Two
steps therefore double that layer's parameter count and let the encoder see velocity, at the cost of
holding two frames per camera in the inference queue. It also shifts the chunk window: `action_delta_indices`
is `range(1 - n_obs_steps, 1 - n_obs_steps + horizon)`, i.e. the chunk starts at the *oldest* observation,
and `generate_actions` compensates by slicing from index `n_obs_steps - 1`.

**`horizon`** — the whole chunk is compressed into one `latent_dim` vector, so this is the parameter that
decides how much the autoencoder has to carry. Raising it (32, 64) buys temporal consistency and fewer
re-plans; past some point the latent becomes the bottleneck and `enc_action_recon_loss` stops falling — that
metric is the honest read on whether `latent_dim` is large enough for the `horizon` you picked. Raising
`horizon` also raises the CNN encoder's flattened conv width, hence the size of `latent_proj`.

**`n_action_steps`** — execution/replanning trade-off, and the single biggest lever on closed-loop latency
cost per environment step: the policy runs `num_sampling_steps` flow evaluations once per `n_action_steps`
actions. Small values react faster to disturbances but pay inference more often; large values are smoother
and can drift open-loop. Constraint: `n_action_steps ≤ horizon - n_obs_steps + 1`.

**`drop_n_last_frames`** — **not derived, and this is a real footgun.** The value must equal
`horizon - n_action_steps - n_obs_steps + 1`; the default `8` is correct only for the default
`16 / 8 / 1`. It is passed straight to `EpisodeAwareSampler` in `lerobot_train.py`, and it exists so the
sampler does not start a window whose action chunk runs off the end of the episode. Change any of the three
shape fields and you must recompute it by hand:

```bash
# horizon=32, n_action_steps=16, n_obs_steps=1  ->  32 - 16 - 1 + 1 = 16
--policy.horizon=32 --policy.n_action_steps=16 --policy.drop_n_last_frames=16
```

Too small and the last windows of each episode are padded/short; too large and you silently throw away
training data (and, if it exceeds the episode length, the sampler logs a warning and skips the episode).

### 2. Normalization

**`normalization_mapping`** — `{"VISUAL": MEAN_STD, "STATE": MIN_MAX, "ACTION": MIN_MAX}`, identical to the
diffusion policy so the two consume the dataset the same way. `MIN_MAX` maps to `[-1, 1]` using the dataset
statistics. This matters more here than in a conditioned policy: the action autoencoder is trained on the
*normalized* scale (see `processor_vita.py`), and the flow runs between two latents built from normalized
quantities, so a dataset whose min/max are set by a few outlier frames compresses everything else into a
narrow band of the latent space. Switching `ACTION` to `MEAN_STD` is a legitimate experiment on data with
heavy-tailed action distributions; if you do, retrain from scratch — the stats are baked into the checkpoint.

### 3. Vision encoder

| field | default | meaning |
|---|---|---|
| `vision_backbone` | `"resnet18"` | Any `torchvision.models` ResNet variant. Validated to start with `resnet`. |
| `pretrained_backbone_weights` | `"ResNet18_Weights.IMAGENET1K_V1"` | `None` trains from scratch. |
| `freeze_backbone_batchnorm` | `True` | Replace BatchNorm with `FrozenBatchNorm2d`. |
| `resize_shape` | `(240, 320)` | `(H, W)` resize applied before cropping. `None` disables. |
| `crop_shape` | `(224, 308)` | `(H, W)` crop. `None` disables. |
| `crop_is_random` | `True` | Random crop while training, center crop at eval. |

**`vision_backbone`** — one backbone is shared by *all* cameras (unlike ACT/diffusion's per-camera
encoders), and its `layer4` map is globally average-pooled to one vector per view: 512-d for
`resnet18`/`resnet34`, 2048-d for `resnet50`+. Moving to `resnet34` roughly doubles backbone FLOPs for the
same feature width; `resnet50` also quadruples the projection layer's input. The dimension is measured with
a dry run (`get_output_shape`), never assumed, so any ResNet works — but remember VITA's selling point is
inference speed, and the backbone runs once per re-plan while the flow net runs `num_sampling_steps` times
on a `latent_dim` vector. On most setups the backbone, not the flow, is the inference cost.

**`freeze_backbone_batchnorm`** — VITA keeps ImageNet's running statistics instead of swapping in GroupNorm.
Keep it `True` at the batch sizes typical here (≤ 64 across several cameras); the per-camera images are
folded into the batch dimension, so BN statistics would otherwise be estimated from a mix of viewpoints.

**`resize_shape` / `crop_shape` / `crop_is_random`** — the crop is the only image augmentation in the
policy, and it runs *after* the resize, so the pair `(240, 320) → (224, 308)` gives roughly ±7% jitter.
Setting `crop_is_random=False` removes that augmentation entirely; on small datasets expect it to overfit
sooner. Note the asymmetry in validation: with `resize_shape=None` the crop is checked against the actual
image shapes in `validate_features`, but when a resize is configured the crop is *not* checked against it —
a crop larger than the resize raises inside torchvision instead. All cameras must share one image shape.

### 4. The shared latent

**`latent_dim`** (default `512`) — width of the space the flow lives in, and the parameter that touches
everything: the observation projection's output, the flow net's input and output, the action encoder's
output, the action decoder's input. It is the bottleneck the whole `horizon × action_dim` chunk must pass
through — 16 steps × 14 DoF = 224 numbers compressed into 512 by default, comfortable; 64 steps × 54 DoF =
3456 into 512 is not. Raise it together with `horizon` and watch `enc_action_recon_loss`, which measures
exactly this compression (it is `decode(encode(a))` vs `a`, with no flow involved). Constraint with the
`simple` action encoder: `latent_dim % horizon == 0`.

### 5. Flow matcher

| field | default | meaning |
|---|---|---|
| `flow_matcher_type` | `"exact"` | Probability path + training objective. |
| `flow_sigma` | `0.0` | Gaussian noise added to the interpolant `x_t`; CFM family only. |
| `num_sampling_steps` | `6` | Euler steps per generation. |

**`flow_matcher_type`** — five choices, defined in `flow_matching.py`:

| value | family | required `flow_net_type` | typical `num_sampling_steps` | notes |
|---|---|---|---|---|
| `exact` | OT-CFM (upstream default) | `simple` | 6 | Re-pairs `(z_img, z_act)` within the minibatch. Needs `scipy`. **Never combine with `decode_flow_latents=false`** — see the caveat section. |
| `conditional` | plain CFM | `simple` | 6 | Keeps the per-sample pairing. The honest baseline; ablate against `exact` on your own data. |
| `mean` | MeanFlow | `simple_mean` | 1 | 1-NFE generation. |
| `improved_mean` | Improved MeanFlow | `simple_mean` | 1 | Adds the auxiliary instantaneous-velocity head and a corrected bootstrap target. |
| `consistency` | consistency FM | `simple` | 1–2 | Piecewise-straight paths, own stochastic sampler. |

The `mean`/`improved_mean` ↔ `simple_mean` pairing is enforced in both directions at startup. Two practical
warnings for the MeanFlow family: its training identity is computed through
`torch.autograd.functional.jvp`, which is numerically fragile in fp16/bf16 — **do not enable `use_amp` with
`mean`/`improved_mean`** — and under DDP, plain `mean` leaves `v_output_layer` without gradient, so it needs
`find_unused_parameters=True`.

`exact` costs one CPU device-synchronisation plus an O(batch³) linear-assignment solve per training step.
Negligible against a ResNet forward at batch 128; visible if you ever train with a tiny backbone.

**`flow_sigma`** — the width of the Gaussian around the straight-line interpolant. `0.0` (upstream) makes
the path exactly `x_t = (1-t)·x0 + t·x1`. Small positive values (1e-3–1e-2) smooth the target field and can
help when latents are tightly clustered; it is read only by `conditional` and `exact`.

**`num_sampling_steps`** — Euler steps at inference *and*, when `decode_flow_latents=True`, inside the
training graph. That second part is the one people miss: training cost and activation memory for the flow
net scale linearly in this value, because FLD backpropagates through every step. Going 6 → 12 does not
double total training time (the ResNet dominates) but it does grow the retained graph. At inference,
6 → 4 → 2 is the cheapest accuracy/latency dial in the whole config; measure the drop before shipping it.

### 6. MeanFlow-only knobs

All ignored unless `flow_matcher_type` is `mean` or `improved_mean` (`make_flow_matcher` filters kwargs by
the matcher's signature — misapplied knobs are silently dropped, not rejected).

| field | default | meaning |
|---|---|---|
| `meanflow_flow_ratio` | `0.5` | Fraction of the batch where `r` is tied to `t`, reducing those samples to ordinary flow matching. This is what anchors the average-velocity field to the instantaneous one; at `0.0` training has nothing pinning it down, at `1.0` you have plain flow matching with no mean-velocity learning. |
| `meanflow_time_dist_mu` | `-0.4` | Mean of the logit-normal distribution `(t, r)` are drawn from. More negative → sampling concentrates near `t=0` (the action end of the reversed path). |
| `meanflow_time_dist_sigma` | `1.0` | Spread of that distribution. |
| `meanflow_adaptive_loss_gamma` | `0.5` | Exponent of the adaptive-L2 down-weighting of high-residual samples (`mean` only; `improved_mean` uses its own adaptive loss). `0` → uniform weighting. |
| `meanflow_aux_v_loss_weight` | `1.0` | Weight of the auxiliary instantaneous-velocity regression. `improved_mean` only. |
| `meanflow_dispersive_loss_weight` | `0.0` | Weight of the MP1 dispersive loss pushing hidden states apart, against latent collapse. Off by default; **enabling it costs an extra flow-net forward pass per step**, which is why it is gated on `> 0`. |
| `meanflow_dispersive_loss_tau` | `1.0` | Temperature of that repulsion term. |

### 7. Velocity network

| field | default | meaning |
|---|---|---|
| `flow_net_type` | `"simple"` | `"simple"` (`SimpleFlowNet`) or `"simple_mean"` (`SimpleMeanFlowNet`). Tied to `flow_matcher_type`. |
| `flow_hidden_dim` | `512` | Width of the residual MLP trunk. |
| `flow_num_layers` | `4` | Number of residual blocks. |
| `flow_mlp_ratio` | `4.0` | Inner MLP expansion, i.e. hidden width `flow_hidden_dim × 4`. |
| `flow_dropout` | `0.0` | Dropout inside the MLP blocks. |
| `flow_time_embed_dim` | `256` | Sinusoidal time-embedding width. **`SimpleFlowNet` only** — `SimpleMeanFlowNet` accepts the argument and ignores it. |

This network is the deliberate architectural point of VITA: it takes `(x_t, t)` and *nothing else*. There is
no cross-attention, no AdaLN over an observation, no FiLM — the observation enters the ODE as the initial
condition. Consequently it is small and cheap, and scaling it is usually the wrong first move; scale
`latent_dim` or the backbone first. `flow_num_layers` 4 → 6/8 is the reasonable direction if the flow loss
plateaus while reconstruction losses are already low. `flow_dropout > 0` is worth trying on small datasets,
but remember it is active during the FLD sampling inside training and inactive at eval, which makes the
training-time and inference-time ODEs slightly different maps.

Note the init quirk documented in the code and in Deviations: `SimpleFlowNet._init_weights` applies a global
Xavier init *after* the blocks are built, overwriting each `FlowNetLayer`'s zero-init of `time_modulator`,
so adaLN-zero is not actually in effect. Preserved for fidelity with upstream.

### 8. Action autoencoder

| field | default | meaning |
|---|---|---|
| `action_encoder_type` | `"cnn"` | `"cnn"` (strided 1D convs over time) or `"simple"` (per-timestep MLP). |
| `action_decoder_type` | `"simple"` | Only `"simple"` exists; validated. |
| `action_enc_hidden_dim` | `512` | Conv channel width of the CNN encoder. Unused by `"simple"`. |
| `action_dec_hidden_dim` | `512` | Decoder trunk width. |
| `action_ae_num_layers` | `4` | Depth — **shared by encoder and decoder**. |
| `action_ae_dropout` | `0.0` | Decoder dropout. Encoders ignore it. |
| `freeze_action_encoder` | `False` | Freeze encoder weights. |
| `freeze_action_decoder` | `False` | Freeze decoder weights. |

**`action_encoder_type`** — `cnn` halves the chunk length at every layer (stride 2) and then flattens, so it
mixes across time and is the upstream default. `simple` gives each timestep its own `latent_dim // horizon`
channels and never mixes across time, which makes the latent trivially decodable but discards temporal
structure; it is mostly useful as a diagnostic. Each carries its own constraint:

* `cnn` requires `horizon ≥ 2 ** action_ae_num_layers` — with the default depth 4, `horizon ≥ 16`. Raising
  `action_ae_num_layers` to 5 forces `horizon ≥ 32`.
* `simple` requires `latent_dim % horizon == 0`.

**`action_ae_num_layers` is shared**, which is easy to forget: bumping it to deepen the decoder also adds a
stride-2 conv stage to the encoder and can trip the `horizon` constraint above.

**`freeze_action_encoder` / `freeze_action_decoder`** — intended for a two-stage recipe (pretrain the
autoencoder, then train only the flow). Read `compute_loss` before using them, because the loss changes
shape:

* both frozen → the function returns immediately after the flow-matching term (plus `enc_contrastive` if
  enabled). Nothing else is computed.
* `freeze_action_encoder=True` alone → **the entire FLD block is skipped**, so `consistency_weight`,
  `flow_recon_weight` and `flow_contrastive_weight` are silently inert regardless of their values. Combined
  with the default `flow_matcher_type="exact"`, that leaves nothing tying an observation to its own
  action — exactly the failure mode described in the caveat section. Use `conditional` if you freeze the
  encoder.
* `freeze_action_decoder=True` alone → both reconstruction terms are dropped; consistency and the
  contrastive terms still apply.

### 9. Loss composition

The total training loss assembled in `VitaModel.compute_loss`:

```
loss = flow_loss(matcher; z_img -> z_act)                          # always
     + enc_contrastive_weight  * InfoNCE(z_img, z_act)             # if > 0
                                                                   # --- FLD block, only if
                                                                   #     decode_flow_latents and
                                                                   #     neither AE half is frozen:
     + consistency_weight      * MSE(ẑ_act, z_act)                 #   if > 0
     + flow_contrastive_weight * InfoNCE(z_img, ẑ_act)             #   if > 0
     + flow_recon_weight       * recon(decode(ẑ_act), a_gt)        #   if > 0
                                                                   # ---
     + enc_recon_weight        * recon(decode(z_act),  a_gt)       # if > 0 and decoder not frozen
```

| field | default | meaning |
|---|---|---|
| `decode_flow_latents` | `True` | Master switch for flow latent decoding (FLD). |
| `consistency_weight` | `1.0` | Flow latent consistency (FLC): MSE between sampled and encoded action latent. |
| `flow_recon_weight` | `0.5` | Reconstruction of the ground-truth chunk from the **sampled** latent. |
| `enc_recon_weight` | `0.5` | Reconstruction from the **encoded** latent — the plain autoencoder term. |
| `recon_loss_type` | `"l1"` | `"l1"` or `"l2"` for both reconstruction terms. |
| `enc_contrastive_weight` | `0.0` | InfoNCE between `z_img` and `z_act`. |
| `flow_contrastive_weight` | `0.0` | InfoNCE between `z_img` and `ẑ_act`. |
| `contrastive_temperature` | `0.07` | Temperature shared by both InfoNCE terms. |

**`decode_flow_latents`** — the most consequential boolean in the file. When `True`, `_sample_action_latents`
runs the full ODE *inside the training graph* and gradients flow back through every Euler step. This is what
ties `ẑ_act[i]` to `gt_actions[i]` index for index, and therefore what makes the default `exact` matcher
safe. Turning it off makes training substantially cheaper and faster, and is only defensible together with
`flow_matcher_type="conditional"`. The failure it causes is silent: the loss curve looks fine while the map
is unpaired.

**`consistency_weight` vs `flow_recon_weight`** — both pull the sampled latent toward the truth, but in
different spaces: FLC in latent space (cheap, no decoder pass), FLD reconstruction in action space (goes
through the decoder, so it also shapes the decoder). Upstream weights them 1.0 / 0.5. If the sampled and
encoded latents agree (`consistency_loss` low) while actions are still wrong, the autoencoder is the problem,
not the flow — check `enc_action_recon_loss`.

**`recon_loss_type`** — `l1` is the default and is more tolerant of the occasional outlier action; `l2`
penalises large errors harder and tends to produce smoother, more averaged chunks. Note the reconstruction
losses are **not masked on padded actions** (matching upstream), so episodes whose tail is padded contribute
padding to the loss — one more reason to keep `drop_n_last_frames` correct.

**Contrastive terms** — both are `0.0` upstream and reported there as an optional boost on top of FLD/FLC.
They pull matching observation/action pairs together across the batch. Worth trying when several tasks or
scenes share one dataset and you suspect the visual latent is not discriminative enough; they need a
reasonably large batch to have negatives worth contrasting against.

**What lands in the training log:** `loss` (total), `flow_loss` (matcher term; for MeanFlow this is the
matcher's own total), plus `consistency_loss`, `flow_action_recon_loss`, `enc_action_recon_loss`,
`enc_contrastive_loss`, `flow_contrastive_loss` when their weights are nonzero, and matcher-specific keys
(`meanflow_loss`, `imf_loss`, `aux_v_loss`, `consistency_f_loss`, `consistency_v_loss`, `dispersive_loss`).

### 10. Optimization

| field | default | meaning |
|---|---|---|
| `optimizer_lr` | `1e-4` | Learning rate for everything except the vision backbone. |
| `optimizer_lr_backbone` | `1e-5` | Backbone learning rate — a separate Adam param group built in `VitaPolicy.get_optim_params`, matched by the `vita.obs_encoder.backbone` name prefix. |
| `optimizer_betas` | `(0.95, 0.999)` | Adam betas. Note `β₁ = 0.95`, not the usual 0.9 — inherited from the diffusion-policy lineage. |
| `optimizer_eps` | `1e-8` | Adam epsilon. |
| `optimizer_weight_decay` | `1e-6` | Plain Adam weight decay (not AdamW decoupled). |
| `scheduler_name` | `"cosine"` | Passed to `diffusers.get_scheduler`; hence the `diffusers` requirement at training time. |
| `scheduler_warmup_steps` | `500` | Linear warmup before the cosine decay. |

The 10× lower backbone rate is the standard "don't destroy ImageNet features early" setting. Set
`optimizer_lr_backbone=0.0` to freeze the backbone outright — cheap and often fine when your cameras look
like natural images and the dataset is small.

These fields are the *only* way to set the optimizer while the default `use_policy_training_preset=true` is
in effect: `TrainPipelineConfig.__post_init__` overwrites `cfg.optimizer` / `cfg.scheduler` from
`get_optimizer_preset()` / `get_scheduler_preset()`, so a `--optimizer.lr=` on the command line is silently
discarded. Use `--policy.optimizer_lr=`, or pass `--use_policy_training_preset=false` and then supply both
`--optimizer.*` and `--scheduler.*` in full (the config refuses to start with only one of them). The preset
is also skipped entirely when `--resume=true`, which restores the optimizer from the checkpoint.

### Inherited from `PreTrainedConfig`

`device`, `use_amp`, `input_features` / `output_features` (inferred from the dataset when left empty),
`push_to_hub`, `repo_id`, `private`, `tags`, `license`, `pretrained_path`, `pretrained_revision`.
`use_amp` deserves repeating: safe for the CFM family, **not** for `mean` / `improved_mean`.

`validate_features` additionally requires at least one image feature (VITA has no non-visual source for its
flow) and an `observation.state` input, and requires every camera to share the same image shape.

### Validation rules (all of these raise at startup)

| rule | triggered by |
|---|---|
| `vision_backbone` starts with `resnet` | any other backbone |
| `flow_matcher_type ∈ {conditional, exact, mean, improved_mean, consistency}` | typo |
| `flow_net_type ∈ {simple, simple_mean}` | typo |
| `mean`/`improved_mean` ⟺ `flow_net_type="simple_mean"` | enforced both ways |
| `recon_loss_type ∈ {l1, l2}` | typo |
| `action_encoder_type ∈ {cnn, simple}`, `action_decoder_type == "simple"` | typo |
| `latent_dim % horizon == 0` | `action_encoder_type="simple"` |
| `horizon ≥ 2 ** action_ae_num_layers` | `action_encoder_type="cnn"` |
| `n_action_steps ≤ horizon - n_obs_steps + 1` | chunk shape |
| `resize_shape` / `crop_shape` are positive pairs | shape typo |
| ≥ 1 image feature, `observation.state` present, all cameras same shape | `validate_features`, at policy construction |

Not validated, and therefore worth a second look every time you touch shapes: `drop_n_last_frames`.

### Hardcoded — not exposed on the config

Reachable only by editing the source:

* **Consistency-matcher hyper-parameters** — `eps`, `num_segments`, `boundary`, `delta`, `alpha`,
  `noise_scale`, `sigma_var` in `ConsistencyFlowMatcher.__init__`. `make_flow_matcher` filters kwargs by
  signature, so `flow_matcher_type="consistency"` always runs with the upstream defaults (except
  `num_sampling_steps`, which is forwarded). Note the class default is 1 step while `VitaConfig` sends 6 —
  set `--policy.num_sampling_steps=1` explicitly if you want its intended few-step regime.
* **Improved MeanFlow `norm_p` / `norm_eps`** — same filtering; fixed at `1.0` / `0.01`.
* **EMA**, the transformer action encoder, and the variational autoencoder — absent by design, see
  Deviations.
* **CNN encoder kernel/stride** (`kernel_size=5, stride=2`) and the fact that reconstruction losses are
  unmasked.

### Tuning starting points

Faster inference, accepting some accuracy loss:

```bash
--policy.num_sampling_steps=2 --policy.n_action_steps=12
# or, 1-NFE:
--policy.flow_matcher_type=mean --policy.flow_net_type=simple_mean --policy.num_sampling_steps=1
```

Longer chunks (remember `drop_n_last_frames` and the `cnn` depth constraint):

```bash
--policy.horizon=32 --policy.n_action_steps=16 --policy.drop_n_last_frames=16 --policy.latent_dim=768
```

Small dataset / overfitting:

```bash
--policy.crop_is_random=true --policy.flow_dropout=0.1 --policy.action_ae_dropout=0.1 \
--policy.optimizer_lr_backbone=0.0
```

Diagnosing which half is failing — train the autoencoder in isolation first by disabling FLD, and use
`conditional` so the pairing survives:

```bash
--policy.flow_matcher_type=conditional --policy.decode_flow_latents=false --policy.enc_recon_weight=1.0
```

If `enc_action_recon_loss` will not fall, the chunk does not fit in `latent_dim` and no amount of flow
tuning will fix it.

Tight GPU memory: lower `batch_size` first, then `num_sampling_steps` (FLD retains the graph across every
step), then `latent_dim` / `flow_hidden_dim`. Note `flow_matcher_type="exact"` scales O(batch³) on CPU, so
smaller batches make it cheaper too — but they also weaken both the OT coupling and the contrastive terms,
which need negatives.

## Requirements

Everything VITA needs is declared by the `vita` extra:

```bash
pip install 'lerobot[vita]'
```

which resolves to:

| package | version | needed for | required? |
|---|---|---|---|
| `scipy` | `>=1.14.0,<2.0.0` | the default `flow_matcher_type="exact"` (optimal-transport coupling) | only for `exact`; `conditional`, `mean`, `improved_mean` and `consistency` work without it |
| `diffusers` | `>=0.27.2,<0.36.0` | the cosine LR schedule preset at training time, as for the diffusion policy | yes, to train via `lerobot-train` |

Nothing else: no `torchcfm`, no `timm`, no `POT`. The flow matchers and every network are
self-contained in this directory.

### Deploying to the robot-platform cluster

Training nodes run the shared venv at `/opt/robot-platform/train-venv`, installed by
`scripts/25-install-training-environment.sh`. Which extras it installs comes from
`LEROBOT_EXTRAS` in `config/site.env` — `vita` is in the default list, so a **fresh** install
already covers both packages.

An **existing** node is the case to watch. `scipy` is already present there (the `pi` extra pulls
it in), but `diffusers` is not, and `--sync-lerobot` copies source files without resolving
dependencies. The sync detects this and stops with the exact command; it is:

```bash
sudo /opt/robot-platform/train-venv/bin/pip install 'diffusers>=0.27.2,<0.36.0'
```

Run it once per node, then re-run the sync. Rolling a code change out to a node:

```bash
cd ~/YING/robot_data_platform
git pull
git submodule update --init --recursive          # move lerobot/ to the recorded commit
sudo ./scripts/25-install-training-environment.sh --sync-lerobot --apply
```

The sync writes the submodule revision it copied to
`<site-packages>/lerobot/SUBMODULE_REVISION`, because pip's own metadata still describes whatever
it last installed and will not reflect a synced tree.

## Caveat: minibatch OT coupling changes the pairing

The default `flow_matcher_type="exact"` reproduces upstream: optimal-transport CFM, which re-pairs
observations and actions **within the minibatch**. The flow-matching term therefore generally trains
`z_img[i] -> z_act[perm[i]]`, not `z_img[i] -> z_act[i]`. For ordinary noise-to-data flow matching
that is harmless — the source is exchangeable noise — but here the source carries the observation.

What restores per-sample correctness is FLD, which compares `decode(sample(start=z_img[i]))` against
`gt_actions[i]` index for index. So:

* **Never combine `flow_matcher_type="exact"` with `decode_flow_latents=false`.** That trains an
  unpaired map and the loss curve will not tell you.
* Ablate `exact` against `conditional` early on your own data.

## Files

| file | contents |
|---|---|
| `flow_matching.py` | Standalone matchers: CFM, exact-OT CFM, MeanFlow / Improved MeanFlow, consistency FM. No `torchcfm` dependency. |
| `modeling_vita.py` | `VitaPolicy`, observation encoder, action autoencoder, `SimpleFlowNet` / `SimpleMeanFlowNet`. |
| `configuration_vita.py` | `VitaConfig`, registered as `"vita"`. |
| `processor_vita.py` | Pre/post-processing pipelines, identical in structure to the diffusion policy's. |

Tests: `tests/policies/vita/test_vita.py`.

## Deviations from the reference implementation

* **No EMA.** Upstream trains with `use_ema: true, ema_power: 0.75`; there is no equivalent hook in
  LeRobot's training loop, so final numbers will differ.
* **No transformer action encoder, no variational action autoencoder.** Both are off upstream's
  default path (`encoder_type: cnn`, `use_variational: false`).
* **`torchcfm` inlined.** Exact-OT coupling is solved with `scipy.optimize.linear_sum_assignment`
  rather than POT's `ot.emd`; for the uniform equal-size marginals used here the EMD optimum is
  attained at a permutation, so the two agree.
* **Consistency flow matcher made shape-generic.** Upstream hardcodes rank-3 tensors, which crashes
  on VITA's rank-2 latents.
* **Naming follows `DiffusionConfig`** (`n_obs_steps` / `horizon` / `n_action_steps` instead of
  `obs_horizon` / `pred_horizon` / `action_horizon`) so that a diffusion-vs-flow-matching comparison
  runs on identical dataloader settings.
* **Not registered for async inference.** `vita` is absent from
  `src/lerobot/async_inference/constants.py::SUPPORTED_POLICIES`; that path is untested here.
* **Upstream init quirk preserved.** `SimpleFlowNet._init_weights` applies a global Xavier init that
  overwrites each `FlowNetLayer`'s zero-init of `time_modulator`, so adaLN-zero is not actually in
  effect. Kept for fidelity — see the note in the code.
* **With `flow_matcher_type="mean"`, the `v_output_layer` head receives no gradient** (only Improved
  MeanFlow uses it). Harmless under Adam; set `find_unused_parameters=True` if you wrap the policy in
  DDP.
