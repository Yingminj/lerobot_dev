# Patch New Policy — Patch Policy's block-causal memory with pi0's injections, made ablatable

`patch_new_policy` keeps [Patch Policy](../patch_policy/README.md)'s one real idea — a frozen ViT's
**dense patch tokens** read through a **block-causal** attention mask — and replaces its fixed
action head with a single conditioned decoder in which **where the state enters** and **where the
denoising time enters** are configuration fields rather than code.

- Patch memory and encoder zoo: [gaoyuezhou/patch_policy](https://github.com/gaoyuezhou/patch_policy), [arXiv:2607.18236](https://arxiv.org/abs/2607.18236)
- Injections, flow matching, Beta time schedule: openpi / lerobot's `pi0`, `pi05` ports
- Registered as `--policy.type=patch_new_policy` (`policies/factory.py:140`)

---

## Why it exists

`patch_policy` ships three heads (`vqbet`, `diffusion`, `act`) that are three *different modules*
with three different trunks, three different state pathways and three different opinions about the
diffusion timestep. Comparing them measures the modules, not the design choices inside them.

`patch_new_policy` is the same visual front end behind **one** trunk, where the design choices are
axes:

| axis | field | values |
|---|---|---|
| head | `action_head` | `flow` \| `diffusion` \| `act` |
| time injection | `time_injection` | `concat_mlp` \| `adaln` \| `additive` \| `memory_token` \| `none` |
| state injection | `state_injection` | `suffix_token` \| `obs_token` \| `action_concat` \| `adaln` \| `none` |
| relative actions | `use_relative_actions` | `False` \| `True` (processor-side, orthogonal) |

A 5 × 4 grid runs without touching model code. `vqbet` is deliberately **absent**: its RVQ codebook
is a second, separately-trained model whose reconstruction ceiling sits below the rest.

---

## Architecture

### End to end

```
observation.images.*  (B, S, V, 3, H, W)        observation.state  (B, S, state_dim)
        │                                                 │
        │ resize -> resize_shape (224, 224)               │
        ▼                                                 │
┌────────────────────────┐                                │
│   FROZEN ViT           │  DINOv2 / DINOv3 / WebSSL /    │
│   .eval(), no grad     │  SigLIP2 / V-JEPA 2 / ResNet18 │
└────────────────────────┘  (patch_encoders.py, reused)   │
        │ ((B·S·V), P, E)                                 │
        ▼                                                 │
  rearrange "(b s n) p e -> b s (n p) e"       ┌──────────┴───────────┐
        │                                      │ state_injection      │
        │ ◄────── obs_token (P2) ──────────────┤   = "obs_token"      │
        ▼                                      └──────────┬───────────┘
  patch tokens (B, S, tokens_per_frame, E)                │
        │  tokens_per_frame = V·P (+1 for obs_token)      │
        │                                                 │
   cond_obs_emb: Linear(E -> d)                           │
        │  [+ 1 leading time token if time_injection == "memory_token"]
        │  + cond_pos_emb, dropout(p_drop_emb)            │
   memory_encoder: Linear(d,4d) -> Mish -> Linear(4d,d)   │
        │  ← an MLP, NOT self-attention: see "Load-bearing details"
        ▼                                                 │
   MEMORY  (B, n_leading + S·tokens_per_frame, d)         │
        │                                                 │
        │        decoder sequence:                        │
        │   [S state tokens?] + [horizon action tokens] ◄──┘ suffix_token (P1)
        │        action tokens = Linear(A -> d)(x_t)   ← flow / diffusion
        │                      = query_emb.weight      ← act
        │                      (+ action_concat (P4) / adaLN cond (P5))
        │                      (+ concat_mlp / adaln / additive time)
        ▼
   n_decoder_layers × CondDecoderLayer
        pre-norm | self-attn(self_mask) | cross-attn(memory, memory_mask) | FFN
        optional 9-way adaLN-Zero modulation (scale/shift/gate per sub-block)
        ▼
   LayerNorm -> Linear(d -> A) -> drop the state rows -> (B, horizon, A)
```

`horizon = n_obs_steps + action_chunk_size - 1`. Decoder position `t` predicts the action at
observation step `min(t, n_obs_steps - 1)`; `predict()` returns
`[n_obs_steps - 1 : n_obs_steps - 1 + action_chunk_size]`, the chunk anchored at the **newest**
frame. That alignment is what makes the block-causal mask mean anything, and it is unchanged from
`patch_policy`.

### The masks — `decoder_masks()`

`patch_policy` has two mask helpers (`causal_mask` for self-attention, `block_causal_memory_mask`
for cross-attention) that assume the decoder holds action tokens only. `patch_new_policy` may put
**state tokens in the same sequence**, so both masks are built together from one rule:

> every row is tagged with an observation step — state token `i` → frame `i`, action token `t` →
> frame `min(t, n_obs_steps - 1)` — and may attend only to columns tagged with an equal or earlier
> frame.

Consequences:

- action token 0 sees state token 0 but **not** state token 1, which is one frame in its future;
- state tokens never read action tokens (`row_group` ordering);
- within a group the frame tag saturates for `t >= n_obs_steps - 1`, so plain index causality is
  still applied on top — otherwise the tail of the chunk would be fully bidirectional;
- every row keeps at least frame 0's memory block, so no row is ever fully masked. A fully masked
  row makes `nn.MultiheadAttention` return **NaN**, not raise.

With `n_state_tokens = 0` these masks are provably identical to `patch_policy`'s pair — the
generalisation is free.

### Heads

| | `flow` (default) | `diffusion` | `act` |
|---|---|---|---|
| decoder input | `Linear(A→d)(x_t)` | `Linear(A→d)(x_t)` | `nn.Embedding(horizon, d)` queries |
| training target | velocity `u_t = ε − a` | `ε` (or `sample`) | the action |
| loss | `mse(pred, u_t)` | `mse(pred, ε)` | `l1(pred, a)` |
| `t` sampling | `Beta(1.5, 1) · 0.999 + 0.001`, one per sample | `randint(0, 100)`, one per sample | — |
| `x_t` | `t·ε + (1−t)·a` (linear interpolation, **not** variance-preserving) | `scheduler.add_noise` | — |
| inference | `num_flow_steps` (10) Euler steps, `dt = −1/K`, `t: 1 → 0` | `num_inference_steps` (100) DDPM steps | one forward pass |
| sinusoid base | openpi `min_period`/`max_period` | base 10000 | none |

**The sinusoid base is not cosmetic.** Base 10000 is right for DDPM's integer `t ∈ [0, 100)` and
collapses to a near-constant for flow matching's `t ∈ [0, 1]`: the time signal silently disappears
and the loss just stops moving. `time_sincos="auto"` (the default) resolves to `openpi` for the
flow head and `ddpm` for the diffusion head. Override it only to reproduce that failure on purpose
— there is a regression test that does exactly this
(`test_ddpm_sinusoid_collapses_on_flow_time_and_the_openpi_one_does_not`).

### `CondDecoderLayer`

Pre-norm block, three residual sub-blocks (self-attn / cross-attn / FFN). `nn.TransformerDecoderLayer`
(used by `patch_policy`) and `ACTDecoderLayer` both hardcode "no conditioning", which is the axis
under test, hence the local block. With `use_adaln` the modulation projection emits `9 · d`:
`(scale, shift, gate)` per sub-block, the MolmoAct2 split; pi0.5 and `multi_task_dit` use 3- and
6-way splits of the same construction.

`adaln_zero_init=True` zero-initialises that projection, so every gate starts at 0, every block
starts as the identity, and the conditioning pathway grows in during training instead of scrambling
the initialisation. Note `self.apply(self._init_weights)` in `PatchNewTransformer.__init__`
overwrites it, so the zeroing is re-applied afterwards — deleting that second loop silently
disables AdaLN-Zero.

---

## Differences from `patch_policy`

### Architecture

| | `patch_policy` | `patch_new_policy` |
|---|---|---|
| visual encoder | frozen ViT, `patch_encoders.py` | **identical module, imported** |
| block-causal patch mask | `generate_mask_matrix` | **identical function, imported** |
| memory encoder | `Linear→Mish→Linear` MLP | same |
| trunk | **three** separate modules: `BlockCausalGPT`+`VQBeTHead`, `TransformerForDiffusion`, `PatchACTHead` | **one** `PatchNewTransformer` for all heads |
| decoder block | `nn.TransformerDecoderLayer` (diffusion) / `ACTDecoderLayer` (act) | `CondDecoderLayer` (pre-norm, optional adaLN) |
| decoder mask | `causal_mask` + `block_causal_memory_mask`, action tokens only | `decoder_masks`, mixed state/action sequence |
| time embedding | `DiffusionSinusoidalPosEmb`, base 10000, **no MLP** | `TimeEmbedding`, two frequency conventions, **+ 2-layer MLP** |
| time injection | fixed: one memory token in front of the patches | 5 options |
| state | `use_robot_state: bool`, one token appended per frame, via vqbet `MLP` | 5 options; `obs_token` reproduces the old one via `nn.Linear` |
| flow matching | absent | `action_head="flow"` |
| relative actions | absent | `use_relative_actions` + `ChunkAnchoredRelativeActionsStep` |
| `vqbet` head | present (two-phase RVQ training) | **removed** |
| FFN width | `4 · n_emb`, hardcoded in the diffusion head | `dim_feedforward`, configurable |
| ACT head | `ACTDecoder` reused verbatim, attention wrapped by `_MaskedAttention` | native — same trunk, `query_emb` instead of `input_emb` |

### Parameters

Shared and unchanged: `n_obs_steps`, `action_chunk_size`, `n_action_steps`, `vision_encoder`,
`vision_encoder_checkpoint`, `resize_shape`, `freeze_vision_encoder`, `n_patches_override`,
`dropout`, the whole DDPM block (`noise_scheduler_type`, `num_train_timesteps`, `beta_*`,
`prediction_type`, `clip_sample*`, `num_inference_steps`), `optimizer_lr`, `optimizer_betas`,
`optimizer_weight_decay`, the `IDENTITY / MIN_MAX / MIN_MAX` normalization map, and `horizon`.

| field | `patch_policy` default | `patch_new_policy` default | note |
|---|---|---|---|
| `n_obs_steps` | 5 | **2** | Push-T preset vs. this platform's |
| `action_head` | `"act"` | **`"flow"`** | |
| `dim_model` | 512 | 512 | in `patch_policy` this is the **ACT head only** |
| `n_heads` | 8 | 8 | |
| `dim_feedforward` | 3200 | **1024** | |
| `n_decoder_layers` | 1 | **8** | `patch_policy` inherits ACT's bug-compatible 1 |
| `p_drop_emb` | `diffusion_p_drop_emb` = 0.0 | `p_drop_emb` = 0.0 | now applies to every head |
| `use_robot_state` | `False` | → `state_injection="suffix_token"` | state is **on** by default here |

Dropped with the VQ-BeT head (25 fields): `gpt_*`, `vqvae_*`, `n_vqvae_training_steps`,
`offset_loss_weight`, `primary_code_loss_weight`, `secondary_code_loss_weight`,
`bet_softmax_temperature`, `sequentially_select`, `optimizer_vqvae_*`, `diffusion_n_layer`,
`diffusion_n_head`, `diffusion_hidden_dim`, `diffusion_p_drop_attn`, `pre_norm`,
`feedforward_activation`.

Added (15 fields): `time_injection`, `time_sincos`, `time_min_period`, `time_max_period`,
`adaln_zero_init`, `state_injection`, `use_relative_actions`, `relative_exclude_joints`,
`action_feature_names`, `p_drop_emb`, `num_flow_steps`, `time_sampling_beta_alpha`,
`time_sampling_beta_beta`, `time_sampling_scale`, `time_sampling_offset`.

### Reproducing `patch_policy`'s diffusion head inside `patch_new_policy`

```yaml
action_head: diffusion
time_injection: memory_token     # the leading timestep token
state_injection: obs_token       # or `none` for use_robot_state=False
dim_model: 256                   # diffusion_hidden_dim
n_heads: 4                       # diffusion_n_head
dim_feedforward: 1024            # 4 * n_emb
n_decoder_layers: 8              # diffusion_n_layer
```

Both masks are then bit-identical and the block structure matches. Two residual differences:
`TimeEmbedding` puts a 2-layer MLP after the sinusoid where `DiffusionSinusoidalPosEmb` does not,
and the state token is projected by `nn.Linear` rather than vqbet's `MLP`. Neither is a
configuration switch — this is an *equivalent*, not a *replica*.

---

## Differences from base ACT (`act` / `act_eef`)

`act_eef` is `ACTPolicy` verbatim plus a 14-D EEF feature check (`modeling_act_eef.py` is 31 lines);
everything below applies to both.

### Architecture

| | ACT | `patch_new_policy` |
|---|---|---|
| vision backbone | ResNet-18, **trained**, `FrozenBatchNorm2d`, ImageNet init | frozen ViT, `.eval()`, `requires_grad=False` |
| visual tokens | conv feature map flattened to `H·W` tokens, 2-D sinusoidal pos | `P` ViT patch tokens/camera, learned pos |
| temporal context | **`n_obs_steps == 1` enforced** — one frame, ever | `n_obs_steps` frames, block-causal across them |
| encoder | 4-layer **bidirectional** transformer encoder over `[latent, state, images…]` | none — the memory is an MLP over patch tokens |
| decoder | 1 layer (bug-compatible with the original ACT), **no masks at all** | `n_decoder_layers`, self-mask + block-causal memory mask |
| norm placement | post-norm (`pre_norm=False`) | pre-norm always |
| latent | **CVAE**: `vae_encoder` (4 layers) → `mu, logσ²`, reparameterised, `latent_dim=32` | none |
| loss | `l1 + kl_weight · KL` (`kl_weight=10`) | `mse` on velocity (flow) / noise (diffusion), or plain `l1` (act) |
| stochasticity | CVAE latent at train time, **zeros at inference** | flow/diffusion sampling from `randn` at inference |
| state | one encoder token, always | 5 injection options |
| time conditioning | n/a | 5 injection options |
| temporal ensembling | `temporal_ensemble_coeff` (`ACTTemporalEnsembler`) | none — chunk queue only |

The `action_head="act"` arm is **not** base ACT: it is base ACT's *decoder objective* (learned
queries + L1) on the block-causal patch memory, with no CVAE and no KL term.

### Parameters

| field | ACT | `patch_new_policy` |
|---|---|---|
| `n_obs_steps` | 1 (enforced) | 2 |
| chunk field name | `chunk_size` = 100 | `action_chunk_size` = 50 |
| `n_action_steps` | 100 | 50 |
| `dim_model` / `n_heads` / `dim_feedforward` | 512 / 8 / 3200 | 512 / 8 / 1024 |
| `n_encoder_layers` / `n_decoder_layers` | 4 / 1 | — / 8 |
| `feedforward_activation` | `relu` | GELU, fixed |
| VISUAL normalization | `MEAN_STD` | **`IDENTITY`** — the ViT applies its own |
| STATE / ACTION normalization | `MEAN_STD` | `MIN_MAX` |
| `optimizer_lr` | 1e-5 (+ `optimizer_lr_backbone` 1e-5) | 5.5e-5, one rate |
| `optimizer_weight_decay` | 1e-4 | 2e-4 |
| optimizer groups | backbone vs. the rest | nanoGPT decay / no-decay split |
| scheduler | none | none |
| `observation_delta_indices` | `None` | `range(1 - n_obs_steps, 1)` |
| `action_delta_indices` | `range(chunk_size)` → `0..99` | `range(1 - n_obs_steps, action_chunk_size)` → `-1..49` |

`action_delta_indices` is the one that bites offline tooling: the ACT batch's action window starts
at delta 0, the patch batch's starts at `1 - n_obs_steps`, so reading the "current" action out of a
patch batch needs an offset of `n_obs_steps - 1`.

---

## Handling details

**Frozen encoder.** `PatchNewPolicyModel.train()` overrides `nn.Module.train()` to force
`self.encoder.eval()` after every call. Without it the encoder returns to training mode on each
epoch: ResNet-18's BatchNorm would update its running statistics and the ViTs would apply dropout,
quietly breaking "the encoder is frozen". Encoder parameters also have `requires_grad=False` and
`get_optim_params` skips them, so they never reach the optimizer.

**Patch count is measured, not declared.** `_measure_n_patches()` runs a dry forward pass through
the encoder. The reference hardcodes `n_patches` per encoder YAML, where a wrong value silently
misaligns the block-causal mask with the token stream. `n_patches_override` exists for when
instantiating the encoder at config time is not possible.

**Memory encoder is an MLP on purpose.** `n_cond_layers=0` in every reference config. A
self-attending memory encoder would mix frame `S-1` into frame 0's tokens and the block causality
downstream would then be reading future observations. Do not "upgrade" it to a transformer.

**Timestep plumbing.** A DDPM scheduler yields one scalar timestep for the whole batch, on CPU even
when the model is on GPU; flow matching passes a per-sample `(B,)` tensor. `PatchNewTransformer.forward`
normalises both (`reshape(-1)`, `.to(device)`, `expand(B)`), which is why
`test_a_scalar_cpu_timestep_reaches_the_model` exists.

**Optimizer groups.** nanoGPT's rule, as in Patch Policy's `configure_optimizers`: decay the
`weight` of `Linear`/`Conv1d`/`Conv2d` only; norm weights, biases, embedding tables and bare
positional `nn.Parameter`s go to the no-decay group. Iteration uses `named_parameters(recurse=False)`
per module, so each parameter lands in exactly one group — asserted by
`test_optimizer_groups_are_a_partition_of_the_trainable_parameters`.

**Execution queues.** `select_action` fills `_queues` (`OBS_IMAGES`, `OBS_STATE` of length
`n_obs_steps`; `ACTION` of length `n_action_steps`), predicts a full chunk when the action queue
drains, and pops one action per call. `predict_action_chunk` reads `self._queues` — a batched
offline pass that never fills them will fail there; call `model.predict()` directly instead.

**Processor pipeline** (`processor_patch_new_policy.py`):

```
rename → add batch dim → [relative actions] → normalize → to device
                                     model
              unnormalize → [absolute actions] → to cpu
```

Note the order differs from `patch_policy`, which does `to device → normalize`. Same result,
different device for the statistics arithmetic.

**Relative actions (P6), if you turn it on.** `ChunkAnchoredRelativeActionsStep` pins the anchor to
the **newest** observation step, because the state here is `(B, n_obs_steps, state_dim)` and the
base `RelativeActionsProcessorStep` assumes `(B, state_dim)`. Two traps, both documented in the
processor's docstring:

1. Normalization statistics are still the **absolute** action distribution's, so `MIN_MAX` squeezes
   a small delta range onto a ruler sized by absolute joint extremes. Set
   `normalization_mapping.ACTION = MEAN_STD` on this arm.
2. `action_feature_names` must be set, or `relative_exclude_joints` cannot be resolved and *every*
   dimension is relativised — including the gripper, whose delta is ~0 almost everywhere, so its
   loss collapses and the policy never learns to open or close.
3. At execution time the post-step re-anchors on the last state seen, so with `n_action_steps > 1`
   the chunk's later steps are added back onto a newer anchor than training subtracted. Same as
   lerobot's pi0.5 — a difference from training, not an identity.

**One state token per observation step.** A deliberate generalisation of pi0: with
`n_obs_steps > 1` the P1/P2 arms carry one state token *per frame*, not one for the batch. A single
latest-state token would be visible to decoder position 0, which is aligned with the oldest frame —
leaking a future observation and destroying the block causality that is the whole point. At
`n_obs_steps == 1` the layout is exactly pi0's.

---

## Configuration reference

```python
# I/O
n_obs_steps = 2                  # observed frames; horizon = n_obs_steps + action_chunk_size - 1
action_chunk_size = 50
n_action_steps = 50              # must be <= action_chunk_size
normalization_mapping = {"VISUAL": IDENTITY, "STATE": MIN_MAX, "ACTION": MIN_MAX}

# head
action_head = "flow"             # "flow" | "diffusion" | "act"

# time axis
time_injection = "concat_mlp"    # concat_mlp | adaln | additive | memory_token | none
time_sincos = "auto"             # auto | openpi | ddpm    -- leave on auto
time_min_period = 4e-3           # openpi variant only
time_max_period = 4.0
adaln_zero_init = True

# state axis
state_injection = "suffix_token" # suffix_token | obs_token | action_concat | adaln | none
use_relative_actions = False
relative_exclude_joints = ["gripper"]
action_feature_names = None      # required when use_relative_actions=True

# encoder (shared with patch_policy)
vision_encoder = "dino_patch"    # 16 presets, see below
vision_encoder_checkpoint = None # required for "dynamo"; overrides a preset's weights
resize_shape = (224, 224)
freeze_vision_encoder = True
n_patches_override = None

# trunk
dim_model = 512                  # must be even (sinusoidal time embedding)
n_heads = 8
dim_feedforward = 1024
n_decoder_layers = 8
dropout = 0.1                    # attention + FFN
p_drop_emb = 0.0                 # memory and decoder embeddings

# diffusion head
noise_scheduler_type = "DDPM"; num_train_timesteps = 100
beta_schedule = "squaredcos_cap_v2"; beta_start = 1e-4; beta_end = 0.02
prediction_type = "epsilon"; clip_sample = True; clip_sample_range = 1.0
num_inference_steps = None       # None -> num_train_timesteps

# flow head
num_flow_steps = 10              # Euler steps at inference
time_sampling_beta_alpha = 1.5; time_sampling_beta_beta = 1.0
time_sampling_scale = 0.999; time_sampling_offset = 0.001

# optim (Patch Policy's `optim:` block; no scheduler)
optimizer_lr = 5.5e-5; optimizer_betas = (0.9, 0.999); optimizer_weight_decay = 2e-4
```

Encoder presets (`PATCH_ENCODER_PRESETS`, re-exported from `patch_policy`):

| preset | model | `output_dim` | `n_patches` |
|---|---|---:|---:|
| `dino_patch` | DINOv2 ViT-S/14 | 384 | 256 |
| `dinov3_patch` | DINOv3 ViT-S/16+ | 384 | 196 |
| `webssl_patch` | WebSSL DINO-300M | 1024 | 256 |
| `siglip2_patch` | SigLIP2 base/16 | 768 | 196 |
| `vjepa2_patch` | V-JEPA 2 ViT-L | 1024 | 256 |
| `resnet18_imagenet` / `resnet18_random` | ResNet-18 | 512 | 1 |
| `dynamo` | from checkpoint | 512 | 1 |
| `*_cls`, `*_patch_avg_pool` | pooled controls | — | 1 |

Only `dynamo` *requires* `vision_encoder_checkpoint` (validation raises without it). `dinov3_*`
takes one optionally — a path to a Meta `dinov3_*_pretrain_*.pth`; without it the hub entrypoint
downloads its own default weights. The `_cls` and
`_avg_pool` variants exist as ablation controls — Patch Policy's Table 4 shows pooling 256 → 64
tokens costs most of the gain, so do not reach for them to save memory.

---

## Train

```bash
lerobot-train \
  --policy.type=patch_new_policy \
  --policy.action_head=flow \
  --policy.vision_encoder=dino_patch \
  --policy.n_obs_steps=2 \
  --policy.action_chunk_size=50 \
  --policy.n_action_steps=50 \
  --dataset.repo_id=<your/dataset>
```

### What the 2026-09-04 run used, and what it found

The `patch_new_policy` checkpoint evaluated in
`~/YING/paper/policy/experiment_report/patch_policy/patch_policy-new-arch-flow-head-2026-09.md` overrode
`dim_model` to **256** (the class default is 512) with `dim_feedforward=1024`,
`n_decoder_layers=8`, `dino_patch`, `flow`, 200 k steps, batch 16, seed 1000 — 32.2 M parameters
against `patch_policy`'s 38.1 M and `act_eef`'s 51.7 M.

Read that report before planning an experiment on this policy. Its findings in one line: on
held-out data the three are indistinguishable (all differences inside the 2.0 % sampling-noise
line), while `patch_new_policy` fits the *training* set 37.7 % better than `patch_policy`
(seen→unseen ratio 1.89× vs 1.16×) and converges by 100 k steps. That is an overfitting shape, and
`p_drop_emb` (currently 0.0, applied to 1536 patch tokens) and `image_transforms` (currently off)
are the untried knobs aimed at it.

Also note that the run changed head, backbone **and** trunk at once, so nothing in it attributes a
result to flow matching. This policy makes the one-factor-at-a-time arms cheap — that is what it is
for.

---

## Files

| file | contents |
|---|---|
| `configuration_patch_new_policy.py` | `PatchNewPolicyConfig`, the injection taxonomies, `sincos_variant` / `horizon` / `encoder_preset` resolution, validation |
| `modeling_patch_new_policy.py` | `decoder_masks`, `TimeEmbedding`, `CondDecoderLayer`, `PatchNewTransformer`, `PatchNewPolicyModel`, `PatchNewPolicy` |
| `processor_patch_new_policy.py` | pre/post pipelines, `ChunkAnchoredRelativeActionsStep` |

Imported rather than copied: `make_patch_encoder` and `PATCH_ENCODER_PRESETS` (`patch_policy`),
`generate_mask_matrix` (`patch_policy`), `_make_noise_scheduler` (`diffusion`),
`create_sinusoidal_pos_embedding` and `sample_beta` (`pi0`).

## Tests

`tests/policies/patch_new_policy/test_patch_new_policy.py` — the mask invariants (no future
observation reaches an earlier decoder position, via state tokens or otherwise), the sinusoid-base
collapse, every `action_head × state_injection × time_injection` combination trains, the state
actually changes the output on each injection arm, adaLN-Zero starts as an identity block *and*
still receives gradients, scalar-CPU timesteps, encoder freezing, and the optimizer-group partition.
