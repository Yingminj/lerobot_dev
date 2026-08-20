# Patch Policy — Efficient Embodied Control via Dense Visual Representations

Port of [gaoyuezhou/patch_policy](https://github.com/gaoyuezhou/patch_policy) onto lerobot.

- Paper: [arXiv:2607.18236](https://arxiv.org/abs/2607.18236) — Zhou, Cui, Langford, Tan, LeCun, Pinto (NYU / Meta-FAIR / AMI Labs)
- Project page: <https://patch-policy.github.io/>

---

## The idea

Every other visual policy in this tree compresses each camera frame to **one** vector before the
policy sees it — `DiffusionRgbEncoder` and `VQBeTRgbEncoder` use SpatialSoftmax over a ResNet-18
feature map, ACT flattens a scratch-trained ResNet's patches into a bidirectional encoder, and the
VLAs inherit a whole language model to reach patch tokens at all.

Patch Policy's claim is that the choice between "pooled and cheap" and "dense and 7.6B parameters"
is architectural inertia, not a real trade-off. A frozen pretrained ViT already emits `P` patch
tokens per frame. Feed *all* of them to a standard transformer policy and you keep the spatial
detail; the only thing standing in the way is the attention mask.

The obstacle is that a sequence policy defaults to a **token-causal** mask. Flatten 256 patches into
the sequence and patch 3 can no longer see patch 200 — the mask has imposed a reading order on an
image, which has none. The two obvious fixes are both wrong: full attention leaks future
observations into past predictions, and token-causal shreds the spatial field.

The fix is one line of masking. **Time needs causality; space does not.**

```
            frame t-2         frame t-1          frame t
          ┌───────────┐    ┌───────────┐    ┌───────────┐
frame t-2 │  1 1 1 1  │    │  0 0 0 0  │    │  0 0 0 0  │   ← bidirectional inside the frame,
          │  1 1 1 1  │    │  0 0 0 0  │    │  0 0 0 0  │     zero to every future frame
          └───────────┘    └───────────┘    └───────────┘
frame t-1 │  1 1 1 1  │    │  1 1 1 1  │    │  0 0 0 0  │
          │  1 1 1 1  │    │  1 1 1 1  │    │  0 0 0 0  │
frame t   │  1 1 1 1  │    │  1 1 1 1  │    │  1 1 1 1  │
          │  1 1 1 1  │    │  1 1 1 1  │    │  1 1 1 1  │
```

Block-lower-triangular instead of element-lower-triangular. That is the paper.

Everything else is a consequence: the encoder stays frozen (so it can be swapped for whatever the
representation-learning community ships next year), the action head is interchangeable, and only
~30M parameters train.

---

## Architecture

### End to end

```
observation.images.*         observation.state (optional, off by default)
   (B, T, V, 3, H, W)              (B, T, S)
        │                                │
        │  resize -> resize_shape        │
        ▼                                │
┌───────────────────┐                    │
│  FROZEN ViT       │  DINOv2 / DINOv3 / WebSSL / SigLIP2 / V-JEPA 2 / ResNet18 / DynaMo
│  no grad, eval    │  never fine-tuned — the paper's whole cost argument rests on this
└───────────────────┘                    │
        │ (B·T·V, P, E)                  │ MLP -> E
        ▼                                ▼
   rearrange "(b s n) p e -> b s (n p) e"   +  one state token appended per frame
        │
        ▼
   observation tokens  (B, T, tokens_per_frame, E)
        │                tokens_per_frame = V·P (+1 with use_robot_state)
        │
        ├──────────────────────┬────────────────────────┐
        ▼                      ▼                        ▼
  action_head="vqbet"   action_head="diffusion"   action_head="act"
  BlockCausalGPT         TransformerForDiffusion   PatchACTHead
  (block-causal SELF-    (block-causal MEMORY      (block-causal MEMORY
   attention trunk)       mask, cross-attn)         mask, cross-attn)
        │                      │                        │
        ▼                      ▼                        ▼
   RVQ code + offset      ε-prediction             direct regression
   focal + L1 loss        MSE loss                 L1 loss
```

### Where block-causality lives, per head

This differs by head, and the difference is inherited from the reference, not invented here.

| head | block-causal mask applied to | patch tokens self-attend? |
|---|---|---|
| `vqbet` | GPT trunk self-attention (`(T·P)²`) | yes, `gpt_n_layer` times |
| `diffusion` | decoder→memory cross-attention | no — the memory encoder is a 2-layer MLP (`n_cond_layers=0` in every reference config) |
| `act` | decoder→memory cross-attention | no |

So the `diffusion` and `act` heads are dramatically cheaper: they never pay the `O((T·P)²)` self-
attention bill, only `O(horizon · T · P)` cross-attention. The `vqbet` head is where the paper's
headline numbers come from, and where the sequence-length cost the paper admits to actually lands.

### 1. Frozen patch encoder — `patch_encoders.py`

One class per `models/encoder/*.py` in the reference. All share `PatchEncoder`, which resizes,
collapses leading dims, runs the backbone, and restores leading dims. A pooled encoder (`CLS`
token, `avg_pool`, ResNet, DynaMo) returns `P = 1` rather than dropping the axis, so the rest of the
policy never branches on encoder type — mirroring the reference's `n_patches: 1` configs.

| `vision_encoder` | backbone | source | `E` | `P` @224 |
|---|---|---|---|---|
| `dino_patch` *(default)* | DINOv2 ViT-S/14 | `torch.hub`, pinned to commit `b48308a` | 384 | 256 |
| `dino_cls`, `dino_patch_avg_pool` | ↑ | ↑ | 384 | 1 |
| `dinov3_patch` | DINOv3 ViT-S/16+ | HF `transformers` | 384 | 196 |
| `dinov3_cls`, `dinov3_patch_avg_pool` | ↑ | ↑ | 384 | 1 |
| `webssl_patch` | WebSSL DINO-300M | HF `Dinov2Model` | 1024 | 256 |
| `webssl_cls`, `webssl_patch_avg_pool` | ↑ | ↑ | 1024 | 1 |
| `siglip2_patch` | SigLIP2 base/16 vision tower | HF `transformers` | 768 | 196 |
| `siglip2_patch_avg_pool` | ↑ | ↑ | 768 | 1 |
| `vjepa2_patch` | V-JEPA 2 ViT-L | HF `transformers` | 1024 | 256 @256px |
| `vjepa2_patch_avg_pool` | ↑ | ↑ | 1024 | 1 |
| `resnet18_imagenet`, `resnet18_random` | ResNet-18 | `torchvision` | 512 | 1 |
| `dynamo` | pickled `nn.Module` | `vision_encoder_checkpoint` | 512 | 1 |

Three notes:

- **`P` is measured, not configured.** The reference hardcodes `n_patches` in each encoder YAML; a
  stale value there silently misaligns the block-causal mask with the token stream and the model
  trains anyway, just wrongly. `PatchPolicyModel._measure_n_patches` runs one dummy forward pass at
  build time instead. `n_patches_override` exists if you must skip that.
- **V-JEPA 2 is a video model.** Each still frame is repeated `n_frames=2` times to form the shortest
  clip it accepts, exactly as the reference does. Set `resize_shape=(256, 256)` for it.
- **`dynamo` runs `torch.load(weights_only=False)`**, which executes whatever the pickle contains.
  Point it only at checkpoints you produced.

Table 7 of the paper ranks these: DINOv2 ≈ WebSSL > V-JEPA 2 > SigLIP 2 on all four sim environments.
Self-supervised beats image-text-aligned; language supervision costs geometry. SigLIP 2 is ported for
the ablation, not as a recommendation.

### 2. Token layout

`train_policy.py` in the reference flattens `N T V P E -> N T (V P) E`: multiple cameras are stacked
along the **patch** dimension, not the time dimension, and the mask is built with
`n_patches = P * views`. A wrist-camera patch may therefore attend to a head-camera patch at the
same instant — they are the same "frame" — but not to either one step later. That is reproduced
exactly.

With `use_robot_state=True` (off by default; see *Deviations*), one projected state token is appended
to each frame's block. The mask needs no change: an intra-frame token is bidirectionally visible by
construction.

### 3. Block-causal mask — `generate_mask_matrix`

Ported verbatim from `models/vq_behavior_transformer/gpt.py`. Stored as `bool` rather than the
reference's `float32` — identical under the `bias == 0` test, and 4× smaller, which matters when the
matrix is `(T·P)² = 2560²` at `T=10, P=256`. One tensor is shared by every layer and registered
non-persistent, so it never enters a checkpoint.

`block_causal_memory_mask` is the cross-attention counterpart, from
`TransformerForDiffusion.__init__`: decoder position `i` predicts the action at observation step `i`
(clamped to the last frame), so it may read the patch tokens of frames `0..i` and nothing later.
`n_leading_tokens=1` accounts for the diffusion head's timestep token, which is always visible.

### 4a. `action_head="vqbet"` — the paper's primary configuration

`BlockCausalGPT` subclasses lerobot's `GPT` (`policies/vqbet/vqbet_utils.py`), which is already the
same nanoGPT the reference forked. Two changes:

1. the `tril` mask becomes the block-causal mask;
2. positions are learned over `gpt_block_size × n_patches` slots, not `gpt_block_size`.

The forward pass takes `(B, T, P, D)`, flattens to `(B, T·P, D)`, and reads out `logits[:, :, -1]` —
**the last token of each frame's block**. That token has attended to every token in its frame and to
all previous frames, so it is the per-frame summary. From there, lerobot's `VQBeTHead`, `VqVae`,
`ResidualVQ`, `MLP` and `FocalLoss` are reused unchanged: RVQ code classification (focal loss) plus a
per-code offset (L1).

Sequence length is `T · V · P`, and attention is quadratic in it. At `T=10, V=1, P=256` that is 2560
tokens. The paper acknowledges this as its main cost and never publishes the training-time-vs-`P`
curve; multi-camera scaling is unvalidated. Budget before you scale.

### 4b. `action_head="diffusion"`

`TransformerForDiffusion`, ported from `models/diffusion_policy/diffusion_policy.py`. lerobot's own
diffusion policy is a 1D UNet with FiLM conditioning on a single pooled vector — it cannot take a
token sequence as memory, let alone mask one, so this is a new module.

Memory is `[timestep token] ++ [T·P patch tokens]`, encoded by a 2-layer MLP. `horizon` action tokens
run through an `nn.TransformerDecoder` with a causal `tgt_mask` and the block-causal `memory_mask`.
Only the branch the reference actually instantiates is kept (`time_as_cond=True`, `obs_as_cond=True`,
`causal_attn=True`, `n_cond_layers=0`); the unused encoder-only/BERT path is dropped.

### 4c. `action_head="act"` — new, no counterpart in the reference

Patch Policy ships VQ-BeT and Diffusion Policy heads only. This head is the diffusion head with the
noisy-action input replaced by learned query embeddings and ε-prediction replaced by L1 — that is,
ACT's decoder reading the same block-causally masked patch memory. lerobot's `ACTDecoder` and
`ACTDecoderLayer` are reused verbatim.

Two design points worth knowing:

- **Why `horizon = n_obs_steps + action_chunk_size - 1` and not just `chunk_size`.** Stock ACT
  predicts one chunk from one frame, so every query would legitimately see every frame and the memory
  mask would be all-ones — block-causality would be *vacuous*. Aligning decoder position `i` with
  observation step `i`, as the reference's diffusion head does (`pred_horizon = action_window_size +
  window_size - 1`), is what gives the mask something to constrain.
- **Why the self-attention mask.** `ACTDecoderLayer` runs self-attention over the queries before
  cross-attention. Without a causal `tgt_mask`, query `i` reads query `j > i`, which has seen frame
  `j` — laundering a later observation into an earlier prediction and defeating the memory mask
  entirely.

`ACTDecoderLayer` calls its attention modules without a mask. Rather than fork its `forward`, the two
`nn.MultiheadAttention` submodules are swapped for `_MaskedAttention`, a wrapper that always supplies
a fixed `attn_mask`. That keeps `ACTDecoder`/`ACTDecoderLayer` reused rather than copied.

---

## Train

```bash
pip install -e ".[patch_policy]"
```

```bash
lerobot-train \
  --policy.type=patch_policy \
  --policy.action_head=vqbet \
  --policy.vision_encoder=dino_patch \
  --policy.n_obs_steps=5 \
  --policy.action_chunk_size=5 \
  --policy.n_action_steps=5 \
  --dataset.repo_id=lerobot/pusht
```

The VQ-BeT head trains in two phases, as in lerobot's VQ-BeT: the first `n_vqvae_training_steps`
optimizer steps fit the residual VQ on actions alone, then the trunk and heads start. `forward`
returns `recon_l1_error` / `n_different_codes` during phase 1 and the policy loss after.

### Reference presets

From `configs/train_*.yaml`. The paper's own per-environment settings — start here rather than from
the dataclass defaults, which are the Push-T column.

| | Push-T | LIBERO Goal | BlockPush | Cube |
|---|---|---|---|---|
| `n_obs_steps` | 5 | 10 | 3 | 5 |
| `action_chunk_size` | 5 | 1 | 1 | 5 |
| `gpt_n_layer` / `gpt_n_head` / `gpt_hidden_dim` | 8 / 8 / 512 | 6 / 6 / 120 | 8 / 8 / 512 | 8 / 8 / 512 |
| `offset_loss_weight` | 10 | 100 | 100 | 10 |
| `optimizer_lr` | 5.5e-5 | 5.5e-5 | 1e-4 | 5.5e-5 |
| `optimizer_weight_decay` | 2e-4 | 2e-4 | 0.0 | 2e-4 |
| batch size | 64 | 32 | 32 | 128 |
| epochs | 400 | 50 | 150 | 200 |
| cameras | 1 | 2 | 2 | 1 |

Shared across all four: `vqvae_n_embed=16`, `vqvae_embedding_dim=512`, 2 RVQ groups,
`betas=(0.9, 0.999)`, `dropout=0.1`, no LR schedule.

Diffusion-head presets (`configs/train_*_diffusion.yaml`) differ: `n_obs_steps=2`,
`action_chunk_size=3`, `diffusion_n_layer=8`, `diffusion_n_head=4`, `diffusion_hidden_dim=256`,
`lr=1e-4`, `weight_decay=0.0`, 100 DDPM steps with `squaredcos_cap_v2` and `clip_sample=True`.

### What to expect

The paper's "~40% relative improvement" is an aggregate and is **not** uniform. Per environment,
patch vs. average-pooled with the same head: BlockPush +100%, Cube +572%, Push-T +26%, LIBERO Goal
−3%. The gain tracks how much spatial precision the task demands — largest on the 2 mm-tolerance
real-robot cable insertion, near zero on LIBERO Goal. Validate on your own task before planning
around the abstract's number.

Table 4 is the more transferable result: pooling the patch grid from 256 → 64 tokens loses 0.17 of
0.69 coverage on Push-T, while 64 → 1 loses only 0.04 more. There is a cliff, not a dial. Do not
average-pool patch features to save memory — you will land back at global-representation performance
and pay for the encoder anyway.

---

## Configuration reference

### I/O structure

| field | default | meaning |
|---|---|---|
| `n_obs_steps` | 5 | frames of history, reference `window_size` |
| `action_chunk_size` | 5 | actions per chunk, reference `action_window_size` |
| `n_action_steps` | 5 | actions executed per re-plan; must be ≤ `action_chunk_size` |
| `horizon` *(derived)* | 9 | `n_obs_steps + action_chunk_size - 1`, the diffusion/ACT sequence length |

`action_delta_indices` is `range(1 - n_obs_steps, action_chunk_size)`, so the dataloader delivers
`horizon` actions per sample. `unpack_actions` slices that into one chunk per observation step for
the VQ-BeT head; the diffusion and ACT heads consume the flat sequence.

### Encoder

`vision_encoder`, `vision_encoder_checkpoint`, `resize_shape` (default `(224, 224)`),
`freeze_vision_encoder` (default `True` — the paper never fine-tunes), `n_patches_override`.

### Trunk / heads

`gpt_block_size` (defaults to `n_obs_steps + action_chunk_size`, matching `bet.py`), `gpt_input_dim`
(defaults to the encoder's feature dim), `gpt_n_layer`, `gpt_n_head`, `gpt_hidden_dim`,
`gpt_output_dim`, `dropout`; the `vqvae_*`, `*_loss_weight`, `bet_softmax_temperature` and
`sequentially_select` fields feed lerobot's `VQBeTHead` directly; `diffusion_*` and the scheduler
fields feed `TransformerForDiffusion`; `dim_model` / `n_heads` / `dim_feedforward` /
`n_decoder_layers` / `pre_norm` / `feedforward_activation` are named to match `ACTConfig` so
`ACTDecoder` can be constructed straight from this config.

### Normalization

`VISUAL` is `IDENTITY` — the frozen ViTs apply their own ImageNet or HF-processor mean/std to pixels
in `[0, 1]`, exactly as the reference encoders do. Normalizing in the pipeline would double-normalize.
`STATE` and `ACTION` are `MIN_MAX`.

---

## Deviations from the reference implementation

Each is deliberate; none is a bug.

1. **No goal conditioning.** The reference concatenates a goal-image embedding onto every patch
   token's feature dim when `goal_dim > 0`. Only LIBERO Goal uses it; Push-T, BlockPush and Cube all
   run `goal_dim: 0`, as does every real-robot task. lerobot has no goal-image convention in its
   dataset pipeline, so it is omitted rather than half-wired.
2. **No `act_scale`.** The reference divides raw actions by a hand-set constant (500 for Push-T and
   Cube, 1 elsewhere) before the VQ-VAE. lerobot's processor already normalizes actions to `[-1, 1]`
   from dataset statistics, which is the same job done from data instead of by hand.
3. **VQ-VAE fitting follows lerobot, not the reference.** The reference collects an epoch of actions
   and fits the codebook in a separate loop (`vqvae_iters` = 300–1000 passes) before BeT training
   starts. Here the first `n_vqvae_training_steps` calls to `forward` train the VQ-VAE alone —
   lerobot's VQ-BeT convention, which needs no changes to the training script.
4. **No EMA on the diffusion head.** The reference samples from an EMA copy of the denoiser. An EMA
   must be updated *after* `optimizer.step()`, and lerobot's training loop has no such hook.
   Expect slightly noisier diffusion rollouts than the paper's.
5. **`use_robot_state` exists and defaults to `False`.** The reference has no proprioception pathway
   at all — `obs_dim` is `encoder.output_dim` and observation tokens are purely visual. The default
   reproduces that. Turning it on appends a state token per frame, which also moves the VQ-BeT readout
   (the last token of the block) from the last patch onto the state token.
6. **`P` is measured rather than declared** (see *Frozen patch encoder* above).
7. **The mask is `bool`, not `float32`**, and shared across layers. Numerically identical.
8. **The unused `TransformerForDiffusion` branches are dropped** — the encoder-only/BERT path and the
   `causal_attn=False` path are never instantiated by any reference config.
9. **`FlashAttention` is not used**, matching the reference. The paper lists this as open work; the
   `O((T·P)²)` bill on the `vqbet` head is the place it would pay off.

---

## Files

| file | contents | provenance |
|---|---|---|
| `configuration_patch_policy.py` | `PatchPolicyConfig`, `PATCH_ENCODER_PRESETS` (one entry per `configs/encoder/*.yaml`) | new |
| `patch_encoders.py` | `PatchEncoder` + 7 backbones | new, ported from `models/encoder/*.py` |
| `modeling_patch_policy.py` | masks, `BlockCausalGPT`, `TransformerForDiffusion`, `PatchACTHead`, `PatchPolicyModel`, `PatchPolicy` | mixed — see the module docstring for the per-symbol reuse/new split |
| `processor_patch_policy.py` | pre/post-processing pipelines | new, structurally identical to VQ-BeT's |

Reused from lerobot without modification: `GPT` and `ResidualVQ` (`policies/vqbet/vqbet_utils.py`),
`VqVae` / `VQBeTHead` / `MLP` / `FocalLoss` (`policies/vqbet/modeling_vqbet.py`), `ACTDecoder` and
`ACTDecoderLayer` (`policies/act/modeling_act.py`), `_make_noise_scheduler`
(`policies/diffusion/modeling_diffusion.py`).

## Tests

```bash
python -m pytest tests/policies/patch_policy/test_patch_policy.py -v
```

Covers the block-causal mask's two defining properties, the memory mask's per-step visibility, the
token layout with and without the state token, `unpack_actions`, and a forward/backward plus a
`select_action` rollout for all three heads. A random-weight stub encoder stands in for the real
backbones so the suite needs no network access.
