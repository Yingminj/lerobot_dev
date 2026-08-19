# `act_dit` — ACT's decoder as a DiT denoiser

Scheme **S1** of `paper/policy/act-diffusion-integration-2026-08.md`: keep every bit of ACT's
conditioning path (~900 spatial tokens, token-level cross-attention from each decoder layer) and
change only what the decoder is asked to produce.

```
ACT      decoder_in = zeros                  →  action,    L1 + KL(CVAE)
ACT-DiT  decoder_in = proj(noisy_action), +t →  velocity,  MSE (flow matching / diffusion)
```

The point: ACT's CVAE latent is forced to zero at inference (`modeling_act.py`, eval branch), so a
deployed ACT is a deterministic chunk regressor — `use_vae=True` is training-time regularisation,
not inference-time multimodality. S1 moves the distribution modelling onto a path that survives
deployment, without giving up the observation bandwidth that `multi_task_dit` throws away when it
pools CLIP CLS tokens into a single adaLN conditioning vector.

| file | what it is |
|---|---|
| `configuration_act_dit.py` | `ACTDiTConfig(ACTConfig)` + generative hyperparameters (names match `multi_task_dit`'s objectives) |
| `modeling_act_dit.py` | `ACTDiTPolicy` / `ACTDiT` / `ACTDiTDecoder` / `ACTDiTDecoderLayer` |
| `selftest.py` | CPU self-check: refactor equivalence, encoder-runs-once, both objectives, both ablation arms |

---

## 1. Reference configuration

Every shape in this document is for the configuration below, and was **measured**, not estimated
(`ACTDiTConfig` defaults + a 3-camera 480×640 rig):

| symbol | meaning | value |
|---|---|---|
| `B` | batch | — |
| `C` = `dim_model` | transformer width | 512 |
| `A` = `action_dim` | action dimensions | 16 |
| `S` = `state_dim` | robot state dimensions | 16 |
| `DS` = `chunk_size` | decoder queries = action chunk length | 100 |
| `ES` | encoder tokens | **902** = 1 latent + 1 state + 3 × 300 |
| `T` = `timestep_embed_dim` | timestep embedding width | 256 |
| `K` | ODE / denoising steps at inference | 10 |
| — | camera feature map after ResNet18 | 15 × 20 = 300 tokens/cam |

`ES = 902` is the number that makes S1 worth doing: the decoder cross-attends 902 observation
tokens at every layer, where `multi_task_dit` sees one pooled vector.

---

## 2. Architecture

```
                          ACT-DiT — producing one action chunk

  observation.images.{cam}            observation.state            timestep t
  (B, 3, 480, 640) × 3 cams           (B, 16)                      (B,)  ∈ [0,1]  (or 0…99)
           │                               │                            │
           ▼                               ▼                            ▼
  ┌─────────────────┐            ┌──────────────────┐        ┌────────────────────────┐
  │ ResNet18 layer4 │            │ Linear  16 → 512 │        │ SinusoidalPosEmb  256  │
  │ FrozenBatchNorm │            │ (encoder_robot_  │        │ Linear 256 → 1024      │
  │ (B,512,15,20)   │            │  state_in_proj)  │        │ SiLU                   │
  └────────┬────────┘            └────────┬─────────┘        │ Linear 1024 → 256      │
           ▼                              │                  └───────────┬────────────┘
  ┌─────────────────┐                     ├────────────┐                 │ (B, 256)
  │ Conv2d 1×1      │                     │            │                 │
  │ 512 → 512       │            token (1, B, 512)   static cond         │
  └────────┬────────┘                     │          (B, 512)            │
     flatten(h·w)                         │            └───────┬─────────┘
   (300, B, 512) × 3                      │                    ▼
           │                              │            cond (B, 768)  ────────────────┐
  latent = zeros(B,32)                    │            [t_emb ‖ state_emb]            │
           │ Linear 32 → 512              │                                           │
           ▼                              ▼                                           │
        (1, B, 512) ──────────► encoder_in  (902, B, 512)                             │
                                + 2-D sinusoidal pos (902, 1, 512)                    │
                                           │                                          │
                          ┌────────────────▼─────────────────┐                        │
                          │ ACTEncoder × 4  — UNCHANGED      │                        │
                          │ self-attn over 902 tokens,       │                        │
                          │ FFN 512 → 3200 → 512             │                        │
                          └────────────────┬─────────────────┘                        │
                                           ▼                                          │
                          encoder_out (902, B, 512)  ──────────────┐                  │
                          encoder_pos (902, 1, 512)  ──────────────┤                  │
                                                                   │                  │
  ══════ ODE / denoising loop, k = 0 … K-1  — ONLY this repeats ═══╪══════════════════╪═══
                                                                   │                  │
   x_k (B, 100, 16) ─► Linear 16 → 512 ─► transpose ─► (100,B,512) │                  │
                       (action_in_proj)                       │    │                  │
   decoder_pos_embed  nn.Embedding(100, 512) ─► (100,1,512) ───┤    │                  │
                                                              ▼    ▼                  ▼
                                    ┌─────────────────────────────────────────────────────┐
                                    │ ACTDiTDecoderLayer × 4        (detail in §3)        │
                                    │   self-attn(100 queries)   ← adaLN(shift,scale,gate)│
                                    │   cross-attn ← encoder_out (902 tokens)             │
                                    │   FFN 512→3200→512         ← adaLN(shift,scale,gate)│
                                    └───────────────────────┬─────────────────────────────┘
                                                            ▼  LayerNorm
                                                     (100, B, 512)
                                                            │  Linear 512 → 16 (action_head)
                                                            ▼
                                              v̂_k  (B, 100, 16)   velocity (or ε̂)
                                                            │
                                 x_{k+1} = x_k + Δt · v̂_k ──┘        (Euler; RK4 optional)
  ═══════════════════════════════════════════════════════════════════════════════════════
                                                            ▼
                                          action chunk  (B, 100, 16)
```

Two structural facts the diagram is drawn to make obvious:

1. **Everything above the double line runs once per chunk.** The ResNet passes, the 902-token
   encoder and the state/timestep projections are hoisted out of the loop — the same hoist
   `modeling_diffusion.py` does with `global_cond`. Only the decoder is iterated.
2. **The observation never becomes a vector.** It reaches the decoder as 902 keys/values that each
   layer attends over independently. `use_cross_attention=False` (ablation D) is the only path that
   pools it, and that is exactly what that arm exists to measure.

---

## 3. `ACTDiTDecoderLayer` — the DiT block

Pre-norm throughout, regardless of `config.pre_norm`: adaLN modulates a branch's *input*, which is
only meaningful pre-norm, and post-norm DiT blocks are not something anyone reports training stably.

```
   cond (B, 768)
        │
        ▼
   ┌──────────────────────────────┐
   │ SiLU → Linear 768 → 6·512    │   adaln, LAST LAYER ZERO-INITIALISED
   └──────────────┬───────────────┘
                  │ chunk(6) → six (1, B, 512), broadcast over the 100 queries
                  ▼
   shift_sa   scale_sa   gate_sa   shift_ff   scale_ff   gate_ff


   x (100, B, 512)
        │
        ├─────────────────────────────────────────────┐  residual
        ▼                                             │
   LayerNorm ─► modulate(·, shift_sa, scale_sa)       │      modulate(x, shift, scale)
        │        h                                    │        = x·(1 + scale) + shift
        ├─► q = k = h + decoder_pos_embed             │      pos added at attention time only,
        ▼                                             │      never into the residual stream
   MultiheadAttention(q, k, v=h)  8 heads             │
        │                                             │
        ▼  · gate_sa  · dropout                       │
        └──────────────────────► + ◄──────────────────┘
                                 │
        ┌────────────────────────┴────────────────────┐  residual        ── skipped entirely
        ▼                                             │                     in ablation D
   LayerNorm ─► h                                     │
        │                                             │
        ├─► query = h + decoder_pos_embed             │
        ├─► key   = encoder_out + encoder_pos         │
        ├─► value = encoder_out            (902 tokens, UNMODULATED)
        ▼                                             │
   MultiheadAttention  8 heads                        │
        │  · dropout   (no gate: high-bandwidth path, no scalar gating)
        └──────────────────────► + ◄──────────────────┘
                                 │
        ┌────────────────────────┴────────────────────┐  residual
        ▼                                             │
   LayerNorm ─► modulate(·, shift_ff, scale_ff)       │
        │                                             │
        ▼                                             │
   Linear 512→3200 → ReLU → dropout → Linear 3200→512 │
        │                                             │
        ▼  · gate_ff  · dropout                       │
        └──────────────────────► + ◄──────────────────┘
                                 │
                                 ▼
                          x (100, B, 512)
```

`adaln[-1]` is zero-initialised, so `gate_sa = gate_ff = 0` at step 0 and each block starts as the
identity. A direct consequence worth knowing before you read a training curve: **`time_mlp` receives
exactly zero gradient at initialisation** (∂out/∂cond = 0 while the gate is shut), and in ablation D
so does the entire observation path. Both start learning once `adaln[-1]` moves off zero. A flat
first few hundred steps is the design, not a bug — `selftest.py` asserts this init.

---

## 4. Module-by-module I/O

### Inference path (`ACTDiTPolicy.predict_action_chunk`)

| # | module | input | output | notes |
|---|---|---|---|---|
| 1 | `_prepare_batch` | `observation.state (B,16)`, `observation.images.* (B,3,480,640)` | same + `observation.images` as a **list** of 3 tensors | inherited packing convention from `ACTPolicy` |
| 2 | `ACT.encode_observations` → `backbone` (ResNet18, frozen BN) | `(B,3,480,640)` per camera | `(B,512,15,20)` feature map | ImageNet weights by default; 480/32=15, 640/32=20 |
| 3 | `encoder_cam_feat_pos_embed` (`ACTSinusoidalPositionEmbedding2d`) | `(B,512,15,20)` | `(B,512,15,20)` positional map | **known ACT defect: identical for every camera** — see §8 |
| 4 | `encoder_img_feat_input_proj` (Conv2d 1×1) | `(B,512,15,20)` | `(B,512,15,20)` → rearranged `(300,B,512)` | one token per feature-map cell |
| 5 | `encoder_robot_state_input_proj` (Linear) | `(B,16)` | `(1,B,512)` token | **reused twice**: as an encoder token *and* as the static adaLN conditioning |
| 6 | `encoder_latent_input_proj` (Linear) | `zeros(B,32)` | `(1,B,512)` token | `use_vae=False`, so this is a constant token. Kept as ACT builds it, which is why every shared module keeps its ACT parameter name — an ACT checkpoint warm-starts act_dit with `strict=False` (scheme S3) |
| 7 | `ACTEncoder` × 4 | tokens `(902,B,512)`, pos `(902,1,512)` | `encoder_out (902,B,512)` | unchanged ACT encoder |
| 8 | `ACTDiT.encode_conditioning` | the above | `(encoder_out (902,B,512), encoder_pos (902,1,512), static_cond (B,512))` | **the loop-invariant bundle**; `static_cond` is `(B,1024)` in ablation D (state ‖ mean-pooled `encoder_out`) |
| 9 | `FlowMatchingObjective.conditional_sample` | `conditioning`, `B` | `(B,100,16)` | draws `x_0 ~ N(0,I)`, integrates `t: 0→1` in `K` Euler (or RK4) steps |
| 10 | `time_mlp` | `t (B,)` | `(B,256)` | `SinusoidalPosEmb → Linear → SiLU → Linear` |
| 11 | `action_in_proj` (Linear) | `x_k (B,100,16)` | `(100,B,512)` after transpose | replaces ACT's `torch.zeros` decoder input |
| 12 | `ACTDiTDecoder` (§3) × 4 layers + final `LayerNorm` | `x (100,B,512)`, `encoder_out`, `cond (B,768)`, both pos embeds | `(100,B,512)` | the only module inside the loop |
| 13 | `action_head` (Linear) | `(B,100,512)` | `v̂ (B,100,16)` | velocity for flow matching; `ε̂` or `â` for diffusion, per `prediction_type` |
| 14 | `ACTPolicy.select_action` (inherited) | the chunk | `(B,16)` per call | action queue of `n_action_steps`, or `ACTTemporalEnsembler` |

### Training path (`ACTDiTPolicy.forward`)

Steps 1–8 are identical and run **once**; the denoiser is then called **once** (not `K` times):

| # | step | input | output |
|---|---|---|---|
| 9 | sample `t` | — | `(B,)`; `Beta(1.5, 1.0)`-derived by default, or uniform |
| 10 | build the noisy chunk | `a (B,100,16)`, `ε ~ N(0,I)` | `x_t = t·a + (1 − (1−σ_min)·t)·ε` |
| 11 | target | same | `v = a − (1−σ_min)·ε`  (flow matching) / `ε` (diffusion, `prediction_type="epsilon"`) |
| 12 | `ACTDiT.forward` | `x_t`, `t`, `conditioning` | `v̂ (B,100,16)` |
| 13 | loss | `v̂`, `v`, `action_is_pad (B,100)` | scalar MSE, padded steps masked out (`do_mask_loss_for_padding=True`) |

Returned as `(loss, {"flow_matching_loss": …})` — the key follows `config.objective`.

**Timestep semantics differ between objectives**: flow matching passes `t ∈ [0,1]` as float, diffusion
passes integers `0…num_train_timesteps-1`. The same `time_mlp` serves both, so the sinusoidal
embedding sees very different input scales in the two modes. Checkpoints are not transferable across
`objective`.

### Parameter budget (measured, reference config)

| module | S1 (`use_cross_attention=True`) | D (`use_cross_attention=False`) |
|---|---:|---:|
| backbone (ResNet18) | 11.17 M | 11.17 M |
| encoder × 4 | 17.33 M | 17.33 M |
| decoder × 4 | 30.99 M | 33.07 M |
| ├ adaLN per layer | 2.36 M | 3.93 M |
| ├ self-attn per layer | 1.05 M | 1.05 M |
| ├ cross-attn per layer | 1.05 M | — (deleted) |
| └ FFN per layer | 3.28 M | 3.28 M |
| `time_mlp` | 0.53 M | 0.53 M |
| `action_in_proj` | 8.7 k | 8.7 k |
| **total** | **60.4 M** | **62.5 M** |
| ACT baseline (`n_dec=1`, `use_vae=True`) | 51.6 M | — |

Note the contrast with §2.3 of the design doc: in `multi_task_dit`, adaLN is 85 % of the DiT
parameters because it eats a 5664-dimensional pooled observation. Here it eats 768 dimensions
(timestep + state) and lands at 30 % of the decoder — the visual conditioning went to
cross-attention instead, where the parameter cost is one KV projection per layer.

---

## 5. Conditioning design

Following DiT-X (arXiv:2509.01819) and Tenma, not `multi_task_dit`:

| signal | dimensionality | route | why |
|---|---|---|---|
| diffusion/flow timestep | 1 scalar → 256 | adaLN-Zero | chunk-constant — a per-layer 6×512 modulation is exactly the right shape for it |
| robot state | 16 → 512 | adaLN-Zero **and** an encoder token | low-dimensional, chunk-constant; free to send both ways |
| visual tokens | 902 × 512 | cross-attention, unmodulated | spatially structured; collapsing it into 6 vectors is the `multi_task_dit` failure mode |

Ablation D deliberately breaks the third row: the observation is mean-pooled into `static_cond` and
cross-attention is removed, degenerating to DiT-Block-Policy. The unused `multihead_attn`/`norm2`/
`dropout2` are **deleted** rather than left dangling, so parameter counts stay honest and DDP does
not trip over parameters that never receive gradient.

---

## 6. What is reused

- `ACTPolicy` — `select_action`, the action queue, `ACTTemporalEnsembler`, `reset`,
  `get_optim_params` are inherited untouched. Chunk semantics are ACT's; only chunk *production*
  differs.
- `ACT.encode_observations` — backbone, encoder, token assembly, all of it.
- `multi_task_dit`'s `FlowMatchingObjective` / `DiffusionObjective` / `SinusoidalPosEmb` / `modulate`
  — their `model(x, t, conditioning_vec=…)` contract is precisely what `ACTDiT.forward` implements,
  which is why `conditioning_vec` can be an opaque 3-tuple instead of a flat vector.
- ACT's processor pipeline, for free: `ACTDiTConfig` subclasses `ACTConfig` and
  `make_pre_post_processors` dispatches on `isinstance(policy_cfg, ACTConfig)`. The action
  representation is unchanged, so this is the correct pipeline — contrast `act_delta`, which needs a
  different one and therefore deliberately does *not* subclass.
- `factory.py` needed no edit at all: the `ACTDiTConfig` → `ACTDiTPolicy` naming convention in
  `_get_policy_cls_from_policy_name` resolves `--policy.type=act_dit` on its own.

### Edits outside this folder

| file | edit |
|---|---|
| `policies/act/modeling_act.py` | `ACT.forward` split into `encode_observations` + `forward` (pure code move, so the encoder can be hoisted out of the sampling loop) |
| `policies/act/modeling_act.py` | two class-attribute hooks: `ACTPolicy.model_class`, `ACT.decoder_class`, bound to `ACT`/`ACTDecoder` at the end of the module |
| `policies/__init__.py` | one import line, so `@PreTrainedConfig.register_subclass("act_dit")` runs |

ACT's numerics are unchanged; `selftest.py::test_act_refactor_is_equivalent` asserts that
`encode_observations` + a manual decode reproduces `ACT.forward` exactly.

---

## 7. Ablation arms

| arm | flags | question |
|---|---|---|
| A baseline | `--policy.type=act` | today's number |
| B shallow denoiser | `--policy.type=act_dit --policy.n_decoder_layers=1` | is 1 layer deep enough to denoise across noise levels? |
| C deep denoiser | `--policy.type=act_dit --policy.n_decoder_layers=4` (default) | — |
| D no cross-attention | `--policy.type=act_dit --policy.use_cross_attention=false` | what are the 902 spatial tokens worth? |

**B/C vs D is the number worth having.** A and B/C differ in generative target; B/C and D differ only
in the conditioning route, with parameter counts within 3 % of each other.

```bash
lerobot-train --policy.type=act_dit --dataset.repo_id=... \
  --policy.objective=flow_matching --policy.num_integration_steps=10
```

---

## 8. Known ceilings

- **Latency** (measured, RTX 4090, batch 1, 3 × 480×640, `n_dec=4`, `K=10`):
  encoder pass **12.3 ms**, one denoiser step **3.4 ms**, full chunk **71 ms**. A hand-written Euler
  loop over the same modules takes **46 ms**, so ~24 ms is host-side overhead in
  `FlowMatchingObjective._euler_integrate` — it calls `.item()` on the time grid twice per step,
  forcing a GPU sync. Fixing that means editing shared `multi_task_dit` code, so it is documented
  rather than silently changed. Until then, budget ~7 ms/step, not 3.4.
- **`num_integration_steps=10` / `num_inference_steps=10`** are literature defaults, not measurements
  on this data. `K` is the one knob that trades latency for sample fidelity — sweep it.
- **`optimizer_lr=1e-4`** is a deliberate bump off ACT's `1e-5`: a velocity/noise target is a harder
  regression than L1, and the DiT literature runs an order of magnitude higher. It is the first
  thing to sweep if arm C underperforms arm A.
- **Camera positional embeddings are identical per camera** (inherited ACT defect, step 3 above:
  `ACTSinusoidalPositionEmbedding2d` sees only the feature-map geometry, which is the same for all
  three). This damages exactly the pathway S1 is built to preserve. Orthogonal fix, worth doing
  before reading too much into the B/C vs D gap.
- **`n_action_steps` still defaults to the full 100-step open loop.** S1 changes nothing about
  execution horizon or temporal ensembling; those remain mutually exclusive as in ACT.
- **`use_vae=True` is refused** — the flow/diffusion objective already models the action
  distribution, so a CVAE latent would be a second, unused one. Use `act` if you want it.
- **Diffusion mode needs `diffusers`** (imported via `multi_task_dit`); flow matching does not.
