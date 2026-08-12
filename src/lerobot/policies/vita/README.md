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

## Requirements

* `scipy` — for the default `flow_matcher_type="exact"` only (`pip install 'lerobot[scipy-dep]'`).
  `conditional`, `mean`, `improved_mean` and `consistency` need nothing extra.
* `diffusers` — for the cosine LR schedule at training time, as for the diffusion policy
  (`pip install 'lerobot[diffusion]'`).

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
