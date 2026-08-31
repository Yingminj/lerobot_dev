#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Configuration for ACT-DiT (scheme S1 of `policy/act-diffusion-integration-2026-08.md`).

Subclasses `ACTConfig` on purpose: every architecture field (backbone, dim_model,
chunk_size, ...) keeps its ACT meaning, and `make_pre_post_processors` dispatches on
`isinstance(policy_cfg, ACTConfig)`, so ACT-DiT gets exactly the ACT normalization
pipeline - which is what we want, since the action representation is unchanged.

Only the *generative target* differs from ACT: the decoder is fed noisy actions plus a
timestep instead of zeros, and regresses a flow-matching velocity (or diffusion noise)
instead of the action itself. The generative hyperparameters below are named to match
`multi_task_dit`'s `FlowMatchingObjective` / `DiffusionObjective`, which this policy reuses.
"""

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("act_dit")
@dataclass
class ACTDiTConfig(ACTConfig):
    """ACT with a DiT-style generative decoder.

    Args (on top of `ACTConfig`):
        objective: "flow_matching" (default, 4-10 inference steps) or "diffusion".
        timestep_embed_dim: Width of the sinusoidal timestep embedding feeding adaLN.
        state_in_adaln: Route `observation.state` into the adaLN conditioning vector on top
            of its encoder token. Default False, and that default is a measured one: with it
            True the decoder gets a full-bandwidth, multiplicative, per-layer road from the
            robot state, which is strictly cheaper to fit than cross-attention over ~900
            visual tokens. Within a few hundred steps the network shuts the observation
            encoder off (the last encoder layer's output LayerNorm gain shrinks toward zero,
            and with it every gradient that would reach the cameras), and the policy
            degenerates into a proprioception -> trajectory map that replays the nominal
            demonstration whatever the scene shows. See
            `paper/policy/experiment_report/act_dit/act_dit-encoder-collapse-2026-08.md`.
            With it False, adaLN carries the flow timestep only and the state reaches the
            decoder exactly the way plain ACT delivers it: as one encoder token.
            Checkpoints written before this field existed were trained with the state on
            adaLN, and their `config.json` has no `state_in_adaln` key, so they load against
            the new default and fail loudly on the adaLN `Linear` shape. Add
            `"state_in_adaln": true` to such a `config.json` to load one as it was trained.
        use_cross_attention: Ablation switch. True (default) keeps ACT's token-level
            cross-attention over the ~900 encoder tokens - the whole point of S1. False
            drops it and mean-pools the encoder output into the adaLN conditioning vector
            instead, degenerating to a DiT-Block-Policy-style model. The gap between the
            two is the "what are spatial tokens worth" measurement (ablation D).
        n_decoder_layers: Inherited, but the default is raised from ACT's 1 to 4 - one
            layer is likely too shallow to act as a denoiser across noise levels. Sweep
            {1, 2, 4} for ablations B/C.
        use_vae: Inherited, but defaults to False - the flow/diffusion objective now
            carries the distribution, so the CVAE latent (which ACT zeroes at inference
            anyway) is redundant.
        use_ema: Keep an exponential moving average of the denoiser weights and sample from
            it in eval mode. Off by default because it doubles the checkpoint and cannot be
            switched on mid-run (see `ema_decay`). The reference flow/diffusion policies all
            train this way - `configuration_patch_policy.py` and `modeling_vita.py` document
            the same gap - because the velocity/noise target is a high-variance regression:
            the same (observation, chunk) pair gets a different target on every step
            depending on the sampled `t` and noise, so the iterates never settle. ACT's L1
            target has no such per-sample randomness, which is why `act` never needed this.
        ema_decay: Averaging horizon, ~1/(1-decay) steps. The default 0.9999 is ~10k steps,
            sized for runs of 100k+; drop it to 0.999 for runs shorter than ~20k or the
            average lags the weights for most of training.
    """

    # ACT overrides: the decoder is now a denoiser, and the CVAE is redundant.
    n_decoder_layers: int = 4
    use_vae: bool = False

    # Generative target.
    objective: str = "flow_matching"  # "flow_matching" or "diffusion"
    timestep_embed_dim: int = 256
    use_cross_attention: bool = True
    state_in_adaln: bool = False

    # --- Flow matching (objective="flow_matching") ---
    sigma_min: float = 0.0
    num_integration_steps: int = 10
    integration_method: str = "euler"  # "euler" or "rk4"
    timestep_sampling_strategy: str = "beta"  # "uniform" or "beta"
    timestep_sampling_s: float = 0.999
    timestep_sampling_alpha: float = 1.5
    timestep_sampling_beta: float = 1.0

    # --- Diffusion (objective="diffusion") ---
    noise_scheduler_type: str = "DDIM"  # "DDPM" or "DDIM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"  # "epsilon" or "sample"
    clip_sample: bool = True
    clip_sample_range: float = 1.0
    num_inference_steps: int | None = 10

    # --- Weight EMA (both objectives) ---
    use_ema: bool = False
    ema_decay: float = 0.9999

    # Training preset. ACT's 1e-5 is tuned for an L1 target; a velocity/noise target is a
    # harder regression and the DiT literature runs an order of magnitude higher.
    optimizer_lr: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    def __post_init__(self):
        super().__post_init__()
        if self.objective not in ("flow_matching", "diffusion"):
            raise ValueError(
                f"`objective` must be 'flow_matching' or 'diffusion', got {self.objective!r}."
            )
        if self.objective == "flow_matching":
            if self.integration_method not in ("euler", "rk4"):
                raise ValueError(
                    f"`integration_method` must be 'euler' or 'rk4', got {self.integration_method!r}."
                )
            if self.timestep_sampling_strategy not in ("uniform", "beta"):
                raise ValueError("`timestep_sampling_strategy` must be 'uniform' or 'beta'.")
            if self.num_integration_steps <= 0:
                raise ValueError(
                    f"`num_integration_steps` must be positive, got {self.num_integration_steps}."
                )
        else:
            if self.noise_scheduler_type not in ("DDPM", "DDIM"):
                raise ValueError(
                    f"`noise_scheduler_type` must be 'DDPM' or 'DDIM', got {self.noise_scheduler_type!r}."
                )
            if self.prediction_type not in ("epsilon", "sample"):
                raise ValueError(
                    f"`prediction_type` must be 'epsilon' or 'sample', got {self.prediction_type!r}."
                )
        if self.use_ema and not 0.0 < self.ema_decay < 1.0:
            raise ValueError(f"`ema_decay` must be in (0, 1), got {self.ema_decay}.")
        if self.use_vae:
            raise ValueError(
                "`use_vae=True` is not supported by act_dit: the flow/diffusion objective already "
                "models the action distribution, so the CVAE latent would be a second, unused one. "
                "Use `act` if you want the CVAE."
            )
