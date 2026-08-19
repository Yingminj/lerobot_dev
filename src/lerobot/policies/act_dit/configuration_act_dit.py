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
    """

    # ACT overrides: the decoder is now a denoiser, and the CVAE is redundant.
    n_decoder_layers: int = 4
    use_vae: bool = False

    # Generative target.
    objective: str = "flow_matching"  # "flow_matching" or "diffusion"
    timestep_embed_dim: int = 256
    use_cross_attention: bool = True

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
        if self.use_vae:
            raise ValueError(
                "`use_vae=True` is not supported by act_dit: the flow/diffusion objective already "
                "models the action distribution, so the CVAE latent would be a second, unused one. "
                "Use `act` if you want the CVAE."
            )
