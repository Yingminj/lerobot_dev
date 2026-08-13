#!/usr/bin/env python

# Portions of this file are derived from VITA
# (https://github.com/ucd-dare/VITA, MIT License, Copyright (c) 2025 the VITA authors).
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
from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamConfig, DiffuserSchedulerConfig

from .flow_matching import FLOW_MATCHER_CLASSES, MEAN_FLOW_MATCHERS


@PreTrainedConfig.register_subclass("vita")
@dataclass
class VitaConfig(PreTrainedConfig):
    """Configuration for `VitaPolicy`.

    Defaults reproduce `flare/configs/policy/vita.yaml` and `flare/configs/default_policy.yaml` from
    the reference implementation, except where noted below.

    Naming follows `DiffusionConfig` rather than VITA's own, so that the two policies consume the
    dataset identically and can be compared without a second dataloader configuration:

    | VITA               | here            |
    |--------------------|-----------------|
    | `obs_horizon: 1`   | `n_obs_steps`   |
    | `pred_horizon: 16` | `horizon`       |
    | `action_horizon: 8`| `n_action_steps`|

    Args:
        n_obs_steps: Observation steps fed to the policy. VITA uses 1; larger values concatenate the
            per-step visual/state features before projecting to the latent.
        horizon: Length of the predicted action chunk.
        n_action_steps: How many actions of each chunk are executed before re-planning.
        latent_dim: Width of the shared vision/action latent. The flow runs entirely in this space.
        flow_matcher_type: Probability path and training objective. See `flow_matching.py`.
            - `exact` (default, matches VITA): OT-CFM. Re-pairs observations and actions within the
              minibatch — read `ExactOptimalTransportConditionalFlowMatcher`'s docstring, and do not
              combine it with `decode_flow_latents=False`.
            - `conditional`: plain CFM, keeps the per-sample pairing.
            - `mean` / `improved_mean`: MeanFlow, 1-NFE generation. Requires
              `flow_net_type="simple_mean"`.
            - `consistency`: consistency flow matching, also few-step.
        num_sampling_steps: Euler steps at inference. 6 for the CFM family, 1 for MeanFlow.
        decode_flow_latents: Enable flow latent decoding (FLD). Runs the sampler inside the training
            graph and backpropagates the action reconstruction loss through every ODE step. This is
            what ties a sampled latent back to *this* sample's ground-truth action.
        consistency_weight: Weight of the flow latent consistency (FLC) term, an MSE between the
            sampled action latent and the encoder's action latent.
        flow_recon_weight: Weight of the FLD action reconstruction term.
        enc_recon_weight: Weight of the plain action autoencoder reconstruction term.
        enc_contrastive_weight: InfoNCE between visual and encoded action latents. 0 in the paper's
            default config; reported as an optional boost on top of FLD/FLC.
        flow_contrastive_weight: Same, but against the *sampled* action latent.
    """

    # Inputs / output structure.
    n_obs_steps: int = 1
    horizon: int = 64 
    n_action_steps: int = 32

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # horizon - n_action_steps - n_obs_steps + 1
    drop_n_last_frames: int = 32

    # Vision backbone. VITA uses a shared ImageNet ResNet-18 with frozen BatchNorm and global
    # average pooling (one 512-d vector per camera), not per-camera encoders with SpatialSoftmax.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    freeze_backbone_batchnorm: bool = True
    resize_shape: tuple[int, int] | None = (240, 320)
    crop_shape: tuple[int, int] | None = (224, 308)
    crop_is_random: bool = True

    # Latent space shared by vision and action.
    latent_dim: int = 512

    # Flow matcher.
    flow_matcher_type: str = "exact"
    flow_sigma: float = 0.0
    num_sampling_steps: int = 6

    # MeanFlow-only knobs (ignored by the other matchers).
    meanflow_flow_ratio: float = 0.5
    meanflow_time_dist_mu: float = -0.4
    meanflow_time_dist_sigma: float = 1.0
    meanflow_adaptive_loss_gamma: float = 0.5
    meanflow_aux_v_loss_weight: float = 1.0
    meanflow_dispersive_loss_weight: float = 0.0
    meanflow_dispersive_loss_tau: float = 1.0

    # Velocity network. Deliberately unconditioned: only the timestep enters, never the observation.
    flow_net_type: str = "simple"  # "simple" | "simple_mean"
    flow_hidden_dim: int = 512
    flow_num_layers: int = 4
    flow_mlp_ratio: float = 4.0
    flow_dropout: float = 0.0
    flow_time_embed_dim: int = 256

    # Action autoencoder.
    action_encoder_type: str = "cnn"  # "cnn" | "simple"
    action_decoder_type: str = "simple"
    action_enc_hidden_dim: int = 512
    action_dec_hidden_dim: int = 512
    action_ae_num_layers: int = 4
    action_ae_dropout: float = 0.0
    freeze_action_encoder: bool = False
    freeze_action_decoder: bool = False

    # Loss composition.
    decode_flow_latents: bool = True
    consistency_weight: float = 1.0
    flow_recon_weight: float = 0.5
    enc_recon_weight: float = 0.5
    recon_loss_type: str = "l1"
    enc_contrastive_weight: float = 0.0
    flow_contrastive_weight: float = 0.0
    contrastive_temperature: float = 0.07

    # Optimization.
    optimizer_lr: float = 1e-4
    optimizer_lr_backbone: float = 1e-5
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()

        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )

        if self.flow_matcher_type not in FLOW_MATCHER_CLASSES:
            raise ValueError(
                f"`flow_matcher_type` must be one of {sorted(FLOW_MATCHER_CLASSES)}. "
                f"Got {self.flow_matcher_type}."
            )

        supported_flow_nets = ["simple", "simple_mean"]
        if self.flow_net_type not in supported_flow_nets:
            raise ValueError(
                f"`flow_net_type` must be one of {supported_flow_nets}. Got {self.flow_net_type}."
            )

        # The MeanFlow identity needs a network exposing both the mean and instantaneous velocity.
        is_mean_flow = self.flow_matcher_type in MEAN_FLOW_MATCHERS
        if is_mean_flow and self.flow_net_type != "simple_mean":
            raise ValueError(
                f"`flow_matcher_type='{self.flow_matcher_type}'` requires "
                f"`flow_net_type='simple_mean'` (the network must return (u, v, features)). "
                f"Got '{self.flow_net_type}'."
            )
        if not is_mean_flow and self.flow_net_type == "simple_mean":
            raise ValueError(
                "`flow_net_type='simple_mean'` is only usable with `flow_matcher_type` in "
                f"{list(MEAN_FLOW_MATCHERS)}. Got '{self.flow_matcher_type}'."
            )

        if self.recon_loss_type not in ["l1", "l2"]:
            raise ValueError(f"`recon_loss_type` must be 'l1' or 'l2'. Got {self.recon_loss_type}.")

        supported_encoders = ["cnn", "simple"]
        if self.action_encoder_type not in supported_encoders:
            raise ValueError(
                f"`action_encoder_type` must be one of {supported_encoders}. "
                f"Got {self.action_encoder_type}."
            )
        if self.action_decoder_type != "simple":
            raise ValueError(
                f"`action_decoder_type` must be 'simple'. Got {self.action_decoder_type}."
            )

        if self.action_encoder_type == "simple" and self.latent_dim % self.horizon != 0:
            raise ValueError(
                "The 'simple' action encoder splits the latent across the chunk, so `latent_dim` must "
                f"be divisible by `horizon`. Got {self.latent_dim=} and {self.horizon=}."
            )

        if self.action_encoder_type == "cnn":
            # Each conv layer halves the chunk length; it must not vanish.
            if self.horizon < 2**self.action_ae_num_layers:
                raise ValueError(
                    "The 'cnn' action encoder halves the chunk length once per layer, so `horizon` "
                    f"must be at least 2**`action_ae_num_layers`. Got {self.horizon=} and "
                    f"{self.action_ae_num_layers=}."
                )

        if self.n_action_steps > self.horizon - self.n_obs_steps + 1:
            raise ValueError(
                "`n_action_steps` must satisfy `n_action_steps <= horizon - n_obs_steps + 1`. Got "
                f"{self.n_action_steps=}, {self.horizon=}, {self.n_obs_steps=}."
            )

        if self.resize_shape is not None and (
            len(self.resize_shape) != 2 or any(d <= 0 for d in self.resize_shape)
        ):
            raise ValueError(f"`resize_shape` must be a pair of positive integers. Got {self.resize_shape}.")
        if self.crop_shape is not None and (self.crop_shape[0] <= 0 or self.crop_shape[1] <= 0):
            raise ValueError(f"`crop_shape` must have positive dimensions. Got {self.crop_shape}.")

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        if len(self.image_features) == 0:
            raise ValueError(
                "VITA flows from a visual latent to an action latent, so at least one image input is "
                "required."
            )
        if self.robot_state_feature is None:
            raise ValueError("`observation.state` is required as an input.")

        if self.crop_shape is not None and self.resize_shape is None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` should fit within the image shapes. Got {self.crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for `{key}`."
                    )

        first_image_key, first_image_ft = next(iter(self.image_features.items()))
        for key, image_ft in self.image_features.items():
            if image_ft.shape != first_image_ft.shape:
                raise ValueError(
                    f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
