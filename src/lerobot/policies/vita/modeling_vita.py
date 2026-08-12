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
"""VITA: Vision-to-Action Flow Matching Policy (Gao et al., ICLR 2026).

Paper: https://huggingface.co/papers/2507.13231
Code:  https://github.com/ucd-dare/VITA

Every other flow-matching policy in this repository treats the observation as a *condition*: noise is
sampled, and the visual features are injected into the velocity network over and over (cross
attention, AdaLN, FiLM) at every denoising step. VITA's observation is that the source distribution
of a flow does not have to be noise. So it flows *from the visual latent itself* to an action latent:

    z_img = obs_encoder(resnet(images), state)      # the source of the probability path
    z_act = action_encoder(action_chunk)            # the target
    v     = flow_net(z_t, t)                        # no conditioning input, only the timestep

That removes the conditioning modules entirely, which is where the reported 1.5-2x inference speedup
comes from. The cost is that the source and target must live in comparable spaces, which is what the
action autoencoder is for, and that the flow-matching loss alone no longer pins a specific
observation to a specific action. Flow latent decoding (FLD) closes that gap: the sampler is run
*inside the training graph* and the action reconstruction loss is backpropagated through every ODE
step, so the sampled latent is tied to this sample's ground truth.

Read `flow_matching.py` before changing anything about the objective, in particular the note on
minibatch OT coupling in `ExactOptimalTransportConditionalFlowMatcher`.

Known deviations from the reference implementation, all deliberate:

* No EMA of the weights. Upstream trains with `use_ema: true, ema_power: 0.75`; LeRobot's training
  loop has no equivalent hook, so final numbers will differ somewhat.
* No transformer action encoder and no variational action autoencoder. Upstream's defaults are
  `encoder_type: cnn`, `use_variational: false`, so both are off the default path.
* Loss is not masked on padded actions, matching upstream (which has no such option).
"""

import math
from collections import deque

import einops
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from ..pretrained import PreTrainedPolicy
from ..utils import get_output_shape, populate_queues
from .configuration_vita import VitaConfig
from .flow_matching import make_flow_matcher


class VitaPolicy(PreTrainedPolicy):
    """Vision-to-Action Flow Matching Policy."""

    config_class = VitaConfig
    name = "vita"

    def __init__(self, config: VitaConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self._queues = None
        self.vita = VitaModel(config)

        self.reset()

    def get_optim_params(self) -> dict:
        # The pretrained backbone is fine-tuned at a lower rate than the rest, as upstream does.
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("vita.obs_encoder.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("vita.obs_encoder.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`."""
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
            OBS_IMAGES: deque(maxlen=self.config.n_obs_steps),
        }

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations.

        Supports two modes:
        - Online (queues populated via `select_action`): stacks observations from internal queues.
        - Offline (empty queues, e.g. a dataloader batch): uses the batch directly.
        """
        queues_populated = any(len(q) > 0 for q in self._queues.values())
        if queues_populated:
            batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        else:
            batch = dict(batch)
            for key in self.config.image_features:
                if batch[key].ndim == 4:
                    batch[key] = batch[key].unsqueeze(1)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        return self.vita.generate_actions(batch)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        `horizon` actions are generated at once, of which `n_action_steps` are executed before the
        policy is queried again.
        """
        # NOTE: for offline evaluation the action is in the batch, so pop it out.
        if ACTION in batch:
            batch.pop(ACTION)

        batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
        batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        # NOTE: It's important that this happens after stacking the images into a single key.
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
        for key in self.config.image_features:
            if self.config.n_obs_steps == 1 and batch[key].ndim == 4:
                batch[key] = batch[key].unsqueeze(1)
        batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        return self.vita.compute_loss(batch)


def compute_contrastive_loss(
    image_features: Tensor, action_features: Tensor, temperature: float = 0.07
) -> Tensor:
    """Symmetric InfoNCE between visual and action latents.

    Off by default (weight 0) in the reference config; reported there as an optional boost on top of
    flow latent decoding and consistency.
    """
    batch_size = image_features.shape[0]
    image_features = F.normalize(image_features, dim=1)
    action_features = F.normalize(action_features, dim=1)

    logits = torch.matmul(image_features, action_features.T) / temperature
    labels = torch.arange(batch_size, device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


class VitaModel(nn.Module):
    """The network stack: observation encoder, action autoencoder, and unconditioned velocity net."""

    def __init__(self, config: VitaConfig):
        super().__init__()
        self.config = config

        self.obs_encoder = VitaObservationEncoder(config)
        self.action_encoder, self.action_decoder = make_action_autoencoder(config)

        flow_net_cls = SimpleMeanFlowNet if config.flow_net_type == "simple_mean" else SimpleFlowNet
        self.flow_net = flow_net_cls(
            input_dim=config.latent_dim,
            hidden_dim=config.flow_hidden_dim,
            output_dim=config.latent_dim,
            num_layers=config.flow_num_layers,
            mlp_ratio=config.flow_mlp_ratio,
            dropout=config.flow_dropout,
            time_embed_dim=config.flow_time_embed_dim,
        )

        self.flow_matcher = make_flow_matcher(
            name=config.flow_matcher_type,
            sigma=config.flow_sigma,
            num_sampling_steps=config.num_sampling_steps,
            flow_ratio=config.meanflow_flow_ratio,
            time_dist_mu=config.meanflow_time_dist_mu,
            time_dist_sigma=config.meanflow_time_dist_sigma,
            adaptive_loss_gamma=config.meanflow_adaptive_loss_gamma,
            aux_v_loss_weight=config.meanflow_aux_v_loss_weight,
            dispersive_loss_weight=config.meanflow_dispersive_loss_weight,
            dispersive_loss_tau=config.meanflow_dispersive_loss_tau,
        )

        self.recon_loss_fn = F.l1_loss if config.recon_loss_type == "l1" else F.mse_loss

        if config.freeze_action_encoder:
            for p in self.action_encoder.parameters():
                p.requires_grad_(False)
        if config.freeze_action_decoder:
            for p in self.action_decoder.parameters():
                p.requires_grad_(False)

    def _encode_observation(self, batch: dict[str, Tensor]) -> Tensor:
        """`batch` -> `(B, latent_dim)` visual latent, the source of the flow."""
        return self.obs_encoder(batch)

    def _sample_action_latents(self, obs_latents: Tensor) -> Tensor:
        return self.flow_matcher.sample(
            self.flow_net,
            shape=(obs_latents.shape[0], self.config.latent_dim),
            device=obs_latents.device,
            num_steps=self.config.num_sampling_steps,
            start=obs_latents,  # the visual latent is the source of the flow
        )

    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Expects:
        {
            "observation.state": (B, n_obs_steps, state_dim),
            "observation.images": (B, n_obs_steps, num_cameras, C, H, W),
        }
        Returns `(B, n_action_steps, action_dim)`.
        """
        n_obs_steps = batch[OBS_STATE].shape[1]
        assert n_obs_steps == self.config.n_obs_steps

        obs_latents = self._encode_observation(batch)
        action_latents = self._sample_action_latents(obs_latents)
        actions = self.action_decoder(action_latents)

        # Extract `n_action_steps` worth of actions, starting at the current observation.
        start = n_obs_steps - 1
        return actions[:, start : start + self.config.n_action_steps]

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """
        Expects, in addition to the observation keys above, `"action"` of shape `(B, horizon, A)`.
        """
        assert set(batch).issuperset({OBS_STATE, OBS_IMAGES, ACTION})
        assert batch[OBS_STATE].shape[1] == self.config.n_obs_steps
        assert batch[ACTION].shape[1] == self.config.horizon

        config = self.config
        gt_actions = batch[ACTION]

        obs_latents = self._encode_observation(batch)

        with torch.no_grad() if config.freeze_action_encoder else torch.enable_grad():
            action_latents = self.action_encoder(gt_actions)

        # The whole point: flow from the visual latent to the action latent.
        loss, metrics = self.flow_matcher.compute_loss(
            self.flow_net, target=action_latents, start=obs_latents
        )

        if config.enc_contrastive_weight > 0:
            contrastive_loss = compute_contrastive_loss(
                obs_latents.flatten(1), action_latents.flatten(1), config.contrastive_temperature
            )
            loss = loss + config.enc_contrastive_weight * contrastive_loss
            metrics["enc_contrastive_loss"] = contrastive_loss.item()

        if config.freeze_action_encoder and config.freeze_action_decoder:
            metrics["loss"] = loss.item()
            return loss, metrics

        # Flow latent decoding: sample inside the training graph so the reconstruction loss
        # backpropagates through every ODE step.
        if (
            config.decode_flow_latents
            and not config.freeze_action_encoder
            and not config.freeze_action_decoder
        ):
            action_latents_pred = self._sample_action_latents(obs_latents)

            if config.consistency_weight > 0:
                consistency_loss = F.mse_loss(action_latents_pred, action_latents)
                loss = loss + config.consistency_weight * consistency_loss
                metrics["consistency_loss"] = consistency_loss.item()

            if config.flow_contrastive_weight > 0:
                contrastive_loss = compute_contrastive_loss(
                    obs_latents.flatten(1),
                    action_latents_pred.flatten(1),
                    config.contrastive_temperature,
                )
                loss = loss + config.flow_contrastive_weight * contrastive_loss
                metrics["flow_contrastive_loss"] = contrastive_loss.item()

            if config.flow_recon_weight > 0 and not config.freeze_action_decoder:
                actions_recon = self.action_decoder(action_latents_pred)
                flow_recon_loss = self.recon_loss_fn(actions_recon, gt_actions)
                loss = loss + config.flow_recon_weight * flow_recon_loss
                metrics["flow_action_recon_loss"] = flow_recon_loss.item()

        if config.enc_recon_weight > 0 and not config.freeze_action_decoder:
            actions_recon = self.action_decoder(action_latents)
            enc_recon_loss = self.recon_loss_fn(actions_recon, gt_actions)
            loss = loss + config.enc_recon_weight * enc_recon_loss
            metrics["enc_action_recon_loss"] = enc_recon_loss.item()

        metrics["loss"] = loss.item()
        return loss, metrics


class VitaObservationEncoder(nn.Module):
    """Images and state -> the `latent_dim` source latent.

    One ResNet is shared by every camera, features are globally average-pooled to 512 per view, then
    concatenated with the state across observation steps and projected by a single linear layer.
    """

    def __init__(self, config: VitaConfig):
        super().__init__()
        self.config = config

        if config.resize_shape is not None:
            self.resize = torchvision.transforms.Resize(config.resize_shape)
        else:
            self.resize = None

        if config.crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(config.crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        # VITA freezes BatchNorm in the backbone rather than replacing it with GroupNorm: the running
        # statistics from ImageNet are kept, which matters at the small batch sizes used here.
        backbone_kwargs = {"weights": config.pretrained_backbone_weights}
        if config.freeze_backbone_batchnorm:
            backbone_kwargs["norm_layer"] = FrozenBatchNorm2d
        backbone_model = getattr(torchvision.models, config.vision_backbone)(**backbone_kwargs)
        # Drop the final avgpool and fc; keep the layer4 feature map.
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        images_shape = next(iter(config.image_features.values())).shape
        if config.crop_shape is not None:
            dummy_shape_h_w = config.crop_shape
        elif config.resize_shape is not None:
            dummy_shape_h_w = config.resize_shape
        else:
            dummy_shape_h_w = images_shape[1:]
        dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
        self.feature_dim = get_output_shape(self.backbone, dummy_shape)[1]

        num_cameras = len(config.image_features)
        state_dim = config.robot_state_feature.shape[0]
        self.obs_dim = config.n_obs_steps * (num_cameras * self.feature_dim + state_dim)
        self.projection = nn.Linear(self.obs_dim, config.latent_dim)

    def _encode_images(self, images: Tensor) -> Tensor:
        """`(N, C, H, W)` in [0, 1] -> `(N, feature_dim)`."""
        if self.resize is not None:
            images = self.resize(images)
        if self.do_crop:
            images = self.maybe_random_crop(images) if self.training else self.center_crop(images)
        return torch.flatten(self.pool(self.backbone(images)), start_dim=1)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]

        img_features = self._encode_images(
            einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")
        )
        # Absorb the camera index into the feature dim, i.e. concatenate the per-camera features.
        img_features = einops.rearrange(
            img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
        )

        features = torch.cat([batch[OBS_STATE], img_features], dim=-1).flatten(start_dim=1)
        return self.projection(features)


class SinusoidalPosEmb(nn.Module):
    """1D sinusoidal embedding of the continuous flow time."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device, dtype=x.dtype) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Mlp(nn.Module):
    """The two-layer MLP block used throughout VITA (equivalent to `timm.layers.Mlp`)."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class FlowNetLayer(nn.Module):
    """Residual MLP block modulated by the timestep only — never by the observation."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=dropout,
        )
        self.time_modulator = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim))

        nn.init.constant_(self.time_modulator[-1].weight, 0)
        nn.init.constant_(self.time_modulator[-1].bias, 0)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        gamma, scale, shift = self.time_modulator(t).view(x.shape[0], 3, self.dim).unbind(1)
        x_norm = self.norm(x) * (scale + 1) + shift
        return x + self.mlp(x_norm) * gamma


class SimpleFlowNet(nn.Module):
    """Velocity field `v(x_t, t)` for the CFM-family matchers.

    Note what is *absent*: there is no conditioning argument. The observation entered the ODE through
    the initial condition, not through the network.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        time_embed_dim: int = 256,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.Mish(),
            nn.Linear(time_embed_dim * 4, hidden_dim),
        )
        self.layers = nn.ModuleList(
            [
                FlowNetLayer(dim=hidden_dim, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # NOTE: upstream applies this after constructing the blocks, which overwrites the zero-init of
        # each FlowNetLayer's `time_modulator` output layer, i.e. the adaLN-zero initialisation is not
        # actually in effect. Kept as-is so results match the reference; re-zeroing those layers here
        # is a one-liner if you want to test whether it helps.
        self.apply(_basic_init)

        nn.init.normal_(self.time_embed[1].weight, std=0.02)
        nn.init.normal_(self.time_embed[3].weight, std=0.02)

    def forward(self, x: Tensor, t: Tensor, **kwargs) -> Tensor:
        x = self.input_proj(x)
        t = self.time_embed(t)
        for block in self.layers:
            x = block(x, t)
        return self.out_proj(self.norm(x))


class SimpleMeanFlowNet(nn.Module):
    """Average-velocity field `u(x_t, r, t)` for the MeanFlow matchers.

    Returns `(u, v, internal_features)`: the average velocity over the interval, an auxiliary
    instantaneous velocity used by Improved MeanFlow, and the per-layer hidden states that the
    dispersive loss operates on. Both output heads are zero-initialised.

    NOTE: plain MeanFlow (`flow_matcher_type="mean"`) never reads `v`, so `v_output_layer` receives
    no gradient. The head is kept anyway so that a checkpoint can be switched between `mean` and
    `improved_mean` without changing shape. It is harmless under Adam, but DDP needs
    `find_unused_parameters=True`.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        time_embed_dim: int = 256,  # unused; kept for a common constructor signature
    ):
        super().__init__()
        self.t_embed = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.h_embed = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        inner_dim = int(hidden_dim * mlp_ratio)
        self.hidden_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, inner_dim),
                    nn.SiLU(),
                    nn.Dropout(p=dropout),
                    nn.Linear(inner_dim, hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.u_output_layer = nn.Linear(hidden_dim, output_dim)
        self.v_output_layer = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        for layer in (self.u_output_layer, self.v_output_layer):
            nn.init.constant_(layer.weight, 0)
            nn.init.constant_(layer.bias, 0)

    def forward(self, x: Tensor, timestep: Tensor, h: Tensor, **kwargs):
        time_cond = self.t_embed(timestep.unsqueeze(-1)) + self.h_embed(h.unsqueeze(-1))
        x = self.input_layer(x)

        internal_features = []
        for layer in self.hidden_layers:
            x = layer(x + time_cond) + x
            internal_features.append(x)

        return self.u_output_layer(x), self.v_output_layer(x), torch.stack(internal_features, dim=0)


class CNNActionEncoder(nn.Module):
    """Action chunk `(B, T, A)` -> latent `(B, latent_dim)` via strided 1D convolutions over time."""

    def __init__(
        self,
        horizon: int,
        action_dim: int,
        latent_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
    ):
        super().__init__()
        layers = []
        current_dim = action_dim
        for _ in range(num_layers):
            layers.append(nn.Conv1d(current_dim, hidden_dim, kernel_size=5, stride=2, padding=2))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)

        # Derive the flattened width from a dry run rather than assuming `horizon // 2**num_layers`,
        # which only holds for powers of two.
        with torch.no_grad():
            conv_output_dim = self.encoder(torch.zeros(1, action_dim, horizon)).flatten(1).shape[1]
        self.latent_proj = nn.Linear(conv_output_dim, latent_dim)

        self.apply(_orthogonal_init)

    def forward(self, actions: Tensor) -> Tensor:
        x = self.encoder(actions.transpose(1, 2))  # (B, hidden, T')
        return self.latent_proj(x.flatten(start_dim=1))


class SimpleActionEncoder(nn.Module):
    """Per-timestep MLP encoder: each of the `horizon` steps owns `latent_dim // horizon` channels."""

    def __init__(self, latent_dim: int, horizon: int, action_dim: int, num_layers: int = 4):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.step_latent_dim = latent_dim // horizon

        self.input_proj = nn.Linear(action_dim, self.step_latent_dim)
        self.layers = nn.ModuleList(
            [
                Mlp(
                    in_features=self.step_latent_dim,
                    hidden_features=4 * self.step_latent_dim,
                    out_features=self.step_latent_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.apply(_xavier_init)

    def forward(self, actions: Tensor) -> Tensor:
        x = self.input_proj(actions)  # (B, T, step_latent_dim)
        for layer in self.layers:
            x = layer(x)
        return x.flatten(start_dim=1)


class SimpleActionDecoder(nn.Module):
    """Latent `(B, latent_dim)` -> action chunk `(B, horizon, action_dim)`."""

    def __init__(
        self,
        dec_hidden_dim: int,
        latent_dim: int,
        horizon: int,
        action_dim: int,
        num_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim

        self.input_proj = nn.Linear(latent_dim, dec_hidden_dim)
        self.layers = nn.ModuleList(
            [
                Mlp(
                    in_features=dec_hidden_dim,
                    hidden_features=dec_hidden_dim,
                    out_features=dec_hidden_dim,
                    drop=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_proj = nn.Linear(dec_hidden_dim, horizon * action_dim)

        self.apply(_xavier_init)

    def forward(self, z: Tensor) -> Tensor:
        x = self.input_proj(z)
        for layer in self.layers:
            x = layer(x)
        return self.output_proj(x).view(-1, self.horizon, self.action_dim)


def _xavier_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def _orthogonal_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear | nn.Conv1d):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def make_action_autoencoder(config: VitaConfig) -> tuple[nn.Module, nn.Module]:
    """Build the action encoder/decoder pair that maps chunks into the shared latent space."""
    action_dim = config.action_feature.shape[0]

    if config.action_encoder_type == "cnn":
        encoder = CNNActionEncoder(
            horizon=config.horizon,
            action_dim=action_dim,
            latent_dim=config.latent_dim,
            hidden_dim=config.action_enc_hidden_dim,
            num_layers=config.action_ae_num_layers,
        )
    else:
        encoder = SimpleActionEncoder(
            latent_dim=config.latent_dim,
            horizon=config.horizon,
            action_dim=action_dim,
            num_layers=config.action_ae_num_layers,
        )

    decoder = SimpleActionDecoder(
        dec_hidden_dim=config.action_dec_hidden_dim,
        latent_dim=config.latent_dim,
        horizon=config.horizon,
        action_dim=action_dim,
        num_layers=config.action_ae_num_layers,
        dropout=config.action_ae_dropout,
    )
    return encoder, decoder
