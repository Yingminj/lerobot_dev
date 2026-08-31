#!/usr/bin/env python

# Portions of this file are derived from Patch Policy
# (https://github.com/gaoyuezhou/patch_policy, MIT License, Copyright (c) 2026 the Patch Policy authors),
# and from VQ-BeT / miniBET, which Patch Policy itself builds on.
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
from lerobot.optim import AdamWConfig

PATCH_ENCODER_PRESETS: dict[str, dict] = {
    # Mirrors `configs/encoder/*.yaml` in the reference repo, one entry per file.
    # `n_patches` records what the reference declared; the real value is measured from the
    # encoder with a dry run in `PatchPolicyModel.__init__`.
    "dino_patch": {
        "encoder_type": "dinov2",
        "name": "dinov2_vits14",
        "feature_key": "x_norm_patchtokens",
        "postprocess": None,
        "output_dim": 384,
        "n_patches": 256,
    },
    "dino_cls": {
        "encoder_type": "dinov2",
        "name": "dinov2_vits14",
        "feature_key": "x_norm_clstoken",
        "postprocess": None,
        "output_dim": 384,
        "n_patches": 1,
    },
    "dino_patch_avg_pool": {
        "encoder_type": "dinov2",
        "name": "dinov2_vits14",
        "feature_key": "x_norm_patchtokens",
        "postprocess": "avg_pool",
        "output_dim": 384,
        "n_patches": 1,
    },
    "dinov3_patch": {
        "encoder_type": "dinov3",
        "name": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
        "feature_key": "x_norm_patchtokens",
        "postprocess": None,
        "output_dim": 384,
        "n_patches": 196,
    },
    "dinov3_cls": {
        "encoder_type": "dinov3",
        "name": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
        "feature_key": "x_norm_clstoken",
        "postprocess": None,
        "output_dim": 384,
        "n_patches": 1,
    },
    "dinov3_patch_avg_pool": {
        "encoder_type": "dinov3",
        "name": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
        "feature_key": "x_norm_patchtokens",
        "postprocess": "avg_pool",
        "output_dim": 384,
        "n_patches": 1,
    },
    "webssl_patch": {
        "encoder_type": "webssl",
        "name": "facebook/webssl-dino300m-full2b-224",
        "feature_key": "x_norm_patchtokens",
        "postprocess": None,
        "output_dim": 1024,
        "n_patches": 256,
    },
    "webssl_cls": {
        "encoder_type": "webssl",
        "name": "facebook/webssl-dino300m-full2b-224",
        "feature_key": "x_norm_clstoken",
        "postprocess": None,
        "output_dim": 1024,
        "n_patches": 1,
    },
    "webssl_patch_avg_pool": {
        "encoder_type": "webssl",
        "name": "facebook/webssl-dino300m-full2b-224",
        "feature_key": "x_norm_patchtokens",
        "postprocess": "avg_pool",
        "output_dim": 1024,
        "n_patches": 1,
    },
    # Table 7's weakest encoder, ported for the ablation rather than as a default.
    "siglip2_patch": {
        "encoder_type": "siglip2",
        "name": "google/siglip2-base-patch16-224",
        "feature_key": "x_norm_patchtokens",
        "postprocess": None,
        "output_dim": 768,
        "n_patches": 196,
    },
    "siglip2_patch_avg_pool": {
        "encoder_type": "siglip2",
        "name": "google/siglip2-base-patch16-224",
        "feature_key": "x_norm_patchtokens",
        "postprocess": "avg_pool",
        "output_dim": 768,
        "n_patches": 1,
    },
    "vjepa2_patch": {
        "encoder_type": "vjepa2",
        "name": "facebook/vjepa2-vitl-fpc64-256",
        "feature_key": "x_norm_patchtokens",
        "postprocess": None,
        "output_dim": 1024,
        "n_patches": 256,
    },
    "vjepa2_patch_avg_pool": {
        "encoder_type": "vjepa2",
        "name": "facebook/vjepa2-vitl-fpc64-256",
        "feature_key": "x_norm_patchtokens",
        "postprocess": "avg_pool",
        "output_dim": 1024,
        "n_patches": 1,
    },
    # Global-pooled baselines: `P == 1`, i.e. the representation Patch Policy argues against.
    "resnet18_imagenet": {
        "encoder_type": "resnet18",
        "name": "resnet18",
        "feature_key": "x_norm_clstoken",
        "postprocess": None,
        "output_dim": 512,
        "n_patches": 1,
        "pretrained": True,
    },
    "resnet18_random": {
        "encoder_type": "resnet18",
        "name": "resnet18",
        "feature_key": "x_norm_clstoken",
        "postprocess": None,
        "output_dim": 512,
        "n_patches": 1,
        "pretrained": False,
    },
    # DynaMo checkpoints (`configs/encoder/*_dynamo.yaml`). Set `vision_encoder_checkpoint`
    # to the .pt file; the reference ships one per environment.
    "dynamo": {
        "encoder_type": "from_ckpt",
        "name": "dynamo",
        "feature_key": "x_norm_clstoken",
        "postprocess": None,
        "output_dim": 512,
        "n_patches": 1,
    },
}


@PreTrainedConfig.register_subclass("patch_policy")
@dataclass
class PatchPolicyConfig(PreTrainedConfig):
    """Configuration for `PatchPolicy`.

    Reproduces https://github.com/gaoyuezhou/patch_policy ("Patch Policy: Efficient Embodied Control
    via Dense Visual Representations", arXiv:2607.18236).

    The reference is driven by Hydra with one YAML per environment. Field names here follow
    lerobot conventions; the mapping to the reference's names is:

    | reference (`configs/train_*.yaml`) | here                |
    |------------------------------------|---------------------|
    | `window_size`                       | `n_obs_steps`       |
    | `action_window_size`                | `action_chunk_size` |
    | `model.n_layer/n_head/n_embd`       | `gpt_n_layer` / `gpt_n_head` / `gpt_hidden_dim` |
    | `encoder.output_dim`                | measured from the encoder, not configured |
    | `encoder.n_patches`                 | measured from the encoder, not configured |
    | `pred_horizon`                      | `horizon` (derived)  |
    | `n_action_steps`                    | `n_action_steps`    |

    Per-environment presets from the reference (see README.md for the full table):

    | env         | n_obs_steps | action_chunk_size | layers/heads/dim | lr     | weight decay |
    |-------------|-------------|-------------------|------------------|--------|--------------|
    | Push-T      | 5           | 5                 | 8 / 8 / 512      | 5.5e-5 | 2e-4         |
    | LIBERO Goal | 10          | 1                 | 6 / 6 / 120      | 5.5e-5 | 2e-4         |
    | BlockPush   | 3           | 1                 | 8 / 8 / 512      | 1e-4   | 0.0          |
    | Cube        | 5           | 5                 | 8 / 8 / 512      | 5.5e-5 | 2e-4         |

    Deviations from the reference, all deliberate:
      - Goal conditioning is not implemented. The reference concatenates a goal-image embedding onto
        every patch token's feature dim (`goal_dim > 0`, used only by LIBERO Goal); lerobot has no
        goal-image convention in its dataset pipeline. Push-T / BlockPush / Cube all run `goal_dim: 0`.
      - Action scaling (`act_scale`, 500 for Push-T/Cube) is dropped: lerobot's processor pipeline
        normalizes actions to [-1, 1] before the policy sees them.
      - The reference fits the VQ-VAE in a separate loop over the whole action set before BeT
        training. Here the lerobot VQ-BeT convention is used instead: the first
        `n_vqvae_training_steps` calls to `forward` train only the VQ-VAE.
      - No EMA on the diffusion head. The reference samples from an EMA copy of the denoiser;
        lerobot's training loop has no post-optimizer-step hook to update one.

    Args:
        action_head: Which head consumes the patch tokens.
            - `"vqbet"`: the reference's primary head. Patch tokens go through the block-causal GPT
              trunk; the last token of each frame's block is read out and mapped to RVQ code + offset.
            - `"diffusion"`: the reference's `TransformerForDiffusion`. Patch tokens are the
              cross-attention memory of a transformer denoiser; block-causality lives in the
              `memory_mask`, not in a self-attention trunk.
            - `"act"`: no counterpart in the reference. Same decoder shape as `"diffusion"` but the
              queries are learned action-position embeddings and the loss is L1, i.e. ACT's head
              reading the same block-causally masked patch memory.
        vision_encoder: Key into `PATCH_ENCODER_PRESETS`, one entry per `configs/encoder/*.yaml`.
        vision_encoder_checkpoint: Path to a `.pt` file, for the `"dynamo"` preset only.
        resize_shape: Images are resized to this before the frozen encoder. 224x224 in every
            reference config; 256x256 for V-JEPA 2.
        freeze_vision_encoder: The paper never fine-tunes the backbone. Set False at your own risk.
        use_robot_state: Off by default, matching the reference, whose observation tokens are purely
            visual — `obs_dim` is `encoder.output_dim` with no proprioception anywhere. When on, one
            projected state token is appended to each frame's block; the block-causal mask needs no
            change because the token is intra-frame. Note that this also moves the VQ-BeT readout
            (the last token of the block) from the last patch onto the state token.
        n_patches_override: Skip the dry run and assert this many patch tokens per camera. Only
            needed if instantiating the encoder at config time is impossible.
    """

    # Inputs / output structure. Defaults are the Push-T preset (`configs/train_pusht.yaml`).
    n_obs_steps: int = 5
    action_chunk_size: int = 50
    n_action_steps: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            # IDENTITY: the frozen ViTs apply their own ImageNet / processor normalization to
            # pixels in [0, 1], exactly as in the reference encoders.
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Which head consumes the patch tokens.
    action_head: str = "act"  # "vqbet" | "diffusion" | "act"

    # Frozen visual encoder.
    vision_encoder: str = "dino_patch"
    vision_encoder_checkpoint: str | None = None
    resize_shape: tuple[int, int] = (224, 224)
    freeze_vision_encoder: bool = True
    n_patches_override: int | None = None

    # Observation tokens.
    use_robot_state: bool = False

    # Block-causal GPT trunk (`action_head="vqbet"` only). Reference: `models/vq_behavior_transformer/gpt.py`.
    gpt_block_size: int | None = None  # None -> n_obs_steps (the trunk only sees observation frames)
    gpt_input_dim: int | None = None  # None -> the encoder's feature dim, measured at build time
    gpt_n_layer: int = 8
    gpt_n_head: int = 8
    gpt_hidden_dim: int = 512
    gpt_output_dim: int = 512
    dropout: float = 0.1

    # VQ-BeT head. Reference: `models/vq_behavior_transformer/{bet,vqvae}.py`.
    n_vqvae_training_steps: int = 5000
    vqvae_n_embed: int = 16
    vqvae_embedding_dim: int = 512  # reference `vqvae_latent_dim: 512`
    vqvae_enc_hidden_dim: int = 128
    offset_loss_weight: float = 10.0  # reference `offset_loss_multiplier`, 10 for Push-T / Cube
    primary_code_loss_weight: float = 5.0
    secondary_code_loss_weight: float = 0.5
    bet_softmax_temperature: float = 0.1
    sequentially_select: bool = False

    # Diffusion head. Reference: `models/diffusion_policy/diffusion_policy.py`.
    diffusion_n_layer: int = 8
    diffusion_n_head: int = 4
    diffusion_hidden_dim: int = 256
    diffusion_p_drop_emb: float = 0.0
    diffusion_p_drop_attn: float = 0.1
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0
    num_inference_steps: int | None = None

    # ACT head. Names match `ACTConfig` so lerobot's `ACTDecoder` can be constructed from this config.
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    n_decoder_layers: int = 1
    pre_norm: bool = False
    feedforward_activation: str = "relu"

    # Training presets. Reference `optim:` block, Push-T / LIBERO Goal / Cube values.
    optimizer_lr: float = 5.5e-5
    optimizer_betas: tuple = (0.9, 0.999)
    optimizer_weight_decay: float = 2e-4
    optimizer_vqvae_lr: float = 1e-3  # reference `_vqvae_optim`, Adam(lr=1e-3, weight_decay=1e-4)
    optimizer_vqvae_weight_decay: float = 1e-4

    def __post_init__(self):
        super().__post_init__()

        if self.action_head not in ("vqbet", "diffusion", "act"):
            raise ValueError(
                f"`action_head` must be one of 'vqbet', 'diffusion', 'act'. Got {self.action_head}."
            )
        if self.vision_encoder not in PATCH_ENCODER_PRESETS:
            raise ValueError(
                f"`vision_encoder` must be one of {sorted(PATCH_ENCODER_PRESETS)}. Got {self.vision_encoder}."
            )
        if self.vision_encoder == "dynamo" and self.vision_encoder_checkpoint is None:
            raise ValueError("The 'dynamo' encoder preset requires `vision_encoder_checkpoint`.")
        if self.n_action_steps > self.action_chunk_size:
            raise ValueError(
                f"`n_action_steps` ({self.n_action_steps}) cannot exceed `action_chunk_size` "
                f"({self.action_chunk_size})."
            )
        if self.gpt_block_size is None:
            # `bet.py` uses obs_window_size + act_window_size because its trunk is fed action
            # tokens too. Here the trunk only ever sees `n_obs_steps` frames, and every extra
            # slot costs a (block_size*n_patches)^2 mask and n_patches*hidden_dim unused
            # position embeddings -- 1.8 GB and 21M dead parameters at 3 cameras / 256 patches.
            self.gpt_block_size = self.n_obs_steps
        if self.gpt_block_size < self.n_obs_steps:
            raise ValueError(
                f"`gpt_block_size` ({self.gpt_block_size}) must be at least `n_obs_steps` "
                f"({self.n_obs_steps})."
            )

    @property
    def encoder_preset(self) -> dict:
        preset = dict(PATCH_ENCODER_PRESETS[self.vision_encoder])
        if self.vision_encoder_checkpoint is not None:
            preset["checkpoint"] = self.vision_encoder_checkpoint
        return preset

    @property
    def horizon(self) -> int:
        """Action-sequence length predicted by the diffusion / ACT heads.

        Reference: `pred_horizon: ${eval:'${action_window_size} + ${window_size} - 1'}`. Decoder
        position i lines up with observation step i, which is what makes the block-causal memory
        mask meaningful.
        """
        return self.n_obs_steps + self.action_chunk_size - 1

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        # The reference trains at a constant learning rate; there is no scheduler in train_policy.py.
        return None

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("Patch Policy requires at least one image among the inputs.")

        first_image_key, first_image_ft = next(iter(self.image_features.items()))
        for key, image_ft in self.image_features.items():
            if image_ft.shape != first_image_ft.shape:
                raise ValueError(
                    f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                )

        if self.use_robot_state and self.robot_state_feature is None:
            raise ValueError("`use_robot_state=True` but the dataset provides no robot state feature.")

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        # n_obs_steps + action_chunk_size - 1 actions, so that `_unpack_actions` can build one
        # action chunk per observation step (reference `BehaviorTransformer._unpack_actions`).
        return list(range(1 - self.n_obs_steps, self.action_chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
