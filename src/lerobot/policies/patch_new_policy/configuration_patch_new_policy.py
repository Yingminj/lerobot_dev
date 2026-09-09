#!/usr/bin/env python

# Portions of this file are derived from Patch Policy
# (https://github.com/gaoyuezhou/patch_policy, MIT License, Copyright (c) 2026 the Patch Policy
# authors) and from openpi / lerobot's pi0 and pi0.5 ports.
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
from lerobot.policies.patch_policy.configuration_patch_policy import PATCH_ENCODER_PRESETS

# The encoder zoo is shared verbatim with `patch_policy`; re-exported so a config file can name
# `patch_new_policy.PATCH_ENCODER_PRESETS` without importing the old policy.
__all__ = ["PATCH_ENCODER_PRESETS", "PatchNewPolicyConfig"]

# Time injection: (position, method) from `time-injection-taxonomy-2026-09.md` §2.
TIME_INJECTIONS = ("concat_mlp", "adaln", "additive", "memory_token", "none")

# State injection: the P-codes of `state-action-fusion-and-anchor-ratio-2026-09.md` §2.1.
STATE_INJECTIONS = ("none", "suffix_token", "obs_token", "action_concat", "adaln")

ACTION_HEADS = ("flow", "diffusion", "act")


@PreTrainedConfig.register_subclass("patch_new_policy")
@dataclass
class PatchNewPolicyConfig(PreTrainedConfig):
    """Patch Policy's block-causal patch memory, with pi0's state and time injection by default.

    Same visual front end as `patch_policy` (frozen ViT -> P patch tokens per camera per frame,
    block-causal mask: bidirectional inside a frame, causal across frames) and the same decoder
    alignment (position `t` predicts the action at observation step `min(t, n_obs_steps - 1)`).
    What is new is that *where the state enters* and *where the diffusion/flow time enters* are
    both configuration, so a 5 x 4 ablation grid runs without touching the model code.

    Defaults reproduce pi0: flow matching, state as a continuous token sitting next to the noisy
    action tokens (P1), time folded into the action tokens by concat + MLP (L1/M1).

    Time injection (`time_injection`), from `time-injection-taxonomy-2026-09.md` §2. All variants
    share one encoder -- sinusoid -> 2-layer MLP -- so an arm differs only in the injection:

    | value           | position | method            | injections | seen in    |
    |-----------------|----------|-------------------|-----------:|------------|
    | `concat_mlp`    | L1       | M1 concat + MLP   |          1 | pi0, SmolVLA, EO-1 |
    | `adaln`         | L3       | M2 adaLN-Zero     | 3 x depth  | pi0.5, DiT heads |
    | `additive`      | L4       | M4 plain add      |      depth | Evo-1      |
    | `memory_token`  | L2       | M5 memory token   |          0 | patch_policy, Diffusion Policy |
    | `none`          | --       | --                |          0 | control arm |

    `act` ignores this: an L1-regression head has no denoising time axis.

    The sinusoid base is the silent bug of §4: base 10000 is right for DDPM's integer
    `t in [0, 100)` and collapses to a constant for flow matching's `t in [0, 1]`. `time_sincos`
    defaults to `"auto"`, which picks openpi's `min_period`/`max_period` pair for the flow head and
    base 10000 for the diffusion head. Override only to run that failure mode on purpose.

    State injection (`state_injection`), from `state-action-fusion-and-anchor-ratio-2026-09.md`
    §2.1:

    | value           | code | where the state goes                                    |
    |-----------------|------|---------------------------------------------------------|
    | `none`          | P0   | not an input; the anchor must be rebuilt from pixels     |
    | `suffix_token`  | P1   | its own token beside the action tokens (pi0)             |
    | `obs_token`     | P2   | appended to each frame's patch block, i.e. the memory (patch_policy, SmolVLA) |
    | `action_concat` | P4   | concatenated onto every action token, then projected (GR00T)     |
    | `adaln`         | P5   | a conditioning vector modulating each block's norm (DiT heads)   |

    P3 (state quantised into a language prompt, pi0.5) has no counterpart here: there is no
    language model to reuse the prompt through.

    `use_relative_actions` is the orthogonal P6, "do not feed the anchor, subtract it from the
    target". It runs in the processor pipeline, before normalization, matching openpi's
    `DeltaActions` order; `relative_exclude_joints` keeps the gripper absolute, which all three of
    openpi, lerobot-pi0.5 and X-VLA do independently. Note that the normalization statistics are
    still those of the absolute action distribution -- consider `MEAN_STD` for `ACTION` when
    turning this on, since `MIN_MAX` maps a small delta range onto a ruler sized by absolute
    joint extremes.

    One deliberate generalisation of pi0: with `n_obs_steps > 1` the P1/P2 arms carry one state
    token *per observation step*, not one for the batch. A single latest-state token would be
    visible to decoder position 0, which is aligned with the oldest frame -- that leaks a future
    observation and quietly destroys the block causality that is Patch Policy's entire
    contribution. At `n_obs_steps == 1` the layout is exactly pi0's.
    """

    # Input / output structure.
    n_obs_steps: int = 2
    action_chunk_size: int = 50
    n_action_steps: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            # IDENTITY: the frozen ViTs apply their own ImageNet / processor normalization to
            # pixels in [0, 1]. Normalizing here would double-normalize.
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Head. `vqbet` is deliberately absent: its RVQ codebook is a second, separately trained model
    # whose reconstruction ceiling sits below the policies compared here.
    action_head: str = "flow"

    # --- ablation axis 1: time -------------------------------------------------------------
    time_injection: str = "concat_mlp"
    time_sincos: str = "auto"  # "auto" | "openpi" | "ddpm"
    time_min_period: float = 4e-3  # openpi variant only
    time_max_period: float = 4.0
    adaln_zero_init: bool = True  # AdaLN-Zero: identity block at init

    # --- ablation axis 2: state ------------------------------------------------------------
    state_injection: str = "suffix_token"
    use_relative_actions: bool = False  # P6, orthogonal to the above
    relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    # Action dimension names, used only to resolve `relative_exclude_joints`. Without them
    # every dimension is relativised -- including the gripper, whose delta is ~0 almost
    # everywhere, so its loss collapses and the policy never learns to open or close.
    action_feature_names: list[str] | None = None

    # Frozen visual encoder (shared with `patch_policy`).
    vision_encoder: str = "dino_patch"
    vision_encoder_checkpoint: str | None = None
    resize_shape: tuple[int, int] = (224, 224)
    freeze_vision_encoder: bool = True
    n_patches_override: int | None = None

    # Decoder trunk. One shape for all three heads, so an arm's only difference is the injection.
    dim_model: int = 256
    n_heads: int = 8
    dim_feedforward: int = 1024
    n_decoder_layers: int = 8
    dropout: float = 0.1
    p_drop_emb: float = 0.0

    # Diffusion head.
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0
    num_inference_steps: int | None = None

    # Flow-matching head. openpi's Beta(1.5, 1) time schedule and 10-step Euler integration.
    num_flow_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001

    # Training presets, from Patch Policy's `optim:` block.
    optimizer_lr: float = 5.5e-5
    optimizer_betas: tuple = (0.9, 0.999)
    optimizer_weight_decay: float = 2e-4

    def __post_init__(self):
        super().__post_init__()

        if self.action_head not in ACTION_HEADS:
            raise ValueError(f"`action_head` must be one of {ACTION_HEADS}. Got {self.action_head}.")
        if self.time_injection not in TIME_INJECTIONS:
            raise ValueError(
                f"`time_injection` must be one of {TIME_INJECTIONS}. Got {self.time_injection}."
            )
        if self.time_sincos not in ("auto", "openpi", "ddpm"):
            raise ValueError(f"`time_sincos` must be 'auto', 'openpi' or 'ddpm'. Got {self.time_sincos}.")
        if self.state_injection not in STATE_INJECTIONS:
            raise ValueError(
                f"`state_injection` must be one of {STATE_INJECTIONS}. Got {self.state_injection}."
            )
        if self.vision_encoder not in PATCH_ENCODER_PRESETS:
            raise ValueError(
                f"`vision_encoder` must be one of {sorted(PATCH_ENCODER_PRESETS)}. "
                f"Got {self.vision_encoder}."
            )
        if self.vision_encoder == "dynamo" and self.vision_encoder_checkpoint is None:
            raise ValueError("The 'dynamo' encoder preset requires `vision_encoder_checkpoint`.")
        if self.n_action_steps > self.action_chunk_size:
            raise ValueError(
                f"`n_action_steps` ({self.n_action_steps}) cannot exceed `action_chunk_size` "
                f"({self.action_chunk_size})."
            )
        if self.dim_model % 2 != 0:
            raise ValueError(
                f"`dim_model` must be even for the sinusoidal time embedding. Got {self.dim_model}."
            )

    @property
    def encoder_preset(self) -> dict:
        preset = dict(PATCH_ENCODER_PRESETS[self.vision_encoder])
        if self.vision_encoder_checkpoint is not None:
            preset["checkpoint"] = self.vision_encoder_checkpoint
        return preset

    @property
    def sincos_variant(self) -> str:
        """Resolve `time_sincos="auto"`: base 10000 needs integer timesteps, openpi's needs [0, 1]."""
        if self.time_sincos != "auto":
            return self.time_sincos
        return "ddpm" if self.action_head == "diffusion" else "openpi"

    @property
    def horizon(self) -> int:
        """Length of the predicted action sequence: one chunk per observation step, overlapped."""
        return self.n_obs_steps + self.action_chunk_size - 1

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        # Patch Policy trains at a constant learning rate; there is no scheduler in train_policy.py.
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

        needs_state = self.state_injection != "none" or self.use_relative_actions
        if needs_state and self.robot_state_feature is None:
            raise ValueError(
                f"`state_injection={self.state_injection!r}` / "
                f"`use_relative_actions={self.use_relative_actions}` need a robot state feature, "
                "but the dataset provides none."
            )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, self.action_chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
