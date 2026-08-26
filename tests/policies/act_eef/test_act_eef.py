# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

import pytest

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.act_eef.configuration_act_eef import (
    EEF_FEATURE_NAMES,
    ACTEEFConfig,
)
from lerobot.policies.act_eef.modeling_act_eef import ACTEEFPolicy
from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE


def _make_config(state_dim: int = 14, action_dim: int = 14) -> ACTEEFConfig:
    return ACTEEFConfig(
        device="cpu",
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(4,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        chunk_size=2,
        n_action_steps=2,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        latent_dim=8,
        pretrained_backbone_weights=None,
    )


def test_factory_resolves_act_eef() -> None:
    cfg = make_policy_config("act_eef", device="cpu")
    assert isinstance(cfg, ACTEEFConfig)
    assert get_policy_class("act_eef") is ACTEEFPolicy


def test_model_projections_follow_14d_eef_features() -> None:
    policy = ACTEEFPolicy(_make_config())
    assert policy.model.encoder_robot_state_input_proj.in_features == 14
    assert policy.model.vae_encoder_robot_state_input_proj.in_features == 14
    assert policy.model.vae_encoder_action_input_proj.in_features == 14
    assert policy.model.action_head.out_features == 14


@pytest.mark.parametrize(("state_dim", "action_dim"), [(16, 14), (14, 16)])
def test_rejects_non_eef_dimensions(state_dim: int, action_dim: int) -> None:
    with pytest.raises(ValueError, match="shape"):
        ACTEEFPolicy(_make_config(state_dim=state_dim, action_dim=action_dim))


def test_rejects_wrong_semantic_order() -> None:
    cfg = _make_config()
    wrong_names = list(EEF_FEATURE_NAMES)
    wrong_names[0], wrong_names[1] = wrong_names[1], wrong_names[0]
    with pytest.raises(ValueError, match="names in this order"):
        cfg.set_dataset_feature_metadata(
            {
                OBS_STATE: {"names": wrong_names},
                ACTION: {"names": list(EEF_FEATURE_NAMES)},
            }
        )
