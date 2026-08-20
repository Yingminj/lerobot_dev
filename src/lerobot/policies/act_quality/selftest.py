#!/usr/bin/env python

"""CPU smoke tests for registration, sampler filtering, CVAE masking, and loss."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE

from .configuration_act_quality import ACTQualityConfig
from .modeling_act_quality import ACTQualityPolicy
from .quality_index import QualityAwareEpisodeSampler, QualityIndex, activate_quality_index


class RecordingModel(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.last_batch = None

    def forward(self, batch):
        self.last_batch = batch
        actions = torch.zeros_like(batch[ACTION], requires_grad=True)
        batch_size = actions.shape[0]
        mu = torch.ones(batch_size, self.latent_dim, device=actions.device)
        log_var = torch.zeros_like(mu)
        return actions, (mu, log_var)


def _config() -> ACTQualityConfig:
    return ACTQualityConfig(
        device="cpu",
        chunk_size=4,
        n_action_steps=4,
        dim_model=16,
        n_heads=4,
        dim_feedforward=32,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        latent_dim=2,
        dropout=0.0,
        pretrained_backbone_weights=None,
        quality_filter_invalid_anchors=False,
        quality_balance_anchor_pools=False,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(2,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
    )


def _quality_index() -> QualityIndex:
    labels = torch.tensor([False, False, True, True, True, True, True, True])
    return QualityIndex(
        labels=labels,
        episode_from_indices=torch.tensor([0, 4]),
        episode_to_indices=torch.tensor([4, 8]),
        first_valid_indices=torch.tensor([2, 4]),
        label_key="action_quality",
    )


def test_registration() -> None:
    assert PreTrainedConfig.get_choice_class("act_quality") is ACTQualityConfig


def test_sampler() -> None:
    quality = _quality_index()
    activate_quality_index(quality)
    sampler = QualityAwareEpisodeSampler([0, 4], [4, 8], shuffle=False)
    assert sampler.indices == [2, 3, 4, 5, 6, 7]


def test_balanced_sampler() -> None:
    quality = _quality_index()
    activate_quality_index(
        quality,
        balance_anchor_pools=True,
        recovery_anchor_fraction=0.5,
    )
    sampler = QualityAwareEpisodeSampler([0, 4], [4, 8], shuffle=True, seed=7)
    epoch_zero = list(iter(sampler))
    assert len(sampler) == 6
    assert sampler.normal_pool_size == 4
    assert sampler.recovery_pool_size == 2
    assert sampler.normal_samples_per_epoch == 3
    assert sampler.recovery_samples_per_epoch == 3
    assert sum(index < 4 for index in epoch_zero) == 3
    assert sum(index >= 4 for index in epoch_zero) == 3

    activate_quality_index(
        quality,
        balance_anchor_pools=True,
        recovery_anchor_fraction=0.5,
    )
    reproduced = QualityAwareEpisodeSampler([0, 4], [4, 8], shuffle=True, seed=7)
    assert list(iter(reproduced)) == epoch_zero

    activate_quality_index(
        quality,
        balance_anchor_pools=True,
        recovery_anchor_fraction=0.5,
    )
    resumed = QualityAwareEpisodeSampler([0, 4], [4, 8], shuffle=True, seed=7)
    resumed.load_state_dict({"epoch": 0, "start_index": 2})
    assert list(iter(resumed)) == epoch_zero[2:]


def test_all_valid_recovery_provenance() -> None:
    quality = QualityIndex(
        labels=torch.ones(8, dtype=torch.bool),
        episode_from_indices=torch.tensor([0, 4]),
        episode_to_indices=torch.tensor([4, 8]),
        first_valid_indices=torch.tensor([0, 4]),
        label_key="action_quality",
        recovery_episode_flags=torch.tensor([True, False]),
    )
    activate_quality_index(
        quality,
        balance_anchor_pools=True,
        recovery_anchor_fraction=0.5,
    )
    sampler = QualityAwareEpisodeSampler([0, 4], [4, 8], shuffle=False)
    assert sampler.recovery_pool_size == 4
    assert sampler.normal_pool_size == 4
    assert quality.recovery_anchor_mask().tolist() == [
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
    ]


def test_masked_forward() -> None:
    config = _config()
    policy = ACTQualityPolicy(config)
    quality = _quality_index()
    policy.set_quality_index(quality)
    recorder = RecordingModel(config.latent_dim)
    policy.model = recorder
    policy.train()

    batch = {
        OBS_STATE: torch.zeros(2, 2),
        OBS_ENV_STATE: torch.zeros(2, 2),
        ACTION: torch.tensor(
            [
                [[10.0, 10.0]] * 4,
                [[1.0, 1.0]] * 4,
            ]
        ),
        "action_is_pad": torch.zeros(2, 4, dtype=torch.bool),
        "index": torch.tensor([0, 4]),
    }
    loss, metrics = policy(batch)
    assert torch.isclose(loss, torch.tensor(11.0))  # L1=1, valid KL=1, kl_weight=10
    assert metrics["quality_invalid_anchor_fraction"] == 0.5
    assert metrics["quality_recovery_anchor_fraction"] == 0.0
    assert metrics["quality_normal_anchor_fraction"] == 0.5
    assert recorder.last_batch["action_is_pad"][0].all()
    assert not recorder.last_batch["action_is_pad"][1].any()
    assert (recorder.last_batch[ACTION][0] == 0).all()
    assert (recorder.last_batch[ACTION][1] == 1).all()

    per_sample, _ = policy(batch, reduction="none")
    assert torch.isclose(per_sample[0], torch.tensor(0.0))
    assert torch.isclose(per_sample[1], torch.tensor(11.0))


def test_parquet_alignment() -> None:
    with tempfile.TemporaryDirectory(prefix="act-quality-selftest-") as temporary:
        root = Path(temporary)
        data_path = root / "data" / "chunk-000" / "file-000.parquet"
        data_path.parent.mkdir(parents=True)
        pd.DataFrame(
            {
                "index": np.arange(8, dtype=np.int64),
                "action_quality": [False, False, True, True, True, True, True, True],
            }
        ).to_parquet(data_path, index=False)
        meta = SimpleNamespace(
            root=root,
            total_frames=8,
            features={"action_quality": {"dtype": "bool", "shape": (1,), "names": None}},
            episodes=pd.DataFrame(
                {"dataset_from_index": [0, 4], "dataset_to_index": [4, 8]}
            ),
        )
        quality = QualityIndex.from_dataset_meta(meta, "action_quality")
        assert quality.invalid_frames == 2
        assert quality.first_valid_indices.tolist() == [2, 4]


def main() -> None:
    test_registration()
    test_sampler()
    test_balanced_sampler()
    test_all_valid_recovery_provenance()
    test_masked_forward()
    test_parquet_alignment()
    print("act_quality selftest: PASS")


if __name__ == "__main__":
    main()
