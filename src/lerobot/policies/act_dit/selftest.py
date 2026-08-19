#!/usr/bin/env python
"""Runnable self-check for ACT-DiT. `python -m lerobot.policies.act_dit.selftest`.

Covers the three things that break silently:
  1. the ACT refactor (`encode_observations`) still reproduces ACT's own forward,
  2. the encoder runs ONCE per chunk while the decoder runs once per integration step
     (the whole cost argument of S1 rests on this),
  3. both conditioning arms train: with cross-attention (S1) and without (ablation D).
"""

import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act_dit.configuration_act_dit import ACTDiTConfig
from lerobot.policies.act_dit.modeling_act_dit import ACTDiTPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_IMAGES, OBS_STATE

STATE_DIM, ACTION_DIM, CHUNK, BATCH = 6, 6, 8, 2


def _features(cfg):
    cfg.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 60, 80)),
    }
    cfg.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))}
    cfg.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.IDENTITY,
        "ACTION": NormalizationMode.IDENTITY,
    }
    cfg.device = "cpu"
    return cfg


def _batch():
    return {
        OBS_STATE: torch.randn(BATCH, STATE_DIM),
        OBS_IMAGE: torch.rand(BATCH, 3, 60, 80),
        ACTION: torch.randn(BATCH, CHUNK, ACTION_DIM),
        "action_is_pad": torch.zeros(BATCH, CHUNK, dtype=torch.bool),
    }


def _act_config(cls, **kw):
    return _features(
        cls(
            chunk_size=CHUNK,
            n_action_steps=CHUNK,
            dim_model=32,
            dim_feedforward=64,
            n_heads=2,
            n_encoder_layers=1,
            n_decoder_layers=2,
            dropout=0.0,
            pretrained_backbone_weights=None,
            **kw,
        )
    )


def test_act_refactor_is_equivalent():
    """`encode_observations` + decode == `ACT.forward`, i.e. the split changed nothing."""
    torch.manual_seed(0)
    policy = ACTPolicy(_act_config(ACTConfig, use_vae=False)).eval()
    model, batch = policy.model, _batch()
    batch[OBS_IMAGES] = [batch[OBS_IMAGE]]  # the image-list packing `ACTPolicy` does

    with torch.no_grad():
        actions, _ = model(batch)
        encoder_out, pos, (mu, log_sigma) = model.encode_observations(batch)
        decoder_out = model.decoder(
            torch.zeros(CHUNK, BATCH, model.config.dim_model),
            encoder_out,
            encoder_pos_embed=pos,
            decoder_pos_embed=model.decoder_pos_embed.weight.unsqueeze(1),
        )
        manual = model.action_head(decoder_out.transpose(0, 1))

    assert mu is None and log_sigma is None, "use_vae=False must yield no latent params"
    assert torch.allclose(actions, manual, atol=1e-6), (actions - manual).abs().max()


def test_encoder_runs_once_per_chunk():
    """Encoder: 1 call. Decoder: one per integration step. This is the cost claim of S1."""
    steps = 4
    torch.manual_seed(0)
    policy = ACTDiTPolicy(_act_config(ACTDiTConfig, num_integration_steps=steps)).eval()

    calls = {"encoder": 0, "decoder": 0}
    policy.model.encoder.register_forward_hook(lambda *_: calls.__setitem__("encoder", calls["encoder"] + 1))
    policy.model.decoder.register_forward_hook(lambda *_: calls.__setitem__("decoder", calls["decoder"] + 1))

    batch = _batch()
    chunk = policy.predict_action_chunk({OBS_STATE: batch[OBS_STATE], OBS_IMAGE: batch[OBS_IMAGE]})

    assert chunk.shape == (BATCH, CHUNK, ACTION_DIM), chunk.shape
    assert torch.isfinite(chunk).all()
    assert calls == {"encoder": 1, "decoder": steps}, calls


def test_both_objectives_and_arms_train():
    """Loss is finite and every branch that should get gradient does."""
    for objective in ("flow_matching", "diffusion"):
        for use_cross_attention in (True, False):
            torch.manual_seed(0)
            cfg = _act_config(
                ACTDiTConfig, objective=objective, use_cross_attention=use_cross_attention
            )
            policy = ACTDiTPolicy(cfg).train()
            loss, log = policy.forward(_batch())
            assert torch.isfinite(loss), (objective, use_cross_attention, loss)
            assert f"{objective}_loss" in log
            loss.backward()

            layer = policy.model.decoder.layers[0]
            named = dict(policy.model.named_parameters())
            assert named["action_in_proj.weight"].grad.abs().sum() > 0
            assert layer.adaln[-1].weight.grad.abs().sum() > 0, "adaLN got no gradient"
            if use_cross_attention:
                # The observation reaches the loss through cross-attention, which is not gated.
                assert named["backbone.conv1.weight"].grad.abs().sum() > 0
            # Note: `time_mlp` (and, in ablation D, the whole observation path) has *zero*
            # gradient at initialisation - that is what adaLN-Zero means, the gate is shut and
            # d(out)/d(cond) = 0. It starts training once `adaln[-1]` moves off zero.
            assert hasattr(layer, "multihead_attn") == use_cross_attention, (
                "ablation D must not carry unused cross-attention parameters"
            )
            # No parameter may be left without gradient: DDP would fail on it.
            dead = [n for n, p in named.items() if p.requires_grad and p.grad is None]
            assert not dead, dead


def test_adaln_starts_as_identity():
    """adaLN-Zero: at init the gates are shut, so the block is the identity (DiT init)."""
    torch.manual_seed(0)
    policy = ACTDiTPolicy(_act_config(ACTDiTConfig))
    for layer in policy.model.decoder.layers:
        assert layer.adaln[-1].weight.abs().sum() == 0
        assert layer.adaln[-1].bias.abs().sum() == 0


def test_vae_is_refused():
    try:
        _act_config(ACTDiTConfig, use_vae=True)
    except ValueError as e:
        assert "use_vae" in str(e)
    else:
        raise AssertionError("use_vae=True should be refused by ACTDiTConfig")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all good")
