#!/usr/bin/env python

# NEW MODULE. No lerobot equivalent exists: every in-tree vision encoder
# (`DiffusionRgbEncoder`, `VQBeTRgbEncoder`, `ACT`'s backbone) returns one pooled vector per frame,
# which is exactly the information bottleneck Patch Policy exists to remove.
#
# Ported from `models/encoder/*.py` of https://github.com/gaoyuezhou/patch_policy
# (MIT License, Copyright (c) 2026 the Patch Policy authors). One class per reference file:
#
#   dino.py            -> DinoV2Encoder
#   dinov3.py          -> DinoV3Encoder
#   webssl.py          -> WebSSLEncoder
#   siglip2.py         -> SigLIP2Encoder
#   vjepa2.py          -> VJEPA2Encoder
#   resnet.py          -> ResNet18Encoder
#   from_ckpt.py       -> FromCheckpointEncoder
#
# The reference repeats the same "collapse leading dims, run backbone, restore leading dims"
# block in all seven files; it lives once in `PatchEncoder.forward` here and the subclasses
# implement only `encode`.
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
import sys

import torch
import torchvision
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from lerobot.utils.import_utils import require_package

PATCH_TOKENS = "x_norm_patchtokens"
CLS_TOKEN = "x_norm_clstoken"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PatchEncoder(nn.Module):
    """Frozen visual encoder returning `(..., P, E)` patch tokens.

    Every reference encoder accepts an arbitrary number of leading dims before `(C, H, W)` and
    preserves them; that bookkeeping is done once here. `latent_ndim == 1` variants (CLS token,
    average pooling, ResNet, DynaMo) return `P == 1`, so the rest of the policy has a single
    code path whether or not the representation is dense — matching the reference, where
    `n_patches: 1` in every non-patch encoder config.

    Args:
        feature_key: `"x_norm_patchtokens"` or `"x_norm_clstoken"`.
        postprocess: `None` or `"avg_pool"`. Average pooling collapses the patch grid to one token,
            which is the reference's `*_patch_avg_pool` ablation (Table 4's `n=1` rung).
        resize_shape: Input images are resized to this before the backbone.
    """

    def __init__(
        self,
        feature_key: str = PATCH_TOKENS,
        postprocess: str | None = None,
        resize_shape: tuple[int, int] = (224, 224),
    ):
        super().__init__()
        if feature_key not in (PATCH_TOKENS, CLS_TOKEN):
            raise ValueError(f"Invalid feature key: {feature_key}")
        if postprocess not in (None, "avg_pool"):
            raise ValueError(f"Invalid postprocess: {postprocess}")
        self.feature_key = feature_key
        self.postprocess = postprocess
        self.resize_shape = resize_shape
        # 2 == dense patch grid, 1 == single vector (a dummy patch dim is added on the way out).
        self.latent_ndim = 1 if (feature_key == CLS_TOKEN or postprocess == "avg_pool") else 2

    @property
    def output_dim(self) -> int:
        raise NotImplementedError

    def encode(self, x: Tensor) -> Tensor:
        """`(B, C, H, W)` in `[0, 1]` -> `(B, P, E)` (or `(B, E)` for the CLS token)."""
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        prefix_shape = x.shape[:-3]
        c, h, w = x.shape[-3:]
        x = x.reshape(-1, c, h, w)
        if (h, w) != tuple(self.resize_shape):
            x = F.interpolate(x, size=self.resize_shape, mode="bilinear", align_corners=False)

        emb = self.encode(x)

        if self.postprocess == "avg_pool":
            emb = torch.mean(emb, dim=-2)
        emb = emb.reshape(*prefix_shape, *emb.shape[1:])
        if self.latent_ndim == 1:
            # Dummy patch dim so downstream code never branches on encoder type.
            emb = emb.unsqueeze(len(prefix_shape))
        return emb


class DinoV2Encoder(PatchEncoder):
    """`models/encoder/dino.py`. DINOv2 via torch.hub, pinned to the same commit as the reference."""

    def __init__(self, name: str = "dinov2_vits14", **kwargs):
        super().__init__(**kwargs)
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        self.base_model = torch.hub.load("facebookresearch/dinov2:b48308a", name)
        self.emb_dim = self.base_model.num_features
        self.patch_size = self.base_model.patch_size
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    @property
    def output_dim(self) -> int:
        return self.emb_dim

    def encode(self, x: Tensor) -> Tensor:
        assert x.max() <= 1.0 and x.min() >= 0, "expect 0..1 range"
        x = (x - self.mean) / self.std
        return self.base_model.forward_features(x)[self.feature_key]


class DinoV3Encoder(PatchEncoder):
    """`models/encoder/dinov3.py`, loaded through torch.hub so a local `.pth` can be used directly.

    The hub backbones expose the same `forward_features` dict as DINOv2, so `x_norm_patchtokens`
    already excludes CLS and the 4 storage tokens - no register slicing needed. `checkpoint` is a
    path (or URL) to one of Meta's `dinov3_*_pretrain_*.pth` files; without it the entrypoint
    downloads its own default weights.
    """

    def __init__(self, name: str = "dinov3_vits16plus", checkpoint: str | None = None, **kwargs):
        super().__init__(**kwargs)
        # `torch.hub.load` is unusable here: the repo's `hubconf.py` imports its segmentation and
        # detection heads, which pull in `torchmetrics` and other deps the backbone never touches.
        # Fetch the same checkout, then import only `dinov3.hub.backbones`.
        # ponytail: private torch API, stable across 1.x/2.x; inline the cache path if it moves.
        repo_dir = torch.hub._get_cache_or_reload(
            "facebookresearch/dinov3:main", force_reload=False, trust_repo=True, verbose=False
        )
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        from dinov3.hub import backbones

        kw = {"weights": checkpoint} if checkpoint else {}
        self.base_model = getattr(backbones, name)(**kw)
        self.emb_dim = self.base_model.num_features
        self.patch_size = self.base_model.patch_size
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    @property
    def output_dim(self) -> int:
        return self.emb_dim

    def encode(self, x: Tensor) -> Tensor:
        assert x.max() <= 1.0 and x.min() >= 0, "expect 0..1 range"
        x = (x - self.mean) / self.std
        return self.base_model.forward_features(x)[self.feature_key]


class WebSSLEncoder(PatchEncoder):
    """`models/encoder/webssl.py`. WebSSL DINO-300M, loaded through `Dinov2Model`.

    The reference runs `AutoImageProcessor` per batch; the resize it performs is already done in
    `PatchEncoder.forward`, so only the mean/std it would apply is reproduced here. That keeps the
    encoder on-device and out of the dataloader's critical path.
    """

    def __init__(self, name: str = "facebook/webssl-dino300m-full2b-224", **kwargs):
        super().__init__(**kwargs)
        require_package("transformers", extra="patch_policy")
        from transformers import AutoImageProcessor, Dinov2Model

        processor = AutoImageProcessor.from_pretrained(name, do_rescale=False)
        self.base_model = Dinov2Model.from_pretrained(name)
        self.emb_dim = self.base_model.config.hidden_size
        self.patch_size = self.base_model.config.patch_size
        self.register_buffer(
            "mean", torch.tensor(processor.image_mean).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer("std", torch.tensor(processor.image_std).view(1, 3, 1, 1), persistent=False)

    @property
    def output_dim(self) -> int:
        return self.emb_dim

    def encode(self, x: Tensor) -> Tensor:
        x = (x - self.mean) / self.std
        hidden = self.base_model(pixel_values=x).last_hidden_state
        if self.feature_key == CLS_TOKEN:
            return hidden[:, 0, :]
        return hidden[:, 1:, :]


class SigLIP2Encoder(PatchEncoder):
    """`models/encoder/siglip2.py`. Vision tower only; the text tower is discarded.

    Patch Policy's Table 7 has this encoder last on all four environments — language-aligned
    features cost geometry. It is ported for that ablation, not as a recommended default.
    """

    def __init__(self, name: str = "google/siglip2-base-patch16-224", **kwargs):
        super().__init__(**kwargs)
        if kwargs.get("feature_key", PATCH_TOKENS) == CLS_TOKEN:
            raise ValueError("SigLIP2 has no CLS token; the reference only supports patch tokens.")
        require_package("transformers", extra="patch_policy")
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(name, trust_remote_code=True)
        self.base_model = AutoModel.from_pretrained(name, trust_remote_code=True).vision_model
        self.emb_dim = self.base_model.config.hidden_size
        self.patch_size = self.base_model.config.patch_size
        self.register_buffer(
            "mean", torch.tensor(processor.image_mean).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer("std", torch.tensor(processor.image_std).view(1, 3, 1, 1), persistent=False)

    @property
    def output_dim(self) -> int:
        return self.emb_dim

    def encode(self, x: Tensor) -> Tensor:
        x = (x - self.mean) / self.std
        return self.base_model(pixel_values=x).last_hidden_state


class VJEPA2Encoder(PatchEncoder):
    """`models/encoder/vjepa2.py`. V-JEPA 2 is a video model, so each still frame is repeated
    `n_frames` times to form the shortest clip the backbone accepts (the reference uses `T = 2`).
    Its native resolution is 256, not 224 — set `resize_shape=(256, 256)`.
    """

    def __init__(self, name: str = "facebook/vjepa2-vitl-fpc64-256", n_frames: int = 2, **kwargs):
        super().__init__(**kwargs)
        if kwargs.get("feature_key", PATCH_TOKENS) == CLS_TOKEN:
            raise ValueError("V-JEPA 2 has no CLS token; the reference only supports patch tokens.")
        require_package("transformers", extra="patch_policy")
        from transformers import AutoModel, AutoVideoProcessor

        processor = AutoVideoProcessor.from_pretrained(name, trust_remote_code=True)
        self.base_model = AutoModel.from_pretrained(name, trust_remote_code=True)
        self.emb_dim = self.base_model.config.hidden_size
        self.patch_size = self.base_model.config.patch_size
        self.n_frames = n_frames
        self.register_buffer(
            "mean", torch.tensor(processor.image_mean).view(1, 1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor(processor.image_std).view(1, 1, 3, 1, 1), persistent=False
        )

    @property
    def output_dim(self) -> int:
        return self.emb_dim

    def encode(self, x: Tensor) -> Tensor:
        video = x.unsqueeze(1).repeat(1, self.n_frames, 1, 1, 1)
        video = (video - self.mean) / self.std
        return self.base_model.get_vision_features(video)


class ResNet18Encoder(PatchEncoder):
    """`models/encoder/resnet.py`. The global-pooled baseline; `P == 1` by construction."""

    def __init__(self, name: str = "resnet18", pretrained: bool = True, unit_norm: bool = False, **kwargs):
        kwargs["feature_key"] = CLS_TOKEN  # a pooled vector, never a patch grid
        super().__init__(**kwargs)
        weights = "ResNet18_Weights.IMAGENET1K_V1" if pretrained else None
        resnet = torchvision.models.resnet18(weights=weights)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.unit_norm = unit_norm
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    @property
    def output_dim(self) -> int:
        return 512

    def encode(self, x: Tensor) -> Tensor:
        x = (x - self.mean) / self.std
        out = torch.flatten(self.resnet(x), start_dim=1)
        if self.unit_norm:
            out = F.normalize(out, p=2, dim=-1)
        return out


class FromCheckpointEncoder(PatchEncoder):
    """`models/encoder/from_ckpt.py`. Loads a pickled `nn.Module`, e.g. the DynaMo encoders the
    reference uses as its global-representation baseline (`configs/encoder/*_dynamo.yaml`).

    The checkpoint is trusted code — `torch.load(weights_only=False)` executes whatever it pickles.
    Point this only at files you produced.
    """

    def __init__(self, checkpoint: str, name: str | None = None, output_dim: int = 512, **kwargs):
        kwargs["feature_key"] = CLS_TOKEN
        super().__init__(**kwargs)
        self.base_model = torch.load(checkpoint, weights_only=False, map_location="cpu")
        self._output_dim = output_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def encode(self, x: Tensor) -> Tensor:
        out = self.base_model(x)
        # DynaMo checkpoints already emit (B, E); tolerate a trailing patch dim of 1.
        return out.squeeze(-2) if out.ndim == 3 else out


_ENCODER_CLASSES = {
    "dinov2": DinoV2Encoder,
    "dinov3": DinoV3Encoder,
    "webssl": WebSSLEncoder,
    "siglip2": SigLIP2Encoder,
    "vjepa2": VJEPA2Encoder,
    "resnet18": ResNet18Encoder,
    "from_ckpt": FromCheckpointEncoder,
}


def make_patch_encoder(preset: dict, resize_shape: tuple[int, int]) -> PatchEncoder:
    """Build the encoder named by a `PATCH_ENCODER_PRESETS` entry."""
    preset = dict(preset)
    encoder_type = preset.pop("encoder_type")
    preset.pop("n_patches", None)  # reference metadata only; the real count is measured
    preset.pop("output_dim", None)  # ditto
    return _ENCODER_CLASSES[encoder_type](resize_shape=resize_shape, **preset)
