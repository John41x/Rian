"""Cross-camera appearance embeddings (OSNet-style Re-ID, spec 07)."""

from __future__ import annotations

from typing import List, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ImageInput = Union[Image.Image, np.ndarray]

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_INPUT_SIZE = (128, 256)  # (width, height) for PIL.resize
_FEATURE_DIM = 512
_STAT_GRID = (4, 8)  # width x height cells
_STAT_DIM = _STAT_GRID[0] * _STAT_GRID[1] * 6  # per-cell RGB mean + std


class _DepthwiseSeparableConv(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return F.relu(self.bn(x), inplace=True)


class _OSNetBackbone(nn.Module):
    """Lightweight multi-scale depthwise backbone (OSNet x0.25 style)."""

    def __init__(self, out_channels: int = 256) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        channels = 32
        blocks: List[nn.Module] = []
        for width, stride in ((64, 1), (96, 2), (128, 2), (160, 1)):
            blocks.extend(
                [
                    nn.Conv2d(channels, width, kernel_size=1, bias=False),
                    nn.BatchNorm2d(width),
                    nn.ReLU(inplace=True),
                    _DepthwiseSeparableConv(width),
                    nn.Conv2d(width, width, kernel_size=1, bias=False),
                    nn.BatchNorm2d(width),
                    nn.ReLU(inplace=True),
                ]
            )
            if stride > 1:
                blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
            channels = width
        self.body = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(self.stem(x))
        x = self.pool(x).flatten(1)
        return self.proj(x)


class _ReIDHead(nn.Module):
    def __init__(self, cnn_dim: int = 256, stat_dim: int = _STAT_DIM) -> None:
        super().__init__()
        self.backbone = _OSNetBackbone(out_channels=cnn_dim)
        rng = np.random.default_rng(42)
        stat_proj = rng.standard_normal((128, stat_dim), dtype=np.float32)
        stat_proj /= np.linalg.norm(stat_proj, axis=0, keepdims=True) + 1e-8
        self.stat_proj = nn.Parameter(
            torch.from_numpy(stat_proj), requires_grad=False
        )
        self.fuse = nn.Linear(cnn_dim + 128, _FEATURE_DIM, bias=False)

    def forward(self, image_tensor: torch.Tensor, stat_features: torch.Tensor) -> torch.Tensor:
        cnn_feat = self.backbone(image_tensor)
        stat_feat = stat_features @ self.stat_proj.t()
        fused = torch.cat([cnn_feat, stat_feat], dim=1)
        emb = self.fuse(fused)
        return F.normalize(emb, p=2, dim=1)


def cosine_similarity(vec1: Sequence[float], vec2: Sequence[float]) -> float:
    """Cosine similarity of two vectors (dot product when L2-normalized)."""
    a = np.asarray(vec1, dtype=np.float64)
    b = np.asarray(vec2, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Embedding shape mismatch: {a.shape} vs {b.shape}")
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _to_pil_rgb(image: ImageInput) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def _spatial_stat_vector(rgb: np.ndarray) -> np.ndarray:
    """Grid RGB mean/std features on the standardized Re-ID resolution."""
    h, w, _ = rgb.shape
    gw, gh = _STAT_GRID
    feats: List[float] = []
    for j in range(gh):
        y0, y1 = j * h // gh, (j + 1) * h // gh
        for i in range(gw):
            x0, x1 = i * w // gw, (i + 1) * w // gw
            patch = rgb[y0:y1, x0:x1]
            if patch.size == 0:
                patch = rgb
            feats.extend(patch.mean(axis=(0, 1)).tolist())
            feats.extend(patch.std(axis=(0, 1)).tolist())
    return np.asarray(feats, dtype=np.float32)


class OSNetFeatureExtractor:
    """512-d L2-normalized appearance embeddings from person/vehicle crops."""

    def __init__(self, device: str | None = None) -> None:
        torch.manual_seed(42)
        self._device = torch.device(device or "cpu")
        self._model = _ReIDHead().to(self._device)
        self._model.eval()
        mean = torch.tensor(_IMAGENET_MEAN, device=self._device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=self._device).view(1, 3, 1, 1)
        self._mean = mean
        self._std = std

    def extract(self, image: ImageInput) -> List[float]:
        pil = _to_pil_rgb(image)
        resized = pil.resize(_INPUT_SIZE, Image.BILINEAR)
        rgb = np.asarray(resized, dtype=np.float32) / 255.0
        stat = _spatial_stat_vector(rgb)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self._device)
        tensor = (tensor - self._mean) / self._std
        stat_t = torch.from_numpy(stat).unsqueeze(0).to(self._device)
        with torch.no_grad():
            emb = self._model(tensor, stat_t).squeeze(0).cpu().numpy()
        return emb.astype(np.float32).tolist()
