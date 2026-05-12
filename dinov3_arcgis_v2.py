"""
DINOv3 + ArcGIS Pro v2.32-rs311 training/deployment helpers.

Target platform:
- ArcGIS Pro 3.6 deep learning environment
- Classify Pixels Using Deep Learning
- Custom ModelExtension + custom Python raster function

Main v2 changes:
- Decoder options: linear head or FPN-lite head
- Optional class-weighted CE and focal loss
- Optional validation split, early-stopping callback attempt, and validation metrics export
- Safer EMD / DLPK packaging with manifest.json
- More robust input scaling metadata for inference

This file intentionally keeps the DINOv3 backbone loading through a local repository snapshot
(torch.hub source="local") because ArcGIS Pro's bundled deep-learning stack may not include
newer DINOv3 registry support. The companion Miniconda validation environment
should use Python 3.11+ because the official DINOv3 package declares this
minimum Python version.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
import traceback
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PathLike = Union[str, os.PathLike]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_COLORS = [
    [0, 0, 0],
    [0, 112, 255],
    [0, 176, 80],
    [255, 192, 0],
    [255, 0, 0],
    [112, 48, 160],
    [0, 176, 240],
    [146, 208, 80],
    [255, 102, 0],
    [166, 166, 166],
    [191, 143, 0],
    [91, 155, 213],
]


def _as_path(path_like: PathLike) -> Path:
    return Path(path_like).expanduser().resolve()


def _strip_outer_quotes(text: str) -> str:
    text = str(text).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1].strip()
    return text


def _split_arcgis_multivalue_paths(value: str) -> List[str]:
    """Split ArcGIS multiValue path text.

    ArcGIS geoprocessing multiValue parameters are commonly returned as a
    semicolon-delimited string. Quoted paths are supported. Windows paths do
    not normally contain semicolons, so this keeps the implementation
    deliberately conservative and dependency-free.
    """
    text = str(value or "").strip()
    if not text:
        return []

    parts: List[str] = []
    buf: List[str] = []
    quote: Optional[str] = None
    for ch in text:
        if ch in ("'", '"'):
            if quote is None:
                quote = ch
                continue
            if quote == ch:
                quote = None
                continue
        if ch == ";" and quote is None:
            part = _strip_outer_quotes("".join(buf))
            if part:
                parts.append(part)
            buf = []
        else:
            buf.append(ch)

    part = _strip_outer_quotes("".join(buf))
    if part:
        parts.append(part)
    return parts


def _as_path_list(value: Union[PathLike, Sequence[PathLike]]) -> List[Path]:
    if isinstance(value, (str, os.PathLike)):
        raw_items = _split_arcgis_multivalue_paths(str(value))
        if not raw_items:
            raw_items = [str(value)]
    else:
        raw_items = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, (str, os.PathLike)):
                raw_items.extend(_split_arcgis_multivalue_paths(str(item)) or [str(item)])
            else:
                raw_items.append(str(item))

    paths: List[Path] = []
    seen = set()
    for item in raw_items:
        path = _as_path(item)
        key = str(path).lower()
        if key not in seen:
            paths.append(path)
            seen.add(key)
    return paths


def _ensure_multiple_of_16(chip_size: int) -> None:
    if int(chip_size) % 16 != 0:
        raise ValueError(f"chip_size must be divisible by 16 for ViT-L/16. Got {chip_size}.")


def _safe_model_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(name).strip())
    return safe or "dinov3_arcgis_model"


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=4)


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _find_template_emd(training_data_dir: Path) -> Optional[Path]:
    candidates = sorted(training_data_dir.glob("*.emd"))
    if candidates:
        return candidates[0]
    candidates = sorted(training_data_dir.rglob("*.emd"))
    return candidates[0] if candidates else None


def _normalize_class_record(cls: Dict[str, Any], idx: int) -> Dict[str, Any]:
    value = cls.get("Value", idx)
    try:
        value = int(value)
    except Exception:
        value = idx
    return {
        "Value": value,
        "Name": str(cls.get("Name", f"class_{idx}")),
        "Color": cls.get("Color", DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]),
    }


def _list_classes_from_template(template_emd: Optional[Path]) -> List[Dict[str, Any]]:
    if template_emd is None:
        return []
    try:
        data = _load_json(template_emd)
        classes = data.get("Classes") or []
        if isinstance(classes, list) and classes:
            return [_normalize_class_record(c, i) for i, c in enumerate(classes)]
    except Exception:
        pass
    return []


def _infer_classes_from_data(data: Any) -> List[Dict[str, Any]]:
    class_names: List[str] = []
    if hasattr(data, "classes") and isinstance(data.classes, (list, tuple)):
        class_names = [str(x) for x in data.classes]
    elif hasattr(data, "class_mapping") and isinstance(data.class_mapping, dict):
        sorted_items = sorted(data.class_mapping.items(), key=lambda kv: int(kv[0]))
        class_names = [str(v) for _, v in sorted_items]

    if not class_names and hasattr(data, "c"):
        class_names = [f"class_{i}" for i in range(int(data.c))]

    classes: List[Dict[str, Any]] = []
    for idx, name in enumerate(class_names):
        classes.append({"Value": idx, "Name": name, "Color": DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]})
    return classes


def resolve_classes(training_data_dir: Path, data: Any) -> List[Dict[str, Any]]:
    template_classes = _list_classes_from_template(_find_template_emd(training_data_dir))
    if template_classes:
        return template_classes
    return _infer_classes_from_data(data)


def resolve_classes_for_training_sources(
    training_data_dirs: Sequence[Path],
    effective_training_data_dir: Path,
    data: Any,
) -> List[Dict[str, Any]]:
    """Resolve class metadata for one or more training data folders.

    For multi-folder training, prefer the `arcgis.learn` data object's resolved
    classes because it is the best representation of what `prepare_data()` is
    actually training. If that is unavailable, use class metadata from the
    effective folder or union compatible EMD class records from all source folders.
    """
    data_classes = _infer_classes_from_data(data)
    data_c = int(getattr(data, "c", 0) or 0)
    if data_classes and (data_c <= 0 or len(data_classes) == data_c):
        return data_classes

    effective_classes = _list_classes_from_template(_find_template_emd(effective_training_data_dir))
    if effective_classes:
        return effective_classes

    class_records: Dict[int, Dict[str, Any]] = {}
    for src_dir in training_data_dirs:
        for idx, cls in enumerate(_list_classes_from_template(_find_template_emd(src_dir))):
            norm = _normalize_class_record(cls, idx)
            value = int(norm.get("Value", idx))
            if value not in class_records:
                class_records[value] = norm
    if class_records:
        return [class_records[k] for k in sorted(class_records)]

    return data_classes


def contiguous_training_class_values(classes: Sequence[Dict[str, Any]]) -> bool:
    values = [int(c.get("Value", i)) for i, c in enumerate(classes)]
    return values == list(range(len(values)))


def parse_sequence_float(value: Optional[Union[str, Sequence[float]]]) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        return [float(x.strip()) for x in text.replace(";", ",").split(",") if x.strip()]
    return [float(x) for x in value]


def parse_sequence_int(value: Optional[Union[str, Sequence[int]]], default: Sequence[int]) -> List[int]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return list(default)
        return [int(x.strip()) for x in text.replace(";", ",").split(",") if x.strip()]
    return [int(x) for x in value]


# -----------------------------
# Normalization and target utils
# -----------------------------


def fastai_batch_to_01(
    batch: torch.Tensor,
    input_scale: str = "fastai",
    max_input_value: Optional[float] = None,
) -> torch.Tensor:
    """
    Convert a training batch to floating-point [0, 1] without histogram stretching.

    ArcGIS/fastai usually passes image tensors in [-1, 1].  For raw-image
    pipelines the caller may pass explicit input_scale values such as 0_255 or
    0_65535.  The function intentionally avoids per-chip contrast stretching;
    fixed scaling preserves relative 16-bit DN/radiometry better than auto min/max.
    """
    x = batch.float()
    mode = str(input_scale or "fastai").strip().lower()
    if mode in ("fastai", "fastai_-1_1", "-1_1", "arcgis", "arcgis_fastai"):
        return ((x.clamp(-1.0, 1.0) + 1.0) / 2.0).clamp(0.0, 1.0)
    return scale_raw_pixels_to_01(x, input_scale=mode, max_input_value=max_input_value)


def _expand_channel_stats(values: Sequence[float], channels: int, fallback: Sequence[float]) -> List[float]:
    """Return channel statistics with length exactly matching the backbone input.

    DINOv3 web RGB backbones normally use 3-channel ImageNet statistics, but
    some local SAT-493M repository/weight combinations expose a 4-channel patch
    embedding.  ArcGIS training chips may also include alpha/nodata channels.
    This helper avoids shape crashes while keeping a deterministic, fixed
    normalization policy.  It does not do per-chip min/max stretching.
    """
    ch = int(channels)
    vals = [float(v) for v in (values or fallback)]
    if len(vals) == ch:
        return vals
    if not vals:
        vals = [float(fallback[0]) if fallback else 0.0]
    if len(vals) > ch:
        return vals[:ch]
    return vals + [vals[-1]] * (ch - len(vals))


def normalize_fastai_batch_to_dino(
    batch: torch.Tensor,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> torch.Tensor:
    """Convert ArcGIS/fastai image tensors from [-1, 1] to DINO-normalized float tensors.

    The statistics length is expanded to the actual tensor channel count.  This
    prevents the RGB mean/std vector from crashing on 4-channel SAT-493M style
    backbones or ArcGIS chips that carry an extra band.
    """
    x = fastai_batch_to_01(batch, input_scale="fastai")
    mean_vals = _expand_channel_stats(mean, int(x.shape[1]), IMAGENET_MEAN)
    std_vals = _expand_channel_stats(std, int(x.shape[1]), IMAGENET_STD)
    mean_t = torch.tensor(mean_vals, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std_vals, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    return (x - mean_t) / std_t


def select_fastai_batch_bands(
    batch: torch.Tensor,
    extract_bands: Optional[Sequence[int]] = None,
    expected_channels: Optional[int] = None,
    context: str = "training batch",
) -> torch.Tensor:
    """Select model input bands from an ArcGIS/fastai tensor.

    ArcGIS ``prepare_data`` can return the full chip band stack even when the
    exported training dataset is intended to be trained with only a subset of
    bands.  A common case is an apparent RGB dataset arriving as 4 channels
    because PNG/TIFF chips include an alpha/nodata band.  This helper applies
    the user-facing Extract Bands setting before DINO/ImageNet normalization so
    RGB mode receives exactly 3 channels and multispectral adapter mode receives
    the configured number of channels.

    Band indices are treated as zero-based first (the toolkit UI examples use
    0,1,2).  If they do not fit but look one-based, the function falls back to
    one-based indexing for compatibility.  No per-chip stretch or dtype scaling
    is performed here.
    """
    if batch is None or not torch.is_tensor(batch) or batch.ndim < 4:
        return batch

    c = int(batch.shape[1])
    bands = [int(b) for b in (extract_bands or [])]
    if expected_channels is None:
        expected_channels = len(bands) if bands else c
    expected_channels = int(expected_channels)

    if c == expected_channels:
        return batch

    if bands:
        # Preferred zero-based indices: 0,1,2 means the first three channels.
        if min(bands) >= 0 and max(bands) < c:
            return batch.index_select(1, torch.tensor(bands, dtype=torch.long, device=batch.device))
        # Compatibility fallback for one-based indices: 1,2,3 means first three channels.
        one_based = [b - 1 for b in bands]
        if min(one_based) >= 0 and max(one_based) < c:
            return batch.index_select(1, torch.tensor(one_based, dtype=torch.long, device=batch.device))

    # Safe fallback for RGB-like exports where ArcGIS provides an extra alpha/nodata
    # channel but Extract Bands was omitted or cannot be mapped.  Make the behavior
    # explicit in the error when we cannot safely infer a subset.
    if expected_channels > 0 and c > expected_channels:
        return batch[:, :expected_channels, ...]

    raise ValueError(
        f"{context} has {c} channel(s), but the model expects {expected_channels}. "
        f"ExtractBands={bands or '(not set)'} could not be applied. "
        "Check training chips, alpha/nodata bands, and the Extract Bands parameter."
    )


def scale_raw_pixels_to_01(batch: torch.Tensor, input_scale: str = "auto", max_input_value: Optional[float] = None) -> torch.Tensor:
    """Scale ArcGIS PRF raw pixels to [0, 1]."""
    x = batch.float()
    scale_mode = str(input_scale or "auto").lower()
    if scale_mode in ("0_1", "unit", "none"):
        return x.clamp(0.0, 1.0)
    if scale_mode in ("0_255", "uint8", "8bit"):
        return (x / 255.0).clamp(0.0, 1.0)
    if scale_mode in ("0_65535", "uint16", "16bit"):
        return (x / 65535.0).clamp(0.0, 1.0)
    if max_input_value is not None and float(max_input_value) > 0:
        return (x / float(max_input_value)).clamp(0.0, 1.0)

    observed_max = float(torch.nan_to_num(x).max().item()) if x.numel() else 1.0
    if observed_max <= 1.5:
        return x.clamp(0.0, 1.0)
    if observed_max <= 255.0:
        return (x / 255.0).clamp(0.0, 1.0)
    # Conservative fallback for 16-bit or already-stretched imagery.
    return (x / max(observed_max, 1.0)).clamp(0.0, 1.0)


def normalize_inference_batch_to_dino(
    batch: torch.Tensor,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> torch.Tensor:
    """Normalize a [0, 1] tensor for DINOv3 with channel-safe statistics."""
    x = batch.float()
    mean_vals = _expand_channel_stats(mean, int(x.shape[1]), IMAGENET_MEAN)
    std_vals = _expand_channel_stats(std, int(x.shape[1]), IMAGENET_STD)
    mean_t = torch.tensor(mean_vals, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std_vals, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    return (x - mean_t) / std_t


def target_one_hot_to_label_map(target: torch.Tensor) -> torch.Tensor:
    if isinstance(target, (list, tuple)):
        target = target[0]
    if target.ndim == 4 and target.shape[1] > 1:
        return torch.argmax(target, dim=1).long()
    if target.ndim == 4 and target.shape[1] == 1:
        return target[:, 0, :, :].long()
    if target.ndim == 3:
        return target.long()
    if target.ndim == 2:
        return target.unsqueeze(0).long()
    raise ValueError(f"Unsupported target shape for segmentation: {tuple(target.shape)}")


# -----------------------------
# DINOv3 backbone and decoders
# -----------------------------


def _extract_patch_tokens(feature_output: Any) -> torch.Tensor:
    """Best-effort extraction of ViT patch tokens from DINOv3 forward_features outputs."""
    if isinstance(feature_output, dict):
        for key in ("x_norm_patchtokens", "x_patchtokens", "patch_tokens", "patchtokens"):
            if key in feature_output:
                return feature_output[key]
        if "x_prenorm" in feature_output:
            x_prenorm = feature_output["x_prenorm"]
            if torch.is_tensor(x_prenorm) and x_prenorm.ndim == 3 and x_prenorm.shape[1] > 5:
                # Many DINO ViT variants: 1 cls token + register tokens + patch tokens.
                return x_prenorm[:, 5:, :]
    if torch.is_tensor(feature_output) and feature_output.ndim == 3:
        return feature_output[:, 5:, :] if feature_output.shape[1] > 5 else feature_output
    raise ValueError(
        "Unable to locate patch tokens in DINOv3 forward_features output. "
        "Inspect your local DINOv3 repo snapshot and adapt _extract_patch_tokens()."
    )


def _temporary_sys_path(path: str):
    """Context manager that prepends a path to sys.path and restores it afterwards."""
    class _SysPathGuard:
        def __enter__(self):
            self.path = path
            self.inserted = False
            if self.path not in sys.path:
                sys.path.insert(0, self.path)
                self.inserted = True
            return self

        def __exit__(self, exc_type, exc, tb):
            if self.inserted:
                try:
                    sys.path.remove(self.path)
                except ValueError:
                    pass
            return False

    return _SysPathGuard()


def _load_dinov3_backbone_direct(repo_dir: str, model_name: str, weights: str) -> nn.Module:
    """
    Load a DINOv3 backbone without importing hubconf.py.

    Some DINOv3 repository snapshots import optional evaluation/segmentation modules from
    hubconf.py. In ArcGIS Pro environments, those optional modules may require packages such
    as torchmetrics even when we only need a plain backbone. Importing dinov3.hub.backbones
    directly avoids that unrelated dependency path.
    """
    with _temporary_sys_path(repo_dir):
        from dinov3.hub import backbones as dinov3_backbones  # type: ignore

        if not hasattr(dinov3_backbones, model_name):
            available = [name for name in dir(dinov3_backbones) if name.startswith("dinov3_")]
            raise AttributeError(
                f"DINOv3 backbone '{model_name}' was not found in dinov3.hub.backbones. "
                f"Available candidates: {available}"
            )
        factory = getattr(dinov3_backbones, model_name)
        return factory(weights=weights)


def load_dinov3_backbone(repo_dir: PathLike, model_name: str, backbone_weights_path: PathLike) -> nn.Module:
    """
    Load DINOv3 backbone for ArcGIS Pro.

    v2.32-rs311 behavior:
    1. Try direct import from dinov3.hub.backbones first.
       This avoids hubconf.py importing optional segmentation/evaluation dependencies.
    2. Fall back to torch.hub.load for repository snapshots that do not expose backbones.py.
    3. If torchmetrics is missing, raise an actionable ArcGIS-friendly error message.
    """
    repo_dir_s = str(_as_path(repo_dir))
    weights_s = str(_as_path(backbone_weights_path))

    direct_error: Optional[BaseException] = None
    try:
        return _load_dinov3_backbone_direct(repo_dir_s, model_name, weights_s)
    except Exception as exc:
        direct_error = exc

    try:
        return torch.hub.load(repo_or_dir=repo_dir_s, model=model_name, source="local", weights=weights_s)
    except ModuleNotFoundError as exc:
        if exc.name == "torchmetrics":
            raise ModuleNotFoundError(
                "DINOv3 hubconf.py imported optional segmentation/evaluation modules that require "
                "torchmetrics. Preferred fix: use this v2.32-rs311 loader, which tries direct backbone loading "
                "first. If your local DINOv3 snapshot still requires hubconf fallback, install it in the "
                "active ArcGIS Pro DL environment: python -m pip install torchmetrics. "
                f"Direct loader failed first with: {type(direct_error).__name__}: {direct_error}"
            ) from exc
        raise


class LinearSegmentationHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, decoder_channels: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(embed_dim, decoder_channels, kernel_size=1),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU(),
            nn.Dropout2d(float(dropout)),
            nn.Conv2d(decoder_channels, num_classes, kernel_size=1),
        )

    def forward(self, feat: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        logits = self.net(feat)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


class FPNLiteSegmentationHead(nn.Module):
    """
    FPN-lite decoder for single-scale ViT patch tokens.

    It does not pretend to be a full multi-layer FPN; instead, it creates multi-resolution branches
    from the patch-token feature map, fuses them, then upsamples to chip size. This is a better
    baseline than a pure 1x1 linear head for remote-sensing boundaries, while keeping deployment simple.
    """

    def __init__(self, embed_dim: int, num_classes: int, decoder_channels: int, dropout: float) -> None:
        super().__init__()
        c = int(decoder_channels)
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, c, kernel_size=1),
            nn.BatchNorm2d(c),
            nn.GELU(),
        )
        self.branch_1 = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.GELU())
        self.branch_2 = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.GELU())
        self.branch_4 = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.GELU())
        self.fuse = nn.Sequential(
            nn.Conv2d(c * 3, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.GELU(),
            nn.Dropout2d(float(dropout)),
            nn.Conv2d(c, num_classes, kernel_size=1),
        )

    def forward(self, feat: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        base = self.proj(feat)
        b1 = self.branch_1(base)
        b2 = F.interpolate(base, scale_factor=2.0, mode="bilinear", align_corners=False)
        b2 = self.branch_2(b2)
        b4 = F.interpolate(base, scale_factor=4.0, mode="bilinear", align_corners=False)
        b4 = self.branch_4(b4)
        target_size = b4.shape[-2:]
        b1 = F.interpolate(b1, size=target_size, mode="bilinear", align_corners=False)
        b2 = F.interpolate(b2, size=target_size, mode="bilinear", align_corners=False)
        logits = self.fuse(torch.cat([b1, b2, b4], dim=1))
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


class SpectralInputAdapter(nn.Module):
    """
    Learnable spectral adapter for 3-band, 4-band, multispectral, and hyperspectral chips.

    v2.19 generalizes the adapter output channel count.  Some SAT-493M local
    DINOv3 backbones expose a 4-channel patch embedding while many training
    datasets are RGB-like.  The adapter can therefore learn N selected input
    bands -> M backbone input channels, where M is read from the loaded backbone.
    """

    def __init__(self, input_channels: int, mode: str = "learned_1x1", output_channels: int = 3) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.mode = str(mode or "none").strip().lower()
        if self.input_channels < 1:
            raise ValueError("input_channels must be >= 1")
        if self.output_channels < 1:
            raise ValueError("output_channels must be >= 1")
        if self.mode in ("learned_1x1", "linear", "1x1", "adapter"):
            self.net = nn.Conv2d(self.input_channels, self.output_channels, kernel_size=1, bias=True)
            self._init_average_projection()
        elif self.mode in ("learned_3x3", "conv3x3", "spatial_spectral"):
            hidden = max(8, min(64, self.input_channels * 2))
            self.net = nn.Sequential(
                nn.Conv2d(self.input_channels, hidden, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.GELU(),
                nn.Conv2d(hidden, self.output_channels, kernel_size=1, bias=True),
            )
        else:
            raise ValueError("input_adapter must be one of: none, learned_1x1, learned_3x3")

    def _init_average_projection(self) -> None:
        # Stable default: each output channel starts as the average of selected
        # source bands.  Training can then learn class-specific spectral weighting.
        if isinstance(self.net, nn.Conv2d):
            with torch.no_grad():
                self.net.weight.zero_()
                self.net.bias.zero_()
                self.net.weight[:, :, 0, 0] = 1.0 / float(self.input_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _looks_like_patch_embed_weight(name: str, tensor: torch.Tensor) -> bool:
    """Return True when a tensor is likely the ViT patch embedding conv weight."""
    if not torch.is_tensor(tensor) or tensor.ndim != 4:
        return False
    lname = str(name).lower()
    if "patch_embed" in lname and "weight" in lname:
        return True
    # DINOv3 ViT-L/16 patch embedding has a large output dimension and 16x16 kernel.
    # This fallback is useful when the module is wrapped/parametrized and the key name differs.
    return bool(tensor.shape[0] >= 128 and tensor.shape[-1] in (14, 16) and tensor.shape[-2] in (14, 16))


def _search_state_dict_for_patch_embed_channels(state_dict: Dict[str, Any]) -> Optional[int]:
    """Find patch embedding input channels from a model or checkpoint state_dict."""
    # Prefer explicit patch_embed keys first.
    for key, value in state_dict.items():
        if torch.is_tensor(value) and value.ndim == 4:
            lk = str(key).lower()
            if "patch_embed" in lk and "weight" in lk:
                return int(value.shape[1])
    # Fallback to likely ViT patch convolution kernels.
    for key, value in state_dict.items():
        if _looks_like_patch_embed_weight(str(key), value):
            return int(value.shape[1])
    return None


def _unwrap_checkpoint_state_dict(obj: Any) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of a tensor state_dict from common checkpoint layouts."""
    if isinstance(obj, dict):
        tensor_items = {str(k): v for k, v in obj.items() if torch.is_tensor(v)}
        if tensor_items:
            return tensor_items
        for key in ("model", "state_dict", "teacher", "student", "backbone", "module"):
            if key in obj:
                found = _unwrap_checkpoint_state_dict(obj[key])
                if found:
                    return found
        # Some checkpoints have nested dicts with unexpected names.
        for value in obj.values():
            if isinstance(value, dict):
                found = _unwrap_checkpoint_state_dict(value)
                if found:
                    return found
    return None


def infer_backbone_input_channels_from_weights(backbone_weights_path: Optional[PathLike]) -> Optional[int]:
    """Infer DINOv3 patch-embedding input channels directly from the .pth checkpoint.

    This protects ArcGIS training from wrapper/parametrization cases where
    ``backbone.patch_embed.proj`` is not directly inspectable even though the
    actual forward pass uses a 4-channel patch embedding, as seen with some
    SAT-493M local checkpoints.
    """
    if not backbone_weights_path:
        return None
    try:
        weights_path = _as_path(backbone_weights_path)
        if not weights_path.exists():
            return None
        try:
            ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(str(weights_path), map_location="cpu")
        sd = _unwrap_checkpoint_state_dict(ckpt)
        if not sd:
            return None
        return _search_state_dict_for_patch_embed_channels(sd)
    except Exception:
        return None


def get_backbone_input_channels(backbone: nn.Module, default: int = 3) -> int:
    """Best-effort detection of the DINOv3 patch-embedding input channel count."""
    candidates = []
    obj = getattr(backbone, "patch_embed", None)
    if obj is not None:
        proj = getattr(obj, "proj", None)
        if proj is not None:
            candidates.append(proj)
        candidates.append(obj)
    candidates.append(backbone)
    for obj in candidates:
        if isinstance(obj, nn.Conv2d):
            return int(obj.in_channels)
        weight = getattr(obj, "weight", None)
        if torch.is_tensor(weight) and weight.ndim == 4:
            return int(weight.shape[1])
        # Handle parametrized/wrapped modules where the tensor lives in a state_dict.
        try:
            sd = obj.state_dict()
            found = _search_state_dict_for_patch_embed_channels(sd)
            if found is not None:
                return int(found)
        except Exception:
            pass
    try:
        found = _search_state_dict_for_patch_embed_channels(backbone.state_dict())
        if found is not None:
            return int(found)
    except Exception:
        pass
    for module in backbone.modules():
        if isinstance(module, nn.Conv2d):
            # Do not trust decoder-like 1x1/3x3 convs; patch embedding is normally 14x14 or 16x16.
            k = getattr(module, "kernel_size", None)
            if k and len(k) == 2 and int(k[0]) in (14, 16) and int(k[1]) in (14, 16):
                return int(module.in_channels)
    return int(default)


def resolve_input_adapter_mode(input_adapter: Optional[str], input_channels: int) -> str:
    """Normalize user-facing adapter mode without prematurely resolving ``auto``.

    The actual auto decision must be made after the DINOv3 backbone is loaded,
    because SAT-493M checkpoints may expose either 3-channel or 4-channel patch
    embeddings.  Therefore ``auto`` is passed through to DINOv3SegmentationNetV2,
    where the loaded backbone input channel count is already known.
    """
    mode = str(input_adapter or "auto").strip().lower()
    if mode in ("", "auto"):
        return "auto"
    if mode in ("rgb", "identity"):
        return "none"
    return mode


class DINOv3SegmentationNetV2(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        embed_dim: int = 1024,
        decoder_channels: int = 256,
        dropout: float = 0.1,
        decoder_type: str = "fpn_lite",
        input_channels: int = 3,
        input_adapter: str = "auto",
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
        backbone_input_channels_override: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_classes = int(num_classes)
        self.embed_dim = int(embed_dim)
        self.decoder_type = str(decoder_type).lower()
        self.input_channels = int(input_channels)
        checkpoint_backbone_ch = int(backbone_input_channels_override) if backbone_input_channels_override else None
        detected_backbone_ch = get_backbone_input_channels(self.backbone, default=int(checkpoint_backbone_ch or 3))
        # v2.32-rs311: prefer the *loaded runtime backbone* over checkpoint probing.
        # Some SAT-493M local .pth files contain 3-channel-looking keys in the
        # checkpoint, while torch.hub/local repo loading produces an actual
        # 4-channel patch_embed.proj.weight. The forward pass is governed by
        # the loaded module, so using the checkpoint override first can wrongly
        # select a 4->3 adapter and then crash at patch_embed.
        self.checkpoint_backbone_input_channels = checkpoint_backbone_ch
        self.detected_backbone_input_channels = int(detected_backbone_ch)
        self.backbone_input_channels = int(detected_backbone_ch or checkpoint_backbone_ch or 3)
        if checkpoint_backbone_ch is not None and int(checkpoint_backbone_ch) != self.backbone_input_channels:
            self.backbone_channel_probe_warning = (
                f"checkpoint_probe={checkpoint_backbone_ch} differs from loaded_backbone_probe={self.backbone_input_channels}; "
                "using loaded runtime backbone channel count."
            )
        else:
            self.backbone_channel_probe_warning = None

        requested_adapter = str(input_adapter or "auto").strip().lower()
        requested_adapter = "none" if requested_adapter in ("rgb", "identity") else requested_adapter
        if requested_adapter in ("", "auto"):
            self.input_adapter_mode = "none" if self.input_channels == self.backbone_input_channels else "learned_1x1"
        else:
            self.input_adapter_mode = requested_adapter

        mean_vals = _expand_channel_stats(mean, self.backbone_input_channels, IMAGENET_MEAN)
        std_vals = _expand_channel_stats(std, self.backbone_input_channels, IMAGENET_STD)
        self.register_buffer("dino_mean", torch.tensor(mean_vals, dtype=torch.float32).view(1, self.backbone_input_channels, 1, 1), persistent=False)
        self.register_buffer("dino_std", torch.tensor(std_vals, dtype=torch.float32).view(1, self.backbone_input_channels, 1, 1), persistent=False)

        if self.input_adapter_mode in ("none", "identity", "rgb", "native"):
            self.input_adapter = None
            if self.input_channels != self.backbone_input_channels:
                raise ValueError(
                    f"input_adapter=none/native requires selected input_channels ({self.input_channels}) "
                    f"to match the loaded DINOv3 backbone input channels ({self.backbone_input_channels}). "
                    "Use input_adapter=auto or learned_1x1 to learn a projection instead."
                )
        else:
            self.input_adapter = SpectralInputAdapter(
                self.input_channels,
                self.input_adapter_mode,
                output_channels=self.backbone_input_channels,
            )
        if self.decoder_type in ("linear", "simple", "head"):
            self.decoder = LinearSegmentationHead(embed_dim, num_classes, decoder_channels, dropout)
        elif self.decoder_type in ("fpn_lite", "fpn", "fpnlite"):
            self.decoder = FPNLiteSegmentationHead(embed_dim, num_classes, decoder_channels, dropout)
        else:
            raise ValueError("decoder_type must be one of: fpn_lite, linear")

    def _prepare_backbone_input(self, x: torch.Tensor) -> torch.Tensor:
        # v2.32-rs311 final safety net: re-probe the loaded runtime backbone at
        # forward time.  This avoids stale checkpoint metadata driving a wrong
        # adapter shape, for example 4 selected bands -> 3 channels while the
        # actual SAT-493M patch embedding expects 4 channels.
        runtime_ch = int(get_backbone_input_channels(self.backbone, default=self.backbone_input_channels))
        if runtime_ch != self.backbone_input_channels:
            self.backbone_input_channels = runtime_ch
            mean_vals = _expand_channel_stats(IMAGENET_MEAN, runtime_ch, IMAGENET_MEAN)
            std_vals = _expand_channel_stats(IMAGENET_STD, runtime_ch, IMAGENET_STD)
            self.dino_mean = torch.tensor(mean_vals, dtype=torch.float32).view(1, runtime_ch, 1, 1).to(x.device)
            self.dino_std = torch.tensor(std_vals, dtype=torch.float32).view(1, runtime_ch, 1, 1).to(x.device)
            if self.input_adapter is not None:
                adapter_out = getattr(self.input_adapter, "output_channels", None)
                if int(adapter_out or -1) != runtime_ch:
                    self.input_adapter = SpectralInputAdapter(
                        input_channels=self.input_channels,
                        mode=self.input_adapter_mode,
                        output_channels=runtime_ch,
                    ).to(x.device)

        if self.input_adapter is None:
            if x.shape[1] != self.backbone_input_channels:
                raise ValueError(
                    f"Expected {self.backbone_input_channels} normalized channel(s) for the loaded DINOv3 backbone, "
                    f"got {x.shape[1]} channel(s)."
                )
            return x
        if x.shape[1] != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} selected bands for input_adapter={self.input_adapter_mode}, got {x.shape[1]}.")
        # Adapter mode expects unit-scaled floating-point input.  We do not apply
        # per-tile histogram stretch here.  The adapter learns a projection from
        # selected bands to the backbone's native input channel count.
        xb = self.input_adapter(x.float().clamp(0.0, 1.0))
        xb = xb.clamp(0.0, 1.0)
        return (xb - self.dino_mean.to(dtype=xb.dtype, device=xb.device)) / self.dino_std.to(dtype=xb.dtype, device=xb.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        backbone_x = self._prepare_backbone_input(x)
        features = self.backbone.forward_features(backbone_x)
        patch_tokens = _extract_patch_tokens(features)
        if patch_tokens.ndim != 3:
            raise ValueError(f"Expected patch tokens [B, N, C], got {tuple(patch_tokens.shape)}")
        batch_size, num_tokens, channels = patch_tokens.shape
        grid_size = int(math.sqrt(num_tokens))
        if grid_size * grid_size != num_tokens:
            raise ValueError(f"Patch token count {num_tokens} is not square; cannot reshape to feature map.")
        feat_map = patch_tokens.transpose(1, 2).contiguous().view(batch_size, channels, grid_size, grid_size)
        return self.decoder(feat_map, output_size=x.shape[-2:])


# -----------------------------
# Loss and ModelExtension config
# -----------------------------


def focal_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, target, weight=weight, ignore_index=ignore_index, reduction="none")
    pt = torch.exp(-ce)
    loss = ((1.0 - pt) ** float(gamma)) * ce
    if ignore_index is not None:
        valid = target != int(ignore_index)
        if valid.any():
            return loss[valid].mean()
    return loss.mean()


class DINOv3SegmentationConfigV2:
    """ArcGIS ModelExtension config for pixel classification."""

    def __init__(self) -> None:
        self.model: Optional[nn.Module] = None
        self.loss_mode = "ce"
        self.focal_gamma = 2.0
        self.ignore_index = -100
        self.class_weights: Optional[List[float]] = None
        self.mean = list(IMAGENET_MEAN)
        self.std = list(IMAGENET_STD)
        self.input_channels = 3
        self.extract_bands = [0, 1, 2]
        self.input_adapter = "auto"
        self.training_input_scale = "fastai"
        self.max_input_value: Optional[float] = None

    def get_model(self, data: Any, backbone: Any = None, **kwargs: Any) -> nn.Module:
        repo_dir = kwargs["repo_dir"]
        backbone_weights_path = kwargs["backbone_weights_path"]
        dinov3_model_name = kwargs.get("dinov3_model_name", "dinov3_vitl16")
        embed_dim = int(kwargs.get("embed_dim", 1024))
        decoder_channels = int(kwargs.get("decoder_channels", 256))
        dropout = float(kwargs.get("dropout", 0.1))
        decoder_type = str(kwargs.get("decoder_type", "fpn_lite"))
        freeze_backbone = bool(kwargs.get("freeze_backbone", True))
        self.extract_bands = parse_sequence_int(kwargs.get("extract_bands"), [0, 1, 2])
        self.input_channels = int(kwargs.get("input_channels", len(self.extract_bands) if self.extract_bands else 3))
        self.input_adapter = resolve_input_adapter_mode(kwargs.get("input_adapter", "auto"), self.input_channels)
        self.training_input_scale = str(kwargs.get("training_input_scale", "fastai") or "fastai")
        self.max_input_value = kwargs.get("max_input_value", None)

        self.loss_mode = str(kwargs.get("loss_mode", "ce")).lower()
        self.focal_gamma = float(kwargs.get("focal_gamma", 2.0))
        self.ignore_index = int(kwargs.get("ignore_index", -100))
        self.class_weights = parse_sequence_float(kwargs.get("class_weights"))
        self.mean = parse_sequence_float(kwargs.get("mean")) or list(IMAGENET_MEAN)
        self.std = parse_sequence_float(kwargs.get("std")) or list(IMAGENET_STD)

        num_classes = int(getattr(data, "c"))
        if num_classes < 2:
            raise ValueError("Pixel classification requires at least 2 classes.")
        if self.class_weights is not None and len(self.class_weights) != num_classes:
            raise ValueError(f"class_weights length must equal num_classes={num_classes}.")

        backbone_model = load_dinov3_backbone(repo_dir, dinov3_model_name, backbone_weights_path)
        backbone_input_channels_override = infer_backbone_input_channels_from_weights(backbone_weights_path)
        model = DINOv3SegmentationNetV2(
            backbone=backbone_model,
            num_classes=num_classes,
            embed_dim=embed_dim,
            decoder_channels=decoder_channels,
            dropout=dropout,
            decoder_type=decoder_type,
            input_channels=self.input_channels,
            input_adapter=self.input_adapter,
            mean=self.mean,
            std=self.std,
            backbone_input_channels_override=backbone_input_channels_override,
        )
        probe_warning = getattr(model, "backbone_channel_probe_warning", None)
        if probe_warning:
            try:
                print(f"[DINOv3 v2.32-rs311] Backbone channel probe warning: {probe_warning}", flush=True)
            except Exception:
                pass
        if freeze_backbone:
            for p in model.backbone.parameters():
                p.requires_grad = False
        # Expose training-time spectral settings on the torch model so validation
        # metrics and deployment export can apply the same band selection.
        model.extract_bands = list(self.extract_bands)
        model.training_input_scale = str(self.training_input_scale)
        model.max_input_value = self.max_input_value
        self.input_adapter = str(model.input_adapter_mode)
        self.backbone_input_channels = int(model.backbone_input_channels)
        self.model = model
        return model

    def on_batch_begin(
        self,
        learn: Any,
        model_input_batch: torch.Tensor,
        model_target_batch: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        selected_input = select_fastai_batch_bands(
            model_input_batch,
            extract_bands=self.extract_bands,
            expected_channels=self.input_channels,
            context="ArcGIS training batch",
        )
        if self.input_adapter in ("none", "identity", "rgb"):
            x = normalize_fastai_batch_to_dino(selected_input, mean=self.mean, std=self.std)
        else:
            x = fastai_batch_to_01(selected_input, input_scale=self.training_input_scale, max_input_value=self.max_input_value)
        y = target_one_hot_to_label_map(model_target_batch)
        return x, y

    def transform_input(self, xb: torch.Tensor) -> torch.Tensor:
        selected_input = select_fastai_batch_bands(
            xb,
            extract_bands=self.extract_bands,
            expected_channels=self.input_channels,
            context="ArcGIS transform_input batch",
        )
        if self.input_adapter in ("none", "identity", "rgb"):
            return normalize_fastai_batch_to_dino(selected_input, mean=self.mean, std=self.std)
        return fastai_batch_to_01(selected_input, input_scale=self.training_input_scale, max_input_value=self.max_input_value)

    def transform_input_multispectral(self, xb: torch.Tensor) -> torch.Tensor:
        if self.input_adapter in ("none", "identity", "rgb"):
            raise ValueError("DINOv3 SAT-493M v2 legacy RGB mode requires exactly 3 selected bands. Set input_adapter=learned_1x1 or learned_3x3 for 4-band/multispectral/hyperspectral input.")
        selected_input = select_fastai_batch_bands(
            xb,
            extract_bands=self.extract_bands,
            expected_channels=self.input_channels,
            context="ArcGIS multispectral transform_input batch",
        )
        return fastai_batch_to_01(selected_input, input_scale=self.training_input_scale, max_input_value=self.max_input_value)

    def loss(self, model_output: torch.Tensor, *model_target: torch.Tensor) -> torch.Tensor:
        if len(model_target) != 1:
            raise ValueError("Expected a single segmentation target tensor.")
        target = model_target[0]
        weight_t = None
        if self.class_weights is not None:
            weight_t = torch.tensor(self.class_weights, dtype=model_output.dtype, device=model_output.device)
        if self.loss_mode in ("focal", "focal_ce", "focal_loss"):
            return focal_cross_entropy(model_output, target, weight=weight_t, gamma=self.focal_gamma, ignore_index=self.ignore_index)
        return F.cross_entropy(model_output, target, weight=weight_t, ignore_index=self.ignore_index)

    def post_process(self, pred: torch.Tensor, thres: float) -> torch.Tensor:
        return torch.argmax(pred, dim=1, keepdim=True).long()


# -----------------------------
# Metrics
# -----------------------------


@dataclass
class MetricSummary:
    overall_accuracy: float
    mean_iou: float
    mean_f1: float
    per_class: List[Dict[str, Any]]
    confusion_matrix: List[List[int]]


def _update_confusion_matrix(conf: np.ndarray, pred: np.ndarray, target: np.ndarray, num_classes: int, ignore_index: int) -> None:
    mask = target != int(ignore_index)
    mask &= target >= 0
    mask &= target < num_classes
    if not np.any(mask):
        return
    pred = pred[mask].astype(np.int64)
    target = target[mask].astype(np.int64)
    pred = np.clip(pred, 0, num_classes - 1)
    k = target * num_classes + pred
    bincount = np.bincount(k, minlength=num_classes * num_classes)
    conf += bincount.reshape(num_classes, num_classes)


def _metrics_from_confusion(conf: np.ndarray, classes: Sequence[Dict[str, Any]]) -> MetricSummary:
    eps = 1e-12
    num_classes = conf.shape[0]
    total = conf.sum()
    oa = float(np.trace(conf) / max(total, eps))
    per_class = []
    ious = []
    f1s = []
    for i in range(num_classes):
        tp = float(conf[i, i])
        fp = float(conf[:, i].sum() - tp)
        fn = float(conf[i, :].sum() - tp)
        precision = tp / max(tp + fp, eps)
        recall = tp / max(tp + fn, eps)
        iou = tp / max(tp + fp + fn, eps)
        f1 = 2.0 * precision * recall / max(precision + recall, eps)
        support = int(conf[i, :].sum())
        ious.append(iou)
        f1s.append(f1)
        per_class.append({
            "value": int(classes[i].get("Value", i)) if i < len(classes) else i,
            "name": str(classes[i].get("Name", f"class_{i}")) if i < len(classes) else f"class_{i}",
            "support_pixels": support,
            "precision": precision,
            "recall": recall,
            "iou": iou,
            "f1": f1,
        })
    return MetricSummary(
        overall_accuracy=oa,
        mean_iou=float(np.nanmean(ious)) if ious else 0.0,
        mean_f1=float(np.nanmean(f1s)) if f1s else 0.0,
        per_class=per_class,
        confusion_matrix=conf.astype(int).tolist(),
    )


def _get_valid_dl(data: Any) -> Optional[Iterable[Any]]:
    for attr in ("valid_dl", "val_dl"):
        if hasattr(data, attr):
            return getattr(data, attr)
    if hasattr(data, "dls") and hasattr(data.dls, "valid"):
        return data.dls.valid
    if hasattr(data, "valid_ds"):
        return None
    return None


def evaluate_model_on_validation(
    model: nn.Module,
    data: Any,
    classes: Sequence[Dict[str, Any]],
    ignore_index: int = -100,
    max_batches: Optional[int] = None,
) -> Optional[MetricSummary]:
    valid_dl = _get_valid_dl(data)
    if valid_dl is None:
        return None
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    conf = np.zeros((len(classes), len(classes)), dtype=np.int64)
    with torch.inference_mode():
        for batch_idx, batch in enumerate(valid_dl):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                xb, yb = batch[0], batch[1]
            else:
                continue
            if isinstance(xb, (list, tuple)):
                xb = xb[0]
            if isinstance(yb, (list, tuple)):
                yb = yb[0]
            xb = xb.to(device)
            yb = yb.to(device)
            selected_xb = select_fastai_batch_bands(
                xb,
                extract_bands=getattr(model, "extract_bands", None),
                expected_channels=getattr(model, "input_channels", None),
                context="ArcGIS validation batch",
            )
            if getattr(model, "input_adapter", None) is None:
                x = normalize_fastai_batch_to_dino(selected_xb)
            else:
                x = fastai_batch_to_01(
                    selected_xb,
                    input_scale=getattr(model, "training_input_scale", "fastai"),
                    max_input_value=getattr(model, "max_input_value", None),
                )
            y = target_one_hot_to_label_map(yb)
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
            _update_confusion_matrix(conf, pred.cpu().numpy(), y.cpu().numpy(), len(classes), ignore_index)
    if was_training:
        model.train()
    return _metrics_from_confusion(conf, classes)


def write_metrics(metrics: Optional[MetricSummary], output_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    if metrics is None:
        return None, None
    json_path = output_dir / "validation_metrics.json"
    csv_path = output_dir / "per_class_metrics.csv"
    _write_json(json_path, asdict(metrics))
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["value", "name", "support_pixels", "precision", "recall", "iou", "f1"])
        writer.writeheader()
        for row in metrics.per_class:
            writer.writerow(row)
    return json_path, csv_path


# -----------------------------
# Packaging
# -----------------------------


@dataclass
class TrainingOutputs:
    model_dir: Path
    training_save_dir: Path
    deployment_dir: Path
    deployment_checkpoint: Path
    emd_path: Path
    dlpk_path: Path
    manifest_path: Path
    metrics_json: Optional[Path] = None
    metrics_csv: Optional[Path] = None
    epoch_log_csv: Optional[Path] = None


def _snapshot_repo(source_repo_dir: Path, target_repo_dir: Path, aggressive_trim: bool = True) -> None:
    if target_repo_dir.exists():
        shutil.rmtree(target_repo_dir)
    ignore_patterns = [
        ".git", ".github", "__pycache__", "*.pyc", "*.pyo", ".ipynb_checkpoints",
        "notebooks", "docs", "*.md", "*.png", "*.jpg", "*.jpeg", "*.gif", "*.ipynb",
    ]
    if not aggressive_trim:
        ignore_patterns = [".git", ".github", "__pycache__", "*.pyc", "*.pyo", ".ipynb_checkpoints"]
    shutil.copytree(source_repo_dir, target_repo_dir, ignore=shutil.ignore_patterns(*ignore_patterns))


def _zip_dir_to_dlpk(folder: Path, dlpk_path: Path) -> None:
    if dlpk_path.exists():
        dlpk_path.unlink()
    zip_tmp = dlpk_path.parent.parent / f"{dlpk_path.stem}__tmp.zip"
    if zip_tmp.exists():
        zip_tmp.unlink()
    with zipfile.ZipFile(zip_tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in folder.rglob("*"):
            if not item.is_file():
                continue
            if item.resolve() == dlpk_path.resolve():
                continue
            if item.suffix.lower() == ".dlpk":
                continue
            zf.write(item, arcname=item.relative_to(folder).as_posix())
    zip_tmp.replace(dlpk_path)




def write_model_configuration_py(deployment_dir: Path) -> Path:
    """Write a minimal, self-contained ArcGIS ModelExtension configuration shim."""
    path = deployment_dir / "ModelConfiguration.py"
    content = (
        "# Auto-generated by DINOv3 ArcGIS Pro Segmentation Toolkit v2.32-rs311\n"
        "# ArcGIS Pro imports this file when initializing the DLPK.\n"
        "import os\n"
        "import sys\n"
        "\n"
        "_HERE = os.path.dirname(os.path.abspath(__file__))\n"
        "if _HERE not in sys.path:\n"
        "    sys.path.insert(0, _HERE)\n"
        "\n"
        "from dinov3_arcgis_v2 import DINOv3SegmentationConfigV2  # noqa: F401\n"
    )
    path.write_text(content, encoding="utf-8")
    return path

def build_custom_emd_v2(
    *,
    deployment_checkpoint_name: str,
    backbone_weights_name: str,
    classes: List[Dict[str, Any]],
    chip_size: int,
    batch_size: int,
    model_name: str,
    repo_dir_name: str,
    dinov3_model_name: str,
    embed_dim: int,
    decoder_channels: int,
    dropout: float,
    decoder_type: str,
    freeze_backbone: bool,
    extract_bands: Sequence[int],
    input_scale: str,
    max_input_value: Optional[float],
    input_channels: int,
    backbone_input_channels: Optional[int] = None,
    input_adapter: str = "auto",
    training_input_scale: str,
    loss_mode: str,
    class_weights: Optional[Sequence[float]],
    focal_gamma: float,
    ignore_index: int,
    model_padding: int,
    confidence_threshold: float,
) -> Dict[str, Any]:
    emd = {
        "Framework": "PyTorch",
        "ModelConfiguration": "Custom",
        "ModelType": "ImageClassification",
        "ModelFile": f".\\{deployment_checkpoint_name}",
        "InferenceFunction": ".\\dinov3_inference_v2.py",
        "ImageHeight": int(chip_size),
        "ImageWidth": int(chip_size),
        "ExtractBands": [int(b) for b in extract_bands],
        "DataRange": [0.0, 1.0],
        "BatchSize": int(batch_size),
        "ModelPadding": int(model_padding),
        "Classes": classes,
        "ModelName": model_name,
        "Version": "2026.05.v2.32.1-rs311",
        "BackboneRepoDir": f".\\{repo_dir_name}",
        "BackboneWeights": f".\\{backbone_weights_name}",
        "DINOv3ModelName": dinov3_model_name,
        "EmbedDim": int(embed_dim),
        "DecoderChannels": int(decoder_channels),
        "DecoderType": str(decoder_type),
        "Dropout": float(dropout),
        "FreezeBackbone": bool(freeze_backbone),
        "Mean": list(IMAGENET_MEAN),
        "StdDev": list(IMAGENET_STD),
        "InputScale": str(input_scale),
        "MaxInputValue": None if max_input_value is None else float(max_input_value),
        "InputChannels": int(input_channels),
        "BackboneInputChannels": int(backbone_input_channels or input_channels),
        "InputAdapter": str(input_adapter),
        "TrainingInputScale": str(training_input_scale),
        "InputRadiometryNote": "Use explicit 0_65535 for raw 16-bit DN. This is fixed numeric normalization, not per-chip display stretching.",
        "LossMode": str(loss_mode),
        "ClassWeights": None if class_weights is None else [float(x) for x in class_weights],
        "FocalGamma": float(focal_gamma),
        "IgnoreIndex": int(ignore_index),
        "ConfidenceThreshold": float(confidence_threshold),
        "ClassValueIsContiguousZeroBased": contiguous_training_class_values(classes),
    }
    return emd


def build_manifest(
    *,
    model_name: str,
    training_data_dir: Path,
    repo_dir: Path,
    backbone_weights_path: Path,
    checkpoint_path: Path,
    emd_path: Path,
    hyperparams: Dict[str, Any],
    metrics: Optional[MetricSummary],
) -> Dict[str, Any]:
    return {
        "toolkit": "DINOv3 ArcGIS Pro Segmentation Toolkit",
        "toolkit_version": "v2.32.1-rs311-unified-hotfix",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_name": model_name,
        "python": sys.version,
        "python_version_info": [int(sys.version_info.major), int(sys.version_info.minor), int(sys.version_info.micro)],
        "python_ge_311": bool(sys.version_info >= (3, 11)),
        "recommended_external_env": "dinov3-rs311 / Python 3.11+",
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "training_data_dir": str(training_data_dir),
        "repo_dir_source": str(repo_dir),
        "backbone_weights_source": str(backbone_weights_path),
        "files": {
            checkpoint_path.name: _sha256(checkpoint_path),
            emd_path.name: _sha256(emd_path),
            backbone_weights_path.name: _sha256(backbone_weights_path),
        },
        "hyperparameters": hyperparams,
        "validation_summary": None if metrics is None else {
            "overall_accuracy": metrics.overall_accuracy,
            "mean_iou": metrics.mean_iou,
            "mean_f1": metrics.mean_f1,
        },
    }


def export_custom_deployment_package_v2(
    *,
    output_root: Path,
    training_data_dir: Path,
    model_name: str,
    repo_dir: Path,
    backbone_weights_path: Path,
    trained_model: nn.Module,
    classes: List[Dict[str, Any]],
    chip_size: int,
    batch_size: int,
    dinov3_model_name: str,
    embed_dim: int,
    decoder_channels: int,
    dropout: float,
    decoder_type: str,
    freeze_backbone: bool,
    extract_bands: Sequence[int],
    input_scale: str,
    max_input_value: Optional[float],
    input_channels: int,
    input_adapter: str,
    training_input_scale: str,
    loss_mode: str,
    class_weights: Optional[Sequence[float]],
    focal_gamma: float,
    ignore_index: int,
    model_padding: int,
    confidence_threshold: float,
    metrics: Optional[MetricSummary],
    training_data_source_dirs: Optional[Sequence[Path]] = None,
    aggressive_repo_trim: bool = True,
) -> Tuple[Path, Path, Path, Path]:
    deployment_dir = output_root / f"{model_name}_deployment"
    if deployment_dir.exists():
        shutil.rmtree(deployment_dir)
    deployment_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_name = f"{model_name}.pth"
    checkpoint_path = deployment_dir / checkpoint_name
    torch.save({
        "state_dict": trained_model.state_dict(),
        "num_classes": len(classes),
        "classes": classes,
        "chip_size": int(chip_size),
        "dinov3_model_name": dinov3_model_name,
        "embed_dim": int(embed_dim),
        "decoder_channels": int(decoder_channels),
        "decoder_type": str(decoder_type),
        "dropout": float(dropout),
        "freeze_backbone": bool(freeze_backbone),
        "input_channels": int(input_channels),
        "backbone_input_channels": int(getattr(trained_model, "backbone_input_channels", input_channels)),
        "input_adapter": str(input_adapter),
        "training_input_scale": str(training_input_scale),
    }, checkpoint_path)

    backbone_weights_name = backbone_weights_path.name
    shutil.copy2(backbone_weights_path, deployment_dir / backbone_weights_name)

    local_support_dir = Path(__file__).resolve().parent
    for support_name in ("dinov3_arcgis_v2.py", "dinov3_inference_v2.py", "README_v2.md"):
        support_path = local_support_dir / support_name
        if support_path.exists():
            shutil.copy2(support_path, deployment_dir / support_name)

    repo_snapshot_dir = deployment_dir / "dinov3_repo"
    _snapshot_repo(repo_dir, repo_snapshot_dir, aggressive_trim=aggressive_repo_trim)

    emd = build_custom_emd_v2(
        deployment_checkpoint_name=checkpoint_name,
        backbone_weights_name=backbone_weights_name,
        classes=classes,
        chip_size=chip_size,
        batch_size=batch_size,
        model_name=model_name,
        repo_dir_name="dinov3_repo",
        dinov3_model_name=dinov3_model_name,
        embed_dim=embed_dim,
        decoder_channels=decoder_channels,
        dropout=dropout,
        decoder_type=decoder_type,
        freeze_backbone=freeze_backbone,
        extract_bands=extract_bands,
        input_scale=input_scale,
        max_input_value=max_input_value,
        input_channels=input_channels,
        backbone_input_channels=int(getattr(trained_model, "backbone_input_channels", input_channels)),
        input_adapter=input_adapter,
        training_input_scale=training_input_scale,
        loss_mode=loss_mode,
        class_weights=class_weights,
        focal_gamma=focal_gamma,
        ignore_index=ignore_index,
        model_padding=model_padding,
        confidence_threshold=confidence_threshold,
    )
    emd_path = deployment_dir / f"{model_name}.emd"
    _write_json(emd_path, emd)

    hyperparams = {
        "chip_size": chip_size,
        "batch_size": batch_size,
        "dinov3_model_name": dinov3_model_name,
        "embed_dim": embed_dim,
        "decoder_channels": decoder_channels,
        "decoder_type": decoder_type,
        "dropout": dropout,
        "freeze_backbone": freeze_backbone,
        "extract_bands": list(extract_bands),
        "input_scale": input_scale,
        "max_input_value": max_input_value,
        "input_channels": int(input_channels),
        "backbone_input_channels": int(getattr(trained_model, "backbone_input_channels", input_channels)),
        "input_adapter": str(input_adapter),
        "training_input_scale": str(training_input_scale),
        "loss_mode": loss_mode,
        "class_weights": None if class_weights is None else list(class_weights),
        "focal_gamma": focal_gamma,
        "ignore_index": ignore_index,
        "model_padding": model_padding,
        "confidence_threshold": confidence_threshold,
        "training_data_source_dirs": [str(p) for p in (training_data_source_dirs or [training_data_dir])],
        "effective_training_data_dir": str(training_data_dir),
        "training_data_source_count": len(training_data_source_dirs or [training_data_dir]),
    }
    manifest = build_manifest(
        model_name=model_name,
        training_data_dir=training_data_dir,
        repo_dir=repo_dir,
        backbone_weights_path=backbone_weights_path,
        checkpoint_path=checkpoint_path,
        emd_path=emd_path,
        hyperparams=hyperparams,
        metrics=metrics,
    )
    manifest_path = deployment_dir / "manifest.json"
    _write_json(manifest_path, manifest)

    model_config_path = write_model_configuration_py(deployment_dir)
    try:
        manifest["files"][model_config_path.name] = _sha256(model_config_path)
        _write_json(manifest_path, manifest)
    except Exception:
        pass

    dlpk_path = deployment_dir / f"{model_name}.dlpk"
    _zip_dir_to_dlpk(deployment_dir, dlpk_path)
    return deployment_dir, emd_path, dlpk_path, manifest_path


# -----------------------------
# Environment validation and train orchestration
# -----------------------------


def verify_runtime_environment(
    repo_dir: PathLike,
    backbone_weights_path: PathLike,
    dinov3_model_name: str = "dinov3_vitl16",
    embed_dim: int = 1024,
    chip_size: int = 224,
) -> Dict[str, Any]:
    repo_dir = _as_path(repo_dir)
    backbone_weights_path = _as_path(backbone_weights_path)
    _ensure_multiple_of_16(chip_size)
    report: Dict[str, Any] = {
        "python": sys.version,
        "python_version_info": [int(sys.version_info.major), int(sys.version_info.minor), int(sys.version_info.micro)],
        "python_ge_311": bool(sys.version_info >= (3, 11)),
        "recommended_external_env": "dinov3-rs311 / Python 3.11+",
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "repo_dir": str(repo_dir),
        "repo_exists": repo_dir.exists(),
        "weights": str(backbone_weights_path),
        "weights_exists": backbone_weights_path.exists(),
        "dinov3_model_name": dinov3_model_name,
        "forward_test": False,
        "error": None,
    }
    try:
        if sys.version_info < (3, 11):
            raise RuntimeError(
                "The official DINOv3 package requires Python >= 3.11. "
                "Create/use a Python 3.11+ validation environment such as dinov3-rs311, "
                "or run this check from ArcGIS Pro 3.6+ Python where applicable."
            )
        if not repo_dir.exists():
            raise FileNotFoundError(f"DINOv3 repository folder not found: {repo_dir}")
        if not backbone_weights_path.exists():
            raise FileNotFoundError(f"Backbone weights not found: {backbone_weights_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backbone = load_dinov3_backbone(repo_dir, dinov3_model_name, backbone_weights_path).to(device).eval()
        backbone_input_channels_override = infer_backbone_input_channels_from_weights(backbone_weights_path)
        net = DINOv3SegmentationNetV2(
            backbone,
            num_classes=2,
            embed_dim=embed_dim,
            decoder_type="linear",
            input_channels=3,
            input_adapter="auto",
            backbone_input_channels_override=backbone_input_channels_override,
        ).to(device).eval()
        report["backbone_input_channels"] = int(net.backbone_input_channels)
        report["input_adapter"] = str(net.input_adapter_mode)
        dummy = torch.zeros(1, int(net.input_channels), int(chip_size), int(chip_size), device=device)
        with torch.inference_mode():
            out = net(dummy)
        report["forward_test"] = tuple(out.shape) == (1, 2, int(chip_size), int(chip_size))
        report["output_shape"] = list(out.shape)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    return report


def _classes_signature(classes: Sequence[Dict[str, Any]]) -> List[Tuple[int, str]]:
    signature: List[Tuple[int, str]] = []
    for idx, cls in enumerate(classes):
        value = cls.get("Value", idx)
        try:
            value = int(value)
        except Exception:
            value = idx
        signature.append((value, str(cls.get("Name", f"class_{idx}")).strip()))
    return signature


def _copy_training_subtree_with_prefix(src_root: Path, dst_root: Path, prefix: str) -> int:
    copied = 0
    for src_file in src_root.rglob("*"):
        if not src_file.is_file():
            continue
        rel = src_file.relative_to(src_root)
        # Root EMD files are handled separately so the merged folder has one authoritative template.
        if len(rel.parts) == 1 and src_file.suffix.lower() == ".emd":
            continue
        if len(rel.parts) == 1:
            # Preserve first root metadata filenames when possible, prefix later duplicates.
            dst = dst_root / rel.name
            if dst.exists():
                dst = dst_root / f"{prefix}{rel.name}"
        else:
            dst = dst_root.joinpath(*rel.parts[:-1]) / f"{prefix}{rel.name}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst)
        copied += 1
    return copied


def _build_merged_training_data_folder(
    training_data_dirs: Sequence[Path],
    staging_dir: Path,
    log_callback: Optional[Callable[[str], None]] = None,
) -> Path:
    """Create a single-folder staging dataset from multiple ArcGIS training-data exports.

    This fallback is used only when the installed arcgis.learn.prepare_data does not
    accept a list of folders directly. It is safest when all selected folders use
    the same class schema, or use non-conflicting class values. It does not remap
    pixel label values.
    """
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    class_by_value: Dict[int, str] = {}
    class_records: Dict[int, Dict[str, Any]] = {}
    source_reports: List[Dict[str, Any]] = []
    template_emd: Optional[Path] = None

    for idx, src_dir in enumerate(training_data_dirs, start=1):
        if not src_dir.exists():
            raise FileNotFoundError(f"Training data folder not found: {src_dir}")
        src_emd = _find_template_emd(src_dir)
        classes = _list_classes_from_template(src_emd)
        for value, name in _classes_signature(classes):
            # Empty / unnamed values are ignored. Conflicting labels are unsafe without mask remapping.
            if value in class_by_value and class_by_value[value] != name:
                raise ValueError(
                    "Selected training data folders contain conflicting class schemas: "
                    f"Class Value {value} is both '{class_by_value[value]}' and '{name}'. "
                    "Please re-export the folders with a consistent class map, or train these datasets separately. "
                    "This toolkit does not automatically remap label-mask pixel values."
                )
            class_by_value[value] = name
            class_records[value] = {
                "Value": value,
                "Name": name,
                "Color": next((c.get("Color") for c in classes if int(c.get("Value", -9999)) == value), DEFAULT_COLORS[value % len(DEFAULT_COLORS)]),
            }
        if template_emd is None and src_emd is not None:
            template_emd = src_emd

        prefix = f"src{idx:02d}_{_safe_model_name(src_dir.name)}_"
        copied = _copy_training_subtree_with_prefix(src_dir, staging_dir, prefix)
        source_reports.append({
            "source_index": idx,
            "source_dir": str(src_dir),
            "template_emd": None if src_emd is None else str(src_emd),
            "classes": classes,
            "copied_files": copied,
        })

    if template_emd is not None:
        merged_emd = _load_json(template_emd)
        if class_records:
            merged_emd["Classes"] = [class_records[k] for k in sorted(class_records)]
        _write_json(staging_dir / template_emd.name, merged_emd)

    _write_json(staging_dir / "dinov3_multi_folder_sources.json", {
        "mode": "merged_training_data_fallback",
        "note": "Generated by DINOv3 ArcGIS toolkit. Pixel label values are copied as-is; class maps must be consistent.",
        "sources": source_reports,
    })
    _emit_training_message(
        f"Merged {len(training_data_dirs)} training data folders into staging folder: {staging_dir}",
        log_callback,
    )
    return staging_dir


def _prepare_data_single_folder(
    prepare_data_fn: Callable[..., Any],
    training_data_dir: Path,
    batch_size: int,
    chip_size: int,
    val_split_pct: float,
    seed: int,
) -> Any:
    try:
        return prepare_data_fn(
            path=str(training_data_dir),
            batch_size=int(batch_size),
            chip_size=int(chip_size),
            val_split_pct=float(val_split_pct),
            seed=int(seed),
        )
    except TypeError:
        # Some ArcGIS/arcgis.learn builds expose fewer prepare_data parameters.
        return prepare_data_fn(path=str(training_data_dir), batch_size=int(batch_size), chip_size=int(chip_size))


def _prepare_data_with_fallback(
    training_data_dirs: Sequence[Path],
    batch_size: int,
    chip_size: int,
    val_split_pct: float,
    seed: int,
    staging_dir: Optional[Path] = None,
    log_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[Any, Path]:
    from arcgis.learn import prepare_data

    if not training_data_dirs:
        raise ValueError("At least one training data folder is required.")

    if len(training_data_dirs) == 1:
        data = _prepare_data_single_folder(prepare_data, training_data_dirs[0], batch_size, chip_size, val_split_pct, seed)
        return data, training_data_dirs[0]

    # Newer ArcGIS/arcgis.learn builds may support list-like input. Try this first because
    # it best matches ArcGIS Pro's native multi-folder training tool behavior.
    try:
        _emit_training_message(
            f"Preparing ArcGIS training data from {len(training_data_dirs)} folders using list-path mode...",
            log_callback,
        )
        try:
            data = prepare_data(
                path=[str(p) for p in training_data_dirs],
                batch_size=int(batch_size),
                chip_size=int(chip_size),
                val_split_pct=float(val_split_pct),
                seed=int(seed),
            )
        except TypeError:
            data = prepare_data(path=[str(p) for p in training_data_dirs], batch_size=int(batch_size), chip_size=int(chip_size))
        return data, training_data_dirs[0]
    except Exception as exc:
        if staging_dir is None:
            raise
        _emit_training_message(
            "arcgis.learn.prepare_data list-path mode failed; building merged staging folder. "
            f"Reason: {type(exc).__name__}: {exc}",
            log_callback,
        )
        merged_dir = _build_merged_training_data_folder(training_data_dirs, staging_dir, log_callback)
        data = _prepare_data_single_folder(prepare_data, merged_dir, batch_size, chip_size, val_split_pct, seed)
        return data, merged_dir


def _format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h:
        return f"{h:d}h {m:02d}m {s:05.2f}s"
    if m:
        return f"{m:d}m {s:05.2f}s"
    return f"{s:.2f}s"


def _emit_training_message(message: str, log_callback: Optional[Callable[[str], None]] = None) -> None:
    text = f"[DINOv3 v2.32-rs311] {message}"
    try:
        print(text, flush=True)
    except Exception:
        pass
    if log_callback is not None:
        try:
            log_callback(text)
        except Exception:
            pass


def _append_epoch_log_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch", "n_epochs", "epoch_seconds", "total_seconds",
        "train_loss", "valid_loss", "metric_1", "metric_2", "metric_3",
        "message",
    ]
    write_header = not path.exists()
    safe_row = {k: row.get(k, "") for k in fieldnames}
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(safe_row)


def _make_epoch_progress_callback(
    *,
    n_epochs: int,
    epoch_log_csv: Path,
    log_callback: Optional[Callable[[str], None]] = None,
) -> Any:
    try:
        from fastai.callback.core import Callback
    except Exception:
        from fastai.callbacks import Callback  # type: ignore

    class EpochProgressCallback(Callback):
        order = 95

        def before_fit(self):
            self._fit_start_time = time.perf_counter()
            self._epoch_start_time = None
            _emit_training_message(
                f"Training started | epochs={int(n_epochs)} | progress CSV={epoch_log_csv}",
                log_callback,
            )

        def before_epoch(self):
            self._epoch_start_time = time.perf_counter()
            epoch_idx = int(getattr(self, "epoch", 0)) + 1
            _emit_training_message(f"Epoch {epoch_idx}/{int(n_epochs)} started", log_callback)

        def after_epoch(self):
            now = time.perf_counter()
            fit_start = float(getattr(self, "_fit_start_time", now))
            epoch_start = getattr(self, "_epoch_start_time", None)
            epoch_seconds = now - float(epoch_start if epoch_start is not None else now)
            total_seconds = now - fit_start
            epoch_idx = int(getattr(self, "epoch", 0)) + 1

            values: List[Any] = []
            metric_names: List[str] = []
            try:
                recorder = getattr(self.learn, "recorder", None)
                raw_values = getattr(recorder, "values", []) if recorder is not None else []
                if raw_values:
                    values = list(raw_values[-1])
                raw_names = getattr(recorder, "metric_names", []) if recorder is not None else []
                metric_names = [str(x) for x in raw_names]
                if metric_names and metric_names[0].lower() == "epoch":
                    metric_names = metric_names[1:]
                if metric_names and metric_names[-1].lower() == "time":
                    metric_names = metric_names[:-1]
            except Exception:
                values = []
                metric_names = []

            # Common fastai order is train_loss, valid_loss, then metrics. Keep fallback labels stable.
            labels = metric_names if metric_names and len(metric_names) == len(values) else []
            if not labels:
                base = ["train_loss", "valid_loss", "metric_1", "metric_2", "metric_3"]
                labels = base[: len(values)]

            metrics: Dict[str, Any] = {}
            for k, v in zip(labels, values):
                try:
                    metrics[str(k)] = float(v)
                except Exception:
                    metrics[str(k)] = str(v)

            train_loss = metrics.get("train_loss", values[0] if len(values) > 0 else "")
            valid_loss = metrics.get("valid_loss", values[1] if len(values) > 1 else "")
            metric_values = values[2:] if len(values) > 2 else []

            def fmt(v: Any) -> str:
                try:
                    return f"{float(v):.6f}"
                except Exception:
                    return str(v) if v not in (None, "") else "NA"

            detail_parts = [
                f"Epoch {epoch_idx}/{int(n_epochs)} finished",
                f"epoch_time={_format_seconds(epoch_seconds)}",
                f"total_time={_format_seconds(total_seconds)}",
                f"train_loss={fmt(train_loss)}",
                f"valid_loss={fmt(valid_loss)}",
            ]
            for i, v in enumerate(metric_values[:3], start=1):
                detail_parts.append(f"metric_{i}={fmt(v)}")
            message = " | ".join(detail_parts)
            _emit_training_message(message, log_callback)

            row = {
                "epoch": epoch_idx,
                "n_epochs": int(n_epochs),
                "epoch_seconds": round(epoch_seconds, 3),
                "total_seconds": round(total_seconds, 3),
                "train_loss": fmt(train_loss),
                "valid_loss": fmt(valid_loss),
                "metric_1": fmt(metric_values[0]) if len(metric_values) > 0 else "",
                "metric_2": fmt(metric_values[1]) if len(metric_values) > 1 else "",
                "metric_3": fmt(metric_values[2]) if len(metric_values) > 2 else "",
                "message": message,
            }
            try:
                _append_epoch_log_csv(epoch_log_csv, row)
            except Exception as exc:
                _emit_training_message(f"Could not append epoch log CSV: {type(exc).__name__}: {exc}", log_callback)

        def after_fit(self):
            total_seconds = time.perf_counter() - float(getattr(self, "_fit_start_time", time.perf_counter()))
            _emit_training_message(f"Training finished | total_time={_format_seconds(total_seconds)}", log_callback)

    return EpochProgressCallback()


def _fit_model_extension(
    model: Any,
    epochs: int,
    learning_rate: float,
    use_early_stopping: bool,
    early_stopping_patience: int,
    epoch_log_csv: Optional[Path] = None,
    log_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    train_report: Dict[str, Any] = {
        "method": "arcgis_model.fit",
        "callbacks_used": False,
        "progress_callback_used": False,
        "epoch_log_csv": str(epoch_log_csv) if epoch_log_csv else None,
        "warning": None,
    }
    learn = getattr(model, "learn", None)
    if epoch_log_csv is None:
        epoch_log_csv = Path.cwd() / "dinov3_epoch_log.csv"

    if learn is not None:
        try:
            callbacks = [_make_epoch_progress_callback(n_epochs=int(epochs), epoch_log_csv=epoch_log_csv, log_callback=log_callback)]
            if use_early_stopping:
                try:
                    from fastai.callback.tracker import EarlyStoppingCallback, SaveModelCallback
                    callbacks.append(EarlyStoppingCallback(monitor="valid_loss", patience=int(early_stopping_patience)))
                    callbacks.append(SaveModelCallback(monitor="valid_loss", fname="dinov3_v2_best"))
                except Exception:
                    from fastai.callbacks import EarlyStoppingCallback, SaveModelCallback  # type: ignore
                    callbacks.append(EarlyStoppingCallback(learn, monitor="valid_loss", patience=int(early_stopping_patience)))
                    callbacks.append(SaveModelCallback(learn, monitor="valid_loss", name="dinov3_v2_best"))

            _emit_training_message(
                f"Using fastai learner path | lr={float(learning_rate):.6g} | early_stopping={bool(use_early_stopping)}",
                log_callback,
            )
            if hasattr(learn, "fit_one_cycle"):
                try:
                    learn.fit_one_cycle(int(epochs), lr_max=float(learning_rate), cbs=callbacks)
                except TypeError:
                    learn.fit_one_cycle(int(epochs), float(learning_rate), callbacks=callbacks)
            else:
                try:
                    learn.fit(int(epochs), lr=float(learning_rate), cbs=callbacks)
                except TypeError:
                    learn.fit(int(epochs), float(learning_rate), callbacks=callbacks)

            train_report["method"] = "fastai learner with per-epoch progress callback"
            train_report["callbacks_used"] = True
            train_report["progress_callback_used"] = True
            if use_early_stopping:
                try:
                    learn.load("dinov3_v2_best")
                    train_report["best_model_loaded"] = True
                except Exception:
                    train_report["best_model_loaded"] = False
            return train_report
        except Exception as exc:
            train_report["warning"] = (
                "fastai callback path failed; fell back to arcgis ModelExtension.fit. "
                f"Reason: {type(exc).__name__}: {exc}"
            )
            _emit_training_message(train_report["warning"], log_callback)

    _emit_training_message(
        "Using fallback arcgis ModelExtension.fit path. Per-epoch progress may be limited by ArcGIS/arcgis.learn output buffering.",
        log_callback,
    )
    model.fit(epochs=int(epochs), lr=float(learning_rate))
    return train_report


def train_and_package_v2(
    *,
    training_data_dir: PathLike,
    output_root: PathLike,
    repo_dir: PathLike,
    backbone_weights_path: PathLike,
    model_name: str = "dinov3_sat493m_seg_v2",
    chip_size: int = 224,
    batch_size: int = 4,
    epochs: int = 30,
    learning_rate: float = 1e-4,
    freeze_backbone: bool = True,
    decoder_channels: int = 256,
    decoder_type: str = "fpn_lite",
    dropout: float = 0.1,
    dinov3_model_name: str = "dinov3_vitl16",
    embed_dim: int = 1024,
    loss_mode: str = "ce",
    class_weights: Optional[Union[str, Sequence[float]]] = None,
    focal_gamma: float = 2.0,
    ignore_index: int = -100,
    val_split_pct: float = 0.2,
    seed: int = 42,
    use_early_stopping: bool = True,
    early_stopping_patience: int = 5,
    evaluate_after_training: bool = True,
    max_validation_batches: Optional[int] = None,
    extract_bands: Optional[Union[str, Sequence[int]]] = None,
    input_scale: str = "auto",
    max_input_value: Optional[float] = None,
    input_channels: Optional[int] = None,
    input_adapter: str = "auto",
    training_input_scale: str = "fastai",
    model_padding: int = 0,
    confidence_threshold: float = 0.0,
    run_environment_check: bool = True,
    aggressive_repo_trim: bool = True,
    log_callback: Optional[Callable[[str], None]] = None,
) -> TrainingOutputs:
    from arcgis.learn import ModelExtension

    training_data_dirs = _as_path_list(training_data_dir)
    if not training_data_dirs:
        raise ValueError("At least one training data folder is required.")
    training_data_dir = training_data_dirs[0]
    output_root = _as_path(output_root)
    repo_dir = _as_path(repo_dir)
    backbone_weights_path = _as_path(backbone_weights_path)
    extract_bands_list = parse_sequence_int(extract_bands, [0, 1, 2])
    class_weights_list = parse_sequence_float(class_weights)
    if input_channels is None:
        input_channels = len(extract_bands_list)
    input_channels = int(input_channels)
    input_adapter = str(input_adapter or "auto").strip().lower()
    if input_adapter in ("rgb", "identity"):
        input_adapter = "none"

    _ensure_multiple_of_16(chip_size)
    if len(extract_bands_list) != input_channels:
        raise ValueError(f"Extract Bands count ({len(extract_bands_list)}) must equal input_channels ({input_channels}).")
    for _src_training_dir in training_data_dirs:
        if not _src_training_dir.exists():
            raise FileNotFoundError(f"Training data folder not found: {_src_training_dir}")
    if not repo_dir.exists():
        raise FileNotFoundError(f"DINOv3 repository folder not found: {repo_dir}")
    if not backbone_weights_path.exists():
        raise FileNotFoundError(f"Backbone weights file not found: {backbone_weights_path}")

    model_name = _safe_model_name(model_name)
    output_root.mkdir(parents=True, exist_ok=True)
    run_log_dir = output_root / f"{model_name}_training_logs"
    run_log_dir.mkdir(parents=True, exist_ok=True)

    _emit_training_message(f"Run folder: {run_log_dir}", log_callback)
    _emit_training_message(
        "Training data folders: " + "; ".join(str(p) for p in training_data_dirs),
        log_callback,
    )
    _emit_training_message(
        f"Spectral input: ExtractBands={extract_bands_list} | input_channels={input_channels} | input_adapter={input_adapter} | inference_input_scale={input_scale} | training_input_scale={training_input_scale}",
        log_callback,
    )

    if run_environment_check:
        _emit_training_message("Running environment check before training...", log_callback)
        env_report = verify_runtime_environment(repo_dir, backbone_weights_path, dinov3_model_name, embed_dim, chip_size)
        _write_json(run_log_dir / "environment_check.json", env_report)
        if not env_report.get("forward_test"):
            raise RuntimeError(f"DINOv3 environment check failed. See {run_log_dir / 'environment_check.json'}")
        _emit_training_message("Environment check passed.", log_callback)

    _emit_training_message("Preparing ArcGIS training data...", log_callback)
    data, effective_training_data_dir = _prepare_data_with_fallback(
        training_data_dirs,
        batch_size,
        chip_size,
        val_split_pct,
        seed,
        staging_dir=run_log_dir / "merged_training_data",
        log_callback=log_callback,
    )
    if getattr(data, "c", 0) < 2:
        raise ValueError("Training data appears to have fewer than 2 classes.")

    classes = resolve_classes_for_training_sources(training_data_dirs, effective_training_data_dir, data)
    if not classes:
        raise ValueError("Could not resolve class definitions from training data or EMD.")
    _emit_training_message(f"Classes resolved: {len(classes)} | data.c={int(getattr(data, 'c'))}", log_callback)

    if len(classes) != int(getattr(data, "c")):
        # Keep training stable: arcgis data.c controls model channels.
        classes = classes[: int(getattr(data, "c"))]
        while len(classes) < int(getattr(data, "c")):
            i = len(classes)
            classes.append({"Value": i, "Name": f"class_{i}", "Color": DEFAULT_COLORS[i % len(DEFAULT_COLORS)]})

    model = ModelExtension(
        data=data,
        model_conf=DINOv3SegmentationConfigV2,
        repo_dir=str(repo_dir),
        backbone_weights_path=str(backbone_weights_path),
        dinov3_model_name=dinov3_model_name,
        embed_dim=int(embed_dim),
        decoder_channels=int(decoder_channels),
        decoder_type=str(decoder_type),
        dropout=float(dropout),
        freeze_backbone=bool(freeze_backbone),
        loss_mode=str(loss_mode),
        class_weights=class_weights_list,
        focal_gamma=float(focal_gamma),
        ignore_index=int(ignore_index),
        input_channels=int(input_channels),
        extract_bands=list(extract_bands_list),
        input_adapter=str(input_adapter),
        training_input_scale=str(training_input_scale),
        max_input_value=max_input_value,
        mean=list(IMAGENET_MEAN),
        std=list(IMAGENET_STD),
    )

    try:
        actual_adapter = getattr(model.learn.model, "input_adapter_mode", input_adapter)
        actual_backbone_ch = getattr(model.learn.model, "backbone_input_channels", None)
        _emit_training_message(
            f"Resolved spectral path: selected_input_channels={input_channels} | "
            f"backbone_input_channels={actual_backbone_ch} | input_adapter={actual_adapter}",
            log_callback,
        )
    except Exception:
        actual_adapter = input_adapter
        actual_backbone_ch = None

    epoch_log_csv = run_log_dir / "epoch_progress.csv"
    fit_report = _fit_model_extension(
        model,
        epochs,
        learning_rate,
        use_early_stopping,
        early_stopping_patience,
        epoch_log_csv=epoch_log_csv,
        log_callback=log_callback,
    )
    _write_json(run_log_dir / "fit_report.json", fit_report)
    _emit_training_message(f"Fit report written: {run_log_dir / 'fit_report.json'}", log_callback)

    _emit_training_message("Saving arcgis.learn training artifacts...", log_callback)
    save_path = output_root / model_name
    try:
        model.save(str(save_path), framework="PyTorch", compute_metrics=True, save_inference_file=False)
    except TypeError:
        model.save(str(save_path), framework="PyTorch")
    training_save_dir = save_path
    training_save_dir.mkdir(parents=True, exist_ok=True)
    try:
        (training_save_dir / "DO_NOT_USE_THIS_FOLDER_DLPK_FOR_PRO_INFERENCE.txt").write_text(
            "Use the DLPK in <output_root>/<model_name>_deployment/<model_name>.dlpk instead.\n"
            "The arcgis.learn save folder is retained only as a training artifact folder.\n",
            encoding="utf-8",
        )
    except Exception:
        pass

    trained_model = model.learn.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model.to(device)

    metrics: Optional[MetricSummary] = None
    metrics_json: Optional[Path] = None
    metrics_csv: Optional[Path] = None
    if evaluate_after_training:
        try:
            _emit_training_message("Running post-training validation metrics...", log_callback)
            metrics = evaluate_model_on_validation(trained_model, data, classes, ignore_index, max_validation_batches)
            metrics_json, metrics_csv = write_metrics(metrics, run_log_dir)
            _emit_training_message(f"Validation metrics written: {metrics_json}", log_callback)
        except Exception as exc:
            _write_json(run_log_dir / "metrics_error.json", {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
            _emit_training_message(f"Validation metrics failed: {type(exc).__name__}: {exc}", log_callback)

    _emit_training_message("Exporting custom deployment DLPK...", log_callback)
    deployment_dir, emd_path, dlpk_path, manifest_path = export_custom_deployment_package_v2(
        output_root=output_root,
        training_data_dir=effective_training_data_dir,
        training_data_source_dirs=training_data_dirs,
        model_name=model_name,
        repo_dir=repo_dir,
        backbone_weights_path=backbone_weights_path,
        trained_model=trained_model,
        classes=classes,
        chip_size=chip_size,
        batch_size=batch_size,
        dinov3_model_name=dinov3_model_name,
        embed_dim=embed_dim,
        decoder_channels=decoder_channels,
        dropout=dropout,
        decoder_type=decoder_type,
        freeze_backbone=freeze_backbone,
        extract_bands=extract_bands_list,
        input_scale=input_scale,
        max_input_value=max_input_value,
        input_channels=int(input_channels),
        input_adapter=str(getattr(trained_model, "input_adapter_mode", input_adapter)),
        training_input_scale=str(training_input_scale),
        loss_mode=loss_mode,
        class_weights=class_weights_list,
        focal_gamma=focal_gamma,
        ignore_index=ignore_index,
        model_padding=model_padding,
        confidence_threshold=confidence_threshold,
        metrics=metrics,
        aggressive_repo_trim=aggressive_repo_trim,
    )

    deployment_checkpoint = deployment_dir / f"{model_name}.pth"
    return TrainingOutputs(
        model_dir=output_root,
        training_save_dir=training_save_dir,
        deployment_dir=deployment_dir,
        deployment_checkpoint=deployment_checkpoint,
        emd_path=emd_path,
        dlpk_path=dlpk_path,
        manifest_path=manifest_path,
        metrics_json=metrics_json,
        metrics_csv=metrics_csv,
        epoch_log_csv=epoch_log_csv,
    )


# Backward-compatible alias for old wrappers.
train_and_package = train_and_package_v2
