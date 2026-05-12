"""
ArcGIS custom Python raster function for DINOv3 semantic segmentation v2.32.

EMD example:
    "InferenceFunction": ".\\dinov3_inference_v2.py"

Use with ArcGIS Pro: Classify Pixels Using Deep Learning.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

# v2.32-rs311: keep each ArcGIS raster worker conservative.  ArcGIS Pro can
# launch multiple raster/parallel workers; each worker loading a ViT-L DINOv3
# model on the same GPU can otherwise exceed the worker timeout.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

import numpy as np
import torch

from dinov3_arcgis_v2 import (
    DINOv3SegmentationNetV2,
    load_dinov3_backbone,
    normalize_inference_batch_to_dino,
    scale_raw_pixels_to_01,
)


def _resolve_relative(base_file: Path, maybe_relative: str) -> Path:
    candidate = Path(str(maybe_relative).replace("\\", "/"))
    if candidate.is_absolute():
        return candidate
    return (base_file.parent / candidate).resolve()


def _safe_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def _safe_int(value: Any, default: int = 0, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        if value is None or str(value).strip() == "":
            out = int(default)
        else:
            out = int(float(value))
    except Exception:
        out = int(default)
    if minimum is not None:
        out = max(int(minimum), out)
    if maximum is not None:
        out = min(int(maximum), out)
    return out


def _truthy_model_as_file(value: Any) -> bool:
    """ArcGIS may pass model_as_file as bool, string, or omit it."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def _resolve_input_adapter_mode(input_adapter: Any, input_channels: int) -> str:
    mode = str(input_adapter or "auto").strip().lower()
    if mode in ("", "auto"):
        return "none" if int(input_channels) == 3 else "learned_1x1"
    if mode in ("identity", "rgb"):
        return "none"
    return mode


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except Exception:
        return default


def _extract_class_values(classes: Any, num_classes: int) -> list[int]:
    """Return deployment class values for mapping argmax index -> class value.

    ArcGIS training labels often use class values such as 1, 2, 3, ... while
    PyTorch logits are zero-based class indices.  Returning raw argmax indices
    can therefore make the first class appear as 0/NoData in ArcGIS.
    """
    values: list[int] = []
    if isinstance(classes, list):
        for i, cls in enumerate(classes):
            value = None
            if isinstance(cls, dict):
                for key in ("Value", "value", "ClassValue", "classValue", "class_value", "id", "Id"):
                    if key in cls:
                        value = cls.get(key)
                        break
            elif isinstance(cls, (list, tuple)) and cls:
                value = cls[0]
            if value is None:
                value = i
            try:
                values.append(int(float(value)))
            except Exception:
                values.append(i)
    if len(values) != int(num_classes):
        values = list(range(int(num_classes)))
    return values


def _load_emd_payload(model: Any, model_as_file: Any = None, **kwargs: Any) -> tuple[Dict[str, Any], Path]:
    """
    Robust EMD loader for ArcGIS Pro.

    Depending on whether the user loads a .dlpk or .emd, ArcGIS Pro may call
    initialize() with:
      1. model = path to .emd and model_as_file=True
      2. model = path to .emd but model_as_file=False/None
      3. model = JSON text
      4. model = an empty/opaque string while the .emd sits next to this PRF

    This helper accepts all four cases and returns both the parsed EMD JSON and
    a base path used for resolving relative files like ModelFile and BackboneRepoDir.
    """
    module_dir = Path(__file__).resolve().parent

    if isinstance(model, dict):
        return model, module_dir / "model.emd"

    model_text = "" if model is None else str(model).strip()

    # Case 1/2: explicit or implicit EMD file path. Do not rely only on
    # model_as_file because ArcGIS Pro sometimes passes a path while this flag
    # is False/None.
    candidate_paths: list[Path] = []
    if model_text:
        try:
            candidate_paths.append(Path(model_text).expanduser())
        except Exception:
            pass

    for key in ("emd_path", "model_path", "model_file", "model"):
        value = kwargs.get(key)
        if value:
            try:
                candidate_paths.append(Path(str(value)).expanduser())
            except Exception:
                pass

    for candidate in candidate_paths:
        try:
            if candidate.exists() and candidate.is_file():
                emd_path = candidate.resolve()
                with emd_path.open("r", encoding="utf-8") as f:
                    return json.load(f), emd_path
        except Exception:
            # Keep trying other representations; raise a clear error later if all fail.
            pass

    # Case 3: JSON string. Use the PRF folder as base for relative support files.
    if model_text.startswith("{"):
        try:
            return json.loads(model_text), module_dir / "model.emd"
        except json.JSONDecodeError:
            pass

    # Case 4: fallback. This is common when running directly from a deployment
    # folder by selecting the .emd. The PRF is in the same folder as the EMD.
    emd_candidates = sorted(module_dir.glob("*.emd"))
    if not emd_candidates:
        emd_candidates = sorted(Path.cwd().glob("*.emd"))

    if emd_candidates:
        emd_path = emd_candidates[0].resolve()
        with emd_path.open("r", encoding="utf-8") as f:
            return json.load(f), emd_path

    preview = model_text[:300].replace("\n", " ").replace("\r", " ")
    raise ValueError(
        "Unable to load EMD payload. ArcGIS did not provide a readable EMD path "
        "or JSON string, and no .emd file was found next to dinov3_inference_v2.py. "
        f"model_as_file={model_as_file!r}; model_preview={preview!r}; kwargs_keys={list(kwargs.keys())}"
    )



def _find_raster_pixel_block(pixel_blocks: Dict[str, Any]) -> Tuple[str, np.ndarray]:
    """
    ArcGIS Pro / ArcGISImageClassifier can call the child PRF with different
    pixel block keys depending on whether the user loads an .emd, a .dlpk,
    or whether the parent raster function template has already wrapped the
    input raster parameter.  Do not assume the key is always 'raster_pixels'.
    """
    preferred_keys = (
        "raster_pixels",
        "image_pixels",
        "input_pixels",
        "input_raster_pixels",
        "in_raster_pixels",
        "rasters_pixels",
        "raster",
        "image",
    )
    for key in preferred_keys:
        value = pixel_blocks.get(key)
        if isinstance(value, np.ndarray):
            return key, value

    # Generic fallback: choose the first ndarray that looks like an image chip,
    # and ignore scalar/tool argument values such as tta/use_half/padding.
    candidates = []
    for key, value in pixel_blocks.items():
        if isinstance(value, np.ndarray) and value.ndim >= 2:
            candidates.append((key, value))
    if candidates:
        # Prefer 3D/4D arrays over 2D scalar masks if several arrays are present.
        candidates.sort(key=lambda kv: kv[1].ndim, reverse=True)
        return candidates[0]

    available = ", ".join(f"{k}:{type(v).__name__}" for k, v in pixel_blocks.items())
    raise KeyError(
        "No raster pixel block was found. Expected a numpy array under a key such as "
        "'raster_pixels', 'image_pixels', or '*_pixels'. Available pixelBlocks: "
        f"{available}"
    )


def _coerce_raster_pixels_to_bchw(raster_pixels: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Return pixels as [B, C, H, W]. The boolean indicates whether the original
    block already had a batch dimension. ArcGIS usually sends [C, H, W], but
    some wrappers may send [B, C, H, W].
    """
    arr = np.asarray(raster_pixels)
    if arr.ndim == 2:
        # Single-band fallback: [H, W] -> [1, 1, H, W]
        return arr[None, None, :, :], False
    if arr.ndim == 3:
        # ArcGIS standard: [bands, height, width]
        return arr[None, :, :, :], False
    if arr.ndim == 4:
        # Preferred batch layout: [batch, bands, height, width]
        if arr.shape[1] in (1, 2, 3, 4, 5, 6, 8, 10, 13):
            return arr, True
        # Rare layout: [bands, batch, height, width]
        if arr.shape[0] in (1, 2, 3, 4, 5, 6, 8, 10, 13):
            return np.transpose(arr, (1, 0, 2, 3)), True
    raise ValueError(
        f"Unsupported raster pixel block shape {arr.shape}. Expected [C,H,W] or [B,C,H,W]."
    )

def _infer_target_shape(shape: Any, output_pixels_template: Any = None) -> Tuple[int, int, int]:
    """
    Infer the output band count / height / width expected by ArcGIS.

    ArcGIS may pass shape as [bands, height, width].  When the caller already
    provided an output pixel template, prefer that template.  Return
    (bands, height, width).  Fall back to (-1, -1, -1) when unknown.
    """
    if isinstance(output_pixels_template, np.ndarray) and output_pixels_template.ndim >= 3:
        return int(output_pixels_template.shape[-3]), int(output_pixels_template.shape[-2]), int(output_pixels_template.shape[-1])
    if isinstance(shape, (tuple, list)):
        try:
            if len(shape) >= 3:
                return int(shape[-3]), int(shape[-2]), int(shape[-1])
            if len(shape) == 2:
                return 1, int(shape[-2]), int(shape[-1])
        except Exception:
            pass
    return -1, -1, -1


def _center_crop_or_pad_chw(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Crop/pad a [C,H,W] output to ArcGIS' expected height/width."""
    out = arr
    if target_h > 0 and target_w > 0:
        h, w = out.shape[-2], out.shape[-1]
        if h > target_h:
            top = (h - target_h) // 2
            out = out[..., top:top + target_h, :]
        if w > target_w:
            left = (w - target_w) // 2
            out = out[..., :, left:left + target_w]
        h, w = out.shape[-2], out.shape[-1]
        if h < target_h or w < target_w:
            pad_top = max(0, (target_h - h) // 2)
            pad_bottom = max(0, target_h - h - pad_top)
            pad_left = max(0, (target_w - w) // 2)
            pad_right = max(0, target_w - w - pad_left)
            out = np.pad(out, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode="edge")
    return out


def _match_expected_band_count_chw(arr: np.ndarray, target_bands: int) -> np.ndarray:
    """
    Make a [C,H,W] array match ArcGIS' expected band count.

    The intended output is one classified band.  updateRasterInfo() below asks
    ArcGIS to expect one band.  Some loader paths still report the inherited
    input band count in the shape argument; in that case repeat the class-id
    band so the raster function read path does not fail on shape unpacking.
    """
    if target_bands <= 0 or arr.shape[0] == target_bands:
        return arr
    if arr.shape[0] == 1 and target_bands > 1:
        return np.repeat(arr, target_bands, axis=0)
    if target_bands == 1 and arr.shape[0] > 1:
        return arr[:1, :, :]
    if arr.shape[0] > target_bands:
        return arr[:target_bands, :, :]
    reps = int(np.ceil(target_bands / arr.shape[0]))
    return np.tile(arr, (reps, 1, 1))[:target_bands, :, :]



def _infer_output_pixel_key(pixel_blocks: Dict[str, Any]) -> str:
    """Return the ArcGIS output pixel block key to write.

    Most Esri Python Raster Function samples write to ``output_pixels``.
    Some Classify Pixels loader paths expose / report the output block as
    ``output``.  Prefer an existing output template key when present.
    """
    for key in ("output_pixels", "output"):
        if key in pixel_blocks:
            return key
    return "output_pixels"


def _infer_output_dtype(pixel_blocks: Dict[str, Any], props: Dict[str, Any] | None, num_classes: int) -> np.dtype:
    """Infer the dtype ArcGIS expects for the classified output block.

    The safest source is the output pixel template ArcGIS passes into
    updatePixels.  When processing 16-bit source rasters, some Pro 3.6 raster
    function chains inherit a uint16 output template even when getConfiguration
    asks for U8.  If the PRF returns uint8 in that case ArcGIS reports:
    ``incoming 50176 bytes, expected 100352 bytes`` for a 224x224 tile.
    """
    for key in ("output_pixels", "output"):
        template = pixel_blocks.get(key)
        if isinstance(template, np.ndarray):
            return np.dtype(template.dtype)
    # Try ArcGIS property dictionaries if they are present.
    def _dtype_from_pixel_type(value: Any):
        text = str(value or "").strip().lower()
        if text in ("u1", "u8", "uint8", "8_bit_unsigned"):
            return np.dtype(np.uint8)
        if text in ("u2", "u16", "uint16", "16_bit_unsigned"):
            return np.dtype(np.uint16)
        if text in ("s2", "i16", "int16", "16_bit_signed"):
            return np.dtype(np.int16)
        return None
    if isinstance(props, dict):
        for key in ("pixelType", "pixel_type"):
            dt = _dtype_from_pixel_type(props.get(key))
            if dt is not None:
                return dt
        for nested_key in ("output_info", "out_raster_info", "rasterInfo"):
            nested = props.get(nested_key)
            if isinstance(nested, dict):
                dt = _dtype_from_pixel_type(nested.get("pixelType"))
                if dt is not None:
                    return dt
    return np.dtype(np.uint8 if int(num_classes) <= 255 else np.uint16)


def _clip_class_ids_for_dtype(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Clip predicted class IDs to the integer range before casting."""
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        arr = np.clip(arr, info.min, info.max)
    return arr.astype(dtype, copy=False)


# -----------------------------
# v2.28 inference checkpoint/runtime channel probes
# -----------------------------

def _state_dict_from_checkpoint(ckpt: Any) -> Dict[str, torch.Tensor]:
    """Return the tensor state_dict saved by the training/export step."""
    if isinstance(ckpt, dict):
        sd = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt.get("model")
        if isinstance(sd, dict):
            return {str(k): v for k, v in sd.items() if torch.is_tensor(v)}
        # Some checkpoints are already a state_dict-like mapping.
        if any(torch.is_tensor(v) for v in ckpt.values()):
            return {str(k): v for k, v in ckpt.items() if torch.is_tensor(v)}
    return {}


def _probe_checkpoint_backbone_channels(sd: Dict[str, torch.Tensor]) -> int | None:
    """Infer backbone patch-embedding input channels from a deployment state_dict."""
    preferred = (
        "backbone.patch_embed.proj.weight",
        "module.backbone.patch_embed.proj.weight",
        "learn.model.backbone.patch_embed.proj.weight",
        "model.backbone.patch_embed.proj.weight",
    )
    for key in preferred:
        w = sd.get(key)
        if torch.is_tensor(w) and w.ndim == 4:
            return int(w.shape[1])
    # Generic fallback: look for patch-embedding conv kernels, avoiding decoder/adapters.
    for key, w in sd.items():
        lk = key.lower()
        if "backbone" in lk and "patch_embed" in lk and lk.endswith("weight") and torch.is_tensor(w) and w.ndim == 4:
            return int(w.shape[1])
    for key, w in sd.items():
        lk = key.lower()
        if "patch_embed" in lk and lk.endswith("weight") and torch.is_tensor(w) and w.ndim == 4:
            return int(w.shape[1])
    return None


def _probe_checkpoint_adapter(sd: Dict[str, torch.Tensor]) -> tuple[str | None, int | None, int | None]:
    """Return (adapter_mode, adapter_in_channels, adapter_out_channels) from checkpoint if present."""
    candidates = (
        "input_adapter.net.weight",
        "module.input_adapter.net.weight",
        "learn.model.input_adapter.net.weight",
        "model.input_adapter.net.weight",
    )
    for key in candidates:
        w = sd.get(key)
        if torch.is_tensor(w) and w.ndim == 4:
            kh, kw = int(w.shape[2]), int(w.shape[3])
            mode = "learned_1x1" if (kh, kw) == (1, 1) else "learned_3x3"
            return mode, int(w.shape[1]), int(w.shape[0])
    for key, w in sd.items():
        lk = key.lower()
        if "input_adapter" in lk and lk.endswith("weight") and torch.is_tensor(w) and w.ndim == 4:
            kh, kw = int(w.shape[2]), int(w.shape[3])
            mode = "learned_1x1" if (kh, kw) == (1, 1) else "learned_3x3"
            return mode, int(w.shape[1]), int(w.shape[0])
    return None, None, None


def _replace_backbone_patch_embed_input_channels(backbone: torch.nn.Module, target_channels: int) -> bool:
    """Resize backbone.patch_embed.proj to match checkpoint channels before strict load_state_dict()."""
    target_channels = int(target_channels)
    patch_embed = getattr(backbone, "patch_embed", None)
    proj = getattr(patch_embed, "proj", None) if patch_embed is not None else None
    if isinstance(proj, torch.nn.Conv2d):
        if int(proj.in_channels) == target_channels:
            return False
        new_proj = torch.nn.Conv2d(
            in_channels=target_channels,
            out_channels=int(proj.out_channels),
            kernel_size=proj.kernel_size,
            stride=proj.stride,
            padding=proj.padding,
            dilation=proj.dilation,
            groups=int(proj.groups),
            bias=proj.bias is not None,
            padding_mode=proj.padding_mode,
            device=proj.weight.device,
            dtype=proj.weight.dtype,
        )
        patch_embed.proj = new_proj
        return True
    raise RuntimeError(
        "Unable to resize DINOv3 patch embedding for checkpoint load: "
        "backbone.patch_embed.proj is not a torch.nn.Conv2d."
    )

class ChildImageClassifier:
    """ArcGIS Python raster function entry point for pixel classification."""

    def initialize(self, model=None, model_as_file=None, **kwargs):
        # ArcGIS Pro may pass additional scalar keyword arguments such as
        # pythonmodule, padding, batch_size, processorType, or tool-specific controls.
        # They are accepted via **kwargs for compatibility.
        self.json_info, emd_path = _load_emd_payload(model, model_as_file, **kwargs)

        self.emd_path = emd_path
        self.padding = int(self.json_info.get("ModelPadding", 0))
        self.image_height = int(self.json_info["ImageHeight"])
        self.image_width = int(self.json_info["ImageWidth"])
        self.extract_bands = tuple(int(b) for b in self.json_info.get("ExtractBands", [0, 1, 2]))
        self.mean = self.json_info.get("Mean", [0.485, 0.456, 0.406])
        self.std = self.json_info.get("StdDev", [0.229, 0.224, 0.225])
        self.embed_dim = int(self.json_info.get("EmbedDim", 1024))
        self.decoder_channels = int(self.json_info.get("DecoderChannels", 256))
        self.decoder_type = self.json_info.get("DecoderType", "fpn_lite")
        self.dropout = float(self.json_info.get("Dropout", 0.1))
        self.model_name = self.json_info.get("DINOv3ModelName", "dinov3_vitl16")
        self.input_scale = self.json_info.get("InputScale", "auto")
        self.max_input_value = self.json_info.get("MaxInputValue", None)
        self.input_channels = int(self.json_info.get("InputChannels", len(self.extract_bands)))
        self.input_adapter = _resolve_input_adapter_mode(self.json_info.get("InputAdapter", "auto"), self.input_channels)
        self.confidence_threshold = float(self.json_info.get("ConfidenceThreshold", 0.0))
        self.background_class = int(self.json_info.get("BackgroundClass", 0))
        # v2.32 diagnostic controls. output_mode can be overridden from GP
        # Arguments, e.g. output_mode raw_index / confidence / nonzero_class_value.
        self.output_mode = str(self.json_info.get("OutputMode", "class_value") or "class_value").strip().lower()
        self.suppress_background = _safe_bool(self.json_info.get("SuppressBackground", False), default=False)
        self.background_index = _safe_int(self.json_info.get("BackgroundIndex", 0), default=0, minimum=0)
        self.debug_first_tile = _safe_bool(self.json_info.get("DebugFirstTile", False), default=False)
        self._debug_tile_printed = False
        # Inference batch size is exposed as a PRF scalar parameter in v2.19.
        # 0 means "auto"; current implementation keeps auto conservative because
        # ArcGIS Pro may already allocate tile buffers outside PyTorch.
        self.inference_batch_size = _safe_int(self.json_info.get("InferenceBatchSize", 1), default=1, minimum=0, maximum=64)
        # v2.30-rs311: hard safety cap for PRF inference batching.  The user may
        # pass batch_size=8/16 from the GP tool, but on ArcGIS parallel raster
        # workers this can cause GPU contention and "Parallel processing job timed out".
        # Default cap is 1 for maximum stability; advanced users may edit the EMD
        # MaxInferenceBatchSize to 2/4 after confirming stable inference.
        self.max_inference_batch_size = _safe_int(self.json_info.get("MaxInferenceBatchSize", 1), default=1, minimum=1, maximum=8)

        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_file = _resolve_relative(emd_path, self.json_info["ModelFile"])
        repo_dir = _resolve_relative(emd_path, self.json_info["BackboneRepoDir"])
        backbone_weights = _resolve_relative(emd_path, self.json_info["BackboneWeights"])

        if not model_file.exists():
            raise FileNotFoundError(f"Deployment checkpoint not found: {model_file}")

        # v2.27-rs311: robust fallback for repaired legacy DLPKs that were
        # created before the DINOv3 repo snapshot was bundled.  A portable DLPK
        # should contain .\dinov3_repo, but for local repair/testing allow an
        # explicit environment variable or the common C:\src\dinov3 location.
        if not repo_dir.exists():
            fallback_repo = os.environ.get("DINOV3_REPO_DIR", "")
            fallback_candidates = []
            if fallback_repo:
                fallback_candidates.append(Path(fallback_repo))
            fallback_candidates.append(Path(r"C:\src\dinov3"))
            for cand in fallback_candidates:
                if cand.exists() and (cand / "dinov3").exists():
                    repo_dir = cand
                    break
        if not repo_dir.exists():
            raise FileNotFoundError(
                f"DINOv3 repo snapshot not found: {repo_dir}. "
                "Repair the DLPK with repair_existing_dlpk_v2_27.py --repo-dir C:\\src\\dinov3, "
                "or set DINOV3_REPO_DIR to a valid local DINOv3 repository."
            )
        if not backbone_weights.exists():
            raise FileNotFoundError(f"Backbone weights not found: {backbone_weights}")

        ckpt = torch.load(model_file, map_location=self.device)
        classes = ckpt.get("classes", self.json_info.get("Classes", [])) if isinstance(ckpt, dict) else self.json_info.get("Classes", [])
        self.num_classes = int(ckpt.get("num_classes", len(classes))) if isinstance(ckpt, dict) else int(len(classes))
        self.class_values = _extract_class_values(classes, self.num_classes)
        # v2.30-rs311: default output maps zero-based argmax indices back to
        # deployment class values. Set OutputRawClassIndex=true in the EMD only
        # when debugging raw logits/indices.
        self.output_raw_class_index = _safe_bool(self.json_info.get("OutputRawClassIndex", False), default=False)
        state_dict = _state_dict_from_checkpoint(ckpt)
        if not state_dict:
            raise RuntimeError(f"Deployment checkpoint does not contain a valid state_dict: {model_file}")

        ckpt_backbone_channels = _probe_checkpoint_backbone_channels(state_dict)
        ckpt_adapter_mode, ckpt_adapter_in_channels, ckpt_adapter_out_channels = _probe_checkpoint_adapter(state_dict)

        # v2.27-rs311: the EMD can contain stale channel metadata when the loaded
        # DINOv3 runtime backbone was repaired during training.  For inference,
        # the checkpoint tensor shapes are the source of truth because strict
        # load_state_dict() must recreate the exact trained module topology.
        if ckpt_adapter_in_channels is not None:
            self.input_channels = int(ckpt_adapter_in_channels)
        if ckpt_adapter_mode is not None:
            self.input_adapter = ckpt_adapter_mode
        elif ckpt_backbone_channels is not None:
            self.input_adapter = "none" if int(self.input_channels) == int(ckpt_backbone_channels) else "learned_1x1"

        backbone = load_dinov3_backbone(repo_dir, self.model_name, backbone_weights)
        if ckpt_backbone_channels is not None:
            _replace_backbone_patch_embed_input_channels(backbone, int(ckpt_backbone_channels))

        self.model = DINOv3SegmentationNetV2(
            backbone=backbone,
            num_classes=self.num_classes,
            embed_dim=self.embed_dim,
            decoder_channels=self.decoder_channels,
            dropout=self.dropout,
            decoder_type=self.decoder_type,
            input_channels=self.input_channels,
            input_adapter=self.input_adapter,
            mean=self.mean,
            std=self.std,
            backbone_input_channels_override=ckpt_backbone_channels,
        )
        self.input_adapter = str(getattr(self.model, "input_adapter_mode", self.input_adapter))
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

    def getParameterInfo(self, required_parameters=None):
        # ArcGIS Pro may call this method either as getParameterInfo() or
        # getParameterInfo(required_parameters). Some loader paths pass None.
        # The return value must always be a Python list.
        #
        # IMPORTANT: If ArcGIS calls this without required_parameters and we
        # return only scalar arguments, no raster input is bound to the PRF;
        # then updatePixels receives an empty pixelBlocks dict. Therefore v2.10
        # explicitly adds a raster parameter fallback named "raster". ArcGIS
        # will pass it to updatePixels as "raster_pixels".
        if required_parameters is None:
            params: List[Dict[str, Any]] = [
                {
                    "name": "raster",
                    "dataType": "raster",
                    "required": True,
                    "displayName": "Raster",
                    "description": "Input raster passed through ArcGISImageClassifier to the DINOv3 PRF.",
                }
            ]
        elif isinstance(required_parameters, list):
            params = list(required_parameters)
        elif isinstance(required_parameters, tuple):
            params = list(required_parameters)
        else:
            # Defensive fallback: do not return a dict/string directly because
            # ArcGIS reports "Parameter information returned ... is not a python list".
            params = [required_parameters]

        # Make sure a raster parameter exists exactly once. Some ArcGIS wrapper
        # paths provide it in required_parameters; direct EMD loading may not.
        if not any(isinstance(p, dict) and str(p.get("dataType", "")).lower() == "raster" for p in params):
            params.insert(0, {
                "name": "raster",
                "dataType": "raster",
                "required": True,
                "displayName": "Raster",
                "description": "Input raster passed through ArcGISImageClassifier to the DINOv3 PRF.",
            })

        # ArcGIS may call getParameterInfo before initialize() in some paths,
        # so use getattr defaults instead of assuming instance attributes exist.
        default_padding = getattr(self, "padding", 0)
        default_confidence = getattr(self, "confidence_threshold", 0.0)
        default_batch_size = getattr(self, "inference_batch_size", 1)
        params.extend([
            {
                "name": "padding",
                "dataType": "numeric",
                "value": default_padding,
                "required": False,
                "displayName": "Padding pixels",
                "description": "Overlap/padding around each chip. Must keep tx/ty > 0.",
            },
            {
                "name": "batch_size",
                "dataType": "numeric",
                "value": default_batch_size,
                "required": False,
                "displayName": "Inference batch size",
                "description": "Requested inference batch size. v2.29 clamps this to MaxInferenceBatchSize in the EMD; default cap is 1 to avoid ArcGIS parallel worker GPU timeouts.",
            },
            {
                "name": "tta",
                "dataType": "string",
                "value": "none",
                "required": False,
                "displayName": "Test-time augmentation",
                "description": "none or hflip. hflip averages original and horizontal-flip logits.",
            },
            {
                "name": "confidence_threshold",
                "dataType": "numeric",
                "value": default_confidence,
                "required": False,
                "displayName": "Confidence threshold",
                "description": "Pixels below threshold are assigned to background class. Use 0 to disable.",
            },
            {
                "name": "input_scale",
                "dataType": "string",
                "value": getattr(self, "input_scale", "auto"),
                "required": False,
                "displayName": "Input scale",
                "description": "auto, 0_1, 0_255, 0_65535, or max. For 16-bit sensors whose useful DN range is not full 0-65535, use input_scale max with max_input_value.",
            },
            {
                "name": "max_input_value",
                "dataType": "numeric",
                "value": getattr(self, "max_input_value", None),
                "required": False,
                "displayName": "Max input value",
                "description": "Optional fixed radiometric divisor, e.g. 4095, 10000, 16383, or 65535. This is numeric tensor scaling, not display stretch.",
            },
            {
                "name": "output_mode",
                "dataType": "string",
                "value": getattr(self, "output_mode", "class_value"),
                "required": False,
                "displayName": "Output mode",
                "description": "class_value, raw_index, confidence, nonzero_class_value, or nonzero_raw_index. Use raw_index/confidence for diagnosing all-0 outputs.",
            },
            {
                "name": "suppress_background",
                "dataType": "string",
                "value": "false",
                "required": False,
                "displayName": "Suppress background",
                "description": "true/false. Diagnostic only: ignores the background index before argmax so you can see whether non-background logits contain useful signal.",
            },
            {
                "name": "background_index",
                "dataType": "numeric",
                "value": getattr(self, "background_index", 0),
                "required": False,
                "displayName": "Background class index",
                "description": "Zero-based model class index treated as background for suppress_background/nonzero_* output modes. Usually 0.",
            },
            {
                "name": "debug_first_tile",
                "dataType": "string",
                "value": "false",
                "required": False,
                "displayName": "Debug first tile",
                "description": "true/false. Prints input min/max and first-tile prediction histogram from the PRF worker if ArcGIS captures stdout.",
            },
            {
                "name": "use_half",
                "dataType": "string",
                "value": "false",
                "required": False,
                "displayName": "Use FP16 on CUDA",
                "description": "true/false. Useful only on CUDA GPUs; keep false for maximum stability.",
            },
        ])
        return params


    def updateRasterInfo(self, **kwargs):
        """
        Declare a single-band classified raster output.

        Without this, some ArcGIS Pro loader paths inherit the input raster band
        count, so ArcGIS expects e.g. 3 x ty x tx while the model correctly
        returns one class-id band.  This method makes the desired output explicit.
        """
        pixel_type = "u8" if int(getattr(self, "num_classes", 255)) <= 255 else "u16"
        for key in ("output_info", "out_raster_info"):
            info = kwargs.get(key)
            if isinstance(info, dict):
                info["bandCount"] = 1
                info["pixelType"] = pixel_type
                info["pixelType"] = pixel_type
        return kwargs

    def getConfiguration(self, **scalars):
        padding = _safe_int(scalars.get("padding", self.padding), default=self.padding, minimum=0)
        max_padding = max(0, min(self.image_width, self.image_height) // 2 - 1)
        padding = max(0, min(padding, max_padding))
        batch_size = _safe_int(scalars.get("batch_size", self.inference_batch_size), default=1, minimum=0, maximum=64)
        if batch_size == 0:
            # Conservative auto. ArcGIS Pro and the raster function chain also use VRAM,
            # so avoid aggressive automatic batch expansion. Users can manually increase.
            batch_size = 1
        batch_size = min(int(batch_size), int(getattr(self, "max_inference_batch_size", 1)))
        return {
            "extractBands": self.extract_bands,
            "padding": padding,
            "tx": max(1, self.image_width - 2 * padding),
            "ty": max(1, self.image_height - 2 * padding),
            "batchSize": batch_size,
            "batch_size": batch_size,
            "bandCount": 1,
            "pixelType": "u8" if int(getattr(self, "num_classes", 255)) <= 255 else "u16",
        }

    def _predict_logits(self, tensor: torch.Tensor, tta: str) -> torch.Tensor:
        logits = self.model(tensor)
        if str(tta).strip().lower() in ("hflip", "horizontal", "horizontal_flip"):
            flipped = torch.flip(tensor, dims=[-1])
            logits_flip = self.model(flipped)
            logits_flip = torch.flip(logits_flip, dims=[-1])
            logits = 0.5 * (logits + logits_flip)
        return logits

    def updatePixels(self, tlc, shape, props, **pixelBlocks):
        block_key, raster_pixels = _find_raster_pixel_block(pixelBlocks)
        pixels_bchw, had_batch_dim = _coerce_raster_pixels_to_bchw(raster_pixels)

        expected_channels = (
            int(getattr(self.model, "backbone_input_channels", self.input_channels))
            if self.input_adapter in ("none", "identity", "rgb", "native")
            else int(self.input_channels)
        )
        if pixels_bchw.shape[1] != expected_channels:
            raise ValueError(
                f"DINOv3 v2.19 expects {expected_channels} selected band(s) after ExtractBands "
                f"for input_adapter={self.input_adapter} and backbone_input_channels="
                f"{getattr(self.model, 'backbone_input_channels', 'unknown')}, but pixel block key '{block_key}' has "
                f"shape {raster_pixels.shape} and was interpreted as [B,C,H,W]={pixels_bchw.shape}. "
                f"Check EMD ExtractBands, InputChannels, input raster band order, and whether the backbone is native 3-channel or 4-channel."
            )

        tta = pixelBlocks.get("tta", "none")
        confidence_threshold = float(pixelBlocks.get("confidence_threshold", self.confidence_threshold))
        use_half = _safe_bool(pixelBlocks.get("use_half", False), default=False)
        inference_batch_size = _safe_int(pixelBlocks.get("batch_size", self.inference_batch_size), default=1, minimum=0, maximum=64)
        if inference_batch_size == 0:
            inference_batch_size = 1
        inference_batch_size = min(int(inference_batch_size), int(getattr(self, "max_inference_batch_size", 1)))

        input_scale_override = str(pixelBlocks.get("input_scale", self.input_scale) or self.input_scale)
        max_input_value_override = _safe_float(pixelBlocks.get("max_input_value", self.max_input_value), default=self.max_input_value)
        output_mode = str(pixelBlocks.get("output_mode", getattr(self, "output_mode", "class_value")) or "class_value").strip().lower()
        suppress_background = _safe_bool(pixelBlocks.get("suppress_background", getattr(self, "suppress_background", False)), default=False)
        background_index = _safe_int(pixelBlocks.get("background_index", getattr(self, "background_index", 0)), default=0, minimum=0)
        debug_first_tile = _safe_bool(pixelBlocks.get("debug_first_tile", getattr(self, "debug_first_tile", False)), default=False)
        if output_mode in ("nonzero", "nonzero_class", "nonzero_class_value", "non_background", "nonbackground"):
            suppress_background = True
            output_mode = "class_value"
        elif output_mode in ("nonzero_raw", "nonzero_raw_index", "non_background_raw", "nonbackground_raw"):
            suppress_background = True
            output_mode = "raw_index"

        tensor = torch.from_numpy(pixels_bchw.astype(np.float32)).to(self.device)
        tensor = scale_raw_pixels_to_01(tensor, input_scale=input_scale_override, max_input_value=max_input_value_override)
        if self.input_adapter in ("none", "identity", "rgb"):
            tensor = normalize_inference_batch_to_dino(tensor, mean=self.mean, std=self.std)
        # Adapter mode expects unit-scaled tensor; model performs adapter projection and DINO normalization.

        if use_half and self.device.type == "cuda":
            tensor = tensor.half()
            self.model.half()
        else:
            self.model.float()

        preds = []
        with torch.inference_mode():
            for start in range(0, tensor.shape[0], max(1, inference_batch_size)):
                tensor_chunk = tensor[start:start + max(1, inference_batch_size)]
                logits = self._predict_logits(tensor_chunk, str(tta)).float()
                if suppress_background and 0 <= int(background_index) < int(logits.shape[1]) and int(logits.shape[1]) > 1:
                    logits = logits.clone()
                    logits[:, int(background_index), :, :] = -1.0e9
                prob = torch.softmax(logits, dim=1)
                conf, pred_idx = torch.max(prob, dim=1, keepdim=True)

                # Diagnostic / production output modes. class_value is the normal
                # ArcGIS classified raster output. raw_index and confidence are
                # intended to diagnose all-background/all-zero predictions.
                if output_mode in ("confidence", "conf", "max_prob", "prob"):
                    pred = torch.round(conf * 10000.0).to(torch.int32)
                elif output_mode in ("raw", "raw_index", "index", "argmax") or getattr(self, "output_raw_class_index", False):
                    pred = pred_idx
                    if confidence_threshold > 0:
                        pred = torch.where(
                            conf >= confidence_threshold,
                            pred,
                            torch.full_like(pred, int(self.background_class)),
                        )
                else:
                    class_values = torch.tensor(getattr(self, "class_values", list(range(self.num_classes))), dtype=pred_idx.dtype, device=pred_idx.device)
                    pred_idx_clamped = pred_idx.clamp(0, max(0, int(class_values.numel()) - 1))
                    pred = class_values[pred_idx_clamped]
                    if confidence_threshold > 0:
                        pred = torch.where(
                            conf >= confidence_threshold,
                            pred,
                            torch.full_like(pred, int(self.background_class)),
                        )

                if debug_first_tile and not getattr(self, "_debug_tile_printed", False):
                    try:
                        vals, cnts = torch.unique(pred_idx.detach().cpu(), return_counts=True)
                        hist = ", ".join(f"{int(v)}:{int(c)}" for v, c in zip(vals[:20], cnts[:20]))
                        tmin = float(torch.nan_to_num(tensor_chunk.detach().float()).min().item())
                        tmax = float(torch.nan_to_num(tensor_chunk.detach().float()).max().item())
                        print(
                            f"[DINOv3 v2.32 PRF DEBUG] mode={output_mode}; suppress_background={suppress_background}; "
                            f"input_scale={input_scale_override}; max_input_value={max_input_value_override}; "
                            f"tensor_min={tmin:.6f}; tensor_max={tmax:.6f}; raw_pred_hist={hist}",
                            flush=True,
                        )
                        self._debug_tile_printed = True
                    except Exception:
                        pass
                preds.append(pred.to(torch.int32).cpu())

        result = torch.cat(preds, dim=0).numpy()  # [B, 1, H, W]
        target_bands, target_h, target_w = _infer_target_shape(shape, pixelBlocks.get("output_pixels"))

        if not had_batch_dim:
            output_pixels = result[0]  # [1, H, W], intended ArcGIS classified output
            output_pixels = _center_crop_or_pad_chw(output_pixels, target_h, target_w)
            output_pixels = _match_expected_band_count_chw(output_pixels, target_bands)
        else:
            # Batched pixel blocks are uncommon in ArcGIS PRF calls, but keep a
            # conservative implementation.  Match H/W and band count per item.
            items = []
            for i in range(result.shape[0]):
                item = _center_crop_or_pad_chw(result[i], target_h, target_w)
                item = _match_expected_band_count_chw(item, target_bands)
                items.append(item)
            output_pixels = np.stack(items, axis=0)

        # ArcGIS Python Raster Functions must return a dictionary from updatePixels/read paths.
        # Returning a raw numpy array causes: "Python raster function's read method returned a non-dictionary object."
        # v2.28: cast to the dtype ArcGIS' output pixel template expects.  This fixes
        # Pro 3.6 cases where a 16-bit input raster causes the output block to expect
        # uint16 even though the predicted class IDs are naturally small integers.
        output_key = _infer_output_pixel_key(pixelBlocks)
        output_dtype = _infer_output_dtype(pixelBlocks, props, int(getattr(self, "num_classes", 255)))
        if str(output_mode).lower() in ("confidence", "conf", "max_prob", "prob"):
            output_dtype = np.dtype(np.uint16)
        pixelBlocks[output_key] = _clip_class_ids_for_dtype(output_pixels, output_dtype)
        if output_key != "output_pixels":
            pixelBlocks["output_pixels"] = pixelBlocks[output_key]
        return pixelBlocks


# -----------------------------------------------------------------------------
# ArcGIS Pro PRF compatibility aliases
# -----------------------------------------------------------------------------
# Some ArcGIS Pro deep learning loaders use the Python file stem as the raster
# function class name.  If InferenceFunction is .\dinov3_inference_v2.py, Pro may
# try getattr(module, "dinov3_inference_v2") instead of ChildImageClassifier.
# Keep these aliases so both loader styles work.
class dinov3_inference_v2(ChildImageClassifier):
    pass


class DINOv3InferenceV2(ChildImageClassifier):
    pass
