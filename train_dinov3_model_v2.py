from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path


def _force_utf8_stdio() -> None:
    """Make stdout/stderr safe for Traditional Chinese Windows paths.

    ArcGIS Pro subprocesses on Windows can inherit a legacy console code page
    such as cp1252. Printing a working directory like D:\\python程式資料區 then
    raises UnicodeEncodeError before training starts.  Reconfigure streams to
    UTF-8 with replacement so diagnostic logging never terminates training.
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_force_utf8_stdio()


def _configure_arcgis_standalone_paths() -> None:
    """Prepare DLL/PATH search order for standalone ArcGIS Pro Python subprocesses.

    Directly launching an ArcGIS Pro cloned environment python.exe from ArcGIS Pro
    can miss PATH entries that the ArcGIS Pro Python Command Prompt normally sets.
    arcgis.learn imports arcpy through the ArcGIS geometry engine; without the ArcGIS
    Pro bin and conda Library\bin folders, Windows may terminate with fatal exception
    0xe0000001.  This function is intentionally called before importing arcgis.learn.
    """
    candidates = []
    try:
        py = Path(sys.executable)
        env_dir = py.parent
        candidates.extend([
            env_dir,
            env_dir / "Scripts",
            env_dir / "Library" / "bin",
            env_dir / "Library" / "usr" / "bin",
            env_dir / "Library" / "mingw-w64" / "bin",
        ])
        os.environ.setdefault("CONDA_PREFIX", str(env_dir))
        os.environ.setdefault("CONDA_DEFAULT_ENV", env_dir.name)
    except Exception:
        pass

    pro_root = Path(os.environ.get("ARCGIS_PRO_ROOT", r"C:\Program Files\ArcGIS\Pro"))
    candidates.extend([
        pro_root / "bin",
        pro_root / "Resources" / "ArcPy",
        pro_root / "bin" / "Python" / "Scripts",
        pro_root / "bin" / "Python" / "envs" / "arcgispro-py3" / "Library" / "bin",
        pro_root / "bin" / "Python" / "envs" / "arcgispro-py3" / "Scripts",
    ])

    seen = set()
    prepend = []
    for c in candidates:
        try:
            c = Path(c)
            if c.exists():
                key = str(c).lower()
                if key not in seen:
                    prepend.append(str(c))
                    seen.add(key)
                    try:
                        os.add_dll_directory(str(c))
                    except Exception:
                        pass
        except Exception:
            pass
    if prepend:
        os.environ["PATH"] = os.pathsep.join(prepend + [os.environ.get("PATH", "")])
    os.environ.setdefault("ESRI_SOFTWARE_CLASS", "Professional")
    try:
        print("[DINOv3 v2.32-rs311 BOOT] ArcGIS DLL/PATH entries prepared:", flush=True)
        for item in prepend:
            print(f"  {item}", flush=True)
    except Exception:
        pass

_configure_arcgis_standalone_paths()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train DINOv3 SAT-493M based semantic segmentation model v2.32-rs311 and export an "
            "ArcGIS Pro custom .dlpk for Classify Pixels Using Deep Learning."
        )
    )
    parser.add_argument(
        "--training-data",
        required=True,
        nargs="+",
        help=(
            "One or more ArcGIS Export Training Data For Deep Learning output folders. "
            "Multiple folders may be provided as separate quoted paths, or as one semicolon-delimited ArcGIS multiValue string."
        ),
    )
    parser.add_argument("--output-root", required=True, help="Output folder for training logs and deployment package.")
    parser.add_argument("--repo-dir", required=True, help="Local clone/snapshot of facebookresearch/dinov3.")
    parser.add_argument("--backbone-weights", required=True, help="Path to dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth.")
    parser.add_argument("--model-name", default="dinov3_sat493m_seg_v2", help="Model name prefix.")

    parser.add_argument("--chip-size", type=int, default=224, help="Chip size; must be divisible by 16.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--val-split-pct", type=float, default=0.2, help="Validation split percentage if supported by arcgis.learn.prepare_data.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used by prepare_data when supported.")

    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze DINOv3 backbone and train decoder only.")
    parser.add_argument("--unfreeze-backbone", action="store_true", help="Override --freeze-backbone and train full model.")
    parser.add_argument("--decoder-type", default="fpn_lite", choices=["fpn_lite", "linear"], help="Segmentation decoder type.")
    parser.add_argument("--decoder-channels", type=int, default=256, help="Decoder channel width.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Decoder dropout rate.")
    parser.add_argument("--dinov3-model-name", default="dinov3_vitl16", help="torch.hub model name in local DINOv3 repo.")
    parser.add_argument("--embed-dim", type=int, default=1024, help="Backbone embedding dimension for ViT-L/16.")

    parser.add_argument("--loss-mode", default="ce", choices=["ce", "focal"], help="Loss function.")
    parser.add_argument("--class-weights", default=None, help="Comma-separated class weights, e.g. '0.5,2,1,3'.")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Gamma for focal loss.")
    parser.add_argument("--ignore-index", type=int, default=-100, help="Ignore index for loss and metrics.")

    parser.add_argument("--no-early-stopping", action="store_true", help="Disable early-stopping callback attempt.")
    parser.add_argument("--early-stopping-patience", type=int, default=5, help="Patience for early-stopping callback when available.")
    parser.add_argument("--no-eval", action="store_true", help="Skip post-training validation metric export.")
    parser.add_argument("--max-validation-batches", type=int, default=None, help="Limit validation metric pass for large datasets.")

    parser.add_argument("--extract-bands", default="0,1,2", help="Comma-separated band indexes for ArcGIS ExtractBands. RGB legacy: '0,1,2'; 4-band example: '0,1,2,3'; hyperspectral example: '0,1,2,...,31'.")
    parser.add_argument("--input-scale", default="auto", choices=["auto", "0_1", "0_255", "0_65535", "max"], help="Raw pixel scale at inference. Use max with --max-input-value for fixed 16-bit DN scaling.")
    parser.add_argument("--max-input-value", type=float, default=None, help="Optional explicit max value for raw pixel scaling.")
    parser.add_argument("--input-channels", type=int, default=None, help="Number of selected input bands. Defaults to len(extract_bands).")
    parser.add_argument("--input-adapter", default="auto", choices=["auto", "none", "learned_1x1", "learned_3x3"], help="auto uses legacy RGB for 3 bands and learned_1x1 for non-3-band input; use learned_3x3 for stronger multispectral adaptation.")
    parser.add_argument("--training-input-scale", default="fastai", choices=["fastai", "0_1", "0_255", "0_65535", "auto", "max"], help="Training tensor scale. ArcGIS prepare_data commonly uses fastai [-1,1]; use max with --max-input-value only for raw DN training tensors.")
    parser.add_argument("--model-padding", type=int, default=0, help="Default PRF padding/overlap pixels.")
    parser.add_argument("--confidence-threshold", type=float, default=0.0, help="Assign low-confidence pixels to class 0 during inference.")

    parser.add_argument("--skip-environment-check", action="store_true", help="Skip DINOv3 repo/weights forward test.")
    parser.add_argument("--no-aggressive-repo-trim", action="store_true", help="Copy a less-trimmed DINOv3 repo snapshot into DLPK.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    print(f"[DINOv3 v2.32-rs311 BOOT] Python: {sys.executable}", flush=True)
    print(f"[DINOv3 v2.32-rs311 BOOT] Working directory: {os.getcwd()}", flush=True)
    print(f"[DINOv3 v2.32-rs311 BOOT] Training data argument count: {len(args.training_data)}", flush=True)
    print(f"[DINOv3 v2.32-rs311 BOOT] Importing dinov3_arcgis_v2 ...", flush=True)
    try:
        from dinov3_arcgis_v2 import train_and_package_v2
    except Exception:
        print("[DINOv3 v2.32-rs311 BOOT] Failed to import dinov3_arcgis_v2", flush=True)
        traceback.print_exc()
        raise
    print(f"[DINOv3 v2.32-rs311 BOOT] dinov3_arcgis_v2 import OK", flush=True)
    freeze_backbone = bool(args.freeze_backbone) and not bool(args.unfreeze_backbone)

    outputs = train_and_package_v2(
        training_data_dir=args.training_data,
        output_root=args.output_root,
        repo_dir=args.repo_dir,
        backbone_weights_path=args.backbone_weights,
        model_name=args.model_name,
        chip_size=args.chip_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        freeze_backbone=freeze_backbone,
        decoder_channels=args.decoder_channels,
        decoder_type=args.decoder_type,
        dropout=args.dropout,
        dinov3_model_name=args.dinov3_model_name,
        embed_dim=args.embed_dim,
        loss_mode=args.loss_mode,
        class_weights=args.class_weights,
        focal_gamma=args.focal_gamma,
        ignore_index=args.ignore_index,
        val_split_pct=args.val_split_pct,
        seed=args.seed,
        use_early_stopping=not args.no_early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        evaluate_after_training=not args.no_eval,
        max_validation_batches=args.max_validation_batches,
        extract_bands=args.extract_bands,
        input_scale=args.input_scale,
        max_input_value=args.max_input_value,
        input_channels=args.input_channels,
        input_adapter=args.input_adapter,
        training_input_scale=args.training_input_scale,
        model_padding=args.model_padding,
        confidence_threshold=args.confidence_threshold,
        run_environment_check=not args.skip_environment_check,
        aggressive_repo_trim=not args.no_aggressive_repo_trim,
        log_callback=lambda msg: print(msg, flush=True),
    )

    print("Training and packaging complete.")
    print(f"Deployment checkpoint: {outputs.deployment_checkpoint}")
    print(f"EMD: {outputs.emd_path}")
    print(f"DLPK TO USE IN ARCGIS PRO: {outputs.dlpk_path}")
    print(f"Manifest: {outputs.manifest_path}")
    if outputs.metrics_json:
        print(f"Metrics JSON: {outputs.metrics_json}")
    if outputs.epoch_log_csv:
        print(f"Epoch progress CSV: {outputs.epoch_log_csv}")
    if outputs.metrics_csv:
        print(f"Per-class metrics CSV: {outputs.metrics_csv}")

    # v2.32-rs311: ArcGIS Pro / arcgis.learn / native CUDA libraries can
    # occasionally raise a fatal interpreter-finalization error *after* all
    # training artifacts and the DLPK have already been written.  Exit the
    # child process immediately after a confirmed successful export so ArcGIS
    # Pro receives exit code 0 and does not mark a successful run as failed.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
