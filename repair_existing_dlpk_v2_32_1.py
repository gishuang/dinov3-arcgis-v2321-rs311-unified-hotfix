# -*- coding: utf-8 -*-
"""
Patch an existing DINOv3 ArcGIS DLPK/deployment folder with the v2.32 unified diagnostic inference
runtime fix and ensure that a DINOv3 repo snapshot is bundled in the DLPK.

This fixes ArcGIS Pro PRF initialization errors such as:
  FileNotFoundError: DINOv3 repo snapshot not found: ...\dinov3_repo

Usage:
  python repair_existing_dlpk_v2_32.py --input path\to\model.dlpk --output path\to\model_v232.dlpk --repo-dir C:\src\dinov3
  python repair_existing_dlpk_v2_32.py --input path\to\deployment_folder --output path\to\deployment_folder_v232 --repo-dir C:\src\dinov3
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATCH_FILES = ["dinov3_inference_v2.py", "dinov3_arcgis_v2.py"]
REPO_DIR_NAME = "dinov3_repo"


def _copy_tree(src: Path, dst: Path, *, ignore_patterns=None) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.mkdir(parents=True, exist_ok=True)
    ignore_patterns = ignore_patterns or []
    for item in src.iterdir():
        if any(item.match(pat) for pat in ignore_patterns):
            continue
        target = dst / item.name
        if item.is_dir():
            if item.name in {".git", ".github", "__pycache__", ".ipynb_checkpoints", "notebooks", "docs"}:
                continue
            _copy_tree(item, target, ignore_patterns=ignore_patterns)
        else:
            if item.suffix.lower() in {".pyc", ".pyo", ".ipynb", ".png", ".jpg", ".jpeg", ".gif"}:
                continue
            shutil.copy2(item, target)


def _zip_dir(src_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in src_dir.rglob("*"):
            if path.is_file():
                if path.suffix.lower() == ".dlpk":
                    continue
                zf.write(path, path.relative_to(src_dir).as_posix())


def _find_deployment_root(work: Path) -> Path:
    emds = list(work.rglob("*.emd"))
    if not emds:
        raise FileNotFoundError(f"No .emd file found under {work}")
    for emd in emds:
        if (emd.parent / "dinov3_inference_v2.py").exists():
            return emd.parent
    return emds[0].parent


def _patch_folder(root: Path, repo_dir: Path | None) -> Path:
    deployment_root = _find_deployment_root(root)

    for name in PATCH_FILES:
        src = HERE / name
        if not src.exists():
            raise FileNotFoundError(f"Patch file missing: {src}")
        matches = list(root.rglob(name))
        if not matches:
            matches = [deployment_root / name]
        for dst in matches:
            shutil.copy2(src, dst)

    repo_snapshot = deployment_root / REPO_DIR_NAME
    if not repo_snapshot.exists():
        if repo_dir is None:
            raise FileNotFoundError(
                f"DLPK/deployment folder does not contain {REPO_DIR_NAME}. "
                "Please pass --repo-dir C:\\src\\dinov3 so the repair script can bundle a repo snapshot."
            )
        if not (repo_dir / "dinov3").exists():
            raise FileNotFoundError(f"Invalid DINOv3 repo; expected subfolder 'dinov3': {repo_dir}")
        _copy_tree(repo_dir, repo_snapshot)
        print(f"Bundled DINOv3 repo snapshot: {repo_snapshot}")
    else:
        print(f"Existing DINOv3 repo snapshot found: {repo_snapshot}")

    for emd in root.rglob("*.emd"):
        try:
            data = json.loads(emd.read_text(encoding="utf-8"))
            data["BackboneRepoDir"] = f".\\{REPO_DIR_NAME}"
            data["Version"] = "2026.05.v2.32.1-rs311"
            data["ToolkitVersion"] = "v2.32.1-rs311-unified-hotfix"
            data["InferenceRuntimeFix"] = "v2.32 unified training/inference, diagnostic output modes, background suppression test, first-tile histogram, class-value mapping, radiometry overrides, bundled repo snapshot, output dtype fix, and parallel timeout-safe inference batch cap"
            data["InferenceBatchSize"] = 1
            data["MaxInferenceBatchSize"] = 1
            data.setdefault("OutputRawClassIndex", False)
            data.setdefault("OutputMode", "class_value")
            data.setdefault("SuppressBackground", False)
            data.setdefault("BackgroundIndex", 0)
            data.setdefault("DebugFirstTile", False)
            data.setdefault("InferenceRadiometryNote", "If 16-bit inference returns mostly 0/background, test input_scale auto or input_scale max with max_input_value matching the sensor DN range, e.g. 4095/10000/16383/65535.")
            emd.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as ex:
            print(f"Warning: unable to update EMD {emd}: {ex}")
    return deployment_root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Existing .dlpk file or deployment folder")
    ap.add_argument("--output", required=True, help="Output .dlpk path or output folder")
    ap.add_argument("--repo-dir", default=None, help="Local DINOv3 repository folder, e.g. C:\\src\\dinov3")
    args = ap.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    repo_dir = Path(args.repo_dir).resolve() if args.repo_dir else None
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    with tempfile.TemporaryDirectory(prefix="dinov3_dlpk_patch_v232_") as td:
        work = Path(td) / "work"
        if input_path.is_file():
            with zipfile.ZipFile(input_path, "r") as zf:
                zf.extractall(work)
        else:
            _copy_tree(input_path, work)

        _patch_folder(work, repo_dir)

        if output_path.suffix.lower() == ".dlpk" or input_path.is_file():
            if output_path.suffix.lower() != ".dlpk":
                output_path = output_path.with_suffix(".dlpk")
            _zip_dir(work, output_path)
            print(f"Patched DLPK written: {output_path}")
        else:
            if output_path.exists():
                shutil.rmtree(output_path)
            _copy_tree(work, output_path)
            print(f"Patched deployment folder written: {output_path}")


if __name__ == "__main__":
    main()
