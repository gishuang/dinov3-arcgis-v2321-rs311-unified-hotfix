from __future__ import annotations

import argparse
import json
from pathlib import Path

from dinov3_arcgis_v2 import verify_runtime_environment


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Verify DINOv3 + ArcGIS Pro deep learning runtime for the v2 toolkit.")
    p.add_argument("--repo-dir", required=True, help="Local DINOv3 repository folder.")
    p.add_argument("--backbone-weights", required=True, help="DINOv3 SAT-493M weights .pth file.")
    p.add_argument("--dinov3-model-name", default="dinov3_vitl16")
    p.add_argument("--embed-dim", type=int, default=1024)
    p.add_argument("--chip-size", type=int, default=224)
    p.add_argument("--out-json", default=None, help="Optional report JSON path.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    report = verify_runtime_environment(
        repo_dir=args.repo_dir,
        backbone_weights_path=args.backbone_weights,
        dinov3_model_name=args.dinov3_model_name,
        embed_dim=args.embed_dim,
        chip_size=args.chip_size,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(text, encoding="utf-8")
    if not report.get("forward_test"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
