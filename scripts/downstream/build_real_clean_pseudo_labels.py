#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""为 real selected clean 数据生成轻量 star/background pseudo masks。

说明:
- 该脚本是 tetra3rs 不可用时的轻量备用方案。
- 真实 clean 阶段只生成 star/background/valid 信息。
- target 会在后续数据构建阶段单独注入，因此这里不生成 target pseudo，也不生成 downstream proxy。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml


def add_repo_path() -> Path:
    """把 STAR_Agent 根目录加入 sys.path。"""

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    return root


def load_yaml(path: str | Path) -> dict:
    """读取 YAML 配置。"""

    p = Path(path)
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Build lightweight star/background pseudo masks for real selected clean images.")
    parser.add_argument("--clean_root", default="data/clean/real_selected_v001")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan images if there is an extra nested folder.")
    parser.add_argument("--star_cfg", default="configs/downstream/star_detection/blob.yaml")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    """脚本入口。

    功能:
    - 用轻量 blob detector 为真实 clean 图生成 star pseudo mask。
    - 由 `valid_mask - star_mask` 得到 background pseudo mask。
    - 不生成 target/proxy，避免把真实 clean 中的未知弱目标误当成 GT。
    """

    root = add_repo_path()
    from star_agent.downstream.common.image_ops import image_files, read_gray_float, save_mask, write_json, append_jsonl
    from star_agent.downstream.star_detection.blob.detector import detect_stars

    args = parse_args()
    clean_root = Path(args.clean_root)
    if not clean_root.is_absolute():
        clean_root = (root / clean_root).resolve()
    image_root = clean_root / "images"
    files = image_files(image_root, recursive=args.recursive)
    if args.max_images is not None:
        files = files[: args.max_images]
    if not files:
        raise RuntimeError(f"No images found in {image_root}")

    star_cfg = load_yaml(root / args.star_cfg)

    out_dirs = {
        "star_mask": clean_root / "masks" / "star_pseudo",
        "background_mask": clean_root / "masks" / "background_pseudo",
        "valid_mask": clean_root / "masks" / "valid_pseudo",
        "stars_label": clean_root / "labels" / "stars_pseudo",
        "metrics": clean_root / "labels" / "star_pseudo_metrics",
    }
    for path in out_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    manifest = clean_root / "manifest_pseudo.jsonl"
    if args.overwrite and manifest.exists():
        manifest.unlink()

    for idx, image_path in enumerate(files, start=1):
        stem = image_path.stem
        star_mask_path = out_dirs["star_mask"] / f"{stem}.png"
        if star_mask_path.exists() and not args.overwrite:
            print(f"[SKIP] {idx}/{len(files)} {stem}: pseudo labels exist")
            continue

        image = read_gray_float(image_path)
        stars, star_mask = detect_stars(image, star_cfg)
        valid_mask = np.ones_like(star_mask, dtype=bool)
        background_mask = valid_mask & (~star_mask)

        bg_mask_path = out_dirs["background_mask"] / f"{stem}.png"
        valid_mask_path = out_dirs["valid_mask"] / f"{stem}.png"
        stars_label_path = out_dirs["stars_label"] / f"{stem}.json"
        metrics_path = out_dirs["metrics"] / f"{stem}.json"

        save_mask(star_mask_path, star_mask)
        save_mask(bg_mask_path, background_mask)
        save_mask(valid_mask_path, valid_mask)
        write_json(stars_label_path, {"mask_source": "pseudo_detector_v001", "stars": stars})
        write_json(
            metrics_path,
            {
                "image_id": stem,
                "image_path": str(image_path),
                "num_stars": len(stars),
                "star_mask_area_px": int(star_mask.sum()),
                "star_mask_area_ratio": float(star_mask.sum() / max(1, star_mask.size)),
                "mask_source": "pseudo_detector_v001",
                "mask_confidence": 0.65,
                "has_target": False,
                "target_policy": "not_injected_yet",
            },
        )

        record = {
            "image_id": stem,
            "image_path": str(image_path),
            "star_mask_path": str(star_mask_path),
            "background_mask_path": str(bg_mask_path),
            "valid_mask_path": str(valid_mask_path),
            "stars_label_path": str(stars_label_path),
            "star_metrics_path": str(metrics_path),
            "num_stars": len(stars),
            "has_target": False,
            "target_policy": "not_injected_yet",
            "mask_source": "pseudo_detector_v001",
            "mask_confidence": 0.65,
        }
        append_jsonl(manifest, record)
        print(f"[OK] {idx}/{len(files)} {stem} | stars={len(stars)}")

    print(f"[DONE] pseudo manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
