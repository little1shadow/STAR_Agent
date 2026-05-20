#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""为 real selected clean 数据生成 pseudo masks / labels。"""

from __future__ import annotations

import argparse
import json
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

    parser = argparse.ArgumentParser(description="Build pseudo masks/labels for real selected clean images.")
    parser.add_argument("--clean_root", default="data/clean/real_selected_v001")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan images if there is an extra nested folder.")
    parser.add_argument("--star_cfg", default="configs/downstream/star_detection/blob.yaml")
    parser.add_argument("--blob_cfg", default="configs/downstream/target_detection/log_blob.yaml")
    parser.add_argument("--streak_cfg", default="configs/downstream/target_detection/streak.yaml")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    """脚本入口。"""

    root = add_repo_path()
    from star_agent.downstream.common.image_ops import image_files, read_gray_float, save_mask, write_json, append_jsonl
    from star_agent.downstream.star_detection.blob.detector import detect_stars, summarize_star_candidates
    from star_agent.downstream.target_detection.log_blob_detector.detector import detect_point_targets
    from star_agent.downstream.target_detection.streak_detector.detector import detect_streak_targets
    from star_agent.downstream.proxy_metrics.extract_proxy_features import build_proxy_features

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
    blob_cfg = load_yaml(root / args.blob_cfg)
    streak_cfg = load_yaml(root / args.streak_cfg)

    out_dirs = {
        "star_mask": clean_root / "masks" / "star_pseudo",
        "target_mask": clean_root / "masks" / "target_pseudo",
        "background_mask": clean_root / "masks" / "background_pseudo",
        "valid_mask": clean_root / "masks" / "valid_pseudo",
        "stars_label": clean_root / "labels" / "stars_pseudo",
        "targets_label": clean_root / "labels" / "targets_pseudo",
        "proxy": clean_root / "labels" / "downstream_proxy",
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
        blobs, blob_mask = detect_point_targets(image, star_mask, blob_cfg)
        streaks, streak_mask = detect_streak_targets(image, star_mask, streak_cfg)
        targets = blobs + streaks
        target_mask = blob_mask | streak_mask
        valid_mask = np.ones_like(star_mask, dtype=bool)
        background_mask = valid_mask & (~star_mask) & (~target_mask)
        proxy = build_proxy_features(summarize_star_candidates(stars), targets, image.shape)

        target_mask_path = out_dirs["target_mask"] / f"{stem}.png"
        bg_mask_path = out_dirs["background_mask"] / f"{stem}.png"
        valid_mask_path = out_dirs["valid_mask"] / f"{stem}.png"
        stars_label_path = out_dirs["stars_label"] / f"{stem}.json"
        targets_label_path = out_dirs["targets_label"] / f"{stem}.json"
        proxy_path = out_dirs["proxy"] / f"{stem}.json"

        save_mask(star_mask_path, star_mask)
        save_mask(target_mask_path, target_mask)
        save_mask(bg_mask_path, background_mask)
        save_mask(valid_mask_path, valid_mask)
        write_json(stars_label_path, {"mask_source": "pseudo_detector_v001", "stars": stars})
        write_json(targets_label_path, {"mask_source": "pseudo_detector_v001", "targets": targets})
        write_json(proxy_path, proxy)

        record = {
            "image_id": stem,
            "image_path": str(image_path),
            "star_mask_path": str(star_mask_path),
            "target_mask_path": str(target_mask_path),
            "background_mask_path": str(bg_mask_path),
            "valid_mask_path": str(valid_mask_path),
            "stars_label_path": str(stars_label_path),
            "targets_label_path": str(targets_label_path),
            "downstream_proxy_path": str(proxy_path),
            "num_stars": len(stars),
            "num_targets": len(targets),
            "mask_source": "pseudo_detector_v001",
            "mask_confidence": 0.7,
        }
        append_jsonl(manifest, record)
        print(f"[OK] {idx}/{len(files)} {stem} | stars={len(stars)} targets={len(targets)}")

    print(f"[DONE] pseudo manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
