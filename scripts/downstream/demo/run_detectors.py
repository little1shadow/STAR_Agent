#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""单张图下游检测器 demo。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def add_repo_path() -> Path:
    """把 STAR_Agent 根目录加入 sys.path。"""

    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root))
    return root


def load_yaml(path: str | Path) -> dict:
    """读取 YAML 配置。"""

    p = Path(path)
    if not p.is_absolute():
        p = Path.cwd() / p
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def parse_args() -> argparse.Namespace:
    """解析命令行。"""

    parser = argparse.ArgumentParser(description="Run lightweight downstream detectors on one image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output_dir", default="runs/downstream/demo")
    parser.add_argument("--star_cfg", default="configs/downstream/star_detection/blob.yaml")
    parser.add_argument("--blob_cfg", default="configs/downstream/target_detection/log_blob.yaml")
    parser.add_argument("--streak_cfg", default="configs/downstream/target_detection/streak.yaml")
    return parser.parse_args()


def main() -> int:
    """脚本入口。"""

    root = add_repo_path()
    from star_agent.downstream.common.image_ops import read_gray_float, save_mask, write_json
    from star_agent.downstream.star_detection.blob.detector import detect_stars, summarize_star_candidates
    from star_agent.downstream.target_detection.log_blob_detector.detector import detect_point_targets
    from star_agent.downstream.target_detection.streak_detector.detector import detect_streak_targets
    from star_agent.downstream.proxy_metrics.extract_proxy_features import build_proxy_features

    args = parse_args()
    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = (root / image_path).resolve() if not image_path.exists() else image_path.resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image = read_gray_float(image_path)
    stars, star_mask = detect_stars(image, load_yaml(root / args.star_cfg))
    blobs, blob_mask = detect_point_targets(image, star_mask, load_yaml(root / args.blob_cfg))
    streaks, streak_mask = detect_streak_targets(image, star_mask, load_yaml(root / args.streak_cfg))
    targets = blobs + streaks
    target_mask = blob_mask | streak_mask
    features = build_proxy_features(summarize_star_candidates(stars), targets, image.shape)

    stem = image_path.stem
    save_mask(output_dir / f"{stem}_star_mask.png", star_mask)
    save_mask(output_dir / f"{stem}_target_mask.png", target_mask)
    write_json(output_dir / f"{stem}_stars.json", stars)
    write_json(output_dir / f"{stem}_targets.json", targets)
    write_json(output_dir / f"{stem}_proxy_features.json", features)
    print(json.dumps({"image": str(image_path), "output_dir": str(output_dir), **features}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
