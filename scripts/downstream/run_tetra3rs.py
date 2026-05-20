#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""单张图 tetra3rs 星点检测与 star pseudo mask 生成脚本。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml


def add_repo_path() -> Path:
    """把 STAR_Agent 根目录加入 `sys.path`。

    输入:
    - 无。

    输出:
    - STAR_Agent 根目录 Path。
    """

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    return root


def load_yaml(path: str | Path) -> dict:
    """读取 YAML 配置。

    输入:
    - `path`: YAML 路径。

    输出:
    - 配置字典。
    """

    p = Path(path)
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    输入:
    - 命令行参数。

    输出:
    - argparse Namespace。
    """

    parser = argparse.ArgumentParser(description="Run tetra3rs centroid extraction and build star pseudo mask.")
    parser.add_argument("--image", required=True, help="Input clean star image.")
    parser.add_argument("--output_dir", required=True, help="Directory for star_mask/centroids/metrics outputs.")
    parser.add_argument("--cfg", default="configs/downstream/star_matching/tetra3rs.yaml")
    return parser.parse_args()


def main() -> int:
    """脚本入口。

    功能:
    - 调用 tetra3rs 提取星点质心。
    - 根据质心生成像素级 star mask。
    - 输出 `star_mask.png`、`centroids.json`、`metrics.json`。
    """

    root = add_repo_path()
    from star_agent.downstream.common.image_ops import ensure_dir, save_mask, write_json
    from star_agent.downstream.star_matching.tetra3rs_adapter.star_mask import detect_stars_with_tetra3rs

    args = parse_args()
    cfg_path = Path(args.cfg)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = load_yaml(cfg_path)

    image_path = Path(args.image).resolve()
    output_dir = ensure_dir(Path(args.output_dir).resolve())
    result = detect_stars_with_tetra3rs(image_path, cfg)

    star_mask_path = output_dir / "star_mask.png"
    centroids_path = output_dir / "centroids.json"
    metrics_path = output_dir / "metrics.json"
    valid_mask_path = output_dir / "valid_mask.png"
    background_mask_path = output_dir / "background_mask.png"

    valid_mask = np.ones_like(result.star_mask, dtype=bool)
    background_mask = valid_mask & (~result.star_mask)

    save_mask(star_mask_path, result.star_mask)
    save_mask(valid_mask_path, valid_mask)
    save_mask(background_mask_path, background_mask)
    write_json(centroids_path, {"centroids": result.centroids, "metrics": result.metrics})
    write_json(metrics_path, result.metrics)

    print(f"[OK] image: {image_path}")
    print(f"[OK] centroids: {len(result.centroids)}")
    print(f"[OK] star_mask: {star_mask_path}")
    print(f"[OK] metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
