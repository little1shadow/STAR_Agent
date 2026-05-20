#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""物理约束版 dead/hot pixels 退化。

模拟探测器坏点、热点、坏点簇和死列。坏点是固定像元响应异常，不应像噪声一样连续分布。
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation

from ._single_common import normalize01, run_standard_batch, set_seed


DEAD_PIXEL_MODES = ["sparse_hot", "sparse_dead", "clustered_hot", "dead_column", "mixed"]

LEVEL_PARAMS = {
    1: {"prob": 0.000015, "clusters": 1, "cols": 0, "hot": 0.45},
    2: {"prob": 0.000040, "clusters": 2, "cols": 0, "hot": 0.65},
    3: {"prob": 0.000100, "clusters": 4, "cols": 1, "hot": 0.82},
    4: {"prob": 0.000220, "clusters": 8, "cols": 2, "hot": 0.94},
    5: {"prob": 0.000430, "clusters": 14, "cols": 3, "hot": 1.00},
}

DEFAULT_MASK_THRESHOLD_BY_LEVEL = {1: 0.08, 2: 0.10, 3: 0.12, 4: 0.14, 5: 0.16}


def _cluster_mask(h: int, w: int, count: int) -> np.ndarray:
    """生成坏点簇 mask。"""

    mask = np.zeros((h, w), dtype=bool)
    for _ in range(count):
        y = np.random.randint(2, h - 2)
        x = np.random.randint(2, w - 2)
        mask[y, x] = True
    return binary_dilation(mask, iterations=np.random.randint(1, 3))


def add_dead_pixels(
    image: np.ndarray,
    level: int,
    seed: int | None = None,
    mode: str | None = None,
    mask_threshold_by_level: dict[int, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """添加 dead/hot pixels 退化。"""

    set_seed(seed)
    if mode is None:
        mode = DEAD_PIXEL_MODES[np.random.randint(len(DEAD_PIXEL_MODES))]
    if mode not in DEAD_PIXEL_MODES:
        raise ValueError(f"Unknown dead pixel mode: {mode}")

    p = LEVEL_PARAMS[level]
    h, w = image.shape[:2]
    degraded = image.copy()
    mask_bool = np.zeros((h, w), dtype=bool)

    if mode in {"sparse_hot", "mixed"}:
        hot = np.random.random((h, w)) < p["prob"]
        degraded[hot] = np.maximum(degraded[hot], p["hot"] * np.random.uniform(0.85, 1.0))
        mask_bool |= hot
    if mode in {"sparse_dead", "mixed"}:
        dead = np.random.random((h, w)) < p["prob"] * 1.15
        degraded[dead] = degraded[dead] * np.random.uniform(0.00, 0.08)
        mask_bool |= dead
    if mode in {"clustered_hot", "mixed"}:
        cluster = _cluster_mask(h, w, p["clusters"])
        degraded[cluster] = np.maximum(degraded[cluster], p["hot"] * np.random.uniform(0.70, 1.0))
        mask_bool |= cluster
    if mode in {"dead_column", "mixed"} and p["cols"] > 0:
        col_count = p["cols"] if mode == "dead_column" else max(1, p["cols"] - 1)
        cols = np.random.choice(w, size=col_count, replace=False)
        for col in cols:
            width = np.random.choice([1, 1, 2])
            c0, c1 = max(0, col - width + 1), min(w, col + width)
            gain = np.random.uniform(0.00, 0.16)
            degraded[:, c0:c1] *= gain
            mask_bool[:, c0:c1] = True

    diff = np.mean(np.abs(degraded - image), axis=2)
    thresholds = mask_threshold_by_level or DEFAULT_MASK_THRESHOLD_BY_LEVEL
    thr = float(thresholds.get(level, DEFAULT_MASK_THRESHOLD_BY_LEVEL[level]))
    mask_bool |= diff >= thr
    mask = mask_bool.astype(np.uint8) * 255
    meta = {
        "degradation": "dead_pixels",
        "physical_model": "defective_detector_pixels_and_columns",
        "mode": mode,
        "level": int(level),
        "seed": seed,
        "defective_area_ratio": float(np.mean(mask_bool)),
        "diff_max": float(diff.max()),
        "mask_threshold": thr,
        "mask_area_ratio": float(np.mean(mask > 0)),
    }
    return degraded.astype(np.float32), mask, normalize01(diff), meta


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Generate dead pixel degradation.")
    parser.add_argument("--config", default="STAR_Agent/configs/data_generation/degradation_single.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_root", default="STAR_Agent/data/degraded/single/dead_pixels")
    parser.add_argument("--num_images", type=int, default=None)
    parser.add_argument("--level", type=int, default=None, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--mode", default=None, choices=DEAD_PIXEL_MODES)
    parser.add_argument("--seed", type=int, default=131)
    return parser.parse_args()


def main() -> int:
    """命令行入口。"""

    summary = run_standard_batch(parse_args(), "dead_pixels", DEAD_PIXEL_MODES, add_dead_pixels, DEFAULT_MASK_THRESHOLD_BY_LEVEL)
    print("[OK] dead pixels batch generated")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
