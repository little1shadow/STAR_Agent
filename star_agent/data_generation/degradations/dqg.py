#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""物理约束版 DQG 退化。

DQG 被建模为方向性大面积弥散光/背景梯度入侵：
- 从某个边或角整体进入画面；
- 没有太阳一角、没有明显点光源；
- 没有规则强弧线，和 solar stray light 保持物理/视觉边界。
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from ._single_common import normalize01, run_standard_batch, set_seed


DQG_MODES = ["left", "right", "top", "bottom", "top_left", "top_right", "bottom_left", "bottom_right", "diffuse"]

LEVEL_PARAMS = {
    1: {"amp": 0.010, "mix": 0.004, "lift": 0.000},
    2: {"amp": 0.025, "mix": 0.010, "lift": 0.001},
    3: {"amp": 0.060, "mix": 0.024, "lift": 0.004},
    4: {"amp": 0.130, "mix": 0.052, "lift": 0.010},
    5: {"amp": 0.230, "mix": 0.092, "lift": 0.020},
}

DEFAULT_MASK_THRESHOLD_BY_LEVEL = {1: 0.010, 2: 0.020, 3: 0.040, 4: 0.075, 5: 0.115}


def _direction_vector(mode: str) -> tuple[float, float]:
    """把 DQG 模式转换成入侵方向向量。"""

    table = {
        "left": (1.0, 0.0),
        "right": (-1.0, 0.0),
        "top": (0.0, 1.0),
        "bottom": (0.0, -1.0),
        "top_left": (1.0, 1.0),
        "top_right": (-1.0, 1.0),
        "bottom_left": (1.0, -1.0),
        "bottom_right": (-1.0, -1.0),
        "diffuse": (np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)),
    }
    vx, vy = table[mode]
    n = math.sqrt(vx * vx + vy * vy) + 1e-8
    return vx / n, vy / n


def _dqg_field(h: int, w: int, mode: str, level: int) -> np.ndarray:
    """生成方向性大面积 DQG 光场。"""

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn = (xx - (w - 1) / 2.0) / max(w, 1)
    yn = (yy - (h - 1) / 2.0) / max(h, 1)
    vx, vy = _direction_vector(mode)
    coord = vx * xn + vy * yn
    coord = normalize01(coord)
    if mode == "diffuse":
        coord = 0.55 + 0.45 * normalize01(gaussian_filter(np.random.randn(h, w).astype(np.float32), sigma=90.0))

    gamma = np.random.uniform(0.85, 1.75)
    field = coord ** gamma
    low = gaussian_filter(np.random.randn(h, w).astype(np.float32), sigma=np.random.uniform(32.0, 96.0))
    low = 0.82 + 0.32 * normalize01(low)
    field = gaussian_filter(field * low, sigma=np.random.uniform(7.0, 20.0))

    # DQG 是整片弥散入侵，因此用一个柔和底座，避免只有边缘一点亮。
    base = 0.10 + 0.05 * level
    field = normalize01(field)
    return np.clip(base + (1.0 - base) * field, 0.0, 1.0).astype(np.float32)


def add_dqg(
    image: np.ndarray,
    level: int,
    seed: int | None = None,
    mode: str | None = None,
    mask_threshold_by_level: dict[int, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """添加 DQG 退化。"""

    set_seed(seed)
    if mode is None:
        mode = DQG_MODES[np.random.randint(len(DQG_MODES))]
    if mode not in DQG_MODES:
        raise ValueError(f"Unknown dqg mode: {mode}")

    p = LEVEL_PARAMS[level]
    h, w = image.shape[:2]
    field = _dqg_field(h, w, mode, level)
    raw = p["amp"] * field
    blurred = gaussian_filter(image, sigma=(2.5, 2.5, 0.0))
    wash = np.clip(p["mix"] * field, 0.0, 0.26)
    degraded = image * (1.0 - wash[:, :, None]) + blurred * wash[:, :, None]
    degraded = np.clip(degraded + raw[:, :, None] + p["lift"], 0.0, 1.0).astype(np.float32)

    thresholds = mask_threshold_by_level or DEFAULT_MASK_THRESHOLD_BY_LEVEL
    thr = float(thresholds.get(level, DEFAULT_MASK_THRESHOLD_BY_LEVEL[level]))
    mask = (raw >= thr).astype(np.uint8) * 255
    meta = {
        "degradation": "dqg",
        "physical_model": "directional_diffuse_glow_without_point_source",
        "mode": mode,
        "level": int(level),
        "seed": seed,
        "field_mean_raw": float(raw.mean()),
        "field_max_raw": float(raw.max()),
        "mask_threshold": thr,
        "mask_area_ratio": float(np.mean(mask > 0)),
    }
    return degraded, mask, field, meta


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Generate DQG degradation.")
    parser.add_argument("--config", default="STAR_Agent/configs/data_generation/degradation_single.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_root", default="STAR_Agent/data/degraded/single/dqg")
    parser.add_argument("--num_images", type=int, default=None)
    parser.add_argument("--level", type=int, default=None, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--mode", default=None, choices=DQG_MODES)
    parser.add_argument("--seed", type=int, default=131)
    return parser.parse_args()


def main() -> int:
    """命令行入口。"""

    summary = run_standard_batch(parse_args(), "dqg", DQG_MODES, add_dqg, DEFAULT_MASK_THRESHOLD_BY_LEVEL)
    print("[OK] dqg batch generated")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
