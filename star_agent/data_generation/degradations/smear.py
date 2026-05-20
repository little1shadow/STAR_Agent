#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""物理约束版 smear 退化。

smear 主要模拟强星/目标饱和后沿传感器读出方向产生的电荷拖尾，以及行列读出偏置。
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

from ._single_common import normalize01, run_standard_batch, set_seed


SMEAR_MODES = ["column_bias", "row_bias", "bright_source_vertical", "mixed"]

LEVEL_PARAMS = {
    1: {"bias": 0.0012, "trail": 0.010, "source_q": 0.9975, "width": 0.45},
    2: {"bias": 0.0025, "trail": 0.020, "source_q": 0.9950, "width": 0.65},
    3: {"bias": 0.0055, "trail": 0.042, "source_q": 0.9900, "width": 0.90},
    4: {"bias": 0.0100, "trail": 0.082, "source_q": 0.9820, "width": 1.25},
    5: {"bias": 0.0170, "trail": 0.145, "source_q": 0.9700, "width": 1.70},
}

DEFAULT_MASK_THRESHOLD_BY_LEVEL = {1: 0.004, 2: 0.007, 3: 0.012, 4: 0.020, 5: 0.032}


def _vertical_trails(luma: np.ndarray, level: int) -> np.ndarray:
    """从强光源生成竖直 smear 拖尾。

    输入:
    - `luma`: 单通道亮度图。
    - `level`: 退化等级。

    输出:
    - HxW 拖尾场。
    """

    p = LEVEL_PARAMS[level]
    h, _w = luma.shape
    bright = luma >= np.quantile(luma, p["source_q"])
    bright = maximum_filter(bright.astype(np.float32), size=(3, 3))
    col_profile = bright.max(axis=0) * np.maximum(luma, bright).max(axis=0)
    col_profile = gaussian_filter(col_profile, sigma=p["width"])
    decay_down = np.linspace(1.0, np.random.uniform(0.30, 0.62), h, dtype=np.float32)[:, None]
    decay_up = np.linspace(np.random.uniform(0.18, 0.44), 1.0, h, dtype=np.float32)[:, None]
    field = np.maximum(decay_down, 0.45 * decay_up) * col_profile[None, :]
    return normalize01(gaussian_filter(field, sigma=(4.0 + level * 1.2, p["width"])))


def _readout_bias(shape: tuple[int, int], axis: str, level: int) -> np.ndarray:
    """生成行/列读出偏置。

    输入:
    - `shape`: 图像高宽。
    - `axis`: `column` 或 `row`。
    - `level`: 退化等级。

    输出:
    - HxW 偏置场。
    """

    h, w = shape
    p = LEVEL_PARAMS[level]
    if axis == "column":
        profile = gaussian_filter(np.random.randn(w).astype(np.float32), sigma=np.random.uniform(0.8, 3.2))
        profile = profile / max(float(np.std(profile)), 1e-6) * p["bias"]
        return np.tile(profile[None, :], (h, 1)).astype(np.float32)
    profile = gaussian_filter(np.random.randn(h).astype(np.float32), sigma=np.random.uniform(0.8, 3.2))
    profile = profile / max(float(np.std(profile)), 1e-6) * p["bias"]
    return np.tile(profile[:, None], (1, w)).astype(np.float32)


def add_smear(
    image: np.ndarray,
    level: int,
    seed: int | None = None,
    mode: str | None = None,
    mask_threshold_by_level: dict[int, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """添加 smear 退化。

    输入:
    - `image`: `[0, 1]` RGB 图像。
    - `level`: 退化等级。
    - `seed`: 随机种子。
    - `mode`: smear 模式。
    - `mask_threshold_by_level`: mask 阈值。

    输出:
    - 退化图、mask、连续场、meta。
    """

    set_seed(seed)
    if mode is None:
        mode = SMEAR_MODES[np.random.randint(len(SMEAR_MODES))]
    if mode not in SMEAR_MODES:
        raise ValueError(f"Unknown smear mode: {mode}")

    p = LEVEL_PARAMS[level]
    luma = image.mean(axis=2)
    h, w = luma.shape
    field = np.zeros((h, w), dtype=np.float32)
    if mode in {"column_bias", "mixed"}:
        field += _readout_bias((h, w), "column", level)
    if mode in {"row_bias", "mixed"}:
        field += _readout_bias((h, w), "row", level)
    if mode in {"bright_source_vertical", "mixed"}:
        field += p["trail"] * _vertical_trails(luma, level)

    degraded = np.clip(image + field[:, :, None], 0.0, 1.0).astype(np.float32)
    diff = np.mean(np.abs(degraded - image), axis=2)
    thresholds = mask_threshold_by_level or DEFAULT_MASK_THRESHOLD_BY_LEVEL
    thr = float(thresholds.get(level, DEFAULT_MASK_THRESHOLD_BY_LEVEL[level]))
    mask = (diff >= thr).astype(np.uint8) * 255
    meta = {
        "degradation": "smear",
        "physical_model": "sensor_readout_bias+saturation_charge_smear",
        "mode": mode,
        "level": int(level),
        "seed": seed,
        "diff_mean": float(diff.mean()),
        "diff_max": float(diff.max()),
        "mask_threshold": thr,
        "mask_area_ratio": float(np.mean(mask > 0)),
    }
    return degraded, mask, normalize01(np.abs(field)), meta


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Generate star-image smear degradation.")
    parser.add_argument("--config", default="STAR_Agent/configs/data_generation/degradation_single.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--num_images", type=int, default=None)
    parser.add_argument("--level", type=int, default=None, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--mode", default=None, choices=SMEAR_MODES)
    parser.add_argument("--seed", type=int, default=131)
    return parser.parse_args()


def main() -> int:
    """命令行入口。"""

    summary = run_standard_batch(parse_args(), "smear", SMEAR_MODES, add_smear, DEFAULT_MASK_THRESHOLD_BY_LEVEL)
    print("[OK] smear batch generated")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
