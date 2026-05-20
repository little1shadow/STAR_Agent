#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""物理约束版星空噪声退化。

包含读出高斯噪声、光子散粒噪声、暗电流/热噪声、背景颗粒噪声和混合噪声。
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from ._single_common import normalize01, run_standard_batch, set_seed


NOISE_MODES = ["gaussian", "poisson", "dark_current", "background", "mixed"]

LEVEL_PARAMS = {
    1: {"gauss": 0.0025, "poisson": 0.006, "dark": 0.0015, "grain": 0.0020, "hot_prob": 0.000003},
    2: {"gauss": 0.0050, "poisson": 0.012, "dark": 0.0030, "grain": 0.0040, "hot_prob": 0.000010},
    3: {"gauss": 0.0100, "poisson": 0.022, "dark": 0.0070, "grain": 0.0080, "hot_prob": 0.000025},
    4: {"gauss": 0.0180, "poisson": 0.040, "dark": 0.0140, "grain": 0.0140, "hot_prob": 0.000060},
    5: {"gauss": 0.0300, "poisson": 0.070, "dark": 0.0260, "grain": 0.0240, "hot_prob": 0.000130},
}

DEFAULT_MASK_THRESHOLD_BY_LEVEL = {1: 0.006, 2: 0.010, 3: 0.017, 4: 0.026, 5: 0.040}


def _hot_pixels(shape: tuple[int, int], prob: float, gain: float) -> np.ndarray:
    """生成暗电流导致的稀疏热像元。

    输入:
    - `shape`: 图像高宽。
    - `prob`: 热像元概率。
    - `gain`: 热像元强度。

    输出:
    - HxW 热像元场。
    """

    h, w = shape
    mask = np.random.random((h, w)) < prob
    field = np.zeros((h, w), dtype=np.float32)
    field[mask] = gain * np.random.uniform(0.45, 1.0, int(mask.sum()))
    return field


def add_noise(
    image: np.ndarray,
    level: int,
    seed: int | None = None,
    mode: str | None = None,
    mask_threshold_by_level: dict[int, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """添加星空传感器噪声。

    输入:
    - `image`: `[0, 1]` RGB 图像。
    - `level`: 退化等级。
    - `seed`: 随机种子。
    - `mode`: 噪声模式。
    - `mask_threshold_by_level`: mask 阈值。

    输出:
    - 退化图、mask、连续噪声场、meta。
    """

    set_seed(seed)
    if mode is None:
        mode = NOISE_MODES[np.random.randint(len(NOISE_MODES))]
    if mode not in NOISE_MODES:
        raise ValueError(f"Unknown noise mode: {mode}")

    p = LEVEL_PARAMS[level]
    h, w = image.shape[:2]
    noise = np.zeros_like(image, dtype=np.float32)

    if mode in {"gaussian", "mixed"}:
        noise += np.random.normal(0.0, p["gauss"], image.shape).astype(np.float32)
    if mode in {"poisson", "mixed"}:
        # 用信号相关噪声近似光子散粒噪声；亮星附近噪声会略大。
        sigma = p["poisson"] * np.sqrt(np.clip(image, 0.0, 1.0) + 0.015)
        noise += np.random.normal(0.0, sigma).astype(np.float32)
    if mode in {"dark_current", "mixed"}:
        dark = np.random.gamma(shape=1.6, scale=p["dark"] / 1.6, size=(h, w)).astype(np.float32)
        dark += _hot_pixels((h, w), p["hot_prob"], p["dark"] * 7.0)
        noise += dark[:, :, None]
    if mode in {"background", "mixed"}:
        grain = gaussian_filter(np.random.randn(h, w).astype(np.float32), sigma=np.random.uniform(0.45, 1.4))
        grain = grain / max(float(np.std(grain)), 1e-6) * p["grain"]
        low = gaussian_filter(np.random.randn(h, w).astype(np.float32), sigma=np.random.uniform(18.0, 50.0))
        low = (normalize01(low) - 0.5) * p["grain"] * 1.8
        noise += (grain + low)[:, :, None]

    degraded = np.clip(image + noise, 0.0, 1.0).astype(np.float32)
    diff = np.mean(np.abs(degraded - image), axis=2)
    field = normalize01(diff)
    thresholds = mask_threshold_by_level or DEFAULT_MASK_THRESHOLD_BY_LEVEL
    thr = float(thresholds.get(level, DEFAULT_MASK_THRESHOLD_BY_LEVEL[level]))
    mask = (diff >= thr).astype(np.uint8) * 255
    meta = {
        "degradation": "noise",
        "physical_model": "read_noise+shot_noise+dark_current+background_grain",
        "mode": mode,
        "level": int(level),
        "seed": seed,
        "diff_mean": float(diff.mean()),
        "diff_max": float(diff.max()),
        "mask_threshold": thr,
        "mask_area_ratio": float(np.mean(mask > 0)),
    }
    return degraded, mask, field, meta


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Generate star-image noise degradation.")
    parser.add_argument("--config", default="STAR_Agent/configs/data_generation/degradation_single.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_root", default="STAR_Agent/data/degraded/single/noise")
    parser.add_argument("--num_images", type=int, default=None)
    parser.add_argument("--level", type=int, default=None, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--mode", default=None, choices=NOISE_MODES)
    parser.add_argument("--seed", type=int, default=131)
    return parser.parse_args()


def main() -> int:
    """命令行入口。"""

    summary = run_standard_batch(parse_args(), "noise", NOISE_MODES, add_noise, DEFAULT_MASK_THRESHOLD_BY_LEVEL)
    print("[OK] noise batch generated")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
