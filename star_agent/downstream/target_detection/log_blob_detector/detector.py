#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""点状弱目标 LoG/DoG 检测器。"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi

from ...common.image_ops import connected_components, robust_background, weighted_centroid, component_shape


def detect_point_targets(
    image: np.ndarray,
    star_mask: np.ndarray | None = None,
    cfg: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """检测点状或微椭圆弱目标。

    输入:
    - `image`: HxW `[0, 1]` 灰度图。
    - `star_mask`: 可选星点 mask，用于抑制普通星点误检。
    - `cfg`: 检测配置。

    输出:
    - `(target_candidates, target_mask)`。
    """

    cfg = cfg or {}
    sigma_small = float(cfg.get("sigma_small", 0.9))
    sigma_large = float(cfg.get("sigma_large", 2.2))
    threshold_sigma = float(cfg.get("threshold_sigma", 5.0))
    min_area = int(cfg.get("min_area", 2))
    max_area = int(cfg.get("max_area", 120))
    min_snr = float(cfg.get("min_snr", 5.0))
    star_exclusion_dilation = int(cfg.get("star_exclusion_dilation", 2))

    dog = ndi.gaussian_filter(image, sigma=sigma_small) - ndi.gaussian_filter(image, sigma=sigma_large)
    dog = np.clip(dog, 0.0, None)
    med, sigma = robust_background(dog)
    binary = dog > (med + threshold_sigma * sigma)
    if star_mask is not None and star_mask.any():
        excluded = ndi.binary_dilation(star_mask > 0, iterations=star_exclusion_dilation)
        binary = binary & (~excluded)

    candidates: list[dict[str, Any]] = []
    mask = np.zeros_like(image, dtype=bool)
    bg, img_sigma = robust_background(image)
    for comp in connected_components(binary):
        coords = comp["coords"]
        area = int(coords.shape[0])
        if area < min_area or area > max_area:
            continue
        yy = coords[:, 0]
        xx = coords[:, 1]
        peak = float((image[yy, xx] - bg).max())
        snr = peak / max(img_sigma, 1e-6)
        if snr < min_snr:
            continue
        cx, cy, flux = weighted_centroid(image, coords, bg)
        shape = component_shape(coords)
        confidence = float(min(1.0, 0.15 * snr + 0.02 * area))
        candidates.append(
            {
                "id": len(candidates),
                "target_type": "point_blob",
                "detector_family": "log_blob",
                "x": cx,
                "y": cy,
                "snr": float(snr),
                "flux": flux,
                "confidence": confidence,
                **shape,
            }
        )
        mask[yy, xx] = True
    if mask.any():
        mask = ndi.binary_dilation(mask, iterations=1)
    return candidates, mask
