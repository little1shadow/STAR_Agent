#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""mask 生成工具。"""

from __future__ import annotations

import numpy as np

from ..clean_simulation.star_projector import ProjectedStars


def build_star_mask(
    stars: ProjectedStars,
    height: int,
    width: int,
    radius_px: int,
) -> np.ndarray:
    """根据星点位置生成星点 mask。

    输入:
    - `stars`: 投影后的星点。
    - `height`, `width`: 图像尺寸。
    - `radius_px`: 每颗星 mask 半径。

    输出:
    - uint8 mask，星点区域为 255，背景为 0。

    说明:
    - 这是用于训练/评价的近似 star mask；
    - 后续可以根据 PSF flux threshold 生成更精细 mask。
    """

    mask = np.zeros((height, width), dtype=np.uint8)
    radius = int(max(1, radius_px))
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    disk = (xx * xx + yy * yy) <= radius * radius

    for x, y in zip(stars.x_px, stars.y_px, strict=False):
        cx = int(round(float(x)))
        cy = int(round(float(y)))
        x1 = max(0, cx - radius)
        y1 = max(0, cy - radius)
        x2 = min(width, cx + radius + 1)
        y2 = min(height, cy + radius + 1)
        if x1 >= x2 or y1 >= y2:
            continue

        dx1 = x1 - (cx - radius)
        dy1 = y1 - (cy - radius)
        dx2 = dx1 + (x2 - x1)
        dy2 = dy1 + (y2 - y1)
        mask[y1:y2, x1:x2][disk[dy1:dy2, dx1:dx2]] = 255
    return mask


def build_background_mask(star_mask: np.ndarray) -> np.ndarray:
    """根据 star mask 生成背景 mask。

    输入:
    - `star_mask`: 星点 mask。

    输出:
    - uint8 mask，背景区域为 255，星点区域为 0。
    """

    return np.where(star_mask > 0, 0, 255).astype(np.uint8)


def build_valid_mask(height: int, width: int) -> np.ndarray:
    """生成有效区域 mask。

    输入:
    - `height`, `width`: 图像尺寸。

    输出:
    - 全 255 的 uint8 mask。
    """

    return np.full((height, width), 255, dtype=np.uint8)


def build_empty_target_mask(height: int, width: int) -> np.ndarray:
    """生成空目标 mask。

    输入:
    - `height`, `width`: 图像尺寸。

    输出:
    - 全 0 的 uint8 mask。

    用途:
    - 当前 clean 仿真阶段只生成星点，不注入空间目标；
    - 但为了数据结构稳定，仍然为每张图保存 target mask；
    - 后续加入 target injector 后，这里会替换为真实目标区域。
    """

    return np.zeros((height, width), dtype=np.uint8)
