#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""图像保存和归一化工具。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .io import ensure_dir


def to_uint8_image(image: np.ndarray, max_adu: float = 255.0) -> np.ndarray:
    """将 float 图像裁剪并转换为 uint8。

    输入:
    - `image`: float 图像。
    - `max_adu`: 最大灰度值。

    输出:
    - uint8 图像。

    说明:
    - 当前 V1 输出 8-bit PNG，方便快速查看；
    - 后续如果要保留科学计数，可以额外保存 16-bit PNG 或 FITS。
    """

    clipped = np.clip(image, 0.0, float(max_adu))
    return np.rint(clipped).astype(np.uint8)


def save_gray_png(path: str | Path, image: np.ndarray, max_adu: float = 255.0) -> None:
    """保存灰度 PNG。

    输入:
    - `path`: 输出路径。
    - `image`: float 或 uint8 图像。
    - `max_adu`: float 图像裁剪上限。

    输出:
    - 无返回值。
    """

    p = Path(path)
    ensure_dir(p.parent)
    if image.dtype != np.uint8:
        image = to_uint8_image(image, max_adu=max_adu)
    Image.fromarray(image, mode="L").save(p)


def save_rgb_png(path: str | Path, image: np.ndarray, max_adu: float = 255.0) -> None:
    """保存 RGB PNG。

    输入:
    - `path`: 输出路径。
    - `image`: 单通道或三通道图像。
    - `max_adu`: float 图像裁剪上限。

    输出:
    - 无返回值。

    说明:
    - 若输入是单通道，则复制成 RGB，方便现有 restoration pipeline 使用。
    """

    p = Path(path)
    ensure_dir(p.parent)
    if image.dtype != np.uint8:
        image = to_uint8_image(image, max_adu=max_adu)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    Image.fromarray(image, mode="RGB").save(p)
