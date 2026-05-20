#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""下游检测器通用图像与几何工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在。

    输入:
    - `path`: 目录路径。

    输出:
    - Path 对象。
    """

    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_gray_float(path: str | Path) -> np.ndarray:
    """读取图像为 `[0, 1]` 灰度 float32。

    输入:
    - `path`: 图像路径。

    输出:
    - HxW float32 灰度图。
    """

    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0


def save_mask(path: str | Path, mask: np.ndarray) -> None:
    """保存二值 mask。

    输入:
    - `path`: 输出路径。
    - `mask`: bool/0-1/0-255 mask。

    输出:
    - 无。
    """

    p = Path(path)
    ensure_dir(p.parent)
    arr = (mask > 0).astype(np.uint8) * 255
    Image.fromarray(arr, mode="L").save(p)


def write_json(path: str | Path, data: Any) -> None:
    """写 JSON 文件。

    输入:
    - `path`: 输出路径。
    - `data`: 可 JSON 序列化对象。

    输出:
    - 无。
    """

    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """追加 JSONL 记录。

    输入:
    - `path`: JSONL 路径。
    - `record`: 一条记录。

    输出:
    - 无。
    """

    p = Path(path)
    ensure_dir(p.parent)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def robust_background(image: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, float]:
    """估计背景中位数和鲁棒噪声。

    输入:
    - `image`: HxW 灰度图。
    - `mask`: 可选有效区域 mask，True 表示参与估计。

    输出:
    - `(median, sigma)`。
    """

    values = image[mask > 0] if mask is not None else image.reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        values = image.reshape(-1)
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    sigma = 1.4826 * mad
    if sigma < 1e-6:
        sigma = float(np.std(values) + 1e-6)
    return med, sigma


def connected_components(binary: np.ndarray) -> list[dict[str, Any]]:
    """提取二值图连通域。

    输入:
    - `binary`: HxW bool 二值图。

    输出:
    - 连通域列表，每个元素包含 label、slice、coords。
    """

    labels, num = ndi.label(binary > 0)
    objects = ndi.find_objects(labels)
    comps: list[dict[str, Any]] = []
    for label_id, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        yy, xx = np.nonzero(labels[slc] == label_id)
        if yy.size == 0:
            continue
        y0 = int(slc[0].start)
        x0 = int(slc[1].start)
        coords = np.stack([yy + y0, xx + x0], axis=1)
        comps.append({"label": label_id, "slice": slc, "coords": coords})
    return comps


def weighted_centroid(image: np.ndarray, coords: np.ndarray, background: float) -> tuple[float, float, float]:
    """计算局部亮度加权质心和总 flux。

    输入:
    - `image`: HxW 灰度图。
    - `coords`: Nx2 像素坐标，列为 y/x。
    - `background`: 背景估计。

    输出:
    - `(cx, cy, flux)`，坐标使用 x/y 顺序。
    """

    yy = coords[:, 0].astype(np.int64)
    xx = coords[:, 1].astype(np.int64)
    weights = np.clip(image[yy, xx] - background, 0.0, None)
    flux = float(weights.sum())
    if flux <= 1e-8:
        return float(xx.mean()), float(yy.mean()), 0.0
    cx = float((xx * weights).sum() / flux)
    cy = float((yy * weights).sum() / flux)
    return cx, cy, flux


def component_shape(coords: np.ndarray) -> dict[str, float]:
    """计算连通域几何形状。

    输入:
    - `coords`: Nx2 像素坐标，列为 y/x。

    输出:
    - bbox、长度、宽度、长宽比和方向角。
    """

    yy = coords[:, 0].astype(np.float32)
    xx = coords[:, 1].astype(np.float32)
    x_min, x_max = float(xx.min()), float(xx.max())
    y_min, y_max = float(yy.min()), float(yy.max())
    centered = np.stack([xx - xx.mean(), yy - yy.mean()], axis=1)
    if coords.shape[0] >= 2:
        cov = np.cov(centered, rowvar=False)
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, 1e-6)
        order = np.argsort(vals)[::-1]
        major = float(np.sqrt(vals[order[0]]) * 4.0)
        minor = float(np.sqrt(vals[order[1]]) * 4.0)
        angle = float(np.degrees(np.arctan2(vecs[1, order[0]], vecs[0, order[0]])))
    else:
        major = minor = 1.0
        angle = 0.0
    aspect = float(major / max(minor, 1e-6))
    return {
        "bbox": [int(x_min), int(y_min), int(x_max + 1), int(y_max + 1)],
        "area": int(coords.shape[0]),
        "length_px": major,
        "width_px": minor,
        "aspect_ratio": aspect,
        "angle_deg": angle,
    }


def image_files(root: str | Path, recursive: bool = False) -> list[Path]:
    """列出图像文件。

    输入:
    - `root`: 图像目录。
    - `recursive`: 是否递归查找。

    输出:
    - 图像路径列表。
    """

    base = Path(root)
    iterator = base.rglob("*") if recursive else base.iterdir()
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
