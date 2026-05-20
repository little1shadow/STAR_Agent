#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ASTRiDE 条纹目标检测适配工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from star_agent.downstream.common.image_ops import component_shape, weighted_centroid


def read_image_float(path: str | Path) -> np.ndarray:
    """读取图像为灰度 float32。

    输入:
    - `path`: PNG/JPG/TIF/FITS 前的普通图像路径。

    输出:
    - HxW float32 图像。
    """

    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., :3].astype(np.float32)
        arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    else:
        arr = arr.astype(np.float32)
    return arr.astype(np.float32)


def image_to_fits(image_path: str | Path, fits_path: str | Path) -> Path:
    """把普通图像转成 ASTRiDE 可读的 FITS。

    输入:
    - `image_path`: 输入图像。
    - `fits_path`: 输出 FITS 路径。

    输出:
    - FITS 路径。
    """

    from astropy.io import fits  # type: ignore

    image = read_image_float(image_path)
    p = Path(fits_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(p, image.astype(np.float32), overwrite=True)
    return p


def _safe_json_value(value: Any) -> Any:
    """把 ASTRiDE 内部对象转换为 JSON 友好的摘要。

    输入:
    - `value`: 任意 Python / numpy 对象。

    输出:
    - JSON 可序列化对象。
    """

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (list, tuple)):
        if len(value) > 20:
            return {"type": type(value).__name__, "len": len(value), "sample": [_safe_json_value(v) for v in value[:3]]}
        return [_safe_json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe_json_value(v) for k, v in list(value.items())[:30]}
    return str(value)[:500]


def _coords_from_border(border: Any) -> np.ndarray | None:
    """从 ASTRiDE contour/border 对象提取 y/x 坐标。

    输入:
    - `border`: ASTRiDE raw_borders 中的一个边界对象。

    输出:
    - Nx2 float 坐标，列为 y/x；失败返回 None。
    """

    arr = np.asarray(border)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, :2].astype(np.float32)
    return None


def _coords_from_streak_item(item: Any, raw_borders: list[Any]) -> np.ndarray | None:
    """从 ASTRiDE streak item 尽量恢复条纹坐标。

    输入:
    - `item`: `streak.streaks` 中的元素。
    - `raw_borders`: ASTRiDE 检测到的原始 contour 列表。

    输出:
    - Nx2 float 坐标，列为 y/x；失败返回 None。

    兼容策略:
    - 若 item 是 contour 数组，直接使用。
    - 若 item 是 raw_borders 的索引列表，则拼接对应 contour。
    - 若 item 是 dict/object，则尝试读取 `coords/x/y/points`。
    """

    direct = _coords_from_border(item)
    if direct is not None:
        return direct

    if isinstance(item, dict):
        for key in ["coords", "points", "pixels", "border"]:
            if key in item:
                coords = _coords_from_border(item[key])
                if coords is not None:
                    return coords
        if "x" in item and "y" in item:
            x = np.asarray(item["x"], dtype=np.float32).reshape(-1)
            y = np.asarray(item["y"], dtype=np.float32).reshape(-1)
            if x.size == y.size and x.size > 0:
                return np.stack([y, x], axis=1)

    for attr in ["coords", "points", "pixels", "border"]:
        if hasattr(item, attr):
            coords = _coords_from_border(getattr(item, attr))
            if coords is not None:
                return coords
    if hasattr(item, "x") and hasattr(item, "y"):
        x = np.asarray(getattr(item, "x"), dtype=np.float32).reshape(-1)
        y = np.asarray(getattr(item, "y"), dtype=np.float32).reshape(-1)
        if x.size == y.size and x.size > 0:
            return np.stack([y, x], axis=1)

    if isinstance(item, (list, tuple)) and item and all(isinstance(v, (int, np.integer)) for v in item):
        coords_list = []
        for idx in item:
            if 0 <= int(idx) < len(raw_borders):
                coords = _coords_from_border(raw_borders[int(idx)])
                if coords is not None:
                    coords_list.append(coords)
        if coords_list:
            return np.concatenate(coords_list, axis=0)

    return None


def build_streak_mask_and_records(
    image: np.ndarray,
    streak_items: list[Any],
    raw_borders: list[Any],
    cfg: dict[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """把 ASTRiDE streak 结果转换成 mask 和统一目标记录。

    输入:
    - `image`: HxW 灰度图。
    - `streak_items`: `streak.streaks`。
    - `raw_borders`: `streak.raw_borders`。
    - `cfg`: ASTRiDE 配置。

    输出:
    - `(mask, records)`。
    """

    h, w = image.shape
    mask = np.zeros((h, w), dtype=bool)
    records: list[dict[str, Any]] = []
    for idx, item in enumerate(streak_items):
        coords = _coords_from_streak_item(item, raw_borders)
        if coords is None or coords.size == 0:
            continue
        yy = np.clip(np.rint(coords[:, 0]).astype(np.int64), 0, h - 1)
        xx = np.clip(np.rint(coords[:, 1]).astype(np.int64), 0, w - 1)
        one = np.zeros_like(mask)
        one[yy, xx] = True
        dilate_px = int(cfg.get("mask", {}).get("dilate_px", 2))
        if dilate_px > 0:
            one = ndi.binary_dilation(one, iterations=dilate_px)
        mask |= one

        pix = np.argwhere(one)
        shape = component_shape(pix)
        cx, cy, flux = weighted_centroid(image, pix, float(np.median(image)))
        records.append(
            {
                "target_id": idx,
                "target_type": "short_streak",
                "detector_family": "astride",
                "x": float(cx),
                "y": float(cy),
                "flux": float(flux),
                "confidence": float(cfg.get("quality", {}).get("mask_confidence", 0.75)),
                **shape,
            }
        )
    return mask, records


def run_astride_detection(image_path: str | Path, output_dir: str | Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """运行 ASTRiDE 并返回统一检测结果。

    输入:
    - `image_path`: 输入图像。
    - `output_dir`: 输出目录。
    - `cfg`: ASTRiDE 配置。

    输出:
    - 包含 `mask`、`records`、`summary` 的字典。
    """

    from astride import Streak  # type: ignore

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fits_path = out / "input_for_astride.fits"
    image_to_fits(image_path, fits_path)
    image = read_image_float(image_path)

    kwargs = cfg.get("streak", {}) or {}
    try:
        streak = Streak(str(fits_path), output_path=str(out / "astride_raw"), **kwargs)
    except TypeError:
        streak = Streak(str(fits_path))
    streak.detect()
    try:
        streak.write_outputs()
    except Exception as exc:  # ASTRiDE 输出图表失败不应阻塞 mask 生成。
        write_error = f"write_outputs failed: {type(exc).__name__}: {exc}"
    else:
        write_error = None

    raw_borders = list(getattr(streak, "raw_borders", []) or [])
    streak_items = list(getattr(streak, "streaks", []) or [])
    mask, records = build_streak_mask_and_records(image, streak_items, raw_borders, cfg)
    summary = {
        "image_path": str(image_path),
        "detector": cfg.get("quality", {}).get("detector", "astride"),
        "mask_source": cfg.get("quality", {}).get("mask_source", "astride_streak_v001"),
        "mask_confidence": float(cfg.get("quality", {}).get("mask_confidence", 0.75)),
        "num_raw_borders": len(raw_borders),
        "num_streak_items": len(streak_items),
        "num_records": len(records),
        "mask_area_px": int(mask.sum()),
        "write_error": write_error,
        "astride_attrs": {
            "streaks": _safe_json_value(streak_items),
            "raw_borders": _safe_json_value(raw_borders),
        },
    }
    return {"mask": mask, "records": records, "summary": summary}
