#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""物理约束版太阳杂散光前向仿真。


1. off-axis extended solar source
   太阳作为视场外扩展强光源，只允许在边缘/角落露出极小的 solar limb/glint。
2. baffle / aperture scattering kernel
   遮光罩、镜筒和光学表面对离轴强光产生 Moffat/Lorentzian 型长尾散射。
3. internal reflection ghost
   镜片/窗口多次反射产生 annular ghost。弧线不是手工画的，而是鬼像环被视场、光阑和遮挡裁剪后的可见部分。
4. sensor column bleed
   当边缘 glint 或 ghost 局部过亮时，传感器读出方向产生弱列泄漏。

输出：
- degraded image
- degradation mask
- continuous stray-light field
- meta JSON
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from ..common.io import ensure_dir, write_json


SOLAR_STRAY_LIGHT_MODES = [
    "grazing_solar_limb",
    "single_internal_ghost",
    "multi_reflection_ghosts",
    "veiling_glare_bleed",
]

SOLAR_SIDES = ["top_left", "top_right", "left", "right", "bottom_left", "bottom_right"]

LEVEL_PARAMS: dict[int, dict[str, float]] = {
    # level 1 基本接近 clean；level 4/5 明显影响背景和目标检测。
    # 2026-05-19: 在保持新版物理形态不变的前提下整体增强强度。
    # 低等级只轻微增强；4/5 级增强更明显，用于产生可被 DepictQA 和下游任务稳定感知的强退化样本。
    1: {"limb": 0.034, "scatter": 0.014, "ghost": 0.021, "bleed": 0.0025, "mix": 0.005, "lift": 0.000},
    2: {"limb": 0.064, "scatter": 0.029, "ghost": 0.045, "bleed": 0.0050, "mix": 0.010, "lift": 0.0015},
    3: {"limb": 0.145, "scatter": 0.078, "ghost": 0.115, "bleed": 0.0130, "mix": 0.026, "lift": 0.008},
    4: {"limb": 0.280, "scatter": 0.165, "ghost": 0.245, "bleed": 0.0260, "mix": 0.062, "lift": 0.022},
    5: {"limb": 0.420, "scatter": 0.275, "ghost": 0.405, "bleed": 0.0450, "mix": 0.105, "lift": 0.042},
}

DEFAULT_SIDE_WEIGHTS = {
    "top_left": 0.34,
    "top_right": 0.34,
    "left": 0.10,
    "right": 0.10,
    "bottom_left": 0.06,
    "bottom_right": 0.06,
}

DEFAULT_MASK_THRESHOLD_BY_LEVEL = {1: 0.020, 2: 0.034, 3: 0.060, 4: 0.095, 5: 0.135}
MASK_AREA_CAP_BY_LEVEL = {1: 0.16, 2: 0.28, 3: 0.46, 4: 0.62, 5: 0.76}


def set_seed(seed: int | None) -> None:
    """设置随机种子。

    输入:
    - `seed`: 随机种子。为 None 时不固定。

    输出:
    - 无返回值。
    """

    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def load_rgb_float(path: str | Path) -> np.ndarray:
    """读取 RGB 图像到 `[0, 1]` float32。

    输入:
    - `path`: 图像路径。

    输出:
    - HxWx3 float32 图像。
    """

    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def save_rgb_float(path: str | Path, image: np.ndarray) -> None:
    """保存 RGB float 图像。

    输入:
    - `path`: 输出路径。
    - `image`: `[0, 1]` 图像。

    输出:
    - 无返回值。
    """

    p = Path(path)
    ensure_dir(p.parent)
    arr = np.clip(image, 0.0, 1.0)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    Image.fromarray(np.rint(arr * 255.0).astype(np.uint8), mode="RGB").save(p)


def save_gray_float(path: str | Path, image: np.ndarray) -> None:
    """保存单通道 float/uint8 图像。

    输入:
    - `path`: 输出路径。
    - `image`: 单通道图像。

    输出:
    - 无返回值。
    """

    p = Path(path)
    ensure_dir(p.parent)
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.rint(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(p)


def normalize01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """将数组归一化到 `[0, 1]`。

    输入:
    - `x`: 任意数组。
    - `eps`: 防止除零。

    输出:
    - float32 归一化数组。
    """

    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn + eps)).astype(np.float32)


def meshgrid_xy(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """生成图像坐标网格。

    输入:
    - `height`, `width`: 图像尺寸。

    输出:
    - `(xx, yy)`。
    """

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    return xx, yy


def choose_side(side_weights: dict[str, float] | None = None) -> str:
    """按概率选择太阳位于视场外的方向。

    输入:
    - `side_weights`: 方向概率。

    输出:
    - side 字符串。
    """

    weights_map = side_weights or DEFAULT_SIDE_WEIGHTS
    sides = [side for side in SOLAR_SIDES if side in weights_map]
    weights = [float(weights_map[side]) for side in sides]
    return random.choices(sides, weights=weights, k=1)[0]


def inward_angle_from_side(side: str) -> float:
    """计算从太阳源指向图像内部的大致方向角。

    输入:
    - `side`: 太阳所在边/角。

    输出:
    - 弧度角。
    """

    table = {
        "left": 0.0,
        "right": math.pi,
        "top_left": math.pi / 4.0,
        "top_right": 3.0 * math.pi / 4.0,
        "bottom_left": -math.pi / 4.0,
        "bottom_right": -3.0 * math.pi / 4.0,
    }
    return table.get(side, math.pi / 4.0) + random.uniform(-0.12, 0.12)


def sample_off_axis_sun(height: int, width: int, side: str, level: int) -> dict[str, float]:
    """采样视场外扩展太阳源。

    输入:
    - `height`, `width`: 图像尺寸。
    - `side`: 太阳所在方向。
    - `level`: 退化等级。

    输出:
    - 太阳源参数字典，包含中心、半径和离轴距离。

    物理约束:
    - 太阳中心在视场外；
    - 太阳半径较大，但边缘只露出小 cap；
    - 高等级不是把太阳露出更多，而是散射强度更强、太阳更靠近遮光临界角。
    """

    base = max(height, width)
    radius = random.uniform(0.070, 0.105) * base
    cap_depth = random.uniform(0.025, 0.085) * radius * (1.0 + 0.08 * max(level - 3, 0))
    offset = radius - cap_depth

    if side == "top_left":
        # 角点几何约束：太阳中心到角点的距离约等于 R - cap_depth，
        # 保证画面里只露出很小一角，而不是完全看不见或露出大半圆。
        phi = random.uniform(math.radians(35.0), math.radians(55.0))
        d = radius - cap_depth
        src_x = -d * math.cos(phi)
        src_y = -d * math.sin(phi)
    elif side == "top_right":
        phi = random.uniform(math.radians(35.0), math.radians(55.0))
        d = radius - cap_depth
        src_x = width + d * math.cos(phi)
        src_y = -d * math.sin(phi)
    elif side == "left":
        src_x = -offset
        src_y = random.uniform(0.08 * height, 0.36 * height)
    elif side == "right":
        src_x = width + offset
        src_y = random.uniform(0.08 * height, 0.36 * height)
    elif side == "bottom_left":
        phi = random.uniform(math.radians(35.0), math.radians(55.0))
        d = radius - cap_depth
        src_x = -d * math.cos(phi)
        src_y = height + d * math.sin(phi)
    else:
        phi = random.uniform(math.radians(35.0), math.radians(55.0))
        d = radius - cap_depth
        src_x = width + d * math.cos(phi)
        src_y = height + d * math.sin(phi)

    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    off_axis_distance = math.sqrt((src_x - cx) ** 2 + (src_y - cy) ** 2)
    return {
        "src_x": float(src_x),
        "src_y": float(src_y),
        "solar_radius_px": float(radius),
        "cap_depth_px": float(cap_depth),
        "off_axis_distance_px": float(off_axis_distance),
    }


def solar_limb_cap(height: int, width: int, sun: dict[str, float]) -> np.ndarray:
    """计算太阳盘边缘进入视场的极小 cap。

    输入:
    - `height`, `width`: 图像尺寸。
    - `sun`: 太阳源参数。

    输出:
    - `[0, 1]` 太阳 limb/glint 场。
    """

    xx, yy = meshgrid_xy(height, width)
    rr = np.sqrt((xx - sun["src_x"]) ** 2 + (yy - sun["src_y"]) ** 2)
    disk = np.clip(1.0 - rr / max(sun["solar_radius_px"], 1e-6), 0.0, 1.0)
    disk = disk ** random.uniform(0.55, 0.90)
    disk = gaussian_filter(disk, sigma=random.uniform(1.0, 2.4))
    return normalize01(disk) if float(disk.max()) > 0 else disk.astype(np.float32)


def aperture_scatter_kernel(height: int, width: int, sun: dict[str, float], side: str, level: int) -> np.ndarray:
    """遮光罩/光机散射长尾核。

    输入:
    - `sun`: 太阳源参数。
    - `side`: 太阳所在方向。
    - `level`: 退化等级。

    输出:
    - `[0, 1]` 大尺度散射场。

    物理近似:
    - 离轴强光被遮光罩和镜筒散射，强度随与太阳方向夹角和距离呈长尾衰减；
    - 使用 Moffat/Lorentzian 型核模拟非理想光学散射。
    """

    xx, yy = meshgrid_xy(height, width)
    dx = xx - sun["src_x"]
    dy = yy - sun["src_y"]
    rr = np.sqrt(dx * dx + dy * dy)
    theta = np.arctan2(dy, dx)
    inward = inward_angle_from_side(side)
    dtheta = np.arctan2(np.sin(theta - inward), np.cos(theta - inward))

    base = max(height, width)
    r0 = random.uniform(0.28, 0.58) * base * (1.0 + 0.03 * level)
    beta = random.uniform(1.25, 2.15)
    radial = np.power(1.0 + (rr / max(r0, 1e-6)) ** 2, -beta)
    angular = np.exp(-0.5 * (dtheta / random.uniform(0.50, 0.95)) ** 2)
    floor = random.uniform(0.12, 0.28)
    field = radial * (floor + (1.0 - floor) * angular)

    # 低频微扰代表表面粗糙度和遮光结构非均匀，不直接画图案。
    rough = gaussian_filter(np.random.randn(height, width).astype(np.float32), sigma=random.uniform(35.0, 95.0))
    rough = np.clip(0.82 + 0.28 * normalize01(rough), 0.72, 1.10)
    field = gaussian_filter(field * rough, sigma=random.uniform(5.0, 13.0))
    return normalize01(field)


def smooth_sector_window(theta: np.ndarray, center: float, span: float, floor: float) -> np.ndarray:
    """光阑裁剪导致的弧段可见窗口。

    输入:
    - `theta`: 相对鬼像中心的角度。
    - `center`: 可见方向中心。
    - `span`: 可见角宽。
    - `floor`: 残余透过底座。

    输出:
    - `[0, 1]` 窗函数。
    """

    dtheta = np.arctan2(np.sin(theta - center), np.cos(theta - center))
    win = np.exp(-0.5 * (dtheta / max(span, 1e-6)) ** 2)
    return np.clip(floor + (1.0 - floor) * win, 0.0, 1.0).astype(np.float32)


def annular_ghost(
    height: int,
    width: int,
    center_x: float,
    center_y: float,
    radius: float,
    width_px: float,
    visible_angle: float,
    visible_span: float,
    edge_darkening: float,
) -> np.ndarray:
    """生成被光阑裁剪后的内部反射鬼像环。

    输入:
    - `center_x`, `center_y`: 鬼像环中心。
    - `radius`: 鬼像环半径。
    - `width_px`: 环宽。
    - `visible_angle`: 光阑允许的可见方向。
    - `visible_span`: 可见角宽。
    - `edge_darkening`: 环两侧暗化程度。

    输出:
    - `[0, 1]` 鬼像弧。

    说明:
    - 这里不手动画弧线；先生成完整 annular ghost，再用光阑 sector window 和遮挡窗口裁剪，
      因此弧段来源仍是物理鬼像环。
    """

    xx, yy = meshgrid_xy(height, width)
    dx = xx - center_x
    dy = yy - center_y
    rr = np.sqrt(dx * dx + dy * dy)
    theta = np.arctan2(dy, dx)

    core = np.exp(-0.5 * ((rr - radius) / max(width_px, 1e-6)) ** 2)
    # 真实鬼像常有多层镀膜/玻璃面反射，产生中心亮或边缘亮的截面。
    inner = np.exp(-0.5 * ((rr - radius + 0.72 * width_px) / max(0.45 * width_px, 1e-6)) ** 2)
    outer = np.exp(-0.5 * ((rr - radius - 0.85 * width_px) / max(0.52 * width_px, 1e-6)) ** 2)
    ring = np.clip((1.0 - edge_darkening) * core + 0.62 * edge_darkening * (inner + outer), 0.0, None)

    # 光阑/遮光结构只允许一部分鬼像环进入探测器，因此这里的 floor 必须很低；
    # 否则会出现完整圆环，视觉上更像人工画出的测试图。
    sector = smooth_sector_window(theta, visible_angle, visible_span, floor=random.uniform(0.015, 0.10))

    # 用平滑遮挡窗模拟镜筒、光阑边缘和内部结构对 ghost 的裁剪。
    # 这不是随机贴纹理，而是把完整 annular ghost 通过 aperture clipping 截成不连续弧段。
    obstruction_noise = gaussian_filter(
        np.random.randn(height, width).astype(np.float32),
        sigma=random.uniform(35.0, 95.0),
    )
    obstruction_noise = normalize01(obstruction_noise)
    obstruction_binary = (obstruction_noise > random.uniform(0.42, 0.68)).astype(np.float32)
    obstruction = gaussian_filter(obstruction_binary, sigma=random.uniform(5.0, 15.0))
    obstruction = np.clip(0.06 + 0.94 * normalize01(obstruction), 0.0, 1.0)
    ghost = ring * sector * obstruction
    ghost = gaussian_filter(ghost, sigma=max(0.8, 0.018 * width_px))
    return normalize01(ghost) if float(ghost.max()) > 0 else ghost.astype(np.float32)


def ghost_center_from_reflection(height: int, width: int, sun: dict[str, float], order: int) -> tuple[float, float]:
    """根据离轴太阳位置估计内部反射鬼像中心。

    输入:
    - `sun`: 太阳源参数。
    - `order`: 反射阶次。

    输出:
    - `(ghost_x, ghost_y)`。

    物理近似:
    - 内部反射鬼像常沿太阳离轴方向和光轴连线出现；
    - 不同玻璃面反射对应不同缩放系数。
    """

    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    vx = sun["src_x"] - cx
    vy = sun["src_y"] - cy
    scale_table = [0.34, -0.18, 0.58]
    scale = scale_table[(order - 1) % len(scale_table)] + random.uniform(-0.05, 0.05)
    return float(cx - scale * vx), float(cy - scale * vy)


def internal_reflection_field(height: int, width: int, sun: dict[str, float], side: str, mode: str, level: int) -> np.ndarray:
    """生成内部反射鬼像场。

    输入:
    - `mode`: 物理模式。
    - `level`: 退化等级。

    输出:
    - `[0, 1]` 鬼像场。
    """

    base = max(height, width)
    if mode == "grazing_solar_limb":
        # grazing 模式主要由边缘太阳 limb、遮光罩散射和近源 caustic 构成；
        # 不额外放置大尺度内部反射环，避免出现不真实的大圆环。
        count = 0
        gain_scale = 0.0
    elif mode == "single_internal_ghost":
        count = 1
        gain_scale = 1.00
    elif mode == "multi_reflection_ghosts":
        count = random.choice([2, 3])
        gain_scale = 0.86
    else:
        count = random.choice([0, 1])
        gain_scale = 0.42

    field = np.zeros((height, width), dtype=np.float32)
    inward = inward_angle_from_side(side)
    for order in range(1, count + 1):
        gx, gy = ghost_center_from_reflection(height, width, sun, order)
        radius = random.uniform(0.22, 0.72) * base * (1.0 + 0.035 * level)
        width_px = random.uniform(0.018, 0.060) * base * (1.0 + 0.06 * max(level - 3, 0))
        visible_angle = inward + math.pi + random.uniform(-0.55, 0.55)
        visible_span = random.uniform(0.30, 0.82)
        edge_darkening = random.uniform(0.20, 0.80)
        gain = gain_scale * random.uniform(0.68, 1.15) / (1.0 + 0.28 * (order - 1))
        field += gain * annular_ghost(
            height=height,
            width=width,
            center_x=gx,
            center_y=gy,
            radius=radius,
            width_px=width_px,
            visible_angle=visible_angle,
            visible_span=visible_span,
            edge_darkening=edge_darkening,
        )
    return normalize01(field) if float(field.max()) > 0 else field.astype(np.float32)


def near_limb_caustic(height: int, width: int, sun: dict[str, float], side: str, level: int) -> np.ndarray:
    """太阳边缘附近的短小高亮 caustic/近源弧。

    输入:
    - `sun`: 太阳源参数。
    - `side`: 太阳方向。
    - `level`: 退化等级。

    输出:
    - `[0, 1]` 近源 caustic。
    """

    base = max(height, width)
    # 近源弧是第一个 ghost/aperture-edge caustic，半径较小且靠近太阳边缘。
    radius = random.uniform(0.10, 0.22) * base
    width_px = random.uniform(0.008, 0.024) * base * (1.0 + 0.08 * max(level - 3, 0))
    visible = inward_angle_from_side(side) + random.uniform(-0.20, 0.20)
    span = random.uniform(0.20, 0.45)
    caustic = annular_ghost(
        height=height,
        width=width,
        center_x=sun["src_x"],
        center_y=sun["src_y"],
        radius=radius,
        width_px=width_px,
        visible_angle=visible,
        visible_span=span,
        edge_darkening=random.uniform(0.15, 0.65),
    )
    return normalize01(caustic + 0.35 * caustic * caustic)


def sensor_column_bleed(height: int, width: int, bright_field: np.ndarray, mode: str, level: int) -> np.ndarray:
    """由局部高亮导致的传感器列向泄漏。

    输入:
    - `bright_field`: limb/ghost 等高亮场。
    - `mode`: 物理模式。
    - `level`: 退化等级。

    输出:
    - `[0, 1]` 列 bleed 场。
    """

    if level <= 1 and mode != "veiling_glare_bleed":
        return np.zeros((height, width), dtype=np.float32)
    profile = 0.72 * bright_field.max(axis=0) + 0.28 * bright_field.mean(axis=0)
    profile = normalize01(gaussian_filter(profile, sigma=random.uniform(0.8, 2.8)))
    q = {1: 0.994, 2: 0.986, 3: 0.972, 4: 0.950, 5: 0.925}[level]
    if mode == "veiling_glare_bleed":
        q -= 0.035
    threshold = float(np.quantile(profile, np.clip(q, 0.80, 0.998)))
    sparse = np.clip(profile - threshold, 0.0, None)
    sparse = normalize01(sparse) if float(sparse.max()) > 0 else sparse
    columns = np.tile(sparse[None, :], (height, 1))
    y_mod = np.linspace(1.0, 0.82, height, dtype=np.float32)[:, None]
    columns = gaussian_filter(columns * y_mod, sigma=(random.uniform(8.0, 22.0), 0.18))
    return normalize01(columns) if float(columns.max()) > 0 else columns.astype(np.float32)


def build_physical_solar_fields(
    height: int,
    width: int,
    level: int,
    mode: str,
    side_weights: dict[str, float] | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """构建所有物理分量。

    输入:
    - `height`, `width`: 图像尺寸。
    - `level`: 退化等级。
    - `mode`: 物理模式。
    - `side_weights`: 太阳方向概率。

    输出:
    - `fields`: 物理分量字典。
    - `meta`: 物理参数记录。
    """

    side = choose_side(side_weights)
    sun = sample_off_axis_sun(height, width, side, level)
    limb = solar_limb_cap(height, width, sun)
    scatter = aperture_scatter_kernel(height, width, sun, side, level)
    caustic = near_limb_caustic(height, width, sun, side, level)
    ghosts = internal_reflection_field(height, width, sun, side, mode, level)
    bleed = sensor_column_bleed(height, width, np.maximum.reduce([limb, caustic, ghosts]), mode, level)

    fields = {
        "solar_limb": limb.astype(np.float32),
        "aperture_scatter": scatter.astype(np.float32),
        "near_limb_caustic": caustic.astype(np.float32),
        "internal_ghost": ghosts.astype(np.float32),
        "sensor_column_bleed": bleed.astype(np.float32),
    }
    meta = {"side": side, **sun}
    return fields, meta


def add_solar_stray_light(
    image: np.ndarray,
    level: int,
    seed: int | None = None,
    mode: str | None = None,
    side_weights: dict[str, float] | None = None,
    mask_threshold_by_level: dict[int, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """给图像添加物理约束版太阳杂散光。

    输入:
    - `image`: `[0, 1]` RGB 图像。
    - `level`: 退化等级 1-5。
    - `seed`: 随机种子。
    - `mode`: 物理模式。
    - `side_weights`: 太阳方向概率。
    - `mask_threshold_by_level`: mask 阈值。

    输出:
    - `degraded`: 退化图。
    - `mask`: 退化 mask。
    - `field`: 连续退化场。
    - `meta`: 参数记录。
    """

    set_seed(seed)
    if level not in LEVEL_PARAMS:
        raise ValueError("level must be in [1, 5]")
    if mode is None:
        mode = random.choice(SOLAR_STRAY_LIGHT_MODES)
    if mode not in SOLAR_STRAY_LIGHT_MODES:
        raise ValueError(f"Unknown solar stray light mode: {mode}")

    height, width = image.shape[:2]
    p = LEVEL_PARAMS[level]
    fields, meta = build_physical_solar_fields(height, width, level, mode, side_weights)

    limb_gain = p["limb"] * (1.18 if mode == "grazing_solar_limb" else 0.90)
    scatter_gain = p["scatter"] * (1.32 if mode == "veiling_glare_bleed" else 0.92)
    ghost_gain = p["ghost"] * (1.30 if mode in {"single_internal_ghost", "multi_reflection_ghosts"} else 0.62)
    caustic_gain = p["ghost"] * (1.05 if mode == "grazing_solar_limb" else 0.72)
    bleed_gain = p["bleed"] * (1.45 if mode == "veiling_glare_bleed" else 0.75)

    raw_field = (
        limb_gain * fields["solar_limb"]
        + scatter_gain * fields["aperture_scatter"]
        + caustic_gain * fields["near_limb_caustic"]
        + ghost_gain * fields["internal_ghost"]
        + bleed_gain * fields["sensor_column_bleed"]
    )
    field = normalize01(raw_field) if float(raw_field.max()) > 0 else raw_field.astype(np.float32)

    # 散射造成的局部洗白来自物理 field，而不是额外手绘区域。
    blurred = gaussian_filter(image, sigma=(2.0, 2.0, 0.0))
    wash = np.clip(p["mix"] * normalize01(raw_field), 0.0, 0.36)
    degraded = image * (1.0 - wash[..., None]) + blurred * wash[..., None]
    degraded = degraded + raw_field[..., None] + p["lift"]

    # 极小太阳 limb 与近源 caustic 允许局部饱和。
    hot = np.clip(1.85 * fields["solar_limb"] + 0.85 * fields["near_limb_caustic"] - 0.92, 0.0, 1.0)
    degraded = degraded + (0.48 * p["limb"] + 0.18 * p["ghost"]) * hot[..., None]
    degraded = np.clip(degraded, 0.0, 1.0).astype(np.float32)

    thresholds = mask_threshold_by_level or DEFAULT_MASK_THRESHOLD_BY_LEVEL
    threshold = float(thresholds.get(level, DEFAULT_MASK_THRESHOLD_BY_LEVEL[level]))
    mask_bool = raw_field >= threshold
    area_cap = MASK_AREA_CAP_BY_LEVEL[level]
    if float(np.mean(mask_bool)) > area_cap:
        threshold = max(threshold, float(np.quantile(raw_field, 1.0 - area_cap)))
        mask_bool = raw_field >= threshold
    mask = mask_bool.astype(np.uint8) * 255

    meta.update(
        {
            "degradation": "solar_stray_light",
            "physical_model": "off_axis_sun+baffle_scatter+internal_reflection_ghost+sensor_column_bleed",
            "mode": mode,
            "level": int(level),
            "seed": seed,
            "field_max_raw": float(raw_field.max()),
            "field_mean_raw": float(raw_field.mean()),
            "mask_threshold": float(threshold),
            "mask_area_ratio": float(np.mean(mask > 0)),
            "component_max": {key: float(value.max()) for key, value in fields.items()},
        }
    )
    return degraded, mask, field.astype(np.float32), meta


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读取 JSONL。

    输入:
    - `path`: JSONL 文件路径。

    输出:
    - 字典列表。
    """

    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def resolve_path(path_value: str | Path, repo_root: Path) -> Path:
    """解析 manifest 中的路径。

    输入:
    - `path_value`: 路径。
    - `repo_root`: 仓库根目录。

    输出:
    - 存在的 Path。
    """

    p = Path(path_value)
    candidates = [p] if p.is_absolute() else [repo_root / p, p]
    for item in candidates:
        if item.exists():
            return item
    raise FileNotFoundError(f"Path not found: {path_value}")


def load_config(path: str | Path | None) -> dict[str, Any]:
    """读取 YAML 配置。

    输入:
    - `path`: YAML 路径。

    输出:
    - 配置字典。
    """

    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def output_paths(output_root: Path, mode: str, level: int, image_id: str) -> dict[str, Path]:
    """构造标准输出路径。

    输入:
    - `output_root`: `single/solar_stray_light` 根目录。
    - `mode`: 模式。
    - `level`: 等级。
    - `image_id`: 图像 ID。

    输出:
    - 路径字典。
    """

    base = output_root / mode / f"level_{level}"
    return {
        "image": base / "images" / f"{image_id}@solar_stray_light@{mode}@l{level}.png",
        "mask": base / "masks" / f"{image_id}@solar_stray_light@{mode}@l{level}.png",
        "field": base / "fields" / f"{image_id}@solar_stray_light@{mode}@l{level}.png",
        "meta": base / "meta" / f"{image_id}@solar_stray_light@{mode}@l{level}.json",
    }


def build_preview_grid(records: list[dict[str, Any]], output_path: Path, draw_mask_box: bool = False) -> None:
    """生成预览网格。

    输入:
    - `records`: 生成记录。
    - `output_path`: 输出预览路径。
    - `draw_mask_box`: 是否画 mask 外接框。

    输出:
    - 无返回值。
    """

    if not records:
        return
    tiles: list[Image.Image] = []
    for record in records:
        image = Image.open(record["image_path"]).convert("RGB")
        image.thumbnail((256, 256))
        draw = ImageDraw.Draw(image)
        draw.text((6, 6), f"L{record['level']} {record['mode']}", fill=(255, 80, 80))
        if draw_mask_box:
            mask = Image.open(record["mask_path"]).convert("L").resize(image.size)
            bbox = mask.getbbox()
            if bbox:
                draw.rectangle(bbox, outline=(255, 60, 60), width=2)
        tiles.append(image.copy())

    cols = 5
    rows = int(math.ceil(len(tiles) / cols))
    canvas = Image.new("RGB", (cols * 256, rows * 256), (0, 0, 0))
    for idx, tile in enumerate(tiles):
        canvas.paste(tile, ((idx % cols) * 256, (idx // cols) * 256))
    ensure_dir(output_path.parent)
    canvas.save(output_path)


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    """批量生成 solar stray light。

    输入:
    - `args`: 命令行参数。

    输出:
    - summary 字典。
    """

    repo_root = Path.cwd()
    cfg = load_config(args.config)
    solar_cfg = cfg.get("solar_stray_light", {}) if isinstance(cfg, dict) else {}
    side_weights = solar_cfg.get("side_weights", DEFAULT_SIDE_WEIGHTS)
    threshold_cfg = solar_cfg.get("mask_threshold_by_level", DEFAULT_MASK_THRESHOLD_BY_LEVEL)
    threshold_by_level = {int(k): float(v) for k, v in threshold_cfg.items()}

    records = read_jsonl(args.manifest)
    if args.num_images is not None:
        records = records[: max(0, int(args.num_images))]
    output_root = Path(args.output_root)
    manifest_out = output_root / "_manifests" / f"solar_stray_light_preview_{len(records)}.jsonl"
    ensure_dir(manifest_out.parent)
    manifest_out.write_text("", encoding="utf-8")

    generated: list[dict[str, Any]] = []
    level_cycle = [1, 2, 3, 4, 5]
    mode_cycle = SOLAR_STRAY_LIGHT_MODES
    for idx, record in enumerate(records):
        image_id = str(record.get("image_id") or f"sample_{idx:06d}")
        image_path = resolve_path(record["image_path"], repo_root)
        level = int(args.level) if args.level is not None else level_cycle[idx % len(level_cycle)]
        mode = str(args.mode) if args.mode else mode_cycle[idx % len(mode_cycle)]
        seed = int(args.seed + idx) if args.seed is not None else None

        image = load_rgb_float(image_path)
        degraded, mask, field, meta = add_solar_stray_light(
            image=image,
            level=level,
            seed=seed,
            mode=mode,
            side_weights=side_weights,
            mask_threshold_by_level=threshold_by_level,
        )
        paths = output_paths(output_root, mode, level, image_id)
        save_rgb_float(paths["image"], degraded)
        save_gray_float(paths["mask"], mask)
        save_gray_float(paths["field"], field)
        meta.update(
            {
                "image_id": image_id,
                "source_image_path": str(image_path),
                "image_path": str(paths["image"]),
                "mask_path": str(paths["mask"]),
                "field_path": str(paths["field"]),
                "meta_path": str(paths["meta"]),
            }
        )
        write_json(paths["meta"], meta)
        with manifest_out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        generated.append(meta)
        print(f"[PROGRESS] solar_stray_light {idx + 1}/{len(records)} | {image_id} | level={level} mode={mode}")

    preview_path = output_root / "_preview" / f"solar_stray_light_preview_{len(records)}.png"
    raw_preview_path = output_root / "_preview" / f"solar_stray_light_preview_{len(records)}_raw.png"
    build_preview_grid(generated, preview_path, draw_mask_box=True)
    build_preview_grid(generated, raw_preview_path, draw_mask_box=False)
    summary = {
        "num_images": len(generated),
        "output_root": str(output_root),
        "manifest_path": str(manifest_out),
        "preview_path": str(preview_path),
        "raw_preview_path": str(raw_preview_path),
        "modes": sorted({item["mode"] for item in generated}),
        "levels": sorted({int(item["level"]) for item in generated}),
    }
    write_json(output_root / "_manifests" / f"solar_stray_light_preview_{len(records)}_summary.json", summary)
    return summary


def run_single(args: argparse.Namespace) -> dict[str, Any]:
    """单张图生成。

    输入:
    - `args`: 命令行参数。

    输出:
    - meta 字典。
    """

    image = load_rgb_float(args.input)
    level = int(args.level or 3)
    degraded, mask, field, meta = add_solar_stray_light(image, level=level, seed=args.seed, mode=args.mode)
    save_rgb_float(args.output, degraded)
    if args.mask:
        save_gray_float(args.mask, mask)
    if args.field:
        save_gray_float(args.field, field)
    if args.meta:
        write_json(args.meta, meta)
    return meta


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    输入:
    - 无，读取命令行。

    输出:
    - argparse Namespace。
    """

    parser = argparse.ArgumentParser(description="Generate physical solar stray light degradation for STAR-Agent.")
    parser.add_argument("--config", default="STAR_Agent/configs/data_generation/degradation_single.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--mask", default=None)
    parser.add_argument("--field", default=None)
    parser.add_argument("--meta", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output_root", default="STAR_Agent/data/degraded/single/solar_stray_light")
    parser.add_argument("--num_images", type=int, default=None)
    parser.add_argument("--level", type=int, default=None, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--mode", default=None, choices=SOLAR_STRAY_LIGHT_MODES)
    parser.add_argument("--seed", type=int, default=131)
    return parser.parse_args()


def main() -> int:
    """命令行入口。

    输入:
    - 无。

    输出:
    - 进程退出码。
    """

    args = parse_args()
    if args.manifest:
        summary = run_batch(args)
        print("[OK] solar stray light batch generated")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if not args.input or not args.output:
        raise SystemExit("Either --manifest or both --input/--output are required.")
    meta = run_single(args)
    print("[OK] solar stray light generated")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
