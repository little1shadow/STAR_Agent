#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""相机和指向模型。

本文件负责描述仿真图像的基础相机参数，并随机采样每张 clean 图的指向。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraConfig:
    """相机参数。

    字段:
    - `width`, `height`: 图像宽高，单位 pixel。
    - `fov_deg`: 水平方向视场角，单位 degree。
    - `bit_depth`: 输出 PNG 位深，目前 V1 使用 8-bit 便于查看。
    - `max_adu`: 输出最大灰度值。
    """

    width: int
    height: int
    fov_deg: float
    bit_depth: int = 8
    max_adu: float = 255.0

    @property
    def pixel_scale_deg(self) -> float:
        """返回每个像素对应的角尺度。

        输入:
        - 无。

        输出:
        - degree / pixel。
        """

        return float(self.fov_deg) / float(self.width)

    @property
    def pixel_scale_arcsec(self) -> float:
        """返回每个像素对应的角秒尺度。

        输入:
        - 无。

        输出:
        - arcsec / pixel。
        """

        return self.pixel_scale_deg * 3600.0


@dataclass(frozen=True)
class PointingConfig:
    """指向采样参数。

    字段:
    - `ra_center_deg`, `dec_center_deg`: 采样中心。
    - `ra_jitter_deg`, `dec_jitter_deg`: 每张图随机偏移范围。
    - `roll_min_deg`, `roll_max_deg`: roll 角采样范围。
    """

    ra_center_deg: float
    dec_center_deg: float
    ra_jitter_deg: float
    dec_jitter_deg: float
    roll_min_deg: float
    roll_max_deg: float


@dataclass(frozen=True)
class Pointing:
    """单张图像的相机指向。

    字段:
    - `ra_center_deg`: 图像中心赤经。
    - `dec_center_deg`: 图像中心赤纬。
    - `roll_deg`: 图像旋转角。
    """

    ra_center_deg: float
    dec_center_deg: float
    roll_deg: float


def sample_pointing(config: PointingConfig, rng: np.random.Generator) -> Pointing:
    """随机采样一张图像的指向。

    输入:
    - `config`: 指向采样配置。
    - `rng`: numpy 随机数生成器。

    输出:
    - `Pointing`。

    说明:
    - 第一版围绕已下载 Gaia 子集的中心小范围 jitter；
    - 后续如果使用全天 tile，可以把 RA/Dec 改成全天采样。
    """

    ra = config.ra_center_deg + rng.uniform(-config.ra_jitter_deg, config.ra_jitter_deg)
    dec = config.dec_center_deg + rng.uniform(-config.dec_jitter_deg, config.dec_jitter_deg)
    roll = rng.uniform(config.roll_min_deg, config.roll_max_deg)
    return Pointing(
        ra_center_deg=float(ra % 360.0),
        dec_center_deg=float(np.clip(dec, -89.0, 89.0)),
        roll_deg=float(roll % 360.0),
    )
