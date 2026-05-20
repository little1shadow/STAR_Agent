#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Gaia DR3 星表读取工具。

本文件只负责把 CSV 星表转换为 numpy 数组，方便后续快速筛选和投影。
每一行星表记录代表一个 Gaia source，clean 仿真时通常把它渲染为一颗星点。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class GaiaCatalog:
    """内存中的 Gaia 星表子集。

    字段:
    - `source_id`: Gaia source 唯一编号。
    - `ra_deg`: 赤经，单位 degree。
    - `dec_deg`: 赤纬，单位 degree。
    - `g_mag`: Gaia G-band 星等，数值越小越亮。
    - `bp_mag`: BP 星等，可选。
    - `rp_mag`: RP 星等，可选。
    - `bp_rp`: 颜色指数，可选。
    - `ruwe`: Gaia 天体测量质量指标，可选。
    """

    source_id: np.ndarray
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    g_mag: np.ndarray
    bp_mag: np.ndarray
    rp_mag: np.ndarray
    bp_rp: np.ndarray
    ruwe: np.ndarray

    def __len__(self) -> int:
        """返回星表行数。

        输入:
        - 无。

        输出:
        - 当前星表中的 source 数量。
        """

        return int(self.ra_deg.shape[0])

    def take(self, indices: np.ndarray) -> "GaiaCatalog":
        """按索引取子集。

        输入:
        - `indices`: numpy 索引数组或布尔 mask。

        输出:
        - 新的 `GaiaCatalog` 子集。
        """

        return GaiaCatalog(
            source_id=self.source_id[indices],
            ra_deg=self.ra_deg[indices],
            dec_deg=self.dec_deg[indices],
            g_mag=self.g_mag[indices],
            bp_mag=self.bp_mag[indices],
            rp_mag=self.rp_mag[indices],
            bp_rp=self.bp_rp[indices],
            ruwe=self.ruwe[indices],
        )


def _to_float(value: str | None) -> float:
    """把 CSV 字符串安全转换为浮点数。

    输入:
    - `value`: CSV 单元格字符串。

    输出:
    - float；空值或非法值返回 `nan`。
    """

    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def load_gaia_csv(
    csv_path: str | Path,
    mag_limit: float | None = None,
    ruwe_max: float | None = None,
) -> GaiaCatalog:
    """读取 Gaia DR3 CSV 子集。

    输入:
    - `csv_path`: Gaia CSV 路径。
    - `mag_limit`: 可选，只保留 `phot_g_mean_mag <= mag_limit` 的星。
    - `ruwe_max`: 可选，只保留 `ruwe <= ruwe_max` 的高质量源。

    输出:
    - `GaiaCatalog`。

    关键逻辑:
    - 必须有 `source_id, ra, dec, phot_g_mean_mag`；
    - 缺少坐标或 G 星等的行会跳过；
    - BP/RP/ruwe 缺失不影响读取。
    """

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Gaia catalog not found: {path}")

    source_id: list[str] = []
    ra_deg: list[float] = []
    dec_deg: list[float] = []
    g_mag: list[float] = []
    bp_mag: list[float] = []
    rp_mag: list[float] = []
    bp_rp: list[float] = []
    ruwe: list[float] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ra = _to_float(row.get("ra"))
            dec = _to_float(row.get("dec"))
            g = _to_float(row.get("phot_g_mean_mag"))
            r = _to_float(row.get("ruwe"))

            if not np.isfinite(ra) or not np.isfinite(dec) or not np.isfinite(g):
                continue
            if mag_limit is not None and g > mag_limit:
                continue
            if ruwe_max is not None and np.isfinite(r) and r > ruwe_max:
                continue

            source_id.append(str(row.get("source_id", "")))
            ra_deg.append(ra)
            dec_deg.append(dec)
            g_mag.append(g)
            bp_mag.append(_to_float(row.get("phot_bp_mean_mag")))
            rp_mag.append(_to_float(row.get("phot_rp_mean_mag")))
            bp_rp.append(_to_float(row.get("bp_rp")))
            ruwe.append(r)

    catalog = GaiaCatalog(
        source_id=np.asarray(source_id, dtype=str),
        ra_deg=np.asarray(ra_deg, dtype=np.float64),
        dec_deg=np.asarray(dec_deg, dtype=np.float64),
        g_mag=np.asarray(g_mag, dtype=np.float32),
        bp_mag=np.asarray(bp_mag, dtype=np.float32),
        rp_mag=np.asarray(rp_mag, dtype=np.float32),
        bp_rp=np.asarray(bp_rp, dtype=np.float32),
        ruwe=np.asarray(ruwe, dtype=np.float32),
    )
    if len(catalog) == 0:
        raise RuntimeError(f"No usable Gaia sources loaded from {path}")
    return catalog


def summarize_catalog(catalog: GaiaCatalog) -> dict:
    """统计星表基本信息。

    输入:
    - `catalog`: GaiaCatalog。

    输出:
    - 字典，包含数量、RA/Dec 范围和星等范围。
    """

    return {
        "num_sources": len(catalog),
        "ra_min_deg": float(np.min(catalog.ra_deg)),
        "ra_max_deg": float(np.max(catalog.ra_deg)),
        "dec_min_deg": float(np.min(catalog.dec_deg)),
        "dec_max_deg": float(np.max(catalog.dec_deg)),
        "g_mag_min": float(np.min(catalog.g_mag)),
        "g_mag_max": float(np.max(catalog.g_mag)),
    }
