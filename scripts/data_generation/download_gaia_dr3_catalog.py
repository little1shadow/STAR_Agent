#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""下载 Gaia DR3 星表子集。

功能:
- 通过 ESA Gaia Archive TAP 接口执行 ADQL 查询；
- 默认下载一个可控的 cone-search 子集，而不是直接下载全量 Gaia DR3；
- 将结果保存为 CSV，并额外保存 manifest，记录查询参数和 ADQL 语句；
- 后续 clean_data 仿真会从该 CSV 读取 `ra/dec/mag` 等字段。

为什么不默认下载全量 DR3:
- Gaia DR3 全量 `gaiadr3.gaia_source` 规模极大，不适合直接放入项目本地；
- clean_data 仿真通常只需要指定天区、星等阈值或若干 tile；
- 先做可控 subset，可以避免磁盘和网络被一次性打满。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


DEFAULT_TAP_SYNC_URL = "https://gea.esac.esa.int/tap-server/tap/sync"


def build_cone_query(
    *,
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    mag_limit: float,
    row_limit: int,
) -> str:
    """构建 Gaia DR3 cone-search ADQL 查询。

    输入:
    - `ra_deg`: 视场中心赤经，单位 degree。
    - `dec_deg`: 视场中心赤纬，单位 degree。
    - `radius_deg`: cone search 半径，单位 degree。
    - `mag_limit`: G-band 星等上限，只保留更亮的星。
    - `row_limit`: 最多返回多少行，防止误下载过大数据。

    输出:
    - ADQL 查询字符串。

    说明:
    - 这里使用 `gaiadr3.gaia_source`；
    - 字段保留 clean_data 仿真最常用的天球坐标、星等、颜色和自行信息；
    - 第一版仿真主要需要 `source_id, ra, dec, phot_g_mean_mag`。
    """

    return f"""
SELECT TOP {int(row_limit)}
  source_id,
  ra,
  dec,
  phot_g_mean_mag,
  phot_bp_mean_mag,
  phot_rp_mean_mag,
  bp_rp,
  parallax,
  pmra,
  pmdec,
  ruwe
FROM gaiadr3.gaia_source
WHERE
  1 = CONTAINS(
    POINT('ICRS', ra, dec),
    CIRCLE('ICRS', {float(ra_deg):.8f}, {float(dec_deg):.8f}, {float(radius_deg):.8f})
  )
  AND phot_g_mean_mag IS NOT NULL
  AND phot_g_mean_mag <= {float(mag_limit):.4f}
ORDER BY phot_g_mean_mag ASC
""".strip()


def build_bright_sample_query(*, mag_limit: float, row_limit: int) -> str:
    """构建全局亮星样本 ADQL 查询。

    输入:
    - `mag_limit`: G-band 星等上限。
    - `row_limit`: 最多返回多少行。

    输出:
    - ADQL 查询字符串。

    用途:
    - 快速拿到一批全局亮星，方便测试投影和渲染；
    - 不适合作为完整全天星表，因为它按亮度排序，只是一个测试样本。
    """

    return f"""
SELECT TOP {int(row_limit)}
  source_id,
  ra,
  dec,
  phot_g_mean_mag,
  phot_bp_mean_mag,
  phot_rp_mean_mag,
  bp_rp,
  parallax,
  pmra,
  pmdec,
  ruwe
FROM gaiadr3.gaia_source
WHERE
  phot_g_mean_mag IS NOT NULL
  AND phot_g_mean_mag <= {float(mag_limit):.4f}
ORDER BY phot_g_mean_mag ASC
""".strip()


def read_query_file(path: Path) -> str:
    """从本地文件读取自定义 ADQL 查询。

    输入:
    - `path`: ADQL 文件路径。

    输出:
    - 查询字符串。

    作用:
    - 后续如果需要更复杂筛选，比如 HEALPix tile、颜色范围、ruwe 质量筛选，
      可以直接写 ADQL 文件，不需要改 Python 代码。
    """

    return path.read_text(encoding="utf-8").strip()


def download_tap_csv(
    *,
    query: str,
    output_path: Path,
    tap_url: str,
    timeout: int,
) -> dict[str, Any]:
    """执行 TAP 查询并保存 CSV。

    输入:
    - `query`: ADQL 查询。
    - `output_path`: CSV 输出路径。
    - `tap_url`: TAP sync endpoint。
    - `timeout`: HTTP 超时时间，单位秒。

    输出:
    - 下载统计信息，包括耗时、文件大小和输出路径。

    关键逻辑:
    - 使用 POST 提交 ADQL；
    - 先写入 `.tmp` 文件，下载成功后再原子替换为正式文件；
    - 如果服务返回 HTML/XML 错误页，主动报错，避免把错误页当 CSV 使用。
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    started = time.time()

    payload = {
        "REQUEST": "doQuery",
        "LANG": "ADQL",
        "FORMAT": "csv",
        "QUERY": query,
    }

    with requests.post(tap_url, data=payload, stream=True, timeout=timeout) as response:
        response.raise_for_status()

        first_chunk = b""
        with tmp_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                if not first_chunk:
                    first_chunk = chunk[:2048]
                f.write(chunk)

    # Gaia TAP 报错时可能返回 XML/HTML/VOTable 错误内容，这里做轻量防护。
    preview = first_chunk.lstrip().lower()
    if preview.startswith(b"<") or b"error" in preview[:512]:
        error_text = tmp_path.read_text(encoding="utf-8", errors="replace")[:2000]
        raise RuntimeError(f"TAP query did not return CSV. Preview:\n{error_text}")

    tmp_path.replace(output_path)
    elapsed = time.time() - started
    return {
        "output_path": str(output_path),
        "bytes": output_path.stat().st_size,
        "elapsed_sec": elapsed,
    }


def write_manifest(
    *,
    manifest_path: Path,
    output_path: Path,
    mode: str,
    args: argparse.Namespace,
    query: str,
    stats: dict[str, Any],
) -> None:
    """保存下载 manifest。

    输入:
    - `manifest_path`: manifest 输出路径。
    - `output_path`: CSV 输出路径。
    - `mode`: 查询模式。
    - `args`: 命令行参数。
    - `query`: 实际执行的 ADQL。
    - `stats`: 下载统计信息。

    输出:
    - 无返回值，写入 JSON 文件。

    用途:
    - 记录数据来源、查询参数和执行时间；
    - 保证后续论文和实验可以复现这批星表。
    """

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "catalog": "Gaia DR3",
        "table": "gaiadr3.gaia_source",
        "tap_url": args.tap_url,
        "mode": mode,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_path": str(output_path),
        "parameters": {
            "ra_deg": args.ra_deg,
            "dec_deg": args.dec_deg,
            "radius_deg": args.radius_deg,
            "mag_limit": args.mag_limit,
            "row_limit": args.row_limit,
            "query_file": str(args.query_file) if args.query_file else None,
        },
        "query": query,
        "stats": stats,
    }
    manifest_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。

    输入:
    - `argv`: 可选命令行参数列表，默认读取 `sys.argv`。

    输出:
    - argparse Namespace。
    """

    parser = argparse.ArgumentParser(
        description="Download a manageable Gaia DR3 subset for STAR-Agent clean-data simulation.",
    )
    parser.add_argument(
        "--mode",
        choices=["cone", "bright_sample", "query_file"],
        default="cone",
        help="下载模式: cone 为指定天区; bright_sample 为全局亮星测试样本; query_file 为自定义 ADQL。",
    )
    parser.add_argument("--ra_deg", type=float, default=180.0, help="cone-search 中心赤经，degree。")
    parser.add_argument("--dec_deg", type=float, default=0.0, help="cone-search 中心赤纬，degree。")
    parser.add_argument("--radius_deg", type=float, default=5.0, help="cone-search 半径，degree。")
    parser.add_argument("--mag_limit", type=float, default=18.0, help="G-band 星等上限。")
    parser.add_argument("--row_limit", type=int, default=50000, help="最多下载行数。")
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("STAR_Agent/data/catalogs/raw/gaia_dr3_cone_sample.csv"),
        help="CSV 输出路径。",
    )
    parser.add_argument(
        "--manifest_path",
        type=Path,
        default=Path("STAR_Agent/data/catalogs/catalog_manifest.json"),
        help="manifest 输出路径。",
    )
    parser.add_argument("--query_file", type=Path, default=None, help="自定义 ADQL 查询文件。")
    parser.add_argument("--tap_url", default=DEFAULT_TAP_SYNC_URL, help="Gaia TAP sync endpoint。")
    parser.add_argument("--timeout", type=int, default=600, help="HTTP 超时秒数。")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """脚本主入口。

    输入:
    - `argv`: 可选命令行参数。

    输出:
    - 进程退出码，0 表示成功。
    """

    args = parse_args(argv)

    if args.mode == "cone":
        query = build_cone_query(
            ra_deg=args.ra_deg,
            dec_deg=args.dec_deg,
            radius_deg=args.radius_deg,
            mag_limit=args.mag_limit,
            row_limit=args.row_limit,
        )
    elif args.mode == "bright_sample":
        query = build_bright_sample_query(
            mag_limit=args.mag_limit,
            row_limit=args.row_limit,
        )
    else:
        if args.query_file is None:
            raise ValueError("--mode query_file requires --query_file")
        query = read_query_file(args.query_file)

    print("[INFO] Gaia DR3 query:")
    print(query)
    print(f"[INFO] downloading to: {args.output_path}")

    stats = download_tap_csv(
        query=query,
        output_path=args.output_path,
        tap_url=args.tap_url,
        timeout=args.timeout,
    )
    write_manifest(
        manifest_path=args.manifest_path,
        output_path=args.output_path,
        mode=args.mode,
        args=args,
        query=query,
        stats=stats,
    )

    print("[OK] download finished")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"[OK] manifest: {args.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
