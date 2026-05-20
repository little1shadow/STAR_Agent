#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""agenticir 主环境调用 tetra3rs 外部环境的适配器。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_CFG = "configs/downstream/star_matching/tetra3rs.yaml"


def resolve_repo_root(repo_root: str | Path | None = None) -> Path:
    """解析 STAR_Agent 根目录。

    输入:
    - `repo_root`: 可选显式根目录。

    输出:
    - STAR_Agent 根目录 Path。
    """

    if repo_root is not None:
        return Path(repo_root).resolve()
    return Path(__file__).resolve().parents[4]


def run_tetra3rs_star_mask(
    image_path: str | Path,
    output_dir: str | Path,
    repo_root: str | Path | None = None,
    conda_env: str = "star_downstream",
    cfg: str | Path = DEFAULT_CFG,
    use_conda: bool = True,
    timeout: int | None = None,
) -> dict[str, Any]:
    """调用 tetra3rs 生成单张图 star pseudo mask。

    输入:
    - `image_path`: 输入真实 clean 图像。
    - `output_dir`: 输出目录。
    - `repo_root`: STAR_Agent 根目录。
    - `conda_env`: 安装 tetra3rs 的 conda 环境。
    - `cfg`: tetra3rs YAML 配置。
    - `use_conda`: 是否通过 `conda run` 调用；本环境已安装 tetra3rs 时可设 False。
    - `timeout`: 子进程超时时间秒数。

    输出:
    - metrics 字典，并额外包含 `star_mask_path`、`centroids_path`、`background_mask_path`。
    """

    root = resolve_repo_root(repo_root)
    script = root / "scripts" / "downstream" / "run_tetra3rs.py"
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = Path(cfg)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path

    base_cmd = [
        "python",
        str(script),
        "--image",
        str(Path(image_path).resolve()),
        "--output_dir",
        str(out),
        "--cfg",
        str(cfg_path),
    ]
    cmd = ["conda", "run", "-n", conda_env, *base_cmd] if use_conda else [sys.executable, *base_cmd[1:]]
    subprocess.run(cmd, cwd=root, check=True, timeout=timeout)

    metrics_path = out / "metrics.json"
    if not metrics_path.exists():
        raise RuntimeError(f"tetra3rs metrics not found: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.update(
        {
            "star_mask_path": str(out / "star_mask.png"),
            "background_mask_path": str(out / "background_mask.png"),
            "valid_mask_path": str(out / "valid_mask.png"),
            "centroids_path": str(out / "centroids.json"),
            "metrics_path": str(metrics_path),
        }
    )
    return metrics
