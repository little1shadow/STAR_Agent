#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""agenticir 主环境调用 ASTRiDE 外部环境的适配器。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_CFG = "configs/downstream/target_detection/astride.yaml"


def resolve_repo_root(repo_root: str | Path | None = None) -> Path:
    """解析 STAR_Agent 根目录。"""

    if repo_root is not None:
        return Path(repo_root).resolve()
    return Path(__file__).resolve().parents[4]


def run_astride_streak_detection(
    image_path: str | Path,
    output_dir: str | Path,
    repo_root: str | Path | None = None,
    conda_env: str = "star_astride",
    cfg: str | Path = DEFAULT_CFG,
    use_conda: bool = True,
    timeout: int | None = None,
) -> dict[str, Any]:
    """调用 ASTRiDE 检测短条纹目标。

    输入:
    - `image_path`: 输入图像。
    - `output_dir`: 输出目录。
    - `repo_root`: STAR_Agent 根目录。
    - `conda_env`: 安装 ASTRiDE 的 conda 环境。
    - `cfg`: ASTRiDE YAML 配置。
    - `use_conda`: 是否通过 conda run 调用。
    - `timeout`: 子进程超时时间秒数。

    输出:
    - ASTRiDE summary 字典，并附加 mask/json 输出路径。
    """

    root = resolve_repo_root(repo_root)
    script = root / "scripts" / "downstream" / "run_astride.py"
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

    summary_path = out / "astride_summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"ASTRiDE summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "streak_mask_path": str(out / "streak_mask.png"),
            "streaks_path": str(out / "streaks.json"),
            "summary_path": str(summary_path),
        }
    )
    return summary
