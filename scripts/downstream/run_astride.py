#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""单张图 ASTRiDE 条纹目标检测脚本。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def add_repo_path() -> Path:
    """把 STAR_Agent 根目录加入 `sys.path`。"""

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    return root


def load_yaml(path: str | Path) -> dict:
    """读取 YAML 配置。"""

    p = Path(path)
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Run ASTRiDE streak detector on one image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cfg", default="configs/downstream/target_detection/astride.yaml")
    return parser.parse_args()


def main() -> int:
    """脚本入口。

    输出:
    - `streak_mask.png`: ASTRiDE 条纹 mask。
    - `streaks.json`: 统一目标记录。
    - `astride_summary.json`: ASTRiDE 原始摘要和运行信息。
    """

    root = add_repo_path()
    from star_agent.downstream.common.image_ops import ensure_dir, save_mask, write_json
    from star_agent.downstream.target_detection.astride_adapter.streak import run_astride_detection

    args = parse_args()
    cfg_path = Path(args.cfg)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = load_yaml(cfg_path)
    output_dir = ensure_dir(Path(args.output_dir).resolve())
    result = run_astride_detection(Path(args.image).resolve(), output_dir, cfg)

    save_mask(output_dir / "streak_mask.png", result["mask"])
    write_json(output_dir / "streaks.json", {"targets": result["records"]})
    write_json(output_dir / "astride_summary.json", result["summary"])
    print(f"[OK] image: {Path(args.image).resolve()}")
    print(f"[OK] streaks: {len(result['records'])}")
    print(f"[OK] mask: {output_dir / 'streak_mask.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
