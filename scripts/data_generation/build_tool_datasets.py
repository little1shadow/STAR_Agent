#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build STAR-Agent executor/tool train-val-test datasets.

这个入口脚本不训练 executor 模型，只负责把已有 degradation 数据转换成各工具可用的
paired dataset。输出会按 domain/source/subtask/tool/split 分层保存。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from star_agent.data_generation.dataset_builders.common import add_common_args, build_tool_dataset, load_config


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    输入:
    - 无，直接读取 CLI。

    输出:
    - argparse.Namespace。
    """

    parser = argparse.ArgumentParser(description="Build executor/tool datasets for STAR-Agent.")
    add_common_args(parser)
    return parser.parse_args()


def main() -> int:
    """CLI 主入口。

    输入:
    - 命令行参数。

    输出:
    - 进程退出码，0 表示成功。
    """

    args = parse_args()
    cfg = load_config(args.config)
    build_tool_dataset(cfg, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
