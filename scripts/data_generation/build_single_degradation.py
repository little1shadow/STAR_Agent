#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""single degradation 数据集生成脚本入口。"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    """脚本主入口。

    输入:
    - 命令行参数，原样透传给核心生成模块。

    输出:
    - 进程退出码。
    """

    star_agent_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(star_agent_root))
    from star_agent.data_generation.composition.build_single_degradation import main as build_main

    return int(build_main())


if __name__ == "__main__":
    raise SystemExit(main())
