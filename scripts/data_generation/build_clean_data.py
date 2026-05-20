#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""clean_data 生成脚本入口。

该文件只负责设置 import path 并调用核心模块。
真正的生成逻辑在:
`STAR_Agent/star_agent/data_generation/clean_simulation/build_clean_data.py`
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    """脚本主入口。

    输入:
    - 无，透传命令行参数给核心模块。

    输出:
    - 进程退出码。
    """

    star_agent_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(star_agent_root))

    from star_agent.data_generation.clean_simulation.build_clean_data import main as build_main

    return int(build_main())


if __name__ == "__main__":
    raise SystemExit(main())
