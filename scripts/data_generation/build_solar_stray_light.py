#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""新版 solar stray light 退化生成脚本入口。

真正逻辑在：
`STAR_Agent/star_agent/data_generation/degradations/solar_stray_light.py`
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

    from star_agent.data_generation.degradations.solar_stray_light import main as solar_main

    return int(solar_main())


if __name__ == "__main__":
    raise SystemExit(main())
