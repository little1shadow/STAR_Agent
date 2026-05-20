#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""DQG 退化生成脚本入口。"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    """脚本主入口。"""

    star_agent_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(star_agent_root))
    from star_agent.data_generation.degradations.dqg import main as dqg_main

    return int(dqg_main())


if __name__ == "__main__":
    raise SystemExit(main())
