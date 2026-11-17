# -*- coding: utf-8 -*-
"""启动演示控制台（等价于 python demo/server.py）。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from demo.server import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
