# -*- coding: utf-8 -*-
"""用 Edge 无头模式批量截图（零第三方依赖）。

用法：
    python scripts/capture_screenshots.py
    python scripts/capture_screenshots.py --base http://127.0.0.1:8802
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "screenshots"

EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/microsoft-edge",
]

PAGES = [
    ("01-chat-product.png", "/?q=" + quote("出差住宿费标准是多少？"), 11000),
    ("02-chat-policy.png", "/?q=" + quote("七天内无理由退货怎么规定？"), 11000),
    ("03-chat-troubleshoot.png", "/?q=" + quote("生鲜坏了怎么申请退款？"), 11000),
    ("04-chat-conflict.png", "/?q=" + quote("差旅报销到底要在 15 个工作日内还是 30 个自然日内提交？"), 11000),
    ("05-chat-refusal.png", "/?q=" + quote("竞品最近的定价策略是什么？"), 11000),
    ("06-dataset.png", "/#dataset", 10000),
    ("07-metrics.png", "/#metrics", 10000),
    ("08-integration.png", "/#integration", 9000),
]


def find_edge(explicit: str | None) -> str:
    if explicit:
        return explicit
    for p in EDGE_CANDIDATES:
        if Path(p).exists():
            return p
    raise SystemExit("未找到 Microsoft Edge，请用 --edge 指定路径")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8802")
    ap.add_argument("--edge", default=None)
    args = ap.parse_args()
    edge = find_edge(args.edge)
    OUT.mkdir(exist_ok=True)
    profile = ROOT / ".edge-profile"
    ok = 0
    for name, path, budget in PAGES:
        out = OUT / name
        url = args.base + path
        cmd = [
            edge, "--headless=old", "--disable-gpu", "--no-sandbox", "--no-proxy-server",
            "--disable-software-rasterizer", "--hide-scrollbars",
            "--window-size=1600,1040", f"--user-data-dir={profile}", "--no-first-run",
            f"--virtual-time-budget={budget}", f"--screenshot={out}", url,
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        time.sleep(0.5)
        exists = out.exists() and out.stat().st_size > 1000
        ok += 1 if exists else 0
        print(f"{'OK  ' if exists else 'FAIL'} {name}  {out.stat().st_size if out.exists() else 0}")
    print(f"{ok}/{len(PAGES)} 张截图完成：{OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
