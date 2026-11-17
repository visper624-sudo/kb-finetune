# -*- coding: utf-8 -*-
"""截取 2.9 知识库平台接入微调模型后的页面（需要 2.9 与 demo 同时运行）。

前置：
    python run_demo.py                       # 8802
    python run.py --port 8803                # 2.9，KB_DATA_DIR 指向临时数据目录
    # 已通过 /api/system/config 把 2.9 的 LLM 配置指到 http://127.0.0.1:8802/v1

用法：
    python scripts/capture_integration.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "screenshots"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"


def login(base: str) -> str:
    data = json.dumps({"username": "admin", "password": "123456"}).encode("utf-8")
    req = urllib.request.Request(base + "/api/auth/login", data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))["data"]["token"]


def shot(edge: str, url: str, out: Path, profile: Path) -> bool:
    cmd = [edge, "--headless=old", "--disable-gpu", "--no-sandbox", "--no-proxy-server",
           "--disable-software-rasterizer", "--hide-scrollbars", "--window-size=1600,1040",
           f"--user-data-dir={profile}", "--no-first-run", "--virtual-time-budget=12000",
           f"--screenshot={out}", url]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    time.sleep(0.5)
    return out.exists() and out.stat().st_size > 1000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8803")
    ap.add_argument("--session", type=int, default=9)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    profile = ROOT / ".edge-profile-29"
    token = login(args.base)
    pages = [
        ("09-29-chat.png", f"{args.base}/?token={token}#/chat?session={args.session}"),
        ("10-29-config.png", f"{args.base}/?token={token}#/org?tab=config"),
    ]
    ok = 0
    for name, url in pages:
        out = OUT / name
        good = shot(EDGE, url, out, profile)
        ok += 1 if good else 0
        print(("OK  " if good else "FAIL"), name, out.stat().st_size if out.exists() else 0)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
