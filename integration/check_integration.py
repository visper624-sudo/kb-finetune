# -*- coding: utf-8 -*-
"""验证 2.9 知识库平台 → 微调模型 的端到端接入。

前置：
  1. 本仓库 Demo 已启动：python run_demo.py（8802）
  2. 2.9 已启动：python run.py --port 8803（建议用独立 KB_DATA_DIR）
  3. 2.9 已初始化演示数据（首次启动自动 seed）

脚本会：
  * 用 admin/123456 登录 2.9；
  * 把 2.9 的 LLM 配置切到本仓库的 OpenAI 兼容端点；
  * 调 /api/chat/ask 提一个知识库问题；
  * 打印回答、引用片段与耗时，验证「2.9 检索 + 四维鉴权 + 微调生成」链路。
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request


def req(base: str, method: str, path: str, obj=None, token: str | None = None):
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8") if obj is not None else None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = urllib.request.Request(base + path, data=data, headers=headers, method=method)
    with urllib.request.urlopen(r, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default="http://127.0.0.1:8803")
    ap.add_argument("--llm", default="http://127.0.0.1:8802/v1")
    ap.add_argument("--question", default="七天内无理由退货怎么规定？")
    args = ap.parse_args()

    try:
        health = req(args.kb, "GET", "/api/system/health")
        print("2.9 health:", health.get("data", {}).get("status"))
    except Exception as exc:  # noqa: BLE001
        print(f"2.9 不可用：{exc}\n请先启动 2.9：python run.py --port 8803")
        return 1

    login = req(args.kb, "POST", "/api/auth/login", {"username": "admin", "password": "123456"})
    token = login["data"]["token"]
    cfg = req(args.kb, "PUT", "/api/system/config", {
        "provider": "openai", "base_url": args.llm, "api_key": "any",
        "model": "kb-finetuned-dph", "temperature": 0.2,
    }, token)
    llm = (cfg.get("data") or {}).get("llm") or {}
    print("2.9 LLM:", llm.get("provider"), llm.get("model"), "engine=", llm.get("engine"))
    if llm.get("provider") != "openai":
        print("配置未生效")
        return 1

    ask = req(args.kb, "POST", "/api/chat/ask", {"question": args.question}, token)["data"]
    print("问题：", args.question)
    print("耗时：", ask.get("latency_ms"), "ms；引用数：", len(ask.get("citations") or []))
    print("回答：\n", ask.get("answer", "")[:800])
    print("引用片段：", [(c["index"], c["title"]) for c in (ask.get("citations") or [])][:6])
    if "【结论】" not in ask.get("answer", "") and "【" not in ask.get("answer", ""):
        print("警告：回答不是微调模型的格式，检查 2.9 是否真的走 OpenAI 端点")
        return 1
    print("\n集成验证通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
