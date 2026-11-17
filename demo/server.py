# -*- coding: utf-8 -*-
"""知识库微调模型 Demo 服务（标准库，零第三方依赖）。

功能：
  1. Web 控制台：同一问题对比「基座模型式 RAG」与「微调策略模型」；
     展示检索片段、引用编号、格式、拒答原因；
  2. 数据集与评测结果浏览；
  3. **OpenAI 兼容接口** `/v1/chat/completions`：2.9 知识库平台把
     `LLM_PROVIDER` 配成 openai、base_url 指向本服务即可零改造接入微调模型。

启动：
    python run_demo.py                 # http://127.0.0.1:8802
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import paths  # noqa: E402
from config.prompts import parse_reference_blocks  # noqa: E402
from eval.generate import BaselineGenerator  # noqa: E402
from kb.compose import citation_indices, sentences_from_contexts  # noqa: E402
from kb.corpus import Corpus, as_context  # noqa: E402
from kb.retrieval import Retriever, context_features  # noqa: E402
from train.policy import FineTunedPolicy  # noqa: E402

STATIC = paths.STATIC_DIR
START = time.time()
#: 公网部署时设置 KB_API_TOKEN，保护 /v1/*（2.9 的 api_key 填同一个值）
API_TOKEN = os.environ.get("KB_API_TOKEN", "")

STATE: dict = {}


def init() -> None:
    corpus = Corpus.load()
    retriever = Retriever(corpus.chunks)
    policy = FineTunedPolicy(vectorizer=retriever.vectorizer)
    STATE.update({
        "corpus": corpus,
        "retriever": retriever,
        "policy": policy,
        "baseline": BaselineGenerator(),
        "chunks": corpus.chunks,
        "docs": corpus.docs,
    })


def retrieve(question: str, top_k: int = 4) -> list[dict]:
    hits = STATE["retriever"].search(question, top_k=top_k)
    return [as_context(h, i + 1) for i, h in enumerate(hits)]


def ask(question: str) -> dict:
    question = (question or "").strip()
    if not question:
        return {"ok": False, "error": "问题不能为空"}
    t0 = time.time()
    contexts = retrieve(question, 4)
    t1 = time.time()
    pred = STATE["policy"].predict(question, contexts)
    t2 = time.time()
    baseline = STATE["baseline"].answer({"question": question, "contexts": contexts, "meta": {}})
    feats = context_features(question, contexts, STATE["retriever"].vectorizer)
    return {
        "ok": True,
        "question": question,
        "contexts": contexts,
        "baseline": {"answer": baseline, "latency_ms": int((t2 - t1) * 1000)},
        "finetuned": {
            "answer": pred["answer"],
            "decision": pred["decision"],
            "format": pred["format"],
            "gate_prob": pred["gate_prob"],
            "citations": citation_indices(pred["answer"]),
            "evidence": pred.get("evidence", []),
            "latency_ms": int((t2 - t1) * 1000),
        },
        "features": {k: feats[k] for k in ("top1_sim", "max_cov", "oov_ratio", "n_parts", "conflict")},
        "timings": {"retrieval_ms": int((t1 - t0) * 1000), "generate_ms": int((t2 - t1) * 1000)},
    }


def stats() -> dict:
    ds = paths.load_json(paths.OUT_DIR / "dataset_stats.json", {}) or {}
    meta = paths.load_json(paths.ADAPTER_DIR / "policy" / "meta.json", {}) or {}
    summary = paths.load_json(paths.RESULTS_DIR / "summary.json", {}) or {}
    per_cap = paths.load_json(paths.RESULTS_DIR / "dpf" / "per_capability.json", {}) or {}
    return {
        "dataset": ds,
        "model": {
            "version": meta.get("version"),
            "gate_threshold": meta.get("gate_threshold"),
            "gate": (meta.get("gate_metrics") or {}).get("valid"),
            "format": (meta.get("format_metrics") or {}),
            "selector": (meta.get("selector_metrics") or {}).get("valid"),
            "train_rows": meta.get("train_rows"),
            "created_at": meta.get("created_at"),
        },
        "eval": summary,
        "per_capability": per_cap,
        "corpus": STATE["corpus"].stats() if STATE else {},
        "uptime_sec": int(time.time() - START),
    }


def samples(capability: str = "", limit: int = 30) -> list[dict]:
    rows: list[dict] = []
    for split in ("train", "valid", "test"):
        rows.extend(paths.read_jsonl(paths.OUT_DIR / f"{split}.jsonl"))
    if capability:
        rows = [r for r in rows if (r.get("meta", {}).get("capability") or r.get("format")) == capability]
    out = []
    for r in rows[:limit]:
        out.append({
            "id": r["id"], "split": r["split"], "question": r["question"],
            "format": r["format"], "capability": r.get("meta", {}).get("capability"),
            "answer": r["answer"], "contexts": r["contexts"], "task": r["task"],
        })
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "KBFinetuneDemo/1.0"

    def log_message(self, fmt: str, *args) -> None:  # 静默访问日志
        return

    # ------------------------------------------------------------ helpers
    def _send(self, status: int, body: bytes, content_type: str = "application/json; charset=utf-8") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status: int = 200) -> None:
        self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def _body(self) -> dict:
        try:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n).decode("utf-8") or "{}")
        except Exception:  # noqa: BLE001
            return {}

    def _static(self, path: str) -> None:
        rel = path.lstrip("/") or "index.html"
        f = (STATIC / rel).resolve()
        if not str(f).startswith(str(STATIC.resolve())) or not f.is_file():
            f = STATIC / "index.html"
        ctype = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
                 ".js": "application/javascript; charset=utf-8", ".svg": "image/svg+xml",
                 ".png": "image/png", ".json": "application/json; charset=utf-8"}.get(
            f.suffix, "application/octet-stream")
        self._send(200, f.read_bytes(), ctype)

    # ------------------------------------------------------------ routes
    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path.startswith("/v1/") and not self._auth_ok():
            self._json({"error": {"message": "invalid api key", "type": "invalid_request_error"}}, 401)
            return
        if path == "/api/health":
            self._json({"ok": True, "model": "kb-finetuned-dph", "uptime_sec": int(time.time() - START)})
        elif path == "/api/stats":
            self._json(stats())
        elif path == "/api/samples":
            qs = parse_qs(urlparse(self.path).query)
            cap = (qs.get("cap") or [""])[0]
            self._json({"ok": True, "samples": samples(cap)})
        elif path == "/v1/models":
            self._json({"object": "list", "data": [{
                "id": "kb-finetuned-dph", "object": "model", "owned_by": "kb-finetune",
                "created": int(START),
            }]})
        else:
            self._static(path)

    def _auth_ok(self) -> bool:
        if not API_TOKEN:
            return True
        return self.headers.get("Authorization", "") == f"Bearer {API_TOKEN}"

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path.startswith("/v1/") and not self._auth_ok():
            self._json({"error": {"message": "invalid api key", "type": "invalid_request_error"}}, 401)
            return
        if path == "/api/ask":
            body = self._body()
            try:
                self._json(ask(body.get("question", "")))
            except Exception as exc:  # noqa: BLE001
                self._json({"ok": False, "error": str(exc)}, 500)
        elif path == "/v1/chat/completions":
            self._chat_completions(self._body())
        else:
            self._json({"ok": False, "error": "not found"}, 404)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    # ------------------------------------------------------------ OpenAI 兼容
    def _chat_completions(self, body: dict) -> None:
        messages = body.get("messages") or []
        user_text = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_text = m.get("content") or ""
                break
        question, contexts = parse_reference_blocks(user_text)
        if not question:
            question = user_text.strip()[:200]
        if not contexts:
            contexts = retrieve(question, 4)
        pred = STATE["policy"].predict(question, contexts)
        answer = pred["answer"]
        prompt_tokens = len(user_text) // 2 + 80
        completion_tokens = len(answer) // 2
        model = body.get("model") or "kb-finetuned-dph"
        cid = f"chatcmpl-{int(time.time()*1000)}"

        if not body.get("stream"):
            self._json({
                "id": cid, "object": "chat.completion", "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": answer},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                          "total_tokens": prompt_tokens + completion_tokens},
            })
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        def emit(obj: dict) -> None:
            self.wfile.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8"))
            self.wfile.flush()

        emit({"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
              "model": model, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
        for i in range(0, len(answer), 18):
            emit({"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                  "model": model, "choices": [{"index": 0, "delta": {"content": answer[i:i + 18]},
                                               "finish_reason": None}]})
            time.sleep(0.008)
        emit({"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
              "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
              "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens}})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8802)
    args = ap.parse_args()
    init()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"知识库微调 Demo: http://{args.host}:{args.port}")
    print(f"OpenAI 兼容端点: http://{args.host}:{args.port}/v1  (model=kb-finetuned-dph)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
