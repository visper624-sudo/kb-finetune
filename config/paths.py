# -*- coding: utf-8 -*-
"""统一路径管理：所有脚本都以项目根目录为基准，换机无需改代码。"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
SPECS_DIR = DATA_DIR / "specs"
OUT_DIR = DATA_DIR / "out"
REPORTS_DIR = DATA_DIR / "reports"

TRAIN_DIR = ROOT / "train"
EVAL_DIR = ROOT / "eval"
RESULTS_DIR = EVAL_DIR / "results"
ADAPTER_DIR = ROOT / "adapter"
DEMO_DIR = ROOT / "demo"
STATIC_DIR = DEMO_DIR / "static"
DOCS_DIR = ROOT / "docs"
SCREENSHOTS_DIR = ROOT / "screenshots"
REPORT_DIR = ROOT / "report"

SPLITS = ("train", "valid", "test")

#: 4 类回答格式（对应 4.1.1 制度解释 / 产品说明 / 操作步骤 / 故障处理）
FORMATS = ("policy", "product", "steps", "troubleshoot")
FORMAT_NAMES = {
    "policy": "制度解释",
    "product": "产品说明",
    "steps": "操作步骤",
    "troubleshoot": "故障处理",
    "refusal": "资料不足拒答",
    "integration": "多片段整合",
}

#: 综合分权重：忠实 30% + 引用 20% + 拒答 20% + 覆盖 20% + 格式 10%
COMPOSITE_WEIGHTS = {
    "faithfulness": 0.30,
    "citation_score": 0.20,
    "refusal_score": 0.20,
    "key_point_coverage": 0.20,
    "format_compliance": 0.10,
}


def ensure_dirs() -> None:
    for d in (DATA_DIR, CORPUS_DIR, SPECS_DIR, OUT_DIR, REPORTS_DIR,
              RESULTS_DIR, ADAPTER_DIR, STATIC_DIR, DOCS_DIR,
              SCREENSHOTS_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_json(path: str | Path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def dump_json(path: str | Path, obj, indent: int = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=indent), encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n
