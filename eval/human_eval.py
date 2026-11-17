# -*- coding: utf-8 -*-
"""人工精评：生成 120 条双人标注表，并计算 Cohen's κ。

用法：
    python eval/human_eval.py                 # 生成 sheet.csv
    python eval/human_eval.py --kappa         # 读取 annotator1/2.csv 计算 κ

标注规则见 docs/评测说明.md 第 5 节。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import paths  # noqa: E402

OUT = paths.EVAL_DIR / "human_eval"
DIMENSIONS = ["faithfulness", "citation", "refusal", "overall"]


def build_sheet() -> Path:
    preds = [json.loads(l) for l in (paths.RESULTS_DIR / "dpf" / "predictions.jsonl")
             .open(encoding="utf-8") if l.strip()]
    # 分层抽样：每类能力取前 N 条，保证 120 条且覆盖所有能力
    buckets: dict[str, list[dict]] = {}
    for r in preds:
        cap = r.get("meta", {}).get("capability") or r.get("format")
        buckets.setdefault(cap, []).append(r)
    quota = {"policy": 35, "product": 15, "steps": 8, "troubleshoot": 10,
             "integration": 10, "refusal_none": 15, "refusal_weak": 10,
             "refusal_conflict": 9, "partial": 8}
    picked = []
    for cap, n in quota.items():
        picked.extend(buckets.get(cap, [])[:n])
    picked = picked[:120]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "sheet.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "capability", "question", "contexts", "gold", "pred",
                    *[f"a1_{d}" for d in DIMENSIONS],
                    *[f"a2_{d}" for d in DIMENSIONS], "notes"])
        for r in picked:
            ctx = "\n".join(f"[{c['n']}] 《{c['title']}》 {c['text'][:160]}" for c in r["contexts"][:6])
            w.writerow([r["id"], r.get("meta", {}).get("capability") or r.get("format"),
                        r["question"], ctx, r["gold"], r["pred"],
                        *[""] * len(DIMENSIONS), *[""] * len(DIMENSIONS), ""])
    (OUT / "README.md").write_text(
        "# 人工精评说明\n\n"
        "1. 两位标注人独立填写 `sheet.csv` 中的 `a1_*` / `a2_*` 列（1~5 分，5 最好）。\n"
        "2. 另存为 `annotator1.csv` / `annotator2.csv`（保留 `id` 列）。\n"
        "3. 运行 `python eval/human_eval.py --kappa`。\n\n"
        "维度：faithfulness（忠实）、citation（引用）、refusal（拒答边界）、overall（整体可交付）。\n",
        encoding="utf-8")
    print(f"生成 {len(picked)} 条：{path}")
    return path


def _read_scores(path: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            out[row["id"]] = {d: int(row.get(d) or 0) for d in DIMENSIONS}
    return out


def weighted_kappa(a: list[int], b: list[int], max_rating: int = 5) -> float:
    """quadratic weighted Cohen's kappa。"""
    n = len(a)
    if n == 0:
        return 0.0
    k = max_rating
    obs = [[0] * (k + 1) for _ in range(k + 1)]
    for x, y in zip(a, b):
        obs[x][y] += 1
    hist_a = [sum(obs[i]) for i in range(k + 1)]
    hist_b = [sum(obs[i][j] for i in range(k + 1)) for j in range(k + 1)]
    num = den = 0.0
    for i in range(k + 1):
        for j in range(k + 1):
            w = ((i - j) ** 2) / (k ** 2)
            e = hist_a[i] * hist_b[j] / n
            num += w * obs[i][j]
            den += w * e
    return 1.0 - num / den if den else 0.0


def compute_kappa() -> int:
    p1, p2 = OUT / "annotator1.csv", OUT / "annotator2.csv"
    if not p1.exists() or not p2.exists():
        print("缺少 annotator1.csv / annotator2.csv；请先按 README 完成双人标注")
        return 1
    a, b = _read_scores(p1), _read_scores(p2)
    ids = sorted(set(a) & set(b))
    if not ids:
        print("两份标注没有共同的 id")
        return 1
    print(f"共同样本：{len(ids)}")
    for d in DIMENSIONS:
        xs = [a[i][d] for i in ids if 1 <= a[i][d] <= 5 and 1 <= b[i][d] <= 5]
        ys = [b[i][d] for i in ids if 1 <= a[i][d] <= 5 and 1 <= b[i][d] <= 5]
        if not xs:
            continue
        k = weighted_kappa(xs, ys)
        print(f"{d:14s} κ={k:.3f}  平均分 a1={sum(xs)/len(xs):.2f} a2={sum(ys)/len(ys):.2f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa", action="store_true")
    args = ap.parse_args()
    return compute_kappa() if args.kappa else (0 if build_sheet() else 1)


if __name__ == "__main__":
    raise SystemExit(main())
