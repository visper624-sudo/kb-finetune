# -*- coding: utf-8 -*-
"""运行评测：逐配置生成回答 → 逐条判定 → 汇总对比表/案例/失败分析。

用法：
    python eval/run_eval.py                      # baseline + dpf + 三个消融
    python eval/run_eval.py --configs dpf        # 只跑微调模型
    python eval/run_eval.py --configs lora --model Qwen/Qwen2.5-1.5B-Instruct --adapter adapter/lora
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import paths  # noqa: E402
from eval.generate import build_generators  # noqa: E402
from eval.metrics import aggregate, judge, per_capability  # noqa: E402
from train.policy import FineTunedPolicy  # noqa: E402

DEFAULT_CONFIGS = ["baseline", "dpf", "ablation_no_refusal",
                   "ablation_no_citation", "ablation_no_retrieval"]
CONFIG_LABELS = {
    "baseline": "基座模型（检索直答）",
    "dpf": "策略模型 DPH（本机实测）",
    "ablation_no_refusal": "消融-无拒答训练",
    "ablation_no_citation": "消融-无引用",
    "ablation_no_retrieval": "消融-无检索",
    "lora": "Qwen LoRA（待 GPU）",
}


def run_one(gen, samples: list[dict]) -> tuple[list[dict], dict]:
    records = []
    for s in samples:
        pred = gen.answer(s)
        rec = {
            "id": s["id"], "split": s["split"], "question": s["question"],
            "pred": pred, "gold": s["answer"], "task": s["task"],
            "format": s["format"], "meta": s.get("meta", {}),
            "contexts": s.get("contexts", []),
            "judge": judge(pred, s),
        }
        records.append(rec)
    return records, aggregate(records)


def markdown_table(results: dict[str, dict]) -> str:
    cols = list(results)
    metrics = [
        ("faithfulness", "忠实性", "高"),
        ("hallucination_rate", "幻觉率", "低"),
        ("citation_accuracy", "引用准确率", "高"),
        ("citation_coverage", "引用覆盖率", "高"),
        ("citation_gold", "引用命中 gold 片段", "高"),
        ("key_point_coverage", "要点覆盖率", "高"),
        ("format_compliance", "格式合规率", "高"),
        ("term_accuracy", "术语准确率", "高"),
        ("refusal_accuracy", "拒答准确率", "高"),
        ("false_refusal_rate", "误拒率", "低"),
        ("composite", "综合分", "高"),
    ]
    lines = ["| 指标 | " + " | ".join(CONFIG_LABELS.get(c, c) for c in cols) + " |",
             "| --- | " + " | ".join("---" for _ in cols) + " |"]
    for key, name, _ in metrics:
        row = [name]
        for c in cols:
            v = results[c].get(key)
            row.append("—" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("| 样本量 | " + " | ".join(str(results[c]["n"]) for c in cols) + " |")
    return "\n".join(lines)


def build_cases(records_by_cfg: dict[str, list[dict]]) -> str:
    dpf = records_by_cfg.get("dpf") or records_by_cfg.get("lora")
    base = {r["id"]: r for r in records_by_cfg.get("baseline", [])}
    if not dpf:
        return ""
    wanted = ["policy", "product", "steps", "troubleshoot", "integration",
              "refusal", "refusal_weak", "refusal_conflict", "partial"]
    lines = ["# 典型案例对比（baseline vs 微调）", ""]
    used = set()
    for fmt in wanted:
        cand = [r for r in dpf if r["format"] == fmt and r["id"] not in used]
        if not cand:
            continue
        r = cand[0]
        used.add(r["id"])
        b = base.get(r["id"], {})
        lines += [
            f"## {fmt}", "",
            f"**问题**：{r['question']}", "",
            f"**上下文**：" + "；".join(f"[{c['n']}] {c['title']}" for c in r["contexts"][:4]), "",
            "**Baseline**：", "", "```", b.get("pred", "（未运行）"), "```", "",
            "**微调模型**：", "", "```", r["pred"], "```", "",
            f"**判定**：忠实性 {r['judge'].get('faithfulness')} / 引用覆盖 {r['judge'].get('citation_coverage')} / "
            f"格式 {r['judge'].get('format_ok')} / 拒答正确 {r['judge'].get('refusal_ok')}", "",
        ]
    return "\n".join(lines)


def build_failures(records: list[dict], limit: int = 10) -> str:
    scored = []
    for r in records:
        j = r["judge"]
        ok = 0
        ok += 1 if j.get("faithfulness") is not None and j["faithfulness"] >= 0.8 else 0
        ok += 1 if j.get("citation_coverage") is not None and j["citation_coverage"] >= 0.8 else 0
        ok += 1 if j.get("format_ok") else 0
        ok += 1 if j.get("refusal_ok") is not False else 0
        ok += 1 if not j.get("false_refusal") else 0
        scored.append((ok, r))
    scored.sort(key=lambda x: x[0])
    lines = ["# 失败案例分析（自动抽取）", ""]
    for ok, r in scored[:limit]:
        lines += [
            f"## {r['id']}（{r['format']}，通过 {ok}/5）", "",
            f"**问题**：{r['question']}", "",
            "**模型回答**：", "", "```", r["pred"], "```", "",
            f"**Gold**：", "", "```", r["gold"], "```", "",
            f"**判定**：{json.dumps(r['judge'], ensure_ascii=False)}", "",
        ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="显存不足时用 4bit 加载基座（LoRA 评测）")
    args = ap.parse_args()

    samples = paths.read_jsonl(paths.OUT_DIR / "test.jsonl")
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit("测试集为空，请先运行 data/build_dataset.py")

    need_policy = any(c != "baseline" and c != "ablation_no_retrieval" for c in args.configs)
    policy = FineTunedPolicy() if need_policy else None
    gens = build_generators(args.configs, policy, args.model, args.adapter,
                            max_new_tokens=args.max_new_tokens,
                            load_in_4bit=args.load_in_4bit)

    out_root = paths.RESULTS_DIR
    summary, all_records = {}, {}
    for gen in gens:
        print(f"[{gen.name}] 生成 {len(samples)} 条 ...")
        records, metrics = run_one(gen, samples)
        out = out_root / gen.name
        out.mkdir(parents=True, exist_ok=True)
        paths.write_jsonl(out / "predictions.jsonl", records)
        paths.dump_json(out / "metrics.json", metrics)
        paths.dump_json(out / "per_capability.json", per_capability(records))
        summary[gen.name] = metrics
        all_records[gen.name] = records

    paths.dump_json(out_root / "summary.json", summary)
    md = ["# 三组对比与消融结果（自动生成）", "",
          "评测口径见 `docs/评测说明.md`；所有数字均由 `eval/run_eval.py` 在当前测试集上计算。", "",
          markdown_table(summary), "",
          "> 说明：忠实性/幻觉率在抽取式实现下接近上限，主要区分力在引用、格式与拒答；",
          "> LoRA 路径用同一脚本 `--configs lora` 运行后会自动追加一列。", ""]
    (out_root / "comparison.md").write_text("\n".join(md), encoding="utf-8")
    cases = build_cases(all_records)
    (out_root / "cases.md").write_text(cases, encoding="utf-8")
    if "dpf" in all_records:
        (out_root / "failures.md").write_text(build_failures(all_records["dpf"]), encoding="utf-8")

    print("\n" + markdown_table(summary))
    print("\n结果目录：", out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
