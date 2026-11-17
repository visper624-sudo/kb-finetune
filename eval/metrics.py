# -*- coding: utf-8 -*-
"""评测指标：忠实性、幻觉率、引用准确率/覆盖率、拒答准确率/误拒率、
要点覆盖率、格式合规率、术语准确率与综合分。

所有判断规则都是确定性的，可复现；人工精评脚本（eval/human_eval.py）使用
同一套抽样与字段，便于算 Cohen's κ。
"""
from __future__ import annotations

import re
from collections import Counter

from kb.format_rules import check_format
from kb.terms import aliases
from kb.textutil import (
    content_terms, fact_numbers, normalize_text, overlap_ratio, wilson_interval,
)

REFUSAL_RE = re.compile(
    r"无法回答|无法给出|资料不足|未找到|未检索到|未收录|没有相关|无法确定|"
    r"关联度较低|存在.{0,6}冲突|口径冲突|请确认适用|缺失信息|建议补充|"
    r"未覆盖|缺少.{0,10}(信息|规则|标准)"
)
CONFLICT_RE = re.compile(r"冲突|不一致|以哪个为准|版本")
MISSING_RE = re.compile(r"【缺失信息】|未覆盖|缺少|建议补充")
_CITE_RE = re.compile(r"\[(\d+)\]")


def is_refusal(text: str) -> bool:
    return bool(REFUSAL_RE.search(text or ""))


def answer_units(answer: str) -> list[dict]:
    """把答案拆成可判定的“事实单元”（句子 / 表格行 / 编号项）。"""
    units: list[dict] = []
    for raw in (answer or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if re.match(r"^\|\s*-{2,}", line):
            continue
        if re.match(r"^\|\s*项目\s*\|", line):
            continue
        citations = [int(x) for x in _CITE_RE.findall(line)]
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            text = " ".join(c for c in cells[:-1] if c)
        else:
            text = line
        text = re.sub(r"^【[^】]{1,12}】", "", text)
        text = re.sub(r"^\d+[.、)]\s*", "", text)
        text = _CITE_RE.sub("", text).strip()
        if len(text) < 8:
            continue
        if text.startswith(("以上内容", "注：", "说明：")):
            continue
        if not content_terms(text):
            continue
        units.append({"text": text, "citations": citations})
    return units


def supports(sentence: str, context_text: str) -> bool:
    """句子是否被片段支撑：数字全命中 + 内容词覆盖率 ≥ 0.5。"""
    nums = fact_numbers(sentence)
    if nums:
        ctx_nums = set(fact_numbers(context_text))
        if any(n not in ctx_nums and n.rstrip("/天") not in ctx_nums for n in nums):
            return False
    terms = content_terms(sentence)
    if not terms:
        return True
    return overlap_ratio(terms, content_terms(context_text)) >= 0.5


def _ctx_map(sample: dict) -> dict[int, dict]:
    return {int(c.get("n", i + 1)): c for i, c in enumerate(sample.get("contexts", []))}


def judge(pred: str, sample: dict) -> dict:
    pred = pred or ""
    ctxs = sample.get("contexts", [])
    ctx_map = _ctx_map(sample)
    all_ctx_text = "\n".join(c.get("text", "") for c in ctxs)
    gold_chunks = set(sample.get("meta", {}).get("gold_chunk_ids", []))
    expected_refusal = sample.get("task") == "refusal"
    expected_format = sample.get("format", "policy")
    detected_refusal = is_refusal(pred)

    units = answer_units(pred)
    factual = [u for u in units if not re.search(r"缺失信息|冲突信息|建议|依据|要点|现象|可能原因|处理步骤", u["text"])]
    supported = 0
    cited_units = 0
    citation_hits = 0
    citation_total = 0
    citation_gold_hits = 0
    hallucinated = False
    for u in factual:
        cites = [n for n in u["citations"] if n in ctx_map]
        if cites:
            cited_units += 1
            for n in cites:
                citation_total += 1
                if supports(u["text"], ctx_map[n].get("text", "")):
                    citation_hits += 1
                if ctx_map[n].get("chunk_id") in gold_chunks:
                    citation_gold_hits += 1
            if any(supports(u["text"], ctx_map[n].get("text", "")) for n in cites):
                supported += 1
        else:
            if all_ctx_text and supports(u["text"], all_ctx_text):
                supported += 1
        nums = fact_numbers(u["text"])
        if nums:
            ctx_nums = set(fact_numbers(all_ctx_text))
            gold_nums = set(sample.get("meta", {}).get("key_points", []))
            if any(n not in ctx_nums and n not in gold_nums for n in nums):
                hallucinated = True

    n_fact = len(factual)
    faithfulness = (supported / n_fact) if n_fact else None
    citation_accuracy = (citation_hits / citation_total) if citation_total else (0.0 if factual else None)
    citation_coverage = (cited_units / n_fact) if n_fact else None
    citation_gold = (citation_gold_hits / citation_total) if citation_total and gold_chunks else None

    # 要点覆盖：gold 数字 + 标准术语
    kps = list(dict.fromkeys(sample.get("meta", {}).get("key_points", [])))
    term = sample.get("meta", {}).get("term")
    if term:
        kps.append(term)
    if not kps:
        kps = sample.get("meta", {}).get("terms_used", [])[:2]
    pred_norm = normalize_text(pred).replace(" ", "")
    found = sum(1 for kp in kps if normalize_text(kp).replace(" ", "") in pred_norm)
    coverage = (found / len(kps)) if kps else None

    fmt_ok, fmt_problems = check_format(pred, expected_format)
    term_ok = None
    if term:
        # 只有“问题里出现标准术语或口语别称”的样本才考术语对齐；
        # 否则问题本身不涉及该术语，不能因为答案没重复它而扣分。
        q = sample.get("question", "")
        if term in q or any(a in q for a in aliases(term)):
            term_ok = term in pred and not any(a in pred for a in aliases(term))

    refusal_ok = None
    if expected_refusal:
        if sample.get("format") == "refusal_conflict":
            refusal_ok = detected_refusal and bool(CONFLICT_RE.search(pred)) and not hallucinated
        elif sample.get("format") == "partial":
            # 部分可答：既要给出有依据的部分，又要明确说明缺失部分
            refusal_ok = (detected_refusal and bool(MISSING_RE.search(pred))
                          and citation_total > 0 and n_fact > 0)
        else:
            refusal_ok = detected_refusal and not hallucinated
    false_refusal = bool(detected_refusal) if not expected_refusal else False

    return {
        "faithfulness": round(faithfulness, 4) if faithfulness is not None else None,
        "hallucination": bool(hallucinated),
        "citation_accuracy": round(citation_accuracy, 4) if citation_accuracy is not None else None,
        "citation_coverage": round(citation_coverage, 4) if citation_coverage is not None else None,
        "citation_gold": round(citation_gold, 4) if citation_gold is not None else None,
        "citation_count": citation_total,
        "key_point_coverage": round(coverage, 4) if coverage is not None else None,
        "format_ok": bool(fmt_ok),
        "format_problems": fmt_problems,
        "term_ok": term_ok,
        "refusal_ok": refusal_ok,
        "false_refusal": false_refusal,
        "detected_refusal": detected_refusal,
        "n_factual": n_fact,
        "n_units": len(units),
    }


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def aggregate(records: list[dict]) -> dict:
    judged = [r["judge"] for r in records]
    answerable = [r for r in records if r.get("task") == "answer"]
    unanswerable = [r for r in records if r.get("task") == "refusal"]

    def vals(key, rows):
        out = []
        for r in rows:
            v = r["judge"].get(key)
            if v is not None:
                out.append(float(v))
        return out

    faithfulness = _mean(vals("faithfulness", answerable))
    halluc = _mean([1.0 if j["hallucination"] else 0.0 for j in judged if j["faithfulness"] is not None])
    cit_acc = _mean(vals("citation_accuracy", answerable))
    cit_cov = _mean(vals("citation_coverage", answerable))
    cit_gold = _mean(vals("citation_gold", answerable))
    coverage = _mean(vals("key_point_coverage", answerable))
    fmt = _mean([1.0 if j["format_ok"] else 0.0 for j in judged])
    term = _mean([1.0 if j["term_ok"] else 0.0 for j in judged if j["term_ok"] is not None])
    refusal = _mean([1.0 if r["judge"]["refusal_ok"] else 0.0
                     for r in unanswerable if r["judge"]["refusal_ok"] is not None])
    false_refusal = _mean([1.0 if r["judge"]["false_refusal"] else 0.0 for r in answerable])
    answer_recall = _mean([0.0 if r["judge"]["false_refusal"] else 1.0 for r in answerable])

    citation_score = ((cit_acc or 0.0) * 0.5 + (cit_cov or 0.0) * 0.5)
    refusal_score = (refusal or 0.0) * 0.7 + (1 - (false_refusal or 0.0)) * 0.3
    composite = (
        0.30 * (faithfulness or 0.0)
        + 0.20 * citation_score
        + 0.20 * refusal_score
        + 0.20 * (coverage or 0.0)
        + 0.10 * (fmt or 0.0)
    )
    return {
        "n": len(records),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "faithfulness": faithfulness,
        "hallucination_rate": halluc,
        "citation_accuracy": cit_acc,
        "citation_coverage": cit_cov,
        "citation_gold": cit_gold,
        "citation_score": round(citation_score, 4),
        "key_point_coverage": coverage,
        "format_compliance": fmt,
        "term_accuracy": term,
        "refusal_accuracy": refusal,
        "false_refusal_rate": false_refusal,
        "answer_recall": answer_recall,
        "refusal_score": round(refusal_score, 4),
        "composite": round(composite, 4),
        "refusal_accuracy_ci": wilson_interval(
            sum(1 for r in unanswerable if r["judge"].get("refusal_ok")), len(unanswerable)),
    }


def per_capability(records: list[dict]) -> dict:
    groups: dict[str, list[dict]] = {}
    for r in records:
        cap = r.get("meta", {}).get("capability") or r.get("format")
        groups.setdefault(cap, []).append(r)
    return {cap: aggregate(rows) for cap, rows in sorted(groups.items())}
