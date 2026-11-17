# -*- coding: utf-8 -*-
"""答案组装器：把「证据句」按 4 类格式组装成带引用的标准答案。

训练数据的 gold answer 与推理时的模型输出**共用同一个组装器**，区别只在于
证据句来源：
  * gold：人工主题锚点命中的句子（保证可溯源）；
  * 推理：证据选择器打分后挑出的句子。
这样评测出来的格式合规率、引用覆盖率反映的是“策略模型选证据 + 组装”的
端到端能力，而不是评测脚本自己拼答案。
"""
from __future__ import annotations

import re

from kb.textutil import clean_sentence, extract_numbers, split_sentences, truncate

_CAUSE_HINTS = ("原因", "导致", "过期", "损坏", "不足", "错误", "未更新", "未送达", "故障")


def _period(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    if s[-1] not in "。！？；.!?;":
        s += "。"
    return s


def _cite(s: str, n: int) -> str:
    s = _period(s)
    return f"{s[:-1]} [{n}]{s[-1]}"


def _ctx_index(evidence: list[dict]) -> int:
    return int(evidence.get("ctx_n") or 1)


def sentences_from_contexts(contexts: list[dict], min_len: int = 8) -> list[dict]:
    out: list[dict] = []
    for c in contexts:
        n = c.get("n", 1)
        text = c.get("text") or c.get("content") or ""
        for s in split_sentences(text):
            s = clean_sentence(s)
            if len(s) < min_len:
                continue
            if re.match(r"^[|#>`]", s):
                continue
            out.append({"text": s, "ctx_n": n, "chunk_id": c.get("chunk_id", ""),
                        "title": c.get("title", ""), "heading_path": c.get("heading_path", "")})
    return out


def _split_row(sentence: str) -> tuple[str, str]:
    """产品说明表格：把“项目：说明”拆成两列。"""
    s = sentence.rstrip("。")
    for sep in ("：", ":"):
        if sep in s and len(s.split(sep, 1)[0]) <= 24:
            a, b = s.split(sep, 1)
            return a.strip(), b.strip()
    for sep, bad_prev in (("为", "分认作成以行"), ("是", "但就总还"), ("按", ""), ("由", "")):
        for m in re.finditer(sep, s):
            idx = m.start()
            if not (1 < idx <= 28):
                continue
            if bad_prev and s[idx - 1] in bad_prev:
                continue
            return s[:idx].strip(), s[idx + 1:].strip()
    return "", s


def _product_items(text: str) -> list[str]:
    """把一条证据拆成适合表格的并列项（键值行 / 顿号清单 / 逗号并列）。"""
    t = text.rstrip("。")
    # “公司全称：… 统一社会信用代码：…” 这类空格分隔的键值行
    if len(re.findall(r"[\u4e00-\u9fff]{2,10}[：:]", t)) >= 3:
        parts = re.split(r"\s+(?=[\u4e00-\u9fff]{2,10}[：:])", t)
        if len(parts) >= 2:
            return [p.strip() for p in parts if len(p.strip()) >= 4]
    # “需提交：A、B、C、D” 这类顿号清单
    if "：" in t:
        head, rest = t.split("：", 1)
        if rest.count("、") >= 3:
            items = [x.strip() for x in rest.split("、") if len(x.strip()) >= 4]
            if len(items) >= 2:
                return [f"{head}：{x}" for x in items]
    parts = [p.strip() for p in re.split(r"[，；;]", t) if len(p.strip()) >= 6]
    return parts if len(parts) >= 2 else [t]


def _product_table(evidence: list[dict], max_rows: int = 6) -> str:
    rows = []
    seen = set()
    for e in evidence:
        text = e["text"]
        if text in seen:
            continue
        seen.add(text)
        for part in _product_items(text):
            head, body = _split_row(part)
            if not body:
                body = part
            if not head:
                head = f"要点 {len(rows) + 1}"
            rows.append((head, body, _ctx_index(e)))
            if len(rows) >= max_rows:
                break
        if len(rows) >= max_rows:
            break
    if len(rows) < 2:
        return ""
    lines = ["| 项目 | 说明 | 依据 |", "| --- | --- | --- |"]
    for head, body, n in rows:
        lines.append(f"| {head} | {_period(body)} | [{n}] |")
    return "\n".join(lines)


def _phenomenon(question: str) -> str:
    q = re.sub(r"[？?。！!]+$", "", question.strip())
    q = re.sub(r"^(请问|想问下|我想问|问一下)[，,：:\s]*", "", q)
    q = re.sub(r"(能|可以|可不可以|能否|需要)[^，。？]{0,20}吗$", "", q)
    q = re.sub(r"(怎么办|如何处理|怎么处理|怎么排查|怎么解决|怎么操作|是什么|是多少)$", "", q).strip()
    return q or "用户反馈异常"


def compose_answer(question: str, contexts: list[dict], fmt: str,
                   evidence: list[dict], max_bullets: int = 4) -> str:
    """按格式组装答案；evidence 需包含 text / ctx_n 两个字段。"""
    evidence = [e for e in evidence if e.get("text")]
    if not evidence:
        return refusal_answer("none", question, "可用证据片段", "相关制度")

    # 去重并保持顺序
    uniq: list[dict] = []
    seen = set()
    for e in evidence:
        key = e["text"][:40]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(e)

    # 单句证据：按逗号拆成“结论 + 分点”，避免为了凑格式补无关内容
    if fmt == "policy" and len(uniq) == 1:
        parts = [p.strip() for p in re.split(r"[，；;]", uniq[0]["text"]) if len(p.strip()) >= 6]
        if len(parts) >= 2:
            uniq = [{"text": p, "ctx_n": uniq[0]["ctx_n"]} for p in parts[:5]]

    if fmt == "product":
        table = _product_table(uniq)
        if table:
            return table
        # 拆分后仍不足两行时退化为制度解释，保证“有引用、不编造”优先
        fmt = "policy"

    if fmt == "steps":
        items: list[tuple[str, int]] = []
        for e in uniq:
            parts = [p.strip() for p in re.split(r"[，；;]", e["text"]) if len(p.strip()) >= 6]
            for p in (parts if len(parts) >= 2 else [e["text"]]):
                items.append((p, _ctx_index(e)))
        if len(items) < 2:
            items = [(e["text"], _ctx_index(e)) for e in uniq]
        lines = ["按以下步骤办理：", ""]
        for i, (text, n) in enumerate(items[:max_bullets + 3], start=1):
            lines.append(f"{i}. {_cite(truncate(text, 160), n)}")
        return "\n".join(lines)

    if fmt == "troubleshoot":
        phenomena = _phenomenon(question)
        cause_sents = [e for e in uniq if any(h in e["text"] for h in _CAUSE_HINTS)]
        cause = cause_sents[0]["text"] if cause_sents else "现有资料未明确说明具体原因。"
        cause_n = _ctx_index(cause_sents[0]) if cause_sents else None
        lines = [f"【现象】{_period(phenomena)}"]
        if cause_n:
            lines.append(f"【可能原因】{_cite(cause, cause_n)}")
        else:
            lines.append("【可能原因】现有资料未明确说明具体原因。")
        lines.append("【处理步骤】")
        for i, e in enumerate(uniq[:max_bullets + 1], start=1):
            lines.append(f"{i}. {_cite(truncate(e['text'], 160), _ctx_index(e))}")
        return "\n".join(lines)

    # policy / integration
    lead = uniq[0]
    lead_text = truncate(lead["text"], 220)
    lines = [f"【结论】{_cite(lead_text, _ctx_index(lead))}", ""]
    bullets = uniq[1:max_bullets + 1]
    if bullets:
        lines.append("【依据】" if fmt == "policy" else "【要点】")
        for i, e in enumerate(bullets, start=1):
            lines.append(f"{i}. {_cite(truncate(e['text'], 180), _ctx_index(e))}")
    return "\n".join(lines)


def refusal_answer(kind: str, question: str, missing: str, suggest: str,
                   extra: str = "") -> str:
    if kind == "weak":
        head = "检索到的资料与问题关联度较低，无法给出可靠答案。"
    elif kind == "conflict":
        head = "检索到的资料存在口径冲突，不能直接给出唯一答案。"
    else:
        head = "现有知识库资料无法回答该问题。"
    if suggest:
        advice = f"请补充《{suggest}》类文档，或联系知识管理员确认后再提问。"
    else:
        advice = "请补充相关制度、操作手册或业务资料后重试，或联系知识管理员确认。"
    lines = [
        f"【结论】{head}",
        f"【缺失信息】{missing or '相关资料'}" + (f"；{extra}" if extra else ""),
        f"【建议】{advice}",
    ]
    return "\n".join(lines)


def conflict_answer(question: str, current: dict, old: dict) -> str:
    """current/old: {text, ctx_n, title, value}"""
    lines = [
        "【结论】检索到的资料存在版本或口径冲突，不能直接给出唯一答案。",
        "【冲突信息】",
        f"1. 《{current['title']}》现行规定：{_period(current['text'])} [{current['ctx_n']}]",
        f"2. 《{old['title']}》规定：{_period(old['text'])} [{old['ctx_n']}]",
        "【建议】请确认适用版本或适用范围；如无特别说明，以现行有效制度为准。",
    ]
    return "\n".join(lines)


def partial_answer(question: str, contexts: list[dict], evidence: list[dict],
                   missing: str, suggest: str) -> str:
    base = compose_answer(question, contexts, "policy", evidence)
    return (base + "\n\n"
            f"【缺失信息】关于「{missing}」，现有知识库资料未覆盖。\n"
            f"【建议】请补充《{suggest}》类文档，或联系知识管理员确认。")


def citation_indices(answer: str) -> list[int]:
    return [int(x) for x in re.findall(r"\[(\d+)\]", answer or "")]


def answer_numbers(answer: str) -> list[str]:
    return [x["canonical"] for x in extract_numbers(answer, require_unit=True)]
