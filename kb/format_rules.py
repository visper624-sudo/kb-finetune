# -*- coding: utf-8 -*-
"""4 类回答格式的机器校验（训练数据质检与评测共用同一口径）。"""
from __future__ import annotations

import re

from kb.textutil import fact_numbers

_NUM_LINE = re.compile(r"^\s*(?:\d+[.、)]|[一二三四五六七八九十]+、)\s*\S+", re.M)
_CITE = re.compile(r"\[\d+\]")


def check_format(answer: str, fmt: str) -> tuple[bool, list[str]]:
    text = answer or ""
    problems: list[str] = []
    has_cite = bool(_CITE.search(text))

    if fmt == "policy":
        if "【结论】" not in text:
            problems.append("缺少【结论】")
        if "【依据】" not in text and "【要点】" not in text:
            problems.append("缺少【依据】/【要点】")
        if len(_NUM_LINE.findall(text)) < 1:
            problems.append("缺少分点依据")
        if not has_cite:
            problems.append("缺少引用编号")
    elif fmt == "product":
        if not re.search(r"\|\s*项目\s*\|", text):
            problems.append("缺少表头“项目”")
        rows = [ln for ln in text.splitlines()
                if ln.strip().startswith("|") and "---" not in ln
                and not re.search(r"\|\s*项目\s*\|", ln)]
        if len(rows) < 2:
            problems.append("表格数据行少于 2 行")
        if not has_cite:
            problems.append("缺少引用编号")
    elif fmt == "steps":
        if len(_NUM_LINE.findall(text)) < 2:
            problems.append("编号步骤少于 2 步")
        if not has_cite:
            problems.append("缺少引用编号")
    elif fmt == "troubleshoot":
        if "【现象】" not in text:
            problems.append("缺少【现象】")
        if "【处理步骤】" not in text and "【处理】" not in text:
            problems.append("缺少【处理步骤】")
        if len(_NUM_LINE.findall(text)) < 1:
            problems.append("缺少编号处理步骤")
        if not has_cite:
            problems.append("缺少引用编号")
    elif fmt == "integration":
        if "【结论】" not in text:
            problems.append("缺少【结论】")
        if "【要点】" not in text and "【依据】" not in text:
            problems.append("缺少【要点】")
        if len(_NUM_LINE.findall(text)) < 2:
            problems.append("整合要点少于 2 条")
        if not has_cite:
            problems.append("缺少引用编号")
    elif fmt in ("refusal", "refusal_none", "refusal_weak", "refusal_conflict", "partial"):
        if "【结论】" not in text:
            problems.append("缺少【结论】")
        if "【缺失信息】" not in text and "【冲突信息】" not in text:
            problems.append("缺少【缺失信息】/【冲突信息】")
    else:
        problems.append(f"未知格式 {fmt}")
    return (not problems), problems


def numbers_supported(answer: str, contexts: list[dict],
                      only_cited: bool = False) -> tuple[bool, list[str]]:
    """逐引用句校验：句子中的数字必须能在被引用片段中找到。

    only_cited=True 时只检查带 [n] 的句子（用于训练数据质检：
    【现象】等复述问题的行不参与溯源校验）。
    """
    by_n = {int(c.get("n", i + 1)): (c.get("text") or c.get("content") or "")
            for i, c in enumerate(contexts)}
    problems: list[str] = []
    current_ctx = 1
    for raw in (answer or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        cites = [int(x) for x in re.findall(r"\[(\d+)\]", line)]
        if only_cited and not cites:
            continue
        ctx_n = cites[-1] if cites else current_ctx
        if cites:
            current_ctx = cites[-1]
        nums = fact_numbers(re.sub(r"\[\d+\]", "", line))
        if not nums:
            continue
        ctx_text = by_n.get(ctx_n, "")
        ctx_nums = set(fact_numbers(ctx_text))
        for n in nums:
            if n not in ctx_nums and n.rstrip("/天") not in ctx_nums:
                problems.append(f"数字 {n} 不在引用片段 [{ctx_n}] 中")
    return (not problems), problems
