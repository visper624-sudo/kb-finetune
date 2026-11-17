# -*- coding: utf-8 -*-
"""术语表：标准术语 ↔ 口语别称。

用途有三处：
1. 合成数据时把一部分问句改写成“口语/简称”问法，训练模型学会对齐标准术语（4.1.1）；
2. 校验 gold answer 是否使用标准术语（术语准确率）；
3. 统计答案中是否残留口语别称（术语一致性）。
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_TERMS_PATH = Path(__file__).resolve().parent.parent / "data" / "specs" / "terms.json"


@lru_cache(maxsize=1)
def load_terms() -> list[dict]:
    if not _TERMS_PATH.exists():
        return []
    return json.loads(_TERMS_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _index() -> tuple[dict[str, dict], list[str]]:
    by_standard: dict[str, dict] = {}
    for t in load_terms():
        # 别称若本身是标准术语的子串（如“抬头”之于“发票抬头”），无法用于“别称残留”判断
        t["aliases"] = [a for a in t.get("aliases", []) if a and a not in t["standard"]]
        by_standard[t["standard"]] = t
    return by_standard, list(by_standard)


def standards() -> list[str]:
    return _index()[1]


def get(standard: str) -> dict | None:
    return _index()[0].get(standard)


def aliases(standard: str) -> list[str]:
    t = get(standard)
    return list(t.get("aliases", [])) if t else []


def find_alias(text: str) -> list[str]:
    """返回文本中出现的、且未被标准术语包含的口语别称。"""
    hits = []
    for t in load_terms():
        for a in t.get("aliases", []):
            if a in text and a not in t["standard"]:
                # 标准术语同时出现时，别称可能只是子串巧合，仍记录由人工抽检
                hits.append(a)
    return sorted(set(hits))


def used_standard(text: str) -> list[str]:
    return [s for s in standards() if s in text]


def term_violation(text: str, required: str | None) -> list[str]:
    """校验：必须出现标准术语。

    说明：gold answer 直接摘录制度原文，个别相邻句可能同时出现口语别称
    （如“未命中问题”），因此标准术语出现即视为通过；别称残留由评测侧的
    “术语一致性”指标单独统计，不影响训练样本入库。
    """
    if required and required not in text:
        return [f"缺少标准术语「{required}」"]
    return []


def rewrite_with_alias(question: str, standard: str, alias: str) -> str:
    return question.replace(standard, alias)
