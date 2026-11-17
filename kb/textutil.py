# -*- coding: utf-8 -*-
"""零依赖中文文本工具：分词、分句、数字事实抽取、相似度、问题拆分。

训练数据合成与评测指标都依赖这里的“确定性口径”，因此实现尽量保守：
宁可少抽一个事实，也不要把标题、章节号、P1/T-14 之类的编号误当成业务数字。
"""
from __future__ import annotations

import math
import re

CJK = re.compile(r"[\u4e00-\u9fff]")
_WS = re.compile(r"[\s\u3000]+")
_PUNCT = re.compile(r"[，。！？；：、“”‘’（）《》【】…—·,.!?;:\"'()\[\]{}<>/\\|@#$%^&*_+=~`]+")
_HEADING = re.compile(
    r"^(第[一二三四五六七八九十百零\d]+[章节部分篇]|[一二三四五六七八九十]+、|\d+[.、]\s*[^\d\s]|#+\s|【[^】]{2,12}】$)"
)
_LIST_PREFIX = re.compile(r"^\s*(?:[-*•·]|(?:\d+[.、)]))\s*")
_QUESTION_PREFIX = re.compile(r"^问\s*[:：]\s*")
_ANSWER_PREFIX = re.compile(r"^答\s*[:：]\s*")

#: 数字 + 单位（按长度降序匹配）
_UNIT = (
    "个工作日|工作日|个自然日|自然日|个百分点|万元|亿元|个月|小时|分钟|"
    "天|日|年|次|人|家|项|张|份|米|条|倍|分|岁|周|月|元|%|万"
)
_NUM_RE = re.compile(
    r"(?<![\dA-Za-z#:./-])(\d+(?:\.\d+)?)\s*(" + _UNIT + r")?"
    r"(?:\s*/\s*(晚|天|人|次))?(?![\dA-Za-z])"
)

_UNIT_CANON = {
    "个工作日": "工作日",
    "个自然日": "自然日",
    "个百分点": "个百分点",
    "万元": "万元",
    "亿元": "亿元",
    "个月": "个月",
    "小时": "小时",
    "分钟": "分钟",
    "天": "天",
    "日": "日",
    "年": "年",
    "次": "次",
    "人": "人",
    "家": "家",
    "项": "项",
    "张": "张",
    "份": "份",
    "米": "米",
    "条": "条",
    "倍": "倍",
    "分": "分",
    "岁": "岁",
    "周": "周",
    "月": "月",
    "元": "元",
    "%": "%",
    "万": "万",
}

#: 参与“关键要点覆盖率”计算的单位（排除纯序号/评分值）
FACT_UNITS = {
    "元", "万元", "亿元", "%", "个百分点", "天", "工作日", "自然日", "小时",
    "分钟", "个月", "年", "次", "人", "家", "项", "张", "份", "米", "条",
    "倍", "岁", "周", "日", "月", "万",
}

STOP_TOKENS = {
    "的", "了", "是", "在", "和", "与", "及", "或", "为", "按", "须", "应",
    "可", "不", "有", "无", "等", "其", "该", "本", "并", "对", "由", "将",
    "以", "于", "中", "内", "后", "前", "时", "如", "若", "则", "也", "都",
}
#: 含这些字的二元组大多是跨词噪声（如“和餐”“的住”），计算覆盖率时剔除
STOP_CHARS = set("的了是在和与及或为按须应可不无等其该本并对由将以于中内后前时如若则也都将")


def normalize_text(text: str) -> str:
    """全角数字/空格归一，便于数字匹配。"""
    if not text:
        return ""
    table = str.maketrans(
        "０１２３４５６７８９．％ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ",
        "0123456789.%ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    text = text.translate(table)
    return _WS.sub(" ", text).strip()


def normalize_question(q: str) -> str:
    q = normalize_text(q or "").lower()
    q = q.replace("？", "").replace("?", "").replace("！", "").replace("!", "")
    return _PUNCT.sub("", q).replace(" ", "")


def tokenize(text: str) -> list[str]:
    """中文二元组 + 拉丁词/数字（与 2.9 检索口径一致）。"""
    text = normalize_text(text).lower()
    tokens: list[str] = []
    for run in re.findall(r"[\u4e00-\u9fff]+|[a-z0-9][a-z0-9._%-]*", text):
        if CJK.match(run[0]):
            if len(run) == 1:
                tokens.append(run)
            else:
                tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
                if len(run) <= 6:
                    tokens.append(run)
        else:
            tokens.append(run)
    return tokens


def content_terms(text: str) -> set[str]:
    out: set[str] = set()
    for t in tokenize(text):
        if t in STOP_TOKENS or t.isdigit():
            continue
        if len(t) == 2 and any(ch in STOP_CHARS for ch in t):
            continue
        out.add(t)
    return out


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = len(CJK.findall(text))
    words = len(re.findall(r"[A-Za-z0-9]+", text))
    return cjk + words + max(0, len(text) - cjk) // 8


def split_sentences(text: str) -> list[str]:
    """按中文句末标点/换行分句，保留标点；过滤标题行。"""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"(?<=[。！？；!?;])\s*|\n+", text)
    out: list[str] = []
    for p in parts:
        s = p.strip()
        if not s:
            continue
        if is_heading(s):
            continue
        out.append(s)
    return out


def is_heading(line: str) -> bool:
    s = (line or "").strip()
    if not s or len(s) > 40:
        return False
    if re.match(r"^第[一二三四五六七八九十百零\d]+[章节部分篇]", s):
        return True
    if re.match(r"^[一二三四五六七八九十]+、", s):
        return True
    if re.match(r"^#+\s", s):
        return True
    if re.match(r"^《[^》]{1,40}》$", s):
        return True
    return False


def clean_sentence(s: str) -> str:
    s = (s or "").strip()
    s = _LIST_PREFIX.sub("", s)
    s = _QUESTION_PREFIX.sub("", s)
    s = _ANSWER_PREFIX.sub("", s)
    s = re.sub(r"^《[^》]*》\s*", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    if s.endswith(("；", ";")):
        s = s[:-1] + "。"
    return s


def split_question_parts(question: str) -> list[str]:
    """拆分复合问题（用于「部分可答」判定与多片段样本）。"""
    q = (question or "").strip()
    q = re.sub(r"^(?:请分别说明|请一起说明|请分别|请对比说明|请说明|请列出)[：:\s]*", "", q)
    raw = re.split(r"[？?；;]|(?:以及|还有|另外|同时|并且|顺便)", q)
    parts = [clean_sentence(p).rstrip("。") for p in raw if p and len(clean_sentence(p)) >= 4]
    return parts if len(parts) >= 2 else [q]


def extract_numbers(text: str, require_unit: bool = False) -> list[dict]:
    """抽取业务数字事实：数值 + 单位 + 规范化文本（用于忠实性/覆盖率校验）。"""
    text = normalize_text(text)
    out: list[dict] = []
    seen = set()
    for m in _NUM_RE.finditer(text):
        value = m.group(1)
        unit = m.group(2) or ""
        suffix = m.group(3) or ""
        if not unit and not suffix:
            if require_unit:
                continue
        unit_canon = _UNIT_CANON.get(unit, unit)
        if suffix:
            unit_canon = f"{unit_canon}/{suffix}" if unit_canon else f"/{suffix}"
        if require_unit and unit_canon.split("/")[0] not in FACT_UNITS:
            continue
        canon = f"{value}{unit_canon}"
        if canon in seen:
            continue
        seen.add(canon)
        out.append({"raw": m.group(0).strip(), "value": value,
                    "unit": unit_canon, "canonical": canon})
    return out


def fact_numbers(text: str) -> list[str]:
    return [x["canonical"] for x in extract_numbers(text, require_unit=True)]


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def overlap_ratio(needle: set[str], haystack: set[str]) -> float:
    if not needle:
        return 0.0
    return len(needle & haystack) / len(needle)


def sentence_similarity(question: str, sentence: str) -> float:
    """问题-句子相关度：内容二元组覆盖 + 数字命中 + 长度惩罚。"""
    q = content_terms(question)
    s = content_terms(sentence)
    if not q or not s:
        return 0.0
    cover = overlap_ratio(q, s)
    jac = jaccard(q, s)
    q_nums = set(fact_numbers(question))
    s_nums = set(fact_numbers(sentence))
    num_bonus = 0.15 if q_nums and q_nums & s_nums else 0.0
    return round(0.65 * cover + 0.35 * jac + num_bonus, 4)


def md_escape(text: str) -> str:
    return (text or "").replace("|", "\\|").replace("\n", " ").strip()


def truncate(text: str, n: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """二项比例置信区间（报告里给指标加区间，避免“100%”没有样本量语境）。"""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4))
