# -*- coding: utf-8 -*-
"""统一 Prompt：训练与推理必须完全一致（换模型/换链路时最容易踩的坑）。

* `SYSTEM_PROMPT`：约束「只依据资料、必须引用、不足要拒答」；
* `build_messages`：把检索片段渲染成 `[n]` 编号参考资料；
* `parse_reference_blocks`：反向解析任意 RAG 前端（含 2.9 知识库平台）拼好的 user 消息，
  使本项目既能独立评测，又能以 OpenAI 兼容服务的方式零改造接入 2.9。
"""
from __future__ import annotations

import re

SYSTEM_PROMPT = (
    "你是星海数字科技企业知识库助手。回答必须严格依据【参考资料】，"
    "禁止使用资料之外的事实、数字、政策或结论。\n\n"
    "输出规则：\n"
    "1. 先给结论，再给依据；每个结论句、要点句、表格行的末尾用 [n] 标注来源，n 与参考资料编号一致。\n"
    "2. 按问题类型选择格式：\n"
    "   - 制度解释：以“【结论】”开头，随后用“【依据】”列 2~4 条分点；\n"
    "   - 产品说明：输出 Markdown 表格，列为「项目 | 说明 | 依据」，依据列写 [n]；\n"
    "   - 操作步骤：输出编号步骤（1. 2. 3.），每步末尾标注 [n]；\n"
    "   - 故障处理：依次输出“【现象】”“【可能原因】”“【处理步骤】”，步骤带编号并标注 [n]；\n"
    "   - 多片段整合：按子问题分主题列要点，每条要点标注对应来源 [n]。\n"
    "3. 使用知识库标准术语（如“差旅住宿费限额”“带薪年休假”），不使用口语化别称。\n"
    "4. 资料不足、弱相关、互相矛盾或只能部分回答时，禁止编造，按以下方式处理：\n"
    "   - 资料不足：输出“【结论】现有知识库资料无法回答该问题。”+“【缺失信息】…”+“【建议】…”；\n"
    "   - 弱相关：说明“检索到的资料与问题关联度较低”，不要给出确定性结论；\n"
    "   - 冲突：列出相互矛盾的两种口径及来源，请用户确认适用版本；\n"
    "   - 部分可答：先回答有依据的部分，再用“【缺失信息】”说明缺少什么。\n"
    "5. 使用简体中文，不重复问题，不输出与问题无关的内容。"
)

USER_TEMPLATE = "【参考资料】\n{contexts}\n\n【问题】{question}"


def render_context(c: dict) -> str:
    n = c.get("n", 0)
    title = c.get("title") or "未命名文档"
    page = c.get("page_no") or 1
    heading = c.get("heading_path") or ""
    text = (c.get("text") or "").strip()
    head = f"《{title}》 第{page}页"
    if heading:
        head += f" · {heading}"
    return f"[{n}] {head}\n{text}"


def render_contexts(contexts: list[dict], max_chars: int = 700) -> str:
    blocks = []
    for i, c in enumerate(contexts, start=1):
        cc = dict(c)
        cc["n"] = i
        text = (cc.get("text") or "").strip()
        if len(text) > max_chars:
            text = text[: max_chars - 1] + "…"
        cc["text"] = text
        blocks.append(render_context(cc))
    return "\n\n".join(blocks) if blocks else "（无可用参考资料）"


def build_user_message(question: str, contexts: list[dict], max_chars: int = 700) -> str:
    return USER_TEMPLATE.format(contexts=render_contexts(contexts, max_chars), question=question.strip())


def build_messages(question: str, contexts: list[dict]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(question, contexts)},
    ]


_HEADER_RE = re.compile(r"^\[(\d+)\]\s*《([^》]+)》\s*(.*)$")


def parse_reference_blocks(user_text: str) -> tuple[str, list[dict]]:
    """从 RAG 前端拼好的 user 消息中解析出 (question, contexts)。

    兼容两种格式：
      * 本项目：`[1] 《标题》 第1页 · 第三章 住宿标准`；
      * 2.9 平台：`[1] 《标题》 P1 第1段 《第三章 住宿标准》`。
    """
    text = user_text or ""
    question = ""
    m = re.search(r"【(?:用户)?问题】\s*(.+?)\s*$", text, re.S)
    if m:
        question = m.group(1).strip()
        text = text[: m.start()]
    elif "【问题】" in text:
        question = text.split("【问题】", 1)[1].strip()
        text = text.split("【问题】", 1)[0]

    contexts: list[dict] = []
    current: dict | None = None
    body: list[str] = []

    def flush() -> None:
        nonlocal current, body
        if current is not None:
            current["text"] = "\n".join(x for x in body if x.strip()).strip()
            contexts.append(current)
        current, body = None, []

    for raw in text.splitlines():
        line = raw.rstrip()
        if line.startswith("【参考资料】"):
            continue
        m2 = _HEADER_RE.match(line.strip())
        if m2:
            flush()
            n = int(m2.group(1))
            title = m2.group(2)
            rest = m2.group(3).strip()
            page = 1
            mp = re.search(r"(?:P|第)\s*(\d+)\s*(?:页)?", rest)
            if mp:
                page = int(mp.group(1))
            heading = ""
            mh = re.search(r"《([^》]+)》", rest)
            if mh:
                heading = mh.group(1)
            elif "·" in rest:
                heading = rest.split("·", 1)[1].strip()
            current = {
                "n": n, "title": title, "page_no": page,
                "heading_path": heading, "chunk_id": "", "text": "",
            }
            # 兼容“标题行 + 正文同一行”的写法：把页/段/标题标记去掉后剩余正文
            leftover = re.sub(r"(?:P|第)\s*\d+\s*(?:页|段)?", "", rest)
            leftover = re.sub(r"第\s*\d+\s*段", "", leftover)
            leftover = re.sub(r"《[^》]*》", "", leftover).strip(" ·—-")
            if len(leftover) >= 8:
                body.append(leftover)
            continue
        if current is not None:
            body.append(line)
    flush()
    if not question:
        question = text.strip().splitlines()[-1] if text.strip() else ""
    contexts = [c for c in contexts if c.get("text")]
    contexts.sort(key=lambda c: c.get("n", 0))
    return question, contexts
