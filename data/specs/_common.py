# -*- coding: utf-8 -*-
"""领域主题规格：人工挑选的“高频真实问题 → 证据句”映射。

每条主题（topic）包含：
  * doc      : 2.9 知识库文档 slug；
  * label    : 主题名（用于生成同义问法与格式标题）；
  * format   : policy / product / steps / troubleshoot；
  * questions: 2~4 个真实口吻的问题（后续还会自动扩展同义问法）；
  * anchors  : 证据句锚点（必须能在 2.9 切片句子中精确命中，构建时会校验）；
  * term     : 需要使用的标准术语（可选）。

设计原则：问题必须能由锚点句**直接回答**，不依赖计算/推理；
需要比较、计算的问题放到多片段整合或负样本中单独构造。
"""
from __future__ import annotations


def T(tid: str, doc: str, label: str, fmt: str, questions: list[str],
      anchors: list[str], term: str | None = None) -> dict:
    return {
        "id": tid, "doc": doc, "label": label, "format": fmt,
        "questions": questions, "anchors": anchors, "term": term,
    }
