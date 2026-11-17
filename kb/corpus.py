# -*- coding: utf-8 -*-
"""语料装载：文档 / 切片 / 句子三级视图。

数据来自 2.9 知识库平台（见 tools/extract_corpus.py）：
  * 一篇文档被切成若干 480 token 切片，切片带 page/para/char/heading 元数据；
  * 句子视图用于事实抽取、证据选择与忠实性判断。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from config import paths
from kb.textutil import clean_sentence, is_heading, normalize_text, split_sentences


@dataclass
class Sentence:
    text: str
    chunk_id: str
    doc_id: str
    title: str
    category: str
    page_no: int
    heading_path: str
    idx: int
    terms: set[str] = field(default_factory=set)

    @property
    def key(self) -> str:
        return f"{self.chunk_id}:{self.idx}"


class Corpus:
    def __init__(self, chunks: list[dict], docs: list[dict]):
        self.chunks = chunks
        self.docs = {d["doc_id"]: d for d in docs}
        self.by_id = {c["chunk_id"]: c for c in chunks}
        self._sentences: dict[str, list[Sentence]] = {}

    # ---------------------------------------------------------------- 装载
    @classmethod
    def load(cls, corpus_dir: str | Path | None = None) -> "Corpus":
        base = Path(corpus_dir) if corpus_dir else paths.CORPUS_DIR
        chunks = paths.read_jsonl(base / "chunks.jsonl")
        docs = paths.read_jsonl(base / "docs.jsonl")
        if not chunks:
            raise SystemExit(
                f"未找到语料 {base / 'chunks.jsonl'}，请先运行 python tools/extract_corpus.py"
            )
        return cls(chunks, docs)

    # ---------------------------------------------------------------- 切片
    def doc_chunks(self, doc_id: str) -> list[dict]:
        return [c for c in self.chunks if c["doc_id"] == doc_id]

    def chunks_for_docs(self, doc_ids: set[str] | list[str]) -> list[dict]:
        s = set(doc_ids)
        return [c for c in self.chunks if c["doc_id"] in s]

    # ---------------------------------------------------------------- 句子
    def doc_sentences(self, doc_id: str) -> list[Sentence]:
        if doc_id in self._sentences:
            return self._sentences[doc_id]
        out: list[Sentence] = []
        for ch in self.doc_chunks(doc_id):
            raw_lines = (ch.get("content") or "").splitlines()
            body = []
            for line in raw_lines:
                s = line.strip()
                if not s:
                    continue
                if s == (ch.get("title") or "").strip():
                    continue
                if re.match(r"^《[^》]{0,40}》$", s):      # chunker 加的标题前缀
                    continue
                if is_heading(s):
                    continue
                body.append(s)
            text = "\n".join(body)
            # Markdown 列表常写成“- 事假：… - 病假：…”，拆成独立句子才能逐条引用
            text = re.sub(r"\s+-\s+", "\n", text)
            for i, sent in enumerate(split_sentences(text)):
                s = clean_sentence(sent)
                if len(s) < 6:
                    continue
                out.append(Sentence(
                    text=s, chunk_id=ch["chunk_id"], doc_id=doc_id,
                    title=ch["title"], category=ch["category"],
                    page_no=ch.get("page_no", 1),
                    heading_path=ch.get("heading_path", ""),
                    idx=i,
                ))
        self._sentences[doc_id] = out
        return out

    def all_sentences(self) -> list[Sentence]:
        out: list[Sentence] = []
        for d in self.docs:
            out.extend(self.doc_sentences(d))
        return out

    # ---------------------------------------------------------------- 查找
    @staticmethod
    def _flat(text: str) -> str:
        return re.sub(r"\s+", "", normalize_text(text))

    def find_sentence(self, anchor: str, doc_id: str | None = None) -> Sentence | None:
        """按锚点子串找到包含该事实的句子（空白不敏感）。"""
        target = self._flat(anchor)
        if not target:
            return None
        docs = [doc_id] if doc_id else list(self.docs)
        for d in docs:
            for s in self.doc_sentences(d):
                if target in self._flat(s.text):
                    return s
        # 句子被换行/标点切断时，退化为在切片正文中找
        for d in docs:
            for ch in self.doc_chunks(d):
                if target in self._flat(ch["content"]):
                    for s in self.doc_sentences(d):
                        if s.chunk_id == ch["chunk_id"] and self._flat(s.text)[:12] in self._flat(ch["content"]):
                            return s
        return None

    def find_chunk(self, anchor: str, doc_id: str | None = None) -> dict | None:
        s = self.find_sentence(anchor, doc_id)
        return self.by_id.get(s.chunk_id) if s else None

    # ---------------------------------------------------------------- 统计
    def stats(self) -> dict:
        cats: dict[str, int] = {}
        for d in self.docs.values():
            cats[d["category"]] = cats.get(d["category"], 0) + 1
        return {
            "docs": len(self.docs),
            "chunks": len(self.chunks),
            "sentences": len(self.all_sentences()),
            "chars": sum(d.get("char_count", 0) for d in self.docs.values()),
            "by_category": cats,
        }


def as_context(chunk: dict, n: int) -> dict:
    return {
        "n": n,
        "chunk_id": chunk["chunk_id"],
        "doc_id": chunk["doc_id"],
        "title": chunk["title"],
        "category": chunk["category"],
        "page_no": chunk.get("page_no", 1),
        "heading_path": chunk.get("heading_path", ""),
        "text": chunk.get("content", ""),
    }
