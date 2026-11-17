# -*- coding: utf-8 -*-
"""从 2.9 知识库管理平台抽取语料（文档 → 解析 → 切片 → chunks.jsonl）。

本项目的数据来源是 2.9 知识库平台的 `seed/docs` 示例知识库：
  * 直接复用 2.9 的 `app.parsers`（PDF / DOCX / Markdown / TXT）与 `app.chunker`
    （480 token + 15% 重叠 + page/para/char 区间），保证训练语料与平台检索片段同源；
  * 若本机没有 2.9 工程，本脚本会失败并提示：仓库内已附带生成好的
    `data/corpus/chunks.jsonl`，无需重复抽取即可复现数据合成。

用法：
    python tools/extract_corpus.py                     # 自动查找同级目录「知识库管理平台」
    python tools/extract_corpus.py --kb-root D:/lesson/实战/知识库管理平台
    python tools/extract_corpus.py --out data/corpus

输出：
    data/corpus/manifest.json   文档清单（标题/分类/格式/权限）
    data/corpus/docs.jsonl      文档级全文与元数据
    data/corpus/chunks.jsonl    切片级语料（训练数据的证据单元）
    data/corpus/corpus_stats.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "corpus"


def find_kb_root(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if (p / "seed" / "docs" / "manifest.json").exists():
            return p
        raise SystemExit(f"--kb-root 下没有 seed/docs/manifest.json：{p}")
    candidates = [
        ROOT.parent / "知识库管理平台",
        ROOT.parent.parent / "知识库管理平台",
        Path(r"D:\lesson\实战\知识库管理平台"),
    ]
    for p in candidates:
        if (p / "seed" / "docs" / "manifest.json").exists():
            return p
    raise SystemExit(
        "未找到 2.9 知识库管理平台。\n"
        "请用 --kb-root 指定其路径；或直接使用仓库内已生成的 data/corpus/chunks.jsonl。"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb-root", default=None)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    kb_root = find_kb_root(args.kb_root)
    sys.path.insert(0, str(kb_root))
    try:
        from app.chunker import chunk_doc  # type: ignore
        from app.parsers import parse_file  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"导入 2.9 解析器失败：{exc}") from exc

    docs_dir = kb_root / "seed" / "docs"
    manifest = json.loads((docs_dir / "manifest.json").read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    docs, chunks = [], []
    for m in manifest:
        path = docs_dir / m["file"]
        doc = parse_file(path)
        parsed_chunks = chunk_doc(doc)
        docs.append({
            "doc_id": m["slug"],
            "title": m["title"],
            "category": m["category"],
            "format": m["format"],
            "page_count": doc.page_count,
            "char_count": doc.text_length,
            "chunk_count": len(parsed_chunks),
            "perms": m.get("perms", {}),
            "text": doc.full_text,
        })
        for i, c in enumerate(parsed_chunks):
            chunks.append({
                "chunk_id": f"{m['slug']}#{i}",
                "doc_id": m["slug"],
                "chunk_index": i,
                "title": m["title"],
                "category": m["category"],
                "doc_format": m["format"],
                "page_no": c["page_no"],
                "para_index": c["para_index"],
                "char_start": c["char_start"],
                "char_end": c["char_end"],
                "heading_path": c["heading_path"],
                "content": c["content"],
                "token_count": c["token_count"],
            })

    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "docs.jsonl").write_text(
        "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs), encoding="utf-8")
    (out / "chunks.jsonl").write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks), encoding="utf-8")

    stats = {
        "source": str(kb_root),
        "docs": len(docs),
        "chunks": len(chunks),
        "chars": sum(d["char_count"] for d in docs),
        "avg_chunk_chars": round(sum(len(c["content"]) for c in chunks) / max(1, len(chunks)), 1),
        "by_category": {},
        "by_format": {},
    }
    for d in docs:
        stats["by_category"][d["category"]] = stats["by_category"].get(d["category"], 0) + 1
        stats["by_format"][d["format"]] = stats["by_format"].get(d["format"], 0) + 1
    (out / "corpus_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"文档 {len(docs)} 篇 / 切片 {len(chunks)} 个 / 原始字符 {stats['chars']}")
    print(f"输出目录：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
