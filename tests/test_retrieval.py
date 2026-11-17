# -*- coding: utf-8 -*-
"""单元测试：语料装载与混合检索。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kb.corpus import Corpus  # noqa: E402
from kb.retrieval import Retriever, context_features  # noqa: E402


class RetrievalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = Corpus.load()
        cls.retriever = Retriever(cls.corpus.chunks)

    def test_corpus(self):
        self.assertGreaterEqual(len(self.corpus.docs), 30)
        self.assertGreaterEqual(len(self.corpus.chunks), 50)
        self.assertGreater(len(self.corpus.all_sentences()), 300)

    def test_search_hits_expected_doc(self):
        hits = self.retriever.search("出差住宿费标准是多少？", top_k=4)
        self.assertTrue(hits)
        titles = [h["title"] for h in hits]
        self.assertTrue(any("差旅" in t for t in titles), titles)

    def test_features(self):
        hits = self.retriever.search("出差住宿费标准是多少？", top_k=4)
        contexts = [dict(h, n=i + 1) for i, h in enumerate(hits)]
        feats = context_features("出差住宿费标准是多少？", contexts, self.retriever.vectorizer)
        self.assertGreater(feats["top1_sim"], 0)
        self.assertGreater(feats["max_cov"], 0.2)
        self.assertEqual(feats["n_parts"], 1)
        self.assertIn("missing_term_count", feats)

    def test_conflict_flag(self):
        a = {"n": 1, "text": "一线城市住宿标准为每人每晚 500 元，超标部分由个人承担。"}
        b = {"n": 2, "text": "一线城市住宿标准为每人每晚 400 元，超标部分由个人承担。"}
        feats = context_features("住宿标准是多少？", [a, b], self.retriever.vectorizer)
        self.assertEqual(feats["conflict"], 1.0)

    def test_no_conflict_for_different_topics(self):
        a = {"n": 1, "text": "员工离职账号保留 30 天审计日志后注销。"}
        b = {"n": 2, "text": "生鲜食品签收后 48 小时内出现破损可申请退款。"}
        feats = context_features("离职账号怎么处理？", [a, b], self.retriever.vectorizer)
        self.assertEqual(feats["conflict"], 0.0)


if __name__ == "__main__":
    unittest.main()
