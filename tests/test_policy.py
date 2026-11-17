# -*- coding: utf-8 -*-
"""策略模型与评测判定测试。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.metrics import aggregate, judge  # noqa: E402
from kb.corpus import Corpus, as_context  # noqa: E402
from kb.retrieval import Retriever  # noqa: E402
from train.policy import FineTunedPolicy  # noqa: E402


class PolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = Corpus.load()
        cls.retriever = Retriever(cls.corpus.chunks)
        cls.policy = FineTunedPolicy(vectorizer=cls.retriever.vectorizer)
        hits = cls.retriever.search("出差住宿费标准是多少？", top_k=4)
        cls.contexts = [as_context(h, i + 1) for i, h in enumerate(hits)]

    def test_answer_with_citations(self):
        out = self.policy.predict("出差住宿费标准是多少？", self.contexts)
        self.assertEqual(out["decision"], "answer")
        self.assertIn("[", out["answer"])
        self.assertTrue(out["gate_prob"] > 0.3)

    def test_refuse_without_context(self):
        out = self.policy.predict("竞品最近的定价策略是什么？", [])
        self.assertEqual(out["decision"], "refuse")
        self.assertIn("缺失信息", out["answer"])

    def test_conflict_detection(self):
        ctxs = [
            {"n": 1, "title": "差旅报销标准", "text": "出差结束后 15 个工作日内提交报销单，逾期超过 30 天原则上不予受理。", "chunk_id": "a"},
            {"n": 2, "title": "费用报销审批流程", "text": "费用发生后 30 个自然日内提交报销单，跨月费用须在次月 15 日前提交。", "chunk_id": "b"},
        ]
        out = self.policy.predict("差旅报销到底要在 15 个工作日内还是 30 个自然日内提交？", ctxs)
        self.assertEqual(out["decision"], "conflict")
        self.assertIn("冲突信息", out["answer"])

    def test_judge_perfect_answer(self):
        contexts = [{"n": 1, "chunk_id": "c1", "title": "差旅报销标准",
                     "text": "一线城市住宿标准为每人每晚 500 元。总监级及以上岗位可在上述标准基础上上浮 50%。"}]
        sample = {
            "question": "出差住宿费标准是多少？", "task": "answer", "format": "policy",
            "contexts": contexts,
            "meta": {"key_points": ["500元"], "term": "住宿标准"},
            "answer": "",
        }
        pred = "【结论】一线城市住宿标准为每人每晚 500 元 [1]。\n\n【依据】\n1. 总监级及以上岗位上浮 50% [1]。"
        j = judge(pred, sample)
        self.assertGreaterEqual(j["faithfulness"] or 0, 0.9)
        self.assertGreaterEqual(j["citation_coverage"] or 0, 0.9)
        self.assertTrue(j["format_ok"])
        self.assertFalse(j["false_refusal"])

    def test_judge_refusal(self):
        sample = {
            "question": "竞品定价策略是什么？", "task": "refusal", "format": "refusal",
            "contexts": [], "meta": {"key_points": []}, "answer": "",
        }
        pred = "【结论】现有知识库资料无法回答该问题。\n【缺失信息】竞品定价策略\n【建议】请补充资料。"
        j = judge(pred, sample)
        self.assertTrue(j["refusal_ok"])
        self.assertTrue(j["detected_refusal"])

    def test_aggregate(self):
        sample = {
            "question": "出差住宿费标准是多少？", "task": "answer", "format": "policy",
            "contexts": self.contexts, "meta": {"key_points": ["500元"]},
        }
        rec = {"judge": judge("【结论】一线城市住宿标准为每人每晚 500 元 [1]。", sample),
               "task": "answer", "meta": {"capability": "policy"}}
        m = aggregate([rec])
        self.assertIn("composite", m)
        self.assertGreaterEqual(m["composite"], 0)
        self.assertLessEqual(m["composite"], 1)


if __name__ == "__main__":
    unittest.main()
