# -*- coding: utf-8 -*-
"""数据集结构与切分校验（对生成产物做回归测试）。"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import paths  # noqa: E402
from data import specs  # noqa: E402


class DatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = paths.read_jsonl(paths.OUT_DIR / "train.jsonl")
        cls.valid = paths.read_jsonl(paths.OUT_DIR / "valid.jsonl")
        cls.test = paths.read_jsonl(paths.OUT_DIR / "test.jsonl")

    def test_sizes(self):
        self.assertGreaterEqual(len(self.train), 2000)
        self.assertGreaterEqual(len(self.valid), 300)
        self.assertGreaterEqual(len(self.test), 500)

    def test_messages(self):
        for s in self.train[:50]:
            msgs = s["messages"]
            self.assertEqual([m["role"] for m in msgs], ["system", "user", "assistant"])
            self.assertTrue(msgs[0]["content"])
            self.assertTrue(msgs[2]["content"])

    def test_citations_in_range(self):
        for s in self.train[:300] + self.test[:100]:
            if s["task"] != "answer":
                continue
            n = len(s["contexts"])
            for idx in re.findall(r"\[(\d+)\]", s["answer"]):
                self.assertLessEqual(int(idx), n, s["id"])

    def test_doc_split_no_leak(self):
        for split, rows, docs in (
            ("train", self.train, specs.TRAIN_DOCS),
            ("valid", self.valid, specs.VALID_DOCS),
            ("test", self.test, specs.TEST_DOCS),
        ):
            for s in rows:
                for d in s["meta"].get("doc_ids", []):
                    self.assertIn(d, docs, f"{split} 泄漏文档 {d}：{s['id']}")

    def test_refusal_ratio(self):
        ratio = sum(1 for s in self.train if s["task"] == "refusal") / len(self.train)
        self.assertGreater(ratio, 0.10)
        self.assertLess(ratio, 0.22)

    def test_no_empty(self):
        for s in self.train:
            self.assertTrue(s["question"].strip())
            self.assertTrue(s["answer"].strip())
            self.assertTrue(s["contexts"] or s["task"] == "refusal")

    def test_capabilities_present(self):
        caps = {s["meta"].get("capability") for s in self.train + self.test}
        for cap in ("policy", "product", "steps", "troubleshoot", "integration",
                    "refusal_none", "refusal_weak", "refusal_conflict", "partial"):
            self.assertIn(cap, caps)


if __name__ == "__main__":
    unittest.main()
