# -*- coding: utf-8 -*-
"""单元测试：文本、数字事实与问题拆分。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kb.textutil import (  # noqa: E402
    content_terms, fact_numbers, normalize_question, split_question_parts,
    split_sentences, tokenize,
)


class TextUtilTest(unittest.TestCase):
    def test_fact_numbers(self):
        nums = fact_numbers("一线城市 500 元/晚，15 个工作日内提交，比例 30%，P1 故障 15 分钟响应")
        self.assertIn("500元/晚", nums)
        self.assertIn("15工作日", nums)
        self.assertIn("30%", nums)
        self.assertIn("15分钟", nums)
        self.assertNotIn("1", nums)          # P1 不能被当成业务数字

    def test_split_sentences_skips_headings(self):
        text = "第一章 适用范围\n本标准适用于全体员工。\n第二章 住宿标准\n一线城市 500 元。"
        sents = split_sentences(text)
        self.assertEqual(len(sents), 2)
        self.assertIn("本标准", sents[0])

    def test_question_parts(self):
        parts = split_question_parts("出差住宿标准是多少？另外海外出差的住宿标准呢？")
        self.assertEqual(len(parts), 2)
        parts2 = split_question_parts("请分别说明：差旅报销材料与流程；报销发票要求。")
        self.assertEqual(len(parts2), 2)
        self.assertNotIn("请分别说明", parts2[0])

    def test_content_terms_filters_cross_word_bigrams(self):
        terms = content_terms("出差住宿和餐补标准是多少")
        self.assertIn("住宿", terms)
        self.assertIn("餐补", terms)
        self.assertNotIn("和餐", terms)

    def test_normalize_question(self):
        self.assertEqual(normalize_question("  出差住宿标准是多少？ "), "出差住宿标准是多少")
        self.assertEqual(normalize_question("ＡＢＣ？"), "abc")

    def test_tokenize(self):
        toks = tokenize("VPN 无法连接")
        self.assertIn("vpn", toks)
        self.assertIn("无法", toks)


if __name__ == "__main__":
    unittest.main()
