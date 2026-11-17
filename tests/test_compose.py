# -*- coding: utf-8 -*-
"""单元测试：4 类格式组装、引用、拒答/冲突/部分可答。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kb.compose import (  # noqa: E402
    compose_answer, conflict_answer, partial_answer, refusal_answer,
)
from kb.format_rules import check_format, numbers_supported  # noqa: E402

CONTEXTS = [
    {"n": 1, "chunk_id": "c1", "title": "差旅报销标准", "page_no": 1,
     "text": "一线城市（北京、上海、广州、深圳）住宿标准为每人每晚 500 元。总监级及以上岗位可在上述标准基础上上浮 50%，需附审批记录。"},
    {"n": 2, "chunk_id": "c2", "title": "费用报销审批流程", "page_no": 1,
     "text": "费用发生后 30 个自然日内提交报销单，每月 25 日为当月报销截止日。"},
]
EV = [
    {"text": "一线城市（北京、上海、广州、深圳）住宿标准为每人每晚 500 元。", "ctx_n": 1},
    {"text": "总监级及以上岗位可在上述标准基础上上浮 50%，需附审批记录。", "ctx_n": 1},
]


class ComposeTest(unittest.TestCase):
    def test_policy(self):
        ans = compose_answer("出差住宿费标准是多少？", CONTEXTS, "policy", EV)
        ok, problems = check_format(ans, "policy")
        self.assertTrue(ok, problems)
        ok_num, p = numbers_supported(ans, CONTEXTS)
        self.assertTrue(ok_num, p)
        self.assertIn("[1]", ans)

    def test_product(self):
        ans = compose_answer("各城市住宿标准是什么？", CONTEXTS, "product", EV)
        ok, problems = check_format(ans, "product")
        self.assertTrue(ok, problems)
        self.assertIn("| 项目 |", ans)

    def test_steps(self):
        ev = [{"text": "提交出差申请", "ctx_n": 1}, {"text": "部门负责人审批", "ctx_n": 1},
              {"text": "财务复核付款", "ctx_n": 1}]
        ans = compose_answer("流程是什么？", CONTEXTS, "steps", ev)
        ok, problems = check_format(ans, "steps")
        self.assertTrue(ok, problems)

    def test_troubleshoot(self):
        ev = [{"text": "无法连接 VPN：检查客户端版本是否为最新，重启客户端后重试。", "ctx_n": 1},
              {"text": "仍失败的，确认账号是否过期。", "ctx_n": 1}]
        ans = compose_answer("VPN 连不上怎么办？", CONTEXTS, "troubleshoot", ev)
        ok, problems = check_format(ans, "troubleshoot")
        self.assertTrue(ok, problems)
        self.assertNotIn("怎么办。", ans)

    def test_integration(self):
        ev = EV + [{"text": "费用发生后 30 个自然日内提交报销单。", "ctx_n": 2}]
        ans = compose_answer("住宿和报销时限分别是什么？", CONTEXTS, "integration", ev)
        ok, problems = check_format(ans, "integration")
        self.assertTrue(ok, problems)
        self.assertIn("[2]", ans)

    def test_refusal(self):
        ans = refusal_answer("none", "竞品定价策略是什么？", "竞品价格策略", "")
        ok, problems = check_format(ans, "refusal")
        self.assertTrue(ok, problems)
        self.assertNotIn("[1]", ans)

    def test_conflict(self):
        ans = conflict_answer("住宿是 500 还是 400？",
                              {"text": "住宿标准 500 元", "ctx_n": 1, "title": "现行制度", "value": "500"},
                              {"text": "住宿标准 400 元", "ctx_n": 2, "title": "2022 旧版（已废止）", "value": "400"})
        ok, problems = check_format(ans, "refusal_conflict")
        self.assertTrue(ok, problems)
        self.assertIn("冲突", ans)

    def test_partial(self):
        ans = partial_answer("住宿标准是多少？海外呢？", CONTEXTS, EV, "海外住宿标准", "境外差旅管理办法")
        ok, problems = check_format(ans, "partial")
        self.assertTrue(ok, problems)
        self.assertIn("缺失信息", ans)


if __name__ == "__main__":
    unittest.main()
