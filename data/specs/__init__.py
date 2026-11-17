# -*- coding: utf-8 -*-
"""数据规格入口：主题、组合、负样本、以及按文档切分的数据集划分。

**按文档切分**是防止训练/测试泄漏的硬约束：
同一篇文档只能整体进入 train / valid / test 之一；多片段组合只有在其
涉及的全部文档属于同一 split 时才允许生成（构建脚本会校验并丢弃跨界组合）。
"""
from __future__ import annotations

from .negatives import CONFLICTS, NO_RECALL, PARTIAL, WEAK_RECALL
from .topics_finance_hr import TOPICS as _FIN_HR
from .topics_it_cs import TOPICS as _IT_CS

TOPICS = _FIN_HR + _IT_CS
TOPIC_BY_ID = {t["id"]: t for t in TOPICS}


def C(cid: str, topic_ids: list[str], questions: list[str]) -> dict:
    return {"id": cid, "topics": topic_ids, "questions": questions}


#: 多片段整合：每个组合至少跨 2 个主题；跨文档组合用于 4.1.3 的“跨段落引用”
COMBOS = [
    C("c-travel", ["tr-lodging", "tr-meal", "ea-approval", "ir-check"],
      ["出差住宿和餐补标准是多少？报销审批和发票查验分别有什么要求？"]),
    C("c-travel-invoice", ["tr-materials", "ea-invoice", "ir-error"],
      ["差旅报销要准备哪些材料？电子发票怎么提交、抬头开错了怎么办？"]),
    C("c-refund", ["rp-seven", "rp-refund", "cl-level"],
      ["客户要求七天无理由退货，退款多久到账？这类问题属于几级投诉？"]),
    C("c-complaint", ["cl-sla", "cl-escalate", "rp-fresh"],
      ["一级投诉多久响应和闭环？客户重复投诉会怎么升级？生鲜破损怎么处理？"]),
    C("c-leave", ["eh-leave", "eh-carry", "eh-overtime"],
      ["公司年假有几天、没休完怎么处理？加班调休有什么规定？"]),
    C("c-return", ["rp-seven", "rp-quality", "rp-fresh"],
      ["七天无理由、质量问题退换和生鲜售后分别怎么规定？"]),
    C("c-attendance", ["eh-attendance", "eh-exception"],
      ["公司考勤时间和迟到豁免怎么规定？忘记打卡怎么办？"]),
    C("c-onboard-it", ["ob-process", "it-channel"],
      ["新员工入职第一周要做什么？电脑坏了怎么报修？"]),
    C("c-it-vpn", ["it-vpn", "vr-apply", "vr-use"],
      ["VPN 连不上怎么排查？VPN 怎么申请、远程办公有什么使用要求？"]),
    C("c-logistics", ["lp-abnormal", "ss-scenes"],
      ["物流超过 7 天没送达怎么处理？客服应该怎么回复客户？"]),
    C("c-procurement", ["pg-approve", "pg-compare", "pg-accept"],
      ["采购审批、比价和验收付款分别有什么要求？"]),
    C("c-training", ["tp-external", "tp-service"],
      ["外部培训怎么申请？培训后要签服务期协议吗？"]),
    C("c-performance", ["pr-apply", "sp-structure", "sc-payout"],
      ["季度考核结果对绩效工资和销售提成有什么影响？"]),
    C("c-data", ["ds-transport", "as-password"],
      ["敏感数据传输有什么要求？账号密码策略是什么？"]),
    C("c-pr", ["pc-level", "pc-voice"],
      ["舆情事件怎么分级？媒体来问应该谁回应？"]),
    C("c-asset", ["fa-issue", "as-recycle"],
      ["离职时电脑怎么归还？账号和权限怎么回收？"]),
    C("c-sales", ["sc-rate", "sc-payout", "ec-structure"],
      ["销售提成比例怎么分档、什么时候发？高管薪酬结构又是什么？"]),
    C("c-knowledge", ["kb-perm", "as-perm"],
      ["知识库权限怎么配置？员工账号权限有什么管理要求？"]),
    C("c-market", ["ad-approve", "ad-effect", "bv-review"],
      ["投放预算审批和效果评估怎么做？对外物料要审核吗？"]),
    C("c-oncall", ["oc-level", "oc-flow"],
      ["线上故障怎么分级、处理流程是什么？"]),
    C("c-quality", ["qs-ratio", "qs-fail"],
      ["商品抽检比例是多少？抽检不合格怎么处理？"]),
    C("c-meeting", ["mr-book", "mr-use"],
      ["会议室怎么预订？订了没用会怎样？"]),
    C("c-recruit", ["rp-referral", "rp-background"],
      ["内部推荐有奖励吗？哪些岗位要做背景调查？"]),
    C("c-budget", ["bm-adjust", "bm-final"],
      ["预算调剂和超支怎么审批？预算执行率怎么考核？"]),
    C("c-exec", ["ec-equity", "ec-confidential"],
      ["限制性股票怎么归属？高管离职后的竞业限制怎么规定？"]),
    C("c-fixed", ["fa-repair", "fa-check"],
      ["设备故障怎么维修？资产盘点和报废怎么规定？"]),
]

#: 文档级切分：test 覆盖财务/差旅、售后、考勤三类，valid 覆盖 IT/物流/采购，
#: train 覆盖其余 18 篇。三份文档集合互不相交。
TEST_DOCS = {
    "travel-reimburse", "expense-approval", "invoice-rules",
    "return-policy", "complaint-level", "employee-handbook",
}
VALID_DOCS = {
    "logistics-policy", "service-scripts", "procurement",
    "onboarding", "it-support", "vpn-remote",
}

ALL_DOCS = {
    "travel-reimburse", "expense-approval", "exec-compensation", "sales-commission",
    "budget-management", "invoice-rules", "fixed-asset", "employee-handbook",
    "onboarding", "recruit-process", "performance-review", "training-policy",
    "salary-payment", "account-security", "data-security", "it-support",
    "vpn-remote", "return-policy", "complaint-level", "service-scripts",
    "logistics-policy", "brand-vi", "ad-delivery", "pr-crisis", "campaign-sop",
    "procurement", "meeting-room", "oncall-process", "quality-standard",
    "knowledge-base-rule",
}
TRAIN_DOCS = ALL_DOCS - TEST_DOCS - VALID_DOCS


def split_of_doc(doc: str) -> str:
    if doc in TEST_DOCS:
        return "test"
    if doc in VALID_DOCS:
        return "valid"
    return "train"


def validate() -> list[str]:
    """静态校验：重复 id、未定义文档、锚点为空、组合引用不存在。"""
    problems: list[str] = []
    seen: set[str] = set()
    for t in TOPICS:
        if t["id"] in seen:
            problems.append(f"重复主题 id：{t['id']}")
        seen.add(t["id"])
        if t["doc"] not in ALL_DOCS:
            problems.append(f"{t['id']} 引用了未知文档 {t['doc']}")
        if not t["anchors"]:
            problems.append(f"{t['id']} 没有证据锚点")
        if t["format"] not in ("policy", "product", "steps", "troubleshoot"):
            problems.append(f"{t['id']} 格式非法：{t['format']}")
    for c in COMBOS:
        for tid in c["topics"]:
            if tid not in TOPIC_BY_ID:
                problems.append(f"组合 {c['id']} 引用了未知主题 {tid}")
    return problems


__all__ = [
    "TOPICS", "TOPIC_BY_ID", "COMBOS", "NO_RECALL", "WEAK_RECALL",
    "CONFLICTS", "PARTIAL", "TEST_DOCS", "VALID_DOCS", "TRAIN_DOCS",
    "ALL_DOCS", "split_of_doc", "validate",
]
