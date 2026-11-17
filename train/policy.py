# -*- coding: utf-8 -*-
"""轻量策略模型（Distilled Policy Head, DPH）与推理入口。

本机离线环境没有 GPU 可用的生成式模型权重（详见 README「两条训练路径」），
因此把 4.1 的能力拆成三层**可训练策略**，用同一份 SFT 数据训练：

  1. Gate（二分类）：判断“资料是否足以回答”→ 该答 / 该拒；
  2. Format（四分类）：判断应使用制度解释 / 产品说明 / 操作步骤 / 故障处理；
  3. Evidence Selector（二分类）：从上下文中挑出真正支撑结论的句子。

答案文本由 `kb.compose.py` 按格式组装（只摘录证据句，不做自由生成），
因此忠实性、引用可追溯性由“选证据 + 引用编号”共同保证。
这一层可视为对 LoRA SFT 目标行为的蒸馏：数据、Prompt、评测口径完全一致，
接入 2.9 时也可以直接替换成 `--backend lora`。
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

from config import paths
from kb.compose import (
    compose_answer, conflict_answer, partial_answer, refusal_answer,
    sentences_from_contexts,
)
from kb.retrieval import FEATURE_NAMES, TextVectorizer, context_features, feature_vector
from kb.terms import used_standard
from kb.textutil import (
    content_terms, extract_numbers, fact_numbers, jaccard, overlap_ratio,
    split_question_parts, tokenize,
)

GATE_CLASSES = ("answer", "refuse")
ROUTE_CLASSES = ("policy", "product", "steps", "troubleshoot")
FORMAT_CLASSES = ROUTE_CLASSES

CATEGORY_SUGGEST = {
    "财务制度": "财务制度", "人力资源": "人力资源制度", "信息技术": "信息技术制度",
    "客服政策": "客服政策", "市场运营": "市场运营制度", "行政": "行政管理制度",
}


# ---------------------------------------------------------------- 数值模型
class Scaler:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, X: np.ndarray) -> "Scaler":
        self.mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-6] = 1.0
        self.std = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_softmax(X: np.ndarray, y: np.ndarray, n_classes: int, epochs: int = 220,
                lr: float = 0.4, l2: float = 1e-3, class_weight: bool = True,
                seed: int = 42) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rng = np.random.default_rng(seed)
    n, d = X.shape
    W = rng.normal(0, 0.01, size=(n_classes, d))
    b = np.zeros(n_classes)
    onehot = np.zeros((n, n_classes))
    onehot[np.arange(n), y] = 1
    if class_weight:
        counts = np.bincount(y, minlength=n_classes).astype(float)
        cw = n / (n_classes * np.maximum(counts, 1))
        sw = cw[y]
    else:
        sw = np.ones(n)
    log = []
    for epoch in range(epochs):
        logits = X @ W.T + b
        probs = _softmax(logits)
        loss = float(-(np.log(np.clip(probs[np.arange(n), y], 1e-9, 1)) * sw).mean()
                     + l2 * np.sum(W * W) / 2)
        grad = (probs - onehot) * sw[:, None] / n
        W -= lr * (grad.T @ X + l2 * W)
        b -= lr * grad.sum(axis=0)
        acc = float((probs.argmax(axis=1) == y).mean())
        if epoch % 10 == 0 or epoch == epochs - 1:
            log.append({"epoch": epoch, "loss": round(loss, 6), "acc": round(acc, 4)})
    return W, b, log


def fit_logistic(X: np.ndarray, y: np.ndarray, epochs: int = 260,
                 lr: float = 0.5, l2: float = 1e-3, seed: int = 42) -> tuple[np.ndarray, float, list[dict]]:
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w = rng.normal(0, 0.01, size=d)
    b = 0.0
    pos = max(1, int(y.sum()))
    neg = max(1, n - pos)
    sw = np.where(y == 1, n / (2 * pos), n / (2 * neg))
    log = []
    for epoch in range(epochs):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        loss = float(-(np.log(np.clip(p, 1e-9, 1)) * y + np.log(np.clip(1 - p, 1e-9, 1)) * (1 - y)).dot(sw) / n
                     + l2 * np.dot(w, w) / 2)
        grad = (p - y) * sw / n
        w -= lr * (X.T @ grad + l2 * w)
        b -= lr * grad.sum()
        acc = float(((p >= 0.5).astype(int) == y).mean())
        if epoch % 10 == 0 or epoch == epochs - 1:
            log.append({"epoch": epoch, "loss": round(loss, 6), "acc": round(acc, 4)})
    return w, b, log


# ---------------------------------------------------------------- 特征
def selector_features(question: str, sentence: str, ctx_rank: int,
                      ctx: dict, idx: int, total: int) -> list[float]:
    q_terms = content_terms(question)
    s_terms = content_terms(sentence)
    q_nums = set(fact_numbers(question))
    s_nums = set(fact_numbers(sentence))
    title_terms = content_terms(f"{ctx.get('title', '')} {ctx.get('heading_path', '')}")
    return [
        overlap_ratio(q_terms, s_terms),
        jaccard(q_terms, s_terms),
        len(q_nums & s_nums),
        1.0 if s_nums else 0.0,
        min(len(sentence), 120) / 120.0,
        1.0 / max(1, ctx_rank),
        overlap_ratio(q_terms, title_terms) if title_terms else 0.0,
        idx / max(1, total),
        1.0 if any(w in sentence for w in ("可以", "不得", "须", "应", "为", "按", "需")) else 0.0,
        1.0 if sentence.endswith(("？", "?")) else 0.0,
    ]


SELECTOR_FEATURES = [
    "cov", "jac", "num_hit", "has_num", "len", "ctx_rank", "title_cov",
    "pos", "modal", "question_like",
]

FORMAT_KEYWORDS = [
    "流程", "步骤", "怎么操作", "如何办理", "申请", "材料", "故障", "无法", "报错",
    "异常", "怎么办", "多少", "标准", "比例", "限额", "哪些", "分类", "分级",
    "表格", "对比", "是否", "可以", "规定", "要求",
]
CATEGORIES = ["财务制度", "人力资源", "信息技术", "客服政策", "市场运营", "行政"]


def format_features(question: str, vocab: dict[str, int], category: str = "",
                    context_text: str = "") -> np.ndarray:
    vec = np.zeros(len(vocab) + len(FORMAT_KEYWORDS) + len(CATEGORIES) + 1, dtype=float)
    counts: dict[str, int] = {}
    for tok in tokenize(question + " " + (context_text or "")[:160]):
        if tok in vocab:
            counts[tok] = counts.get(tok, 0) + 1
    for tok, cnt in counts.items():
        vec[vocab[tok]] = min(cnt, 3) / 3.0
    base = len(vocab)
    for i, kw in enumerate(FORMAT_KEYWORDS):
        vec[base + i] = 1.0 if kw in question else 0.0
    base += len(FORMAT_KEYWORDS)
    if category in CATEGORIES:
        vec[base + CATEGORIES.index(category)] = 1.0
    vec[-1] = min(len(question), 60) / 60.0
    return vec


def build_format_vocab(questions: list[str], limit: int = 500) -> dict[str, int]:
    from collections import Counter
    cnt: Counter = Counter()
    for q in questions:
        cnt.update({t: 1 for t in set(tokenize(q))})
    # 选区分度高的二元组：按出现次数排序即可（问题文本较短）
    items = [t for t, _ in cnt.most_common(limit)]
    return {t: i for i, t in enumerate(items)}


# ---------------------------------------------------------------- 推理
class FineTunedPolicy:
    def __init__(self, model_dir: str | Path | None = None,
                 vectorizer: TextVectorizer | None = None):
        self.dir = Path(model_dir) if model_dir else paths.ADAPTER_DIR / "policy"
        self.meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        data = np.load(self.dir / "model.npz", allow_pickle=False)
        self.gate_w = data["gate_w"]
        self.gate_b = float(data["gate_b"])
        self.gate_scaler = Scaler(data["gate_mean"], data["gate_std"])
        self.fmt_w = data["fmt_w"]
        self.fmt_b = data["fmt_b"]
        self.fmt_scaler = Scaler(data["fmt_mean"], data["fmt_std"])
        self.sel_w = data["sel_w"]
        self.sel_b = float(data["sel_b"])
        self.sel_scaler = Scaler(data["sel_mean"], data["sel_std"])
        self.vocab: dict[str, int] = self.meta["format_vocab"]
        self.gate_threshold = float(self.meta["gate_threshold"])
        self.none_sim_threshold = float(self.meta.get("none_sim_threshold", 0.05))
        self.vectorizer = vectorizer or TextVectorizer(
            [c["content"] for c in CorpusLite.chunks()])

    # ------------------------------------------------------------ 工具
    @staticmethod
    def _proba_logistic(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
        z = X @ w + b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def _gate(self, feats: dict) -> float:
        x = np.array([feature_vector(feats)])
        x = self.gate_scaler.transform(x)
        return float(self._proba_logistic(x, self.gate_w, self.gate_b)[0])

    def _format(self, question: str, category: str, context_text: str = "") -> str:
        x = format_features(question, self.vocab, category, context_text)
        x = self.fmt_scaler.transform(np.array([x]))
        logits = x @ self.fmt_w.T + self.fmt_b
        return FORMAT_CLASSES[int(logits.argmax(axis=1)[0])]

    def _select(self, question: str, contexts: list[dict], fmt: str) -> list[dict]:
        cands = sentences_from_contexts(contexts)
        # FAQ 里的“问：……”是设问句，不能当作答案证据
        cands = [c for c in cands if not c["text"].endswith(("？", "?"))] or cands
        # 产品说明优先保留“含数字”的可枚举句；若该主题本身没有数字
        # （如数据分级、故障分级），则保留全部句子，避免表格为空
        if fmt == "product":
            numeric = [c for c in cands if fact_numbers(c["text"])]
            if len(numeric) >= 2:
                cands = numeric
        if not cands:
            return []
        rows = []
        for i, c in enumerate(cands):
            same_ctx = [x for x in cands if x["ctx_n"] == c["ctx_n"]]
            pos = same_ctx.index(c) if c in same_ctx else 0
            ctx = next((x for x in contexts if x.get("n") == c["ctx_n"]), {})
            rows.append(selector_features(question, c["text"], c["ctx_n"], ctx, pos, len(same_ctx)))
        X = self.sel_scaler.transform(np.array(rows))
        scores = self._proba_logistic(X, self.sel_w, self.sel_b)
        # 术语优先：包含标准术语的句子更可能是该主题的答案句
        term_bonus = np.array([min(len(used_standard(c["text"])), 2) for c in cands], dtype=float)
        scores = scores + 0.06 * term_bonus

        # 单主题问答：保留得分接近最高分的片段，既避免拼接无关制度，
        # 又不丢掉同一制度的相邻切片；多片段整合则保留全部上下文
        if fmt != "integration":
            ctx_best: dict[int, float] = {}
            for i, c in enumerate(cands):
                ctx_best[c["ctx_n"]] = max(ctx_best.get(c["ctx_n"], 0.0), float(scores[i]))
            if ctx_best:
                best = max(ctx_best.values())
                band = 0.75 if fmt in ("product", "troubleshoot") else 0.60
                allowed = {ctx for ctx, sc in ctx_best.items() if sc >= best * band}
                kept = [(float(scores[i]), c) for i, c in enumerate(cands)
                        if c["ctx_n"] in allowed]
                kept.sort(key=lambda x: -x[0])
                cands = [c for _, c in kept]
                scores = np.array([s for s, _ in kept])
                order = np.arange(len(cands))
            else:
                order = np.argsort(-scores)
        else:
            order = np.argsort(-scores)
        limit = {"policy": 5, "product": 6, "steps": 6, "troubleshoot": 5,
                 "integration": 6}.get(fmt, 5)
        best_score = float(scores.max()) if len(scores) else 0.0
        picked, seen_ctx = [], set()
        for idx in order:
            if fmt != "integration":
                ratio = 0.45 if fmt == "product" else 0.30
                if float(scores[int(idx)]) < best_score * ratio:
                    continue
            s = cands[int(idx)]
            if s["text"] in {p["text"] for p in picked}:
                continue
            if fmt == "integration":
                if s["ctx_n"] in seen_ctx and len(picked) >= 3:
                    continue
            picked.append({**s, "score": round(float(scores[int(idx)]), 4)})
            seen_ctx.add(s["ctx_n"])
            if len(picked) >= limit:
                break
        if fmt == "integration" and len({p["ctx_n"] for p in picked}) < 2:
            # 尽量覆盖多个片段
            for c in cands:
                if c["ctx_n"] != picked[0]["ctx_n"] if picked else True:
                    picked.append({**c, "score": 0.0})
                    break
        return picked

    @staticmethod
    def _category(contexts: list[dict]) -> str:
        return contexts[0].get("category", "") if contexts else ""

    def _missing_suggest(self, question: str, contexts: list[dict],
                         kind: str = "weak") -> tuple[str, str]:
        q = re.sub(r"^(请问|想问下|我想问|问一下)[，,：:\s]*", "", question).rstrip("？?")
        q = re.sub(r"(怎么办|怎么规定|有哪些要求|是什么|是多少|吗)$", "", q).strip()
        if kind == "none":
            return (q or question, "")
        category = self._category(contexts)
        return (q or question, CATEGORY_SUGGEST.get(category, "相关制度"))

    @staticmethod
    def _snippet(context: dict, question: str) -> str:
        """从长片段中抽出真正包含冲突数值的那一句，避免把整篇制度贴进答案。"""
        nums = set(fact_numbers(question))
        for s in sentences_from_contexts([context]):
            if nums & set(fact_numbers(s["text"])):
                return s["text"]
        return (context.get("text") or "").strip()

    # ------------------------------------------------------------ 冲突/部分可答
    @staticmethod
    def _find_conflict(contexts: list[dict], question: str) -> tuple[dict, dict] | None:
        """只在“问题显式给出两个数值、且分别落在不同片段”时判定冲突。

        这比“文本相似 + 数值不同”保守得多：多片段整合里不同制度经常
        同时出现金额/天数，不能据此判为冲突。
        """
        if len(contexts) < 2:
            return None
        q_nums = list(dict.fromkeys(fact_numbers(question)))
        if len(q_nums) < 2:
            return None
        holds: dict[str, list[dict]] = {}
        for num in q_nums:
            holds[num] = [c for c in contexts if num in set(fact_numbers(c.get("text", "")))]
        nums_ctx = [(num, cs) for num, cs in holds.items() if cs]
        if len(nums_ctx) >= 2:
            c1 = nums_ctx[0][1][0]
            for _, cs in nums_ctx[1:]:
                for c2 in cs:
                    if c1.get("n") != c2.get("n"):
                        return (c1, c2)
        return None

    def _partial_parts(self, question: str, contexts: list[dict]) -> tuple[list[str], int] | None:
        parts = split_question_parts(question)
        if len(parts) < 2 or not contexts:
            return None
        supported = []
        for part in parts:
            pt = content_terms(part)
            cov = max(overlap_ratio(pt, content_terms(c.get("text", ""))) for c in contexts)
            oov = [t for t in pt if t not in self.vectorizer.idf]
            oov_ratio = len(oov) / max(1, len(pt))
            supported.append(cov >= 0.35 and oov_ratio <= 0.20)
        if any(supported) and not all(supported):
            return parts, supported.index(False)
        return None

    # ------------------------------------------------------------ 主入口
    def predict(self, question: str, contexts: list[dict],
                force_answer: bool = False, with_citations: bool = True) -> dict:
        feats = context_features(question, contexts, self.vectorizer)
        gate_prob = self._gate(feats)
        n_parts = len(split_question_parts(question))

        # 多子问题：先尝试识别“部分可答”，再决定是否放行
        partial = self._partial_parts(question, contexts)
        if partial:
            parts, lo = partial
            ev = self._select(question, contexts, "policy")
            answer = partial_answer(question, contexts, ev, parts[lo],
                                    CATEGORY_SUGGEST.get(self._category(contexts), "相关制度"))
            return {"answer": answer, "decision": "partial", "format": "partial",
                    "gate_prob": round(gate_prob, 4), "evidence": ev}

        # 多片段整合问题：只要有子问题被覆盖，就应尝试整合而不是整体拒答
        if (not force_answer) and n_parts >= 2 and feats.get("part_max_cov", 0.0) >= 0.30:
            gate_prob = max(gate_prob, self.gate_threshold + 0.01)

        if (not force_answer) and gate_prob < self.gate_threshold:
            conflict = self._find_conflict(contexts, question)
            if conflict:
                a, b = conflict
                old = next((c for c in (a, b) if re.search(r"废止|旧版|2022|2023|旧", c.get("title", ""))), b)
                cur = b if old is a else a
                ans = conflict_answer(question,
                                      {"text": self._snippet(cur, question), "ctx_n": cur.get("n", 1),
                                       "title": cur.get("title", ""), "value": ""},
                                      {"text": self._snippet(old, question), "ctx_n": old.get("n", 1),
                                       "title": old.get("title", ""), "value": ""})
                return {"answer": ans, "decision": "conflict", "format": "refusal_conflict",
                        "gate_prob": round(gate_prob, 4), "evidence": []}
            kind = "none" if (not contexts or (feats["top1_sim"] < self.none_sim_threshold
                                               and feats["max_cov"] < 0.30)) else "weak"
            missing, suggest = self._missing_suggest(question, contexts, kind)
            extra = "" if kind == "none" else "检索到的片段只覆盖相邻主题，缺少该问题的关键口径"
            answer = refusal_answer(kind, question, missing, suggest, extra=extra)
            return {"answer": answer, "decision": "refuse", "format": "refusal",
                    "gate_prob": round(gate_prob, 4), "evidence": []}

        # 应当回答
        if n_parts >= 2:
            fmt = "integration"
        else:
            ctx_text = contexts[0].get("text", "") if contexts else ""
            fmt = self._format(question, self._category(contexts), ctx_text)
            # 强格式线索兜底（与训练标签口径一致，避免分类器在少数类别上抖动）
            if re.search(r"故障|无法|报错|异常|打不开|连不上|坏了", question):
                fmt = "troubleshoot"
            elif re.search(r"流程|步骤|怎么操作|如何办理|怎么走|手续", question):
                fmt = "steps"
            elif re.search(r"哪几类|哪些类型|分为", question):
                fmt = "policy"
        ev = self._select(question, contexts, fmt)
        answer = compose_answer(question, contexts, fmt, ev)
        if not with_citations:
            answer = re.sub(r"\s*\[\d+\]", "", answer)
        return {"answer": answer, "decision": "answer", "format": fmt,
                "gate_prob": round(gate_prob, 4), "evidence": ev}


class CorpusLite:
    """推理侧只需切片正文来拟合 IDF 向量器（避免与数据集构建耦合）。"""
    _chunks: list[dict] | None = None

    @classmethod
    def chunks(cls) -> list[dict]:
        if cls._chunks is None:
            cls._chunks = paths.read_jsonl(paths.CORPUS_DIR / "chunks.jsonl")
        return cls._chunks
