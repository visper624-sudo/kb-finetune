# -*- coding: utf-8 -*-
"""检索与上下文特征：BM25 + signed-hashing 向量 + RRF 融合。

与 2.9 知识库平台的检索口径保持一致（中文二元组、512 维哈希向量、RRF k=60），
但这里只依赖标准库，便于训练/评测/演示三处复用。

除了“召回”，本模块还负责把一组检索片段转换为**策略模型特征**：
  top1/top2 相似度、分数间隔、词覆盖、冲突标记、问题类型标记等。
"""
from __future__ import annotations

import math
import re
from collections import Counter

from kb.textutil import (
    content_terms, extract_numbers, jaccard, normalize_text, overlap_ratio,
    split_question_parts, tokenize,
)

RRF_K = 60
VEC_DIM = 512
VECTOR_TOP_K = 20
KEYWORD_TOP_K = 10

_FAULT_HINTS = ("故障", "报错", "无法", "失败", "异常", "打不开", "连不上", "坏了", "怎么办")
_FLOW_HINTS = ("流程", "步骤", "怎么操作", "如何办理", "怎么申请", "怎么走", "手续")
_AMOUNT_HINTS = ("多少", "标准", "限额", "比例", "上限", "额度", "费用", "价格")
_POLICY_HINTS = ("规定", "要求", "是否", "可以", "允许", "禁止", "必须", "需要")


def _unused_placeholder() -> None:  # 保留模块级占位，避免旧版本导入报错
    return None


class TextVectorizer:
    """IDF 加权稀疏词项向量（中文二元组）。

    与 2.9 的 signed-hashing 向量同源（都是「中文二元组 + 归一化余弦」），
    但离线评测/特征计算改用**精确稀疏余弦**：57 个切片的语料下哈希碰撞会把
    相关与无关片段的相似度压到同一区间（实测均约 0.1），不适合作为阈值特征。
    两种实现可互相替换，检索排序仍由 BM25 + RRF 主导。
    """

    def __init__(self, corpus_texts: list[str]):
        df: Counter = Counter()
        for t in corpus_texts:
            df.update(set(tokenize(t)))
        self.N = max(1, len(corpus_texts))
        self.idf = {
            tok: math.log(1 + (self.N - n + 0.5) / (n + 0.5))
            for tok, n in df.items()
        }
        self.default_idf = math.log(1 + (self.N + 0.5) / 0.5)
        self._cache: dict[str, dict[str, float]] = {}

    def vector(self, text: str) -> dict[str, float]:
        key = text or ""
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        vec: dict[str, float] = {}
        for tok, tf in Counter(tokenize(key)).items():
            w = self.idf.get(tok, self.default_idf) * (1.0 + math.log(tf))
            vec[tok] = vec.get(tok, 0.0) + w
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm > 1e-9:
            vec = {t: v / norm for t, v in vec.items()}
        self._cache[key] = vec
        return vec

    @staticmethod
    def cosine(a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        if len(a) > len(b):
            a, b = b, a
        return float(sum(v * b.get(t, 0.0) for t, v in a.items()))

    def similarity(self, query: str, text: str) -> float:
        return self.cosine(self.vector(query), self.vector(text))


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.avgdl = sum(len(d) for d in docs) / max(1, self.N)
        self.tf: list[Counter] = [Counter(d) for d in docs]
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {
            t: math.log(1 + (self.N - n + 0.5) / (n + 0.5))
            for t, n in df.items()
        }

    def scores(self, query: list[str]) -> list[float]:
        q = [t for t in query if t in self.idf]
        if not q:
            return [0.0] * self.N
        out = []
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i])
            s = 0.0
            for t in q:
                f = tf.get(t, 0)
                if not f:
                    continue
                s += self.idf[t] * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * dl / max(1.0, self.avgdl))
                )
            out.append(s)
        return out


class Retriever:
    def __init__(self, chunks: list[dict], dim: int = VEC_DIM):
        self.chunks = chunks
        self.vectorizer = TextVectorizer([c.get("content", "") for c in chunks])
        self.hasher = self.vectorizer          # 兼容旧命名
        self._corpus_tokens = [
            tokenize(f"{c.get('title', '')} {c.get('heading_path', '')} {c.get('content', '')}")
            for c in chunks
        ]
        self.bm25 = BM25(self._corpus_tokens)
        self._vecs = [self.vectorizer.vector(c.get("content", "")) for c in chunks]

    # ---------------------------------------------------------------- 检索
    def search(self, query: str, top_k: int = 6,
               allowed_doc_ids: set[str] | None = None) -> list[dict]:
        q_tokens = tokenize(query)
        if not q_tokens or not self.chunks:
            return []
        q_vec = self.vectorizer.vector(query)
        vec_scores = [self.vectorizer.cosine(q_vec, v) for v in self._vecs]
        bm_scores = self.bm25.scores(q_tokens)

        vec_rank = sorted(range(len(self.chunks)), key=lambda i: -vec_scores[i])
        bm_rank = sorted(range(len(self.chunks)), key=lambda i: -bm_scores[i])
        vec_pos = {i: r for r, i in enumerate(vec_rank[: max(VECTOR_TOP_K, top_k * 3)])}
        bm_pos = {i: r for r, i in enumerate(bm_rank[: max(KEYWORD_TOP_K, top_k * 2)])}

        cands = set(vec_pos) | set(bm_pos)
        scored = []
        for i in cands:
            if allowed_doc_ids is not None and self.chunks[i]["doc_id"] not in allowed_doc_ids:
                continue
            rrf = 0.0
            if i in vec_pos:
                rrf += 1.0 / (RRF_K + vec_pos[i] + 1)
            if i in bm_pos:
                rrf += 1.0 / (RRF_K + bm_pos[i] + 1)
            scored.append((rrf, i))
        scored.sort(key=lambda x: -x[0])
        out = []
        for rrf, i in scored[:top_k]:
            c = dict(self.chunks[i])
            c["sim"] = round(vec_scores[i], 4)
            c["bm25"] = round(bm_scores[i], 4)
            c["rrf"] = round(rrf, 6)
            c["rank"] = len(out) + 1
            out.append(c)
        return out

    def best_similarity(self, query: str, text: str) -> float:
        return self.vectorizer.similarity(query, text)


def _char_ngrams(text: str, n: int = 4) -> set[str]:
    t = re.sub(r"\s+", "", text or "")
    return {t[i:i + n] for i in range(max(0, len(t) - n + 1))}


def _conflict_flag(contexts: list[dict]) -> bool:
    """两个**高度相似**的片段给出了同一单位的两个不同数值 → 冲突召回。

    这里用字符 4-gram 相似度而不是词项 Jaccard：多片段整合里的不同制度
    经常共享“报销/审批”等词，但文本差异很大，不能判为冲突。
    """
    if len(contexts) < 2:
        return False
    facts = [extract_numbers(c.get("text", ""), require_unit=True) for c in contexts]
    grams = [_char_ngrams(c.get("text", "")) for c in contexts]
    for i in range(len(contexts)):
        for j in range(i + 1, len(contexts)):
            if jaccard(grams[i], grams[j]) < 0.4:
                continue
            by_unit: dict[str, set[str]] = {}
            for f in facts[i]:
                by_unit.setdefault(f["unit"], set()).add(f["value"])
            for f in facts[j]:
                if by_unit.get(f["unit"]) and f["value"] not in by_unit[f["unit"]]:
                    return True
    return False


def context_features(question: str, contexts: list[dict],
                     vectorizer: TextVectorizer | None = None) -> dict:
    """把（问题, 检索片段）转成策略模型可用的低维特征。"""
    vectorizer = vectorizer or TextVectorizer([c.get("text", "") or c.get("content", "") for c in contexts])
    q = normalize_text(question)
    q_terms = content_terms(q)
    q_vec = vectorizer.vector(q)

    sims, covs, title_hits = [], [], []
    for c in contexts:
        text = c.get("text") or c.get("content") or ""
        sims.append(vectorizer.cosine(q_vec, vectorizer.vector(text)))
        covs.append(overlap_ratio(q_terms, content_terms(text)))
        title_terms = content_terms(f"{c.get('title', '')} {c.get('heading_path', '')}")
        title_hits.append(overlap_ratio(q_terms, title_terms) if title_terms else 0.0)

    sims_sorted = sorted(sims, reverse=True)
    top1 = sims_sorted[0] if sims_sorted else 0.0
    top2 = sims_sorted[1] if len(sims_sorted) > 1 else 0.0
    mean3 = sum(sims_sorted[:3]) / max(1, len(sims_sorted[:3]))
    n_nums = len(extract_numbers(q, require_unit=False))
    parts = split_question_parts(question)
    q_tokens = set(tokenize(q))
    oov = [t for t in q_tokens if t not in vectorizer.idf]
    q_ct = content_terms(q)
    ctx_ct: set[str] = set()
    for c in contexts:
        ctx_ct |= content_terms(c.get("text") or c.get("content") or "")
    missing_terms = q_ct - ctx_ct
    part_covs = []
    for part in parts:
        pt = content_terms(part)
        best = 0.0
        for c in contexts:
            best = max(best, overlap_ratio(pt, content_terms(c.get("text") or c.get("content") or "")))
        part_covs.append(best)
    feats = {
        "n_ctx": len(contexts),
        "top1_sim": round(top1, 4),
        "top2_sim": round(top2, 4),
        "sim_gap": round(top1 - top2, 4),
        "mean3_sim": round(mean3, 4),
        "max_cov": round(max(covs), 4) if covs else 0.0,
        "top1_cov": round(covs[0], 4) if covs else 0.0,
        "max_title_hit": round(max(title_hits), 4) if title_hits else 0.0,
        "n_above": sum(1 for s in sims if s >= 0.25),
        "conflict": 1.0 if _conflict_flag(contexts) else 0.0,
        "q_len": len(re.sub(r"\s", "", question)),
        "q_nums": n_nums,
        "n_parts": len(parts),
        "part_min_cov": round(min(part_covs), 4) if part_covs else 0.0,
        "part_max_cov": round(max(part_covs), 4) if part_covs else 0.0,
        "part_gap": round((max(part_covs) - min(part_covs)), 4) if part_covs else 0.0,
        "oov_ratio": round(len(oov) / max(1, len(q_tokens)), 4),
        "oov_count": len(oov),
        "missing_term_count": len(missing_terms),
        "missing_term_ratio": round(len(missing_terms) / max(1, len(q_ct)), 4),
        "is_fault": 1.0 if any(h in question for h in _FAULT_HINTS) else 0.0,
        "is_flow": 1.0 if any(h in question for h in _FLOW_HINTS) else 0.0,
        "is_amount": 1.0 if any(h in question for h in _AMOUNT_HINTS) else 0.0,
        "is_policy": 1.0 if any(h in question for h in _POLICY_HINTS) else 0.0,
    }
    return feats


FEATURE_NAMES = [
    "n_ctx", "top1_sim", "top2_sim", "sim_gap", "mean3_sim", "max_cov",
    "top1_cov", "max_title_hit", "n_above", "conflict", "q_len", "q_nums",
    "n_parts", "part_min_cov", "part_max_cov", "part_gap",
    "oov_ratio", "oov_count", "missing_term_count", "missing_term_ratio",
    "is_fault", "is_flow", "is_amount", "is_policy",
]


def feature_vector(feats: dict) -> list[float]:
    return [float(feats.get(k, 0.0)) for k in FEATURE_NAMES]
