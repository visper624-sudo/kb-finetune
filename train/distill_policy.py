# -*- coding: utf-8 -*-
"""训练轻量策略模型（DPH）：Gate / Format / Evidence Selector。

在无 GPU、无外网模型权重的环境里，这是本仓库可实际跑通、可复现实验指标的
训练路径；有 GPU 时用 `train/train_lora.py` 训练 Qwen LoRA，评测脚本不变。

用法：
    python train/distill_policy.py
    python train/distill_policy.py --epochs 120 --output adapter/policy
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import paths  # noqa: E402
from kb.compose import sentences_from_contexts  # noqa: E402
from kb.corpus import Corpus  # noqa: E402
from kb.retrieval import (  # noqa: E402
    FEATURE_NAMES, TextVectorizer, context_features, feature_vector,
)
from kb.textutil import content_terms, normalize_text, split_question_parts  # noqa: E402
from train.policy import (  # noqa: E402
    CATEGORIES, FORMAT_CLASSES, FORMAT_KEYWORDS, SELECTOR_FEATURES, ROUTE_CLASSES,
    Scaler, build_format_vocab, fit_logistic, fit_softmax, format_features,
    selector_features,
)


def _norm(text: str) -> str:
    return normalize_text(text).replace(" ", "")


def gate_matrix(samples: list[dict], vectorizer: TextVectorizer) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for s in samples:
        feats = context_features(s["question"], s["contexts"], vectorizer)
        X.append(feature_vector(feats))
        y.append(1 if s["task"] == "answer" else 0)
    return np.array(X, dtype=float), np.array(y, dtype=int)


def format_dataset(samples: list[dict], vocab: dict[str, int]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    X, y, kept = [], [], []
    for s in samples:
        if s["task"] != "answer" or s["format"] not in ROUTE_CLASSES:
            continue
        if len(split_question_parts(s["question"])) >= 2:
            continue
        category = s["contexts"][0].get("category", "") if s["contexts"] else ""
        ctx_text = s["contexts"][0].get("text", "") if s["contexts"] else ""
        X.append(format_features(s["question"], vocab, category, ctx_text))
        y.append(ROUTE_CLASSES.index(s["format"]))
        kept.append(s)
    return np.array(X, dtype=float), np.array(y, dtype=int), kept


def selector_dataset(samples: list[dict], rng: np.random.Generator,
                     max_neg: int = 8) -> tuple[np.ndarray, np.ndarray, int]:
    X, y = [], []
    for s in samples:
        if s["task"] != "answer":
            continue
        gold = {_norm(t) for t in s["meta"].get("gold_evidence", [])}
        cands = sentences_from_contexts(s["contexts"])
        if not cands:
            continue
        pos, neg = [], []
        for i, c in enumerate(cands):
            same = [x for x in cands if x["ctx_n"] == c["ctx_n"]]
            pos_in = same.index(c) if c in same else 0
            ctx = next((x for x in s["contexts"] if x.get("n") == c["ctx_n"]), {})
            row = selector_features(s["question"], c["text"], c["ctx_n"], ctx, pos_in, len(same))
            if _norm(c["text"]) in gold:
                pos.append(row)
            else:
                neg.append(row)
        for row in pos:
            X.append(row)
            y.append(1)
        rng.shuffle(neg)
        for row in neg[:max_neg]:
            X.append(row)
            y.append(0)
    return np.array(X, dtype=float), np.array(y, dtype=int), len(y)


def evaluate_gate(w: np.ndarray, b: float, scaler: Scaler, X: np.ndarray,
                  y: np.ndarray, threshold: float) -> dict:
    p = 1.0 / (1.0 + np.exp(-np.clip(scaler.transform(X) @ w + b, -30, 30)))
    pred = (p >= threshold).astype(int)
    ans = y == 1
    ref = y == 0
    return {
        "n": int(len(y)),
        "accuracy": round(float((pred == y).mean()), 4),
        "answer_recall": round(float((pred[ans] == 1).mean()), 4) if ans.any() else 0.0,
        "refusal_recall": round(float((pred[ref] == 0).mean()), 4) if ref.any() else 0.0,
        "false_refusal": round(float((pred[ans] == 0).mean()), 4) if ans.any() else 0.0,
        "false_answer": round(float((pred[ref] == 1).mean()), 4) if ref.any() else 0.0,
        "auc": round(_auc(p, y), 4),
    }


def _auc(p: np.ndarray, y: np.ndarray) -> float:
    pos = p[y == 1]
    neg = p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    # 采样近似，避免 O(n^2)
    rng = np.random.default_rng(0)
    a = rng.choice(pos, size=min(800, len(pos)), replace=False)
    b = rng.choice(neg, size=min(800, len(neg)), replace=False)
    return float(((a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum())
                 / (len(a) * len(b)))


def tune_threshold(w, b, scaler, X, y) -> tuple[float, dict]:
    p = 1.0 / (1.0 + np.exp(-np.clip(scaler.transform(X) @ w + b, -30, 30)))
    ans = y == 1
    ref = y == 0
    best_t, best = 0.5, None
    for t in np.arange(0.02, 0.99, 0.01):
        pred = (p >= t).astype(int)
        false_refusal = float((pred[ans] == 0).mean()) if ans.any() else 0.0
        if false_refusal > 0.09:
            continue
        recall = float((pred[ref] == 0).mean()) if ref.any() else 0.0
        score = 0.4 * (1 - false_refusal) + 0.6 * recall
        if best is None or score > best["score"]:
            best = {"score": round(score, 4), "false_refusal": round(false_refusal, 4),
                    "refusal_recall": round(recall, 4)}
            best_t = float(t)
    if best is None:
        best = {"score": 0.0, "false_refusal": None, "refusal_recall": None}
    return round(best_t, 2), best


def selector_eval(w, b, scaler, samples: list[dict], top_k: int = 4) -> dict:
    rec, prec, n = [], [], 0
    for s in samples:
        if s["task"] != "answer":
            continue
        gold = {_norm(t) for t in s["meta"].get("gold_evidence", [])}
        cands = sentences_from_contexts(s["contexts"])
        if not cands or not gold:
            continue
        rows = []
        for i, c in enumerate(cands):
            same = [x for x in cands if x["ctx_n"] == c["ctx_n"]]
            ctx = next((x for x in s["contexts"] if x.get("n") == c["ctx_n"]), {})
            rows.append(selector_features(s["question"], c["text"], c["ctx_n"], ctx,
                                          same.index(c) if c in same else 0, len(same)))
        p = 1.0 / (1.0 + np.exp(-np.clip(scaler.transform(np.array(rows)) @ w + b, -30, 30)))
        order = np.argsort(-p)[:top_k]
        picked = {_norm(cands[int(i)]["text"]) for i in order}
        hit = len(gold & picked)
        rec.append(hit / max(1, len(gold)))
        prec.append(hit / max(1, len(picked)))
        n += 1
    return {"n": n, "evidence_recall": round(float(np.mean(rec)), 4) if rec else 0.0,
            "evidence_precision": round(float(np.mean(prec)), 4) if prec else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=220)
    ap.add_argument("--output", default=str(paths.ADAPTER_DIR / "policy"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    train = paths.read_jsonl(paths.OUT_DIR / "train.jsonl")
    valid = paths.read_jsonl(paths.OUT_DIR / "valid.jsonl")
    if not train:
        raise SystemExit("请先运行 python data/build_dataset.py")
    corpus = Corpus.load()
    vectorizer = TextVectorizer([c["content"] for c in corpus.chunks])

    # ---------- Gate
    Xg, yg = gate_matrix(train, vectorizer)
    Xgv, ygv = gate_matrix(valid, vectorizer)
    g_scaler = Scaler().fit(Xg)
    gw, gb, glog = fit_logistic(g_scaler.transform(Xg), yg, epochs=args.epochs,
                                 lr=0.5, l2=2e-3, seed=args.seed)
    threshold, tune = tune_threshold(gw, gb, g_scaler, Xgv, ygv)
    gate_metrics = {
        "train": evaluate_gate(gw, gb, g_scaler, Xg, yg, threshold),
        "valid": evaluate_gate(gw, gb, g_scaler, Xgv, ygv, threshold),
        "threshold": threshold, "tune": tune,
    }

    # ---------- Format
    vocab = build_format_vocab(
        [s["question"] + " " + (s["contexts"][0].get("text", "")[:160] if s["contexts"] else "")
         for s in train], limit=700)
    Xf, yf, _ = format_dataset(train, vocab)
    Xfv, yfv, _ = format_dataset(valid, vocab)
    f_scaler = Scaler().fit(Xf)
    fw, fb, flog = fit_softmax(f_scaler.transform(Xf), yf, len(FORMAT_CLASSES),
                               epochs=max(120, args.epochs), lr=0.5, l2=1e-3, seed=args.seed)
    f_pred = (f_scaler.transform(Xfv) @ fw.T + fb).argmax(axis=1)
    fmt_metrics = {
        "n_train": int(len(yf)), "n_valid": int(len(yfv)),
        "valid_accuracy": round(float((f_pred == yfv).mean()), 4),
        "per_class": {
            ROUTE_CLASSES[i]: {
                "support": int((yfv == i).sum()),
                "recall": round(float((f_pred[yfv == i] == i).mean()), 4) if (yfv == i).any() else 0.0,
            } for i in range(len(ROUTE_CLASSES))
        },
    }

    # ---------- Selector
    Xs, ys, n_sel = selector_dataset(train, rng, max_neg=8)
    s_scaler = Scaler().fit(Xs)
    sw, sb, slog = fit_logistic(s_scaler.transform(Xs), ys, epochs=max(180, args.epochs),
                                lr=0.5, l2=1e-3, seed=args.seed)
    sel_metrics = {
        "rows": int(n_sel), "positives": int(ys.sum()),
        "valid": selector_eval(sw, sb, s_scaler, valid),
    }

    # ---------- 无召回阈值
    no_recall_sims = []
    for s in train:
        if s["meta"].get("negative_type") == "no_recall":
            feats = context_features(s["question"], s["contexts"], vectorizer)
            no_recall_sims.append(feats["top1_sim"])
    none_thr = float(np.percentile(no_recall_sims, 90)) if no_recall_sims else 0.05
    none_thr = round(max(0.01, min(0.15, none_thr)), 4)

    # ---------- 保存
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "model.npz",
        gate_w=gw, gate_b=np.float64(gb), gate_mean=g_scaler.mean, gate_std=g_scaler.std,
        fmt_w=fw, fmt_b=fb, fmt_mean=f_scaler.mean, fmt_std=f_scaler.std,
        sel_w=sw, sel_b=np.float64(sb), sel_mean=s_scaler.mean, sel_std=s_scaler.std,
    )
    meta = {
        "version": "dph-1.0",
        "created_at": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "feature_names": FEATURE_NAMES,
        "selector_features": SELECTOR_FEATURES,
        "format_keywords": FORMAT_KEYWORDS,
        "categories": CATEGORIES,
        "format_classes": list(FORMAT_CLASSES),
        "format_vocab": vocab,
        "gate_threshold": threshold,
        "none_sim_threshold": none_thr,
        "gate_metrics": gate_metrics,
        "format_metrics": fmt_metrics,
        "selector_metrics": sel_metrics,
        "train_rows": {"gate": int(len(yg)), "format": int(len(yf)), "selector": int(n_sel)},
        "output": str(out),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log = {
        "gate_loss": glog, "format_loss": flog, "selector_loss": slog,
        "gate_metrics": gate_metrics, "format_metrics": fmt_metrics,
        "selector_metrics": sel_metrics,
    }
    (paths.TRAIN_DIR / "train_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    print("gate:", json.dumps(gate_metrics["valid"], ensure_ascii=False))
    print("threshold:", threshold, tune)
    print("format valid acc:", fmt_metrics["valid_accuracy"])
    print("selector:", json.dumps(sel_metrics, ensure_ascii=False))
    print("saved:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
