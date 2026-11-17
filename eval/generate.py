# -*- coding: utf-8 -*-
"""评测用的回答生成器。

配置说明（对应难度方案 §八.5 的三组对比 + 两个消融）：
  * baseline        ：基座模型式 RAG 摘要——检索后直接拼句子，不训练、不引用、无格式；
  * dpf             ：本仓库训练的策略模型 + 组装器（本机可跑）；
  * ablation_no_refusal ：去掉拒答训练（Gate 恒答），验证“拒答来自数据”；
  * ablation_no_citation：回答保留内容但删除引用，验证引用指标；
  * ablation_no_retrieval：不检索，直接复述训练集里最相似问题的答案（参数记忆）；
  * lora            ：有 GPU 时加载 Qwen LoRA adapter，与 dpf 走同一评测。
"""
from __future__ import annotations

import math
import re
from collections import Counter

from config import paths
from config.prompts import build_messages
from kb.compose import sentences_from_contexts
from kb.textutil import content_terms, jaccard, sentence_similarity, tokenize
from train.policy import FineTunedPolicy


class BaselineGenerator:
    name = "baseline"
    label = "基座模型（检索后直答，无训练）"

    def answer(self, sample: dict) -> str:
        question = sample["question"]
        contexts = sample.get("contexts", [])
        if not contexts:
            return "公司制度中暂无该事项的明确说明，建议咨询相关业务部门。"
        cands = sentences_from_contexts(contexts)
        cands = [c for c in cands if len(c["text"]) >= 12 and not c["text"].endswith(("？", "?"))]
        if not cands:
            return "根据检索到的资料，建议咨询相关业务部门确认。"
        cands.sort(key=lambda c: -sentence_similarity(question, c["text"]))
        picked, seen = [], set()
        for c in cands:
            if c["text"][:24] in seen:
                continue
            seen.add(c["text"][:24])
            picked.append(c["text"])
            if len(picked) >= 3:
                break
        return "根据检索到的资料：" + "".join(picked)


class DPFGenerator:
    name = "dpf"
    label = "策略模型 DPH（本机实测）"

    def __init__(self, policy: FineTunedPolicy):
        self.policy = policy

    def answer(self, sample: dict) -> str:
        return self.policy.predict(sample["question"], sample.get("contexts", []))["answer"]


class NoRefusalGenerator:
    name = "ablation_no_refusal"
    label = "消融：去掉拒答训练（Gate 恒答）"

    def __init__(self, policy: FineTunedPolicy):
        self.policy = policy

    def answer(self, sample: dict) -> str:
        return self.policy.predict(sample["question"], sample.get("contexts", []),
                                   force_answer=True)["answer"]


class NoCitationGenerator:
    name = "ablation_no_citation"
    label = "消融：回答不带引用编号"

    def __init__(self, policy: FineTunedPolicy):
        self.policy = policy

    def answer(self, sample: dict) -> str:
        return self.policy.predict(sample["question"], sample.get("contexts", []),
                                   with_citations=False)["answer"]


class MemoryGenerator:
    name = "ablation_no_retrieval"
    label = "消融：不检索（参数记忆最近邻）"

    def __init__(self):
        rows = [r for r in paths.read_jsonl(paths.OUT_DIR / "train.jsonl")
                if r.get("task") == "answer"]
        self.questions = [r["question"] for r in rows]
        self.answers = [re.sub(r"\s*\[\d+\]", "", r["answer"]) for r in rows]
        self.tokens = [content_terms(q) for q in self.questions]

    def answer(self, sample: dict) -> str:
        q = content_terms(sample["question"])
        best_i, best = -1, 0.0
        for i, t in enumerate(self.tokens):
            s = len(q & t) / max(1, len(q | t))
            if s > best:
                best, best_i = s, i
        if best_i < 0 or best < 0.12:
            return "根据一般行业惯例，公司应参照相关制度执行，建议咨询业务部门。"
        return self.answers[best_i]


class LoRAGenerator:
    """GPU 路径：加载 base + LoRA adapter，使用与训练一致的 system/user prompt。"""
    name = "lora"
    label = "Qwen LoRA（GPU）"

    def __init__(self, model: str, adapter: str | None = None,
                 max_new_tokens: int = 512, load_in_4bit: bool = False):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"加载 LoRA 需要 transformers/torch：{exc}")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        kwargs = {"trust_remote_code": True, "device_map": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
        else:
            kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model_obj = AutoModelForCausalLM.from_pretrained(model, **kwargs)
        if adapter:
            from peft import PeftModel
            model_obj = PeftModel.from_pretrained(model_obj, adapter)
        self.model = model_obj.eval()
        self.max_new_tokens = max_new_tokens

    def answer(self, sample: dict) -> str:
        messages = build_messages(sample["question"], sample.get("contexts", []))
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        device = next(self.model.parameters()).device
        inputs = self.tokenizer([text], return_tensors="pt").to(device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                      do_sample=False, temperature=None, top_p=None,
                                      pad_token_id=self.tokenizer.eos_token_id)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()


def build_generators(names: list[str], policy: FineTunedPolicy,
                     model: str = "Qwen/Qwen2.5-1.5B-Instruct",
                     adapter: str | None = None,
                     max_new_tokens: int = 512,
                     load_in_4bit: bool = False) -> list:
    gens = []
    for name in names:
        if name == "baseline":
            gens.append(BaselineGenerator())
        elif name == "dpf":
            gens.append(DPFGenerator(policy))
        elif name == "ablation_no_refusal":
            gens.append(NoRefusalGenerator(policy))
        elif name == "ablation_no_citation":
            gens.append(NoCitationGenerator(policy))
        elif name == "ablation_no_retrieval":
            gens.append(MemoryGenerator())
        elif name == "lora":
            gens.append(LoRAGenerator(model=model, adapter=adapter,
                                      max_new_tokens=max_new_tokens,
                                      load_in_4bit=load_in_4bit))
        else:
            raise SystemExit(f"未知生成器：{name}")
    return gens
