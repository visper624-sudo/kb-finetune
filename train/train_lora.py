# -*- coding: utf-8 -*-
"""Qwen LoRA SFT 训练脚本（GPU 路径）。

数据：`data/out/train.jsonl` / `valid.jsonl`，字段 messages 使用与推理完全一致的
system prompt（见 config/prompts.py）。

用法（单卡 24GB 可跑 7B QLoRA；8GB 建议 1.5B + cutoff_len=1024）：
    pip install -r train/requirements-gpu.txt
    python train/train_lora.py --model Qwen/Qwen2.5-1.5B-Instruct \
        --output adapter/lora --epochs 2 --cutoff-len 2048 --qlora

先做格式校验（不需要 GPU / transformers）：
    python train/train_lora.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import paths  # noqa: E402
from kb.textutil import estimate_tokens  # noqa: E402


def load_rows(split: str) -> list[dict]:
    return paths.read_jsonl(paths.OUT_DIR / f"{split}.jsonl")


def dry_run(args) -> int:
    rows = load_rows("train")
    valid = load_rows("valid")
    if not rows:
        raise SystemExit("训练集为空：请先运行 python data/build_dataset.py")
    lengths = []
    for r in rows[: min(500, len(rows))]:
        text = "".join(m["content"] for m in r["messages"])
        lengths.append(estimate_tokens(text))
    lengths.sort()
    over = sum(1 for x in lengths if x > args.cutoff_len)
    print(f"train={len(rows)} valid={len(valid)}")
    print(f"估算 token：P50={lengths[len(lengths)//2]} P95={lengths[int(len(lengths)*0.95)]} "
          f"max={lengths[-1]} 超过 cutoff({args.cutoff_len}) 的比例={over/len(lengths):.1%}")
    print("messages 字段校验：", all(set(m) == {"role", "content"} for r in rows[:20] for m in r["messages"]))
    print("样例 assistant：", rows[0]["messages"][-1]["content"][:80].replace("\n", " "))
    print("dry-run OK（真正训练请安装 requirements-gpu.txt 并去掉 --dry-run）")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--output", default=str(paths.ADAPTER_DIR / "lora"))
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--valid-split", default="valid")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--cutoff-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--qlora", action="store_true")
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--log-steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.dry_run:
        return dry_run(args)

    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  BitsAndBytesConfig, get_cosine_schedule_with_warmup)
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "缺少 GPU 训练依赖。请先执行：\n"
            "  pip install -r train/requirements-gpu.txt\n"
            f"原始错误：{exc}"
        )

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision,
                                              trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16 if args.fp16 else torch.bfloat16,
    }
    if args.qlora:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16 if args.fp16 else torch.bfloat16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, **model_kwargs)
    if args.qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM", target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    class ChatDataset(Dataset):
        def __init__(self, rows: list[dict]):
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, i: int) -> dict:
            msgs = self.rows[i]["messages"]
            prompt_text = tokenizer.apply_chat_template(
                msgs[:-1], tokenize=False, add_generation_prompt=True)
            full_text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False)
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"][: args.cutoff_len]
            labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids):]
            labels = labels[: len(full_ids)]
            return {"input_ids": full_ids, "labels": labels,
                    "attention_mask": [1] * len(full_ids)}

    def collate(batch: list[dict]) -> dict:
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            pad = maxlen - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [tokenizer.pad_token_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append(b["attention_mask"] + [0] * pad)
        return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attn)}

    train_rows = load_rows(args.train_split)
    valid_rows = load_rows(args.valid_split)
    if not train_rows:
        raise SystemExit("训练集为空")
    train_loader = DataLoader(ChatDataset(train_rows), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate, num_workers=0)
    valid_loader = DataLoader(ChatDataset(valid_rows[: min(200, len(valid_rows))]),
                              batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.qlora:
        device = "cuda"
    else:
        model.to(device)
    total_steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    total_steps = int(total_steps)
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    log: list[dict] = []
    global_step, epoch_done = 0, 0.0
    model.train()
    for epoch in range(int(math.ceil(args.epochs))):
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % args.log_steps == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    log.append({"step": global_step, "epoch": epoch + step / len(train_loader),
                                "loss": round(float(out.loss.detach()), 6), "lr": lr_now})
                    print(f"step {global_step}/{total_steps} loss {float(out.loss):.4f} lr {lr_now:.2e}")
                if global_step % args.save_steps == 0:
                    model.save_pretrained(out_dir / f"checkpoint-{global_step}")
                if args.max_steps > 0 and global_step >= args.max_steps:
                    break
        epoch_done = epoch + 1
        if global_step >= total_steps:
            break

    # ---------- 验证 loss
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in valid_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            losses.append(float(model(**batch).loss))
    valid_loss = sum(losses) / max(1, len(losses))
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    summary = {
        "base_model": args.model, "adapter": str(out_dir),
        "epochs": args.epochs, "total_steps": total_steps, "valid_loss": round(valid_loss, 6),
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout},
        "cutoff_len": args.cutoff_len, "qlora": args.qlora,
        "train_rows": len(train_rows), "valid_rows": len(valid_rows),
    }
    (out_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (paths.TRAIN_DIR / "lora_train_log.json").write_text(
        json.dumps({"log": log, "summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("训练完成：", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
