"""``sqltune <command>``: prepare data, fine-tune with LoRA, and compare against prompting baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .data import load_jsonl, load_sql_create_context, save_jsonl, split_examples, synthetic_examples
from .evaluate import results_table, run_mode, save_results
from .modeling import TrainConfig, load_tokenizer, train_lora


def cmd_prepare(args):
    splits = load_sql_create_context(seed=args.seed, n_val=args.n_val, n_test=args.n_test)
    for name, rows in splits.items():
        save_jsonl(rows, Path(args.data_dir) / f"{name}.jsonl")
        print(f"{name}: {len(rows)}")


def _splits(data_dir):
    d = Path(data_dir)
    return load_jsonl(d / "train.jsonl"), load_jsonl(d / "validation.jsonl"), load_jsonl(d / "test.jsonl")


def cmd_train(args):
    train, val, _ = _splits(args.data_dir)
    cfg = TrainConfig(
        base_model=args.base_model,
        output_dir=args.output_dir,
        max_train_examples=args.max_train_examples,
        epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    path = train_lora(cfg, train, val[: args.max_val_examples])
    print(f"Adapter saved to {path}")


def cmd_evaluate(args):
    import torch
    from transformers import AutoModelForCausalLM

    train, _, test = _splits(args.data_dir)
    test = test[: args.limit] if args.limit else test
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    tokenizer = load_tokenizer(args.base_model)
    base = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype).to(device).eval()
