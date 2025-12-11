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

    results = {}
    for mode in args.modes:
        if mode == "lora":
            continue
        print(f"evaluating {mode} ...")
        results[mode] = run_mode(base, tokenizer, test, mode, train, args.batch_size)
    if "lora" in args.modes:
        from peft import PeftModel

        model = PeftModel.from_pretrained(base, args.adapter).merge_and_unload().eval()
        print("evaluating lora ...")
        results["lora"] = run_mode(model, tokenizer, test, "lora", train, args.batch_size)
    print(results_table(results))
    print(f"wrote {save_results(results, args.out_dir, args.tag)}")


def cmd_smoke(args):
    """End-to-end CPU run on synthetic data with a tiny random model. Checks plumbing, not quality."""
    from .smoke import build_tiny_model, build_tokenizer

    examples = synthetic_examples(240, seed=0)
    splits = split_examples(examples, seed=0, n_val=20, n_test=20)
    tokenizer = build_tokenizer(examples)
    model = build_tiny_model(len(tokenizer))
    cfg = TrainConfig(
        base_model="tiny-random",
        output_dir=args.output_dir,
        max_length=128,
        max_steps=args.steps,
        batch_size=8,
        learning_rate=5e-3,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.0,
    )
    adapter = train_lora(cfg, splits["train"], splits["validation"], model=model, tokenizer=tokenizer)

    from peft import PeftModel

    reloaded = PeftModel.from_pretrained(build_tiny_model(len(tokenizer)), adapter).merge_and_unload().eval()
    result = run_mode(reloaded, tokenizer, splits["test"], "lora", splits["train"], batch_size=8, max_new_tokens=24)
    summary = json.loads((Path(args.output_dir) / "train_summary.json").read_text())
    print(
        json.dumps(
            {
                "steps": summary["steps"],
                "trainable_pct": summary["trainable_pct"],
                "first_loss": summary["log_history"][0].get("loss"),
                "final_eval_loss": summary["final_eval_loss"],
                "test_n": result["n"],
                "valid_sql": result["valid_sql"],
                "exact_match": result["exact_match"],
                "sample_predictions": [{"gold": r["gold"], "pred": r["pred"]} for r in result["per_example"][:3]],
            },
