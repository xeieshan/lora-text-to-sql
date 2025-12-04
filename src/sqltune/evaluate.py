"""Compare zero-shot, few-shot and LoRA fine-tuned generation on the same test examples."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

from .data import Example
from .prompts import few_shot_prefix, format_prompt
from .sqlmetrics import execution_match, is_valid, normalized_exact_match


def score(examples: list[Example], predictions: list[str]) -> dict:
    per = []
    for ex, pred in zip(examples, predictions, strict=True):
        per.append(
            {
                "question": ex.question,
                "gold": ex.sql,
                "pred": pred,
                "exact": normalized_exact_match(ex.sql, pred),
                "exec": execution_match(ex.context, ex.sql, pred),
                "valid": is_valid(ex.context, pred),
            }
        )
    n = len(per)
    return {
        "n": n,
        "exact_match": round(sum(r["exact"] for r in per) / n, 4),
        "execution_match": round(sum(r["exec"] for r in per) / n, 4),
        "valid_sql": round(sum(r["valid"] for r in per) / n, 4),
        "per_example": per,
    }


def build_prompts(test: list[Example], mode: str, shots_pool: list[Example], n_shots: int = 3, seed: int = 0):
    if mode in ("zero_shot", "lora"):
        return [format_prompt(ex) for ex in test]
    if mode == "few_shot":
        rng = random.Random(seed)
        shots = rng.sample(shots_pool, n_shots)
        prefix = few_shot_prefix(shots)
        return [prefix + format_prompt(ex) for ex in test]
    raise ValueError(f"unknown mode {mode}")


def run_mode(model, tokenizer, test, mode, shots_pool, batch_size=16, max_new_tokens=128) -> dict:
    from .modeling import generate_sql

    prompts = build_prompts(test, mode, shots_pool)
