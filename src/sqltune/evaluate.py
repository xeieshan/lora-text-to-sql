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
