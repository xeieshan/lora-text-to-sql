"""Tokenisation with prompt masking, LoRA training, and batched greedy generation."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from .data import Example
from .prompts import extract_sql, format_full, format_prompt

DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class TrainConfig:
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    output_dir: str = "outputs/lora"
    max_train_examples: int = 10_000
    max_length: int = 384
    epochs: float = 1.0
    max_steps: int = -1
    learning_rate: float = 2e-4
    batch_size: int = 16
    grad_accum: int = 1
    warmup_fraction: float = 0.03
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: list(DEFAULT_TARGET_MODULES))
    seed: int = 42
    gradient_checkpointing: bool = False


def tokenize_example(tokenizer, ex: Example, max_length: int) -> dict[str, list[int]] | None:
    """Labels are -100 on the prompt, so the loss only covers the SQL answer (plus EOS)."""
    prompt_ids = tokenizer(format_prompt(ex), add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(format_full(ex), add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        # Tokenizer merged across the prompt/answer boundary; fall back to concatenation.
        answer_ids = tokenizer(ex.sql, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        full_ids = prompt_ids + answer_ids
    if len(full_ids) > max_length:
        return None  # drop rather than truncate the answer
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


class PadCollator:
    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        width = max(len(b["input_ids"]) for b in batch)

        def pad(seq, value):
            return seq + [value] * (width - len(seq))

