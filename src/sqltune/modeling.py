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

        return {
            "input_ids": torch.tensor([pad(b["input_ids"], self.pad_id) for b in batch]),
            "attention_mask": torch.tensor([pad(b["attention_mask"], 0) for b in batch]),
            "labels": torch.tensor([pad(b["labels"], -100) for b in batch]),
        }


def load_tokenizer(name_or_path: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name_or_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def _precision_flags() -> dict[str, bool]:
    if not torch.cuda.is_available():
        return {"bf16": False, "fp16": False}
    bf16 = torch.cuda.is_bf16_supported()
    return {"bf16": bf16, "fp16": not bf16}


def train_lora(cfg: TrainConfig, train: list[Example], val: list[Example], model=None, tokenizer=None) -> Path:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, set_seed

    set_seed(cfg.seed)
    tokenizer = tokenizer or load_tokenizer(cfg.base_model)
    if model is None:
        dtype = torch.bfloat16 if _precision_flags()["bf16"] else torch.float32
        model = AutoModelForCausalLM.from_pretrained(cfg.base_model, dtype=dtype)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    trainable, total = model.get_nb_trainable_parameters()

    def encode(examples):
        rows = [tokenize_example(tokenizer, ex, cfg.max_length) for ex in examples]
        return [r for r in rows if r is not None]

    train_rows = encode(train[: cfg.max_train_examples])
    val_rows = encode(val)
    steps_per_epoch = math.ceil(len(train_rows) / (cfg.batch_size * cfg.grad_accum))
    total_steps = cfg.max_steps if cfg.max_steps > 0 else math.ceil(steps_per_epoch * cfg.epochs)

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
