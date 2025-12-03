import json

import pytest

from sqltune.cli import main
from sqltune.data import Example, synthetic_examples
from sqltune.modeling import PadCollator, tokenize_example
from sqltune.prompts import format_prompt
from sqltune.smoke import build_tokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return build_tokenizer(synthetic_examples(200, seed=0))


def test_labels_mask_the_prompt(tokenizer):
    ex = Example(
        "How many employee rows are there?", "CREATE TABLE employee (name VARCHAR)", "SELECT COUNT(*) FROM employee"
    )
    row = tokenize_example(tokenizer, ex, max_length=256)
    n_prompt = len(tokenizer(format_prompt(ex), add_special_tokens=False)["input_ids"])
    assert row["labels"][:n_prompt] == [-100] * n_prompt
    assert row["labels"][n_prompt:] == row["input_ids"][n_prompt:]
    assert row["input_ids"][-1] == tokenizer.eos_token_id
    answer = tokenizer.decode(row["input_ids"][n_prompt:-1])
    assert answer.strip() == ex.sql


def test_long_examples_are_dropped(tokenizer):
    ex = Example("q " * 400, "CREATE TABLE t (a INTEGER)", "SELECT a FROM t")
    assert tokenize_example(tokenizer, ex, max_length=64) is None


def test_collator_pads(tokenizer):
    rows = [
        {"input_ids": [5, 6, 7], "attention_mask": [1, 1, 1], "labels": [-100, 6, 7]},
        {"input_ids": [5], "attention_mask": [1], "labels": [5]},
    ]
    batch = PadCollator(pad_id=1)(rows)
    assert batch["input_ids"].tolist() == [[5, 6, 7], [5, 1, 1]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert batch["labels"].tolist() == [[-100, 6, 7], [5, -100, -100]]


def test_smoke_run_end_to_end(tmp_path):
    """Tiny model: LoRA train -> save adapter -> reload -> generate -> score, all on CPU."""
    main(["smoke", "--output-dir", str(tmp_path), "--steps", "60"])
    summary = json.loads((tmp_path / "train_summary.json").read_text())
    assert (tmp_path / "adapter" / "adapter_config.json").exists()
    assert summary["steps"] == 60
    losses = [h["loss"] for h in summary["log_history"] if "loss" in h]
    assert losses[-1] < losses[0]


def test_cli_train_and_evaluate_with_a_local_model(tmp_path, tokenizer):
    """Exercises the real `train` and `evaluate` commands (from_pretrained on a local path)."""
    from sqltune.data import save_jsonl, split_examples
    from sqltune.smoke import build_tiny_model

    model_dir = tmp_path / "tiny-base"
    build_tiny_model(len(tokenizer)).save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    splits = split_examples(synthetic_examples(200, seed=2), seed=0, n_val=16, n_test=16)
    for name, rows in splits.items():
        save_jsonl(rows, tmp_path / "data" / f"{name}.jsonl")

    common = ["--data-dir", str(tmp_path / "data"), "--base-model", str(model_dir)]
    main(["train", *common, "--output-dir", str(tmp_path / "out"), "--max-steps", "20", "--batch-size", "8"])
    main(
        [
            "evaluate",
            *common,
            "--adapter",
            str(tmp_path / "out" / "adapter"),
            "--out-dir",
            str(tmp_path / "res"),
            "--batch-size",
            "8",
        ]
    )
    results = json.loads((tmp_path / "res" / "comparison.json").read_text())
    assert set(results) == {"zero_shot", "few_shot", "lora"}
    assert all(r["n"] == 16 for r in results.values())
    assert (tmp_path / "res" / "comparison.md").read_text().startswith("| Mode |")
