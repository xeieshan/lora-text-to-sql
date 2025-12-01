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
