"""Text-to-SQL examples: the public ``b-mc2/sql-create-context`` dataset, or a synthetic set for offline tests.

sql-create-context (CC BY 4.0) pairs a natural-language question with the CREATE TABLE
statements it refers to and a gold SQL query. It only ships a ``train`` split, so we
de-duplicate and carve deterministic train / validation / test splits ourselves.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

HF_DATASET = "b-mc2/sql-create-context"


@dataclass(frozen=True)
class Example:
    question: str
    context: str  # CREATE TABLE statements
    sql: str


def load_sql_create_context(seed: int = 42, n_val: int = 500, n_test: int = 1000) -> dict[str, list[Example]]:
    from datasets import load_dataset

    rows = load_dataset(HF_DATASET, split="train")
    seen: set[tuple[str, str]] = set()
    examples = []
    for r in rows:
        key = (r["question"].strip().lower(), r["context"].strip())
        if key in seen:
            continue
        seen.add(key)
        examples.append(Example(r["question"].strip(), r["context"].strip(), r["answer"].strip()))
    return split_examples(examples, seed, n_val, n_test)


def split_examples(examples: list[Example], seed: int, n_val: int, n_test: int) -> dict[str, list[Example]]:
    shuffled = examples[:]
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) <= n_val + n_test:
        raise ValueError("not enough examples for the requested validation and test sizes")
    return {
        "test": shuffled[:n_test],
        "validation": shuffled[n_test : n_test + n_val],
        "train": shuffled[n_test + n_val :],
    }


_TABLES = {
    "employee": [("name", "VARCHAR"), ("age", "INTEGER"), ("city", "VARCHAR"), ("salary", "INTEGER")],
    "product": [("title", "VARCHAR"), ("price", "INTEGER"), ("category", "VARCHAR"), ("stock", "INTEGER")],
    "match": [("home_team", "VARCHAR"), ("away_team", "VARCHAR"), ("score", "INTEGER"), ("year", "INTEGER")],
}
_VALUES = {"city": ["Paris", "Lagos", "Lima"], "category": ["toys", "books"], "home_team": ["Ajax", "Inter"]}


def synthetic_examples(n: int = 300, seed: int = 0) -> list[Example]:
    """Template-generated questions over three small schemas (used by tests and the CPU smoke run)."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        table = rng.choice(list(_TABLES))
