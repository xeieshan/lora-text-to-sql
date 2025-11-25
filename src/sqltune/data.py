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
