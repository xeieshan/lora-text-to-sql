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
        cols = _TABLES[table]
        context = f"CREATE TABLE {table} ({', '.join(f'{c} {t}' for c, t in cols)})"
        num_cols = [c for c, t in cols if t == "INTEGER"]
        str_cols = [c for c, t in cols if t == "VARCHAR"]
        kind = rng.randrange(4)
        if kind == 0:
            out.append(Example(f"How many {table} rows are there?", context, f"SELECT COUNT(*) FROM {table}"))
        elif kind == 1:
            c = rng.choice(num_cols)
            out.append(Example(f"What is the highest {c} in {table}?", context, f"SELECT MAX({c}) FROM {table}"))
        elif kind == 2:
            c = rng.choice(num_cols)
            v = rng.randint(1, 99)
            q = f"List the {str_cols[0]} of {table} with {c} greater than {v}."
            out.append(Example(q, context, f"SELECT {str_cols[0]} FROM {table} WHERE {c} > {v}"))
        else:
            c = rng.choice([s for s in str_cols if s in _VALUES] or str_cols)
            v = rng.choice(_VALUES.get(c, ["x"]))
            q = f"What is the average {num_cols[0]} of {table} whose {c} is {v}?"
            out.append(Example(q, context, f"SELECT AVG({num_cols[0]}) FROM {table} WHERE {c} = '{v}'"))
    return out


def save_jsonl(examples: list[Example], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex)) + "\n")


def load_jsonl(path: str | Path) -> list[Example]:
    return [Example(**json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
