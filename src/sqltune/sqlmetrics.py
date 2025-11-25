"""SQL metrics.

* ``normalized_exact_match``  parse both queries with sqlglot and compare canonical forms,
                              so whitespace, keyword case and quoting don't count as errors.
* ``execution_match``         run both queries on several small random databases built from the
                              CREATE TABLE context and compare results. The databases are seeded
                              with the literals from the gold query so WHERE clauses actually
                              select rows. This approximates "test-suite accuracy" (Zhong et al., 2020):
                              two different queries rarely agree on every random database.
* ``is_valid``                the prediction runs without error on the schema.
"""

from __future__ import annotations

import random
import re
import sqlite3
from collections import Counter

import sqlglot
from sqlglot import exp

_CREATE_RE = re.compile(r"CREATE TABLE\s+([\w\"`\[\]]+)\s*\((.*?)\)\s*(?:;|$)", re.IGNORECASE | re.DOTALL)
_WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]


def canonical(sql: str) -> str | None:
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except (sqlglot.errors.ParseError, sqlglot.errors.TokenError):
        return None
    if tree is None:
        return None
    return tree.sql(dialect="sqlite", normalize=True, pretty=False).lower()


def normalized_exact_match(gold: str, pred: str) -> bool:
    g, p = canonical(gold), canonical(pred)
    if g is None or p is None:
        return " ".join(gold.lower().split()) == " ".join(pred.lower().split())
    return g == p


def parse_schema(context: str) -> dict[str, list[tuple[str, str]]]:
    tables: dict[str, list[tuple[str, str]]] = {}
    for name, body in _CREATE_RE.findall(context):
        cols = []
        for part in body.split(","):
            bits = part.strip().split()
            if not bits:
                continue
            col = bits[0].strip('"`[]')
            ctype = bits[1].upper() if len(bits) > 1 else "VARCHAR"
            cols.append((col, ctype))
        tables[name.strip('"`[]')] = cols
    return tables


def gold_literals(sql: str) -> tuple[list[str], list[float]]:
    strings, numbers = [], []
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except (sqlglot.errors.ParseError, sqlglot.errors.TokenError):
        return strings, numbers
    for lit in tree.find_all(exp.Literal):
        if lit.is_string:
            strings.append(lit.this)
        else:
            try:
                numbers.append(float(lit.this))
            except ValueError:
                pass
    # double-quoted "strings" (common in this dataset) parse as identifiers in some dialects
    for ident in tree.find_all(exp.Identifier):
        if ident.quoted:
            strings.append(ident.this)
    return strings, numbers


def _build_db(schema: dict[str, list[tuple[str, str]]], strings, numbers, rng: random.Random, n_rows: int):
    conn = sqlite3.connect(":memory:")
    for table, cols in schema.items():
        cols_sql = ", ".join(f'"{c}" {t}' for c, t in cols)
        conn.execute(f'CREATE TABLE "{table}" ({cols_sql})')
        rows = []
        for _ in range(n_rows):
            row = []
            for _, ctype in cols:
                if "INT" in ctype or "REAL" in ctype or "NUM" in ctype or "FLOAT" in ctype:
                    pool = numbers or [0.0]
                    row.append(rng.choice(pool) if rng.random() < 0.5 else rng.randint(-5, 100))
                else:
