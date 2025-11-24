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
