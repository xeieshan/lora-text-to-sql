import pytest

from sqltune.data import Example, split_examples, synthetic_examples
from sqltune.prompts import extract_sql, few_shot_prefix, format_full, format_prompt
from sqltune.sqlmetrics import (
    canonical,
    execution_match,
    gold_literals,
    is_valid,
    normalized_exact_match,
    parse_schema,
)

CTX = "CREATE TABLE head (name VARCHAR, age INTEGER, born_state VARCHAR)"
WIKI_CTX = "CREATE TABLE table_name_15 (player VARCHAR, year VARCHAR, points INTEGER)"


def test_canonical_ignores_case_and_whitespace():
    assert canonical("select  name from head WHERE age>56") == canonical("SELECT name FROM head WHERE age > 56")
    assert canonical("SELEC broken (") is None


@pytest.mark.parametrize(
    "gold,pred,expected",
    [
        ("SELECT name FROM head WHERE age > 56", "select name from head where age > 56;".rstrip(";"), True),
        ("SELECT name FROM head WHERE age > 56", "SELECT name FROM head WHERE age >= 56", False),
        ("SELECT COUNT(*) FROM head", "SELECT count(*) FROM head", True),
    ],
)
def test_normalized_exact_match(gold, pred, expected):
    assert normalized_exact_match(gold, pred) is expected


def test_execution_match_accepts_equivalent_queries():
    gold = "SELECT name FROM head WHERE age > 56"
    assert execution_match(CTX, gold, "SELECT h.name FROM head AS h WHERE NOT h.age <= 56")
    assert execution_match(CTX, "SELECT COUNT(*) FROM head", "SELECT COUNT(name) + 0 FROM head WHERE 1 = 1")


def test_execution_match_rejects_different_queries():
    gold = "SELECT name FROM head WHERE born_state = 'California'"
    assert not execution_match(CTX, gold, "SELECT name FROM head")
    assert not execution_match(CTX, gold, "SELECT name FROM head WHERE born_state = 'Texas'")
