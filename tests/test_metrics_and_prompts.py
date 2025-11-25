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
    assert not execution_match(CTX, gold, "SELECT nope FROM head")
    assert not execution_match(CTX, gold, "")


def test_double_quoted_literals_from_wikisql_style_gold():
    gold = 'SELECT player FROM table_name_15 WHERE year = "1990"'
    strings, _ = gold_literals(gold)
    assert "1990" in strings
    assert execution_match(WIKI_CTX, gold, "SELECT player FROM table_name_15 WHERE year = '1990'")
    assert not execution_match(WIKI_CTX, gold, "SELECT player FROM table_name_15 WHERE year = '1991'")


def test_parse_schema_and_validity():
    assert parse_schema(CTX) == {"head": [("name", "VARCHAR"), ("age", "INTEGER"), ("born_state", "VARCHAR")]}
    multi = CTX + "; CREATE TABLE management (head_id INTEGER, temporary_acting VARCHAR)"
    assert set(parse_schema(multi)) == {"head", "management"}
    assert is_valid(CTX, "SELECT name FROM head")
    assert not is_valid(CTX, "SELECT name FROM people")
    assert not is_valid(CTX, "   ")


def test_prompt_format_and_extraction():
    ex = Example("How many heads?", CTX, "SELECT COUNT(*) FROM head")
    assert format_full(ex) == format_prompt(ex) + ex.sql
    assert format_prompt(ex).endswith("### SQL\n")
    assert extract_sql(" SELECT COUNT(*) FROM head;\n### Schema\nCREATE ...") == "SELECT COUNT(*) FROM head"
    assert extract_sql("SELECT 1\nextra text") == "SELECT 1"
    prefix = few_shot_prefix([ex, ex])
    assert prefix.count("### SQL") == 2 and prefix.endswith("\n\n")


def test_splits_are_deterministic_and_disjoint():
    data = synthetic_examples(200, seed=1)
    a = split_examples(data, seed=7, n_val=20, n_test=30)
    b = split_examples(data, seed=7, n_val=20, n_test=30)
    assert a == b
    assert len(a["train"]) == 150
    with pytest.raises(ValueError):
        split_examples(data[:40], seed=0, n_val=20, n_test=30)


def test_synthetic_gold_queries_run():
    for ex in synthetic_examples(50, seed=3):
        assert is_valid(ex.context, ex.sql), ex
