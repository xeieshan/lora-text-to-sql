"""One plain-text prompt format shared by training, zero-shot and few-shot evaluation.

Using the same format everywhere keeps the comparison fair: the fine-tuned model is
not helped (or hurt) by a different template than the baselines see.
"""

from __future__ import annotations

from .data import Example

INSTRUCTION = "Write one SQLite query that answers the question using the schema."


def format_prompt(ex: Example) -> str:
    return f"### Schema\n{ex.context}\n### Question\n{ex.question}\n### SQL\n"


def format_full(ex: Example) -> str:
    return format_prompt(ex) + ex.sql


def few_shot_prefix(shots: list[Example]) -> str:
    blocks = [format_full(s) for s in shots]
    return INSTRUCTION + "\n\n" + "\n\n".join(blocks) + "\n\n"


def extract_sql(generated: str) -> str:
    """Keep the first line of the continuation and cut any echoed next block."""
    text = generated.split("###", 1)[0]
    line = text.strip().split("\n", 1)[0]
    return line.strip().rstrip(";").strip()
