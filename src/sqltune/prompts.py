"""One plain-text prompt format shared by training, zero-shot and few-shot evaluation.

Using the same format everywhere keeps the comparison fair: the fine-tuned model is
not helped (or hurt) by a different template than the baselines see.
"""

from __future__ import annotations

from .data import Example

INSTRUCTION = "Write one SQLite query that answers the question using the schema."


def format_prompt(ex: Example) -> str:
    return f"### Schema\n{ex.context}\n### Question\n{ex.question}\n### SQL\n"
