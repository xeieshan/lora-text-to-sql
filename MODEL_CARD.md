---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
license: apache-2.0
datasets:
  - b-mc2/sql-create-context
tags:
  - text-to-sql
  - lora
---

# Qwen2.5-0.5B text-to-SQL LoRA adapter

A LoRA adapter (r=16, α=32, all attention and MLP projections) for Qwen2.5-0.5B-Instruct, trained to write SQLite queries from a question plus `CREATE TABLE` context.

## Prompt format

```text
### Schema
<CREATE TABLE statements>
### Question
<question>
### SQL
```

The model continues with a single query and then the end-of-sequence token.

## Evaluation

1,000 held-out questions from a de-duplicated, seeded split of sql-create-context. Fill this in from `results/comparison.md` after training:

| Mode | Exact match | Execution match | Valid SQL |
|---|---:|---:|---:|
| Base, zero-shot | | | |
| Base, 3-shot | | | |
| This adapter | | | |

## Training

- Examples: see `train_examples_used` in `train_summary.json`
- 1 epoch, learning rate 2e-4, cosine schedule, batch size 16
- Loss on SQL tokens only (prompt tokens masked)

## Limitations

Trained mostly on single-table questions. Don't expect it to handle joins across many tables or messy real-world schemas. Generated SQL should be reviewed before it runs against a production database.
