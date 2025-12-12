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
