# lora-text-to-sql

Does fine-tuning a 0.5B-parameter model beat simply prompting it? This repo answers that for text-to-SQL. It fine-tunes **Qwen2.5-0.5B-Instruct** with LoRA on `b-mc2/sql-create-context`, then compares it against the same base model with zero-shot and few-shot prompts, on the same 1,000 held-out questions and with the same prompt format.

The whole run fits on a free Colab T4: [`notebooks/colab_finetune.ipynb`](notebooks/colab_finetune.ipynb).

## Results

| Mode | Exact match | Execution match | Valid SQL | n |
|---|---:|---:|---:|---:|
| Zero-shot | pending GPU run | | | 1000 |
| Few-shot (3 examples) | pending GPU run | | | 1000 |
| LoRA (r=16, 10k examples, 1 epoch) | pending GPU run | | | 1000 |

**Status:** the code, tests and notebook are complete. This table is filled in by running the notebook on a GPU, which writes `results/comparison.md`. No numbers are listed until that run happens.

What *has* been run is the CPU plumbing check (`sqltune smoke`, [`results/smoke_cpu_run.json`](results/smoke_cpu_run.json)). A tiny randomly initialised model with a locally trained tokenizer goes through LoRA training (300 steps), saving the adapter, reloading it, generating and scoring. On synthetic template questions it reaches 0.75 exact match, which only shows that the training signal reaches the model and that every step is wired correctly. It says nothing about real performance.

## Method

**Data.** sql-create-context pairs a question with the `CREATE TABLE` statements it needs and a gold query. It ships only a `train` split, so `sqltune prepare` removes duplicate (question, schema) pairs and makes a seeded 1,000 / 500 / rest test / validation / train split.

**One prompt format for every mode:**

```text
### Schema
CREATE TABLE head (name VARCHAR, age INTEGER)
### Question
How many heads of departments are older than 56?
### SQL
SELECT COUNT(*) FROM head WHERE age > 56
```

Few-shot mode prepends three solved examples from the training split. Because the format doesn't change, the comparison isolates what fine-tuning adds.

**Training.** The loss covers only the SQL tokens, because prompt tokens get label `-100`; otherwise the model spends capacity learning to reproduce schemas. Examples longer than the token budget are dropped, never truncated, since a truncated answer teaches the model to stop early. LoRA uses r=16 and α=32 on all attention and MLP projections, about 1.8% of parameters, with a cosine schedule. bf16 or fp16 is chosen to match the GPU.

**Metrics.** Three numbers, because each one misleads on its own:

| Metric | What it checks | Weakness it covers |
|---|---|---|
| Exact match | Queries are identical after sqlglot normalisation (case, whitespace, quoting) | strict: misses correct queries written differently |
| **Execution match** | Both queries return the same rows on 3 random databases built from the schema | catches equivalent rewrites that exact match rejects |
| Valid SQL | The prediction runs on the schema | separates "wrong" from "broken" |

The tables in this dataset come empty, so naive execution comparison would score every pair of queries as equal (empty = empty). The random databases are **seeded with literals from the gold query**, so `WHERE name = 'Paris'` actually selects rows. This is an approximation of test-suite accuracy (Zhong et al., 2020); the unit tests show it accepts equivalent rewrites and rejects changed filters.

## Quickstart

```bash
pip install -e ".[dev]"
pytest                     # 16 tests, CPU only, no downloads (tiny local model)
sqltune smoke              # end-to-end plumbing run, ~15 s on CPU

# On a GPU (or open the Colab notebook):
sqltune prepare
sqltune train --max-train-examples 10000 --batch-size 16
sqltune evaluate --modes zero_shot few_shot lora     # writes results/comparison.{md,json}
```

`--base-model` takes any causal LM on the Hugging Face Hub that uses the standard `q_proj/k_proj/...` module names (Qwen, Llama, Mistral, SmolLM).

## Project layout

```
src/sqltune/
  data.py        dataset loading, de-duplication, seeded splits, synthetic examples
  prompts.py     the shared prompt format, few-shot prefix, SQL extraction
  modeling.py    tokenisation with prompt masking, padding collator, LoRA training, batched generation
  sqlmetrics.py  normalised exact match, seeded-database execution match, validity
  evaluate.py    zero-shot / few-shot / LoRA comparison and result tables
  smoke.py       tiny random model + local tokenizer for offline tests
notebooks/colab_finetune.ipynb   the GPU run, loss curve and failure inspection
```

## Limitations

- sql-create-context is mostly single-table WikiSQL-style questions. High scores here don't transfer to multi-table benchmarks like Spider or BIRD.
- Execution match on random databases can still accept a wrong query that happens to agree on all three databases. Raising `n_dbs` makes this rarer at the cost of time.
- One seed and one hyperparameter setting. Before claiming a gap between modes, run 3 seeds and report the spread.

## License

MIT for the code. Base model: Qwen2.5-0.5B-Instruct (Apache 2.0). Dataset: b-mc2/sql-create-context (CC BY 4.0), downloaded at runtime.
