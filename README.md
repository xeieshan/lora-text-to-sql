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
