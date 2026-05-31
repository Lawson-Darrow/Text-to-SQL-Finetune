# Text-to-SQL: what actually moves accuracy — size, fine-tuning, or schema representation?

A small, reproducible study behind the [Text-to-SQL Fine-Tune](README.md) project. I
fine-tuned open Qwen2.5-Coder models for natural-language → SQL and measured three levers
with the **official Spider / test-suite evaluator** (execution accuracy — run the query,
check the rows), reporting bootstrap confidence intervals on the headline numbers.

## TL;DR

- **Model size is the dominant lever.** Base execution accuracy on Spider dev (n=1034):
  **1.5B 0.56 → 3B 0.69 → 7B 0.79**, all CIs non-overlapping.
- **Open matches frontier.** The open **7B (0.79)** is statistically tied with
  **Claude-haiku-4-5 (0.79)** and within noise of **GPT-4.1-mini (0.81)** — a model you can
  run on your own 24GB GPU reaches frontier-tier execution accuracy on this task.
- **LoRA fine-tuning a 1.5B gives a real, significant lift: 0.56 → 0.63 (+7 pts)** — the
  fine-tuned 1.5B closes **~half the gap to a 2×-larger 3B base**. The efficiency story.
- **Schema representation matters less but consistently:** adding column *types* hurt
  execution at both sizes; PK/FK keys helped exact-match; a `minimal` schema already
  reaches ~0.91 on the 7B.
- **A lesson in rigor:** on a 100-example slice the fine-tuning effect looked *flat
  (even negative)*; on the full dev set with CIs it's a clear, significant gain. Small
  eval slices lie — which is exactly why the CIs are here.

## Question

Most "fine-tune an LLM for SQL" projects report one number. The useful question is
*which lever actually buys accuracy* — so this study isolates three: model size,
fine-tuning, and how the database schema is serialized into the prompt.

## Method

- **Data:** Spider (train for SFT; dev for eval). One self-consistent package (166 SQLite
  DBs + `tables.json` + dev gold).
- **Models:** Qwen2.5-Coder-{1.5B, 3B, 7B}-Instruct (open weights).
- **Fine-tuning:** LoRA bf16 SFT (1 epoch, full train) on the 4090 via WSL2.
- **Eval:** the **official `test-suite-sql-eval`** — execution accuracy + exact-match — not
  a homegrown row-compare. Headline numbers carry 95% bootstrap CIs (per-example via the
  official `eval_exec_match`).
- **Schema serialization** is an ablated component: `minimal` / `with_types` / `with_keys`.

## Results

### Size, fine-tuning, and frontier (full dev, n=1034, execution accuracy, 95% CI)

![accuracy vs size](results/ladder.png)

| config | tier | execution acc | 95% CI |
|---|---|---|---|
| 1.5B base | open | 0.558 | [0.527, 0.587] |
| 1.5B fine-tuned | open | 0.629 | [0.599, 0.659] |
| 3B base | open | 0.691 | [0.664, 0.721] |
| **7B base** | open | **0.793** | [0.766, 0.817] |
| claude-haiku-4-5 | frontier | 0.794 | [0.768, 0.819] |
| gpt-4.1-mini | frontier | 0.811 | [0.787, 0.836] |

Size scales cleanly and significantly. Fine-tuning the 1.5B lifts it +7 points into the gap
between 1.5B and 3B base — roughly half a "size step" from a one-epoch LoRA. And the **open
7B (0.79) is statistically tied with the frontier models** (Claude-haiku 0.79, GPT-4.1-mini
0.81 — CIs overlap): frontier-tier text-to-SQL runs locally. Frontier was evaluated through
the same prompt, schema format, decoding, and official evaluator for a fair comparison.

### Schema representation (n=100, execution accuracy)

![schema ablation](results/schema_ablation_combined.png)

| schema style | 1.5B | 7B |
|---|---|---|
| minimal | 0.57 | 0.91 |
| with_types | 0.55 | 0.86 |
| with_keys | 0.60 | 0.91 |

Adding column *types* consistently hurt; PK/FK keys gave the best exact-match (7B 0.68);
`minimal` already reaches ~0.91 on the 7B. Schema format is a smaller lever than size, but
not free.

### Exact-match vs execution (the metric trap)

Fine-tuning raised 1.5B exact-match 0.28 → 0.45 (it learned Spider's gold SQL *style*).
Whether that meant *functional* improvement depended entirely on sample size: flat at
n=100, clearly positive (+7) at n=1034. Reporting only exact-match — or only a small
slice — would have told the wrong story.

## Limitations (honest)

- Schema ablation is n=100 (directional); the size/FT headline is full dev (n=1034) with CIs.
- One fine-tune per config, 1 epoch, fixed hparams — no hparam sweep or seed averaging.
- Fine-tuned only the 1.5B; 3B/7B fine-tunes (and a fine-tuned size ladder) are future work.
- **Contamination caveat:** Qwen2.5-Coder may have seen Spider in pretraining; treat
  absolute numbers as in-distribution. A held-out unseen-schema set would harden this.
- Frontier comparison uses the mid-tier `gpt-4.1-mini` / `claude-haiku-4-5` (the
  plan-available models), not the largest flagships — so "matches frontier" means this tier.

## Reproduce

```bash
bash scripts/prepare_data.sh                                   # Spider DBs + tables + evaluator (WSL2)
PYTHONPATH=src python scripts/run_ablation.py --n 100 --model Qwen/Qwen2.5-Coder-7B-Instruct
PYTHONPATH=src python scripts/run_finetune.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct
PYTHONPATH=src python scripts/run_ladder.py                    # full-dev ladder + CIs
PYTHONPATH=src python scripts/plot_ladder.py
```

Trained on a single RTX 4090 (24GB) under WSL2; env via `uv`. See `docs/stack_derisk.md`.

## What I'd do next

Fine-tune the full size ladder; add larger frontier flagships; a held-out unseen-schema
set for contamination; schema ablation at full dev with CIs; and the harder BIRD benchmark.
