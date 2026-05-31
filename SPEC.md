# Project Spec — Text-to-SQL: size vs. fine-tuning vs. schema representation

*Status: scope approved 2026-05-31; revised after a Codex plan review. Project #2 of the
LLM/NLP arc (fine-tuning, fresh domain).*

## One-liner

Fine-tune small open **Qwen2.5-Coder** models for text-to-SQL and answer a sharper
question than "is bigger better": **what actually moves accuracy on small text-to-SQL
models — model size, fine-tuning, or how you represent the schema?** Measured with the
official Spider / test-suite evaluator.

## Why this project

- Carries forward the "measure everything" identity from Project #1 ([Biomedical RAG Agent]),
  now with a multi-axis ablation rather than a single leaderboard number.
- Fresh, non-medical domain (code-gen + schema grounding); shows range beyond #1's retrieval.

## Thesis & axes (the artifact)

The headline is an ablation across three axes, not just a size curve:
1. **Schema representation** (highest-signal per the review): serialization format, PK/FK
   inclusion, column descriptions, sample/retrieved cell values. Ablate these.
2. **Fine-tuning**: base zero-shot vs base few-shot vs LoRA fine-tuned.
3. **Model size**: 1.5B / 3B / 7B (one axis, not the whole story).

## Scope

**In:**
- Models: **Qwen2.5-Coder-{1.5B, 3B, 7B}-Instruct**.
- Method: **LoRA bf16** (MVP). QLoRA only if bitsandbytes works on sm_120 (see risks) — optional.
- Data: **Spider** (train/dev). Schema serialization is a first-class, ablated component.
- Eval: the **official Spider evaluator + test-suite accuracy** — not a homegrown row-compare.
  Report: execution accuracy, exact-set-match, **invalid-SQL rate, exec-error rate,
  timeout rate**, and **per-hardness** breakdown (easy/medium/hard/extra).
- Baselines: base zero-shot, base few-shot, **frontier zero/few-shot** (via LLMGateway,
  prompt/schema/decoding matched), LoRA fine-tuned. Optional: an open text-to-SQL baseline
  (e.g. SQLCoder/Defog) if license permits.
- **Contamination caveat**: Qwen2.5-Coder likely saw Spider. State it explicitly; add a
  **small held-out set with unseen schemas** (synthetic or hand-built) as a cleaner signal.
- **Statistical rigor**: fixed seeds, bootstrap CIs on dev accuracy, McNemar/sign test for
  paired model comparisons.
- Pin generation settings (temperature, max tokens, stop seqs, SQL-extraction + repair policy).

**Out (v1):** BIRD (stretch), DPO (stretch), full fine-tuning, serving/deployment.

## Compute & environment

- **Hardware:** local **RTX 5090** (Blackwell, sm_120, 32GB). Colab Pro fallback.
- **Run on WSL2 Ubuntu / Linux, not native Windows** — bitsandbytes/QLoRA kernels on sm_120
  are the weak link, and the broader CUDA stack is far smoother on Linux.
- Pin: torch (CUDA 12.8 wheel), driver, unsloth, transformers, triton, bitsandbytes. Python 3.11/3.12.
- **MVP = LoRA bf16** (7B fits comfortably in 32GB); treat 4-bit QLoRA as a nice-to-have.

## Risks (from the review)

- **bitsandbytes on sm_120** may lack working quantization kernels (esp. Windows) → don't bet
  the project on QLoRA; LoRA bf16 is the safe path.
- **Execution accuracy is objective but not simple** — ordering, NULLs, value formatting,
  casing, float precision, duplicate rows, per-DB setup, timeouts all bite. Use the official
  evaluator and report error/invalid/timeout rates alongside accuracy.
- **Schema linking is the actual hard part** of Spider; isolate it as a failure mode.
- **Frontier comparison is only fair** if prompt, schema format, and decoding budget match;
  frontier APIs may have Spider memorization too.

## Milestones

1. **Scaffold** ✅
2. **Stack derisk** — a pass/fail matrix (see `docs/stack_derisk.md`): torch sees `sm_120`,
   one forward pass, one **LoRA bf16 backward** pass, one generation, one eval query against
   the Spider evaluator; optional 4-bit load. On WSL2.
3. **Data + eval harness** — Spider load, schema serialization (v1 format), wire the
   **official evaluator** first (before training), with the metric suite above.
4. **Schema-representation ablation** — prompt-only / few-shot, varying schema formats, on the base 7B.
5. **Fine-tune** — LoRA bf16 SFT; compare to base + frontier on the full metric suite.
6. **Size ladder** — 1.5B/3B/7B × best schema format; the 3-axis result + CIs + per-hardness.
7. **Writeup** — report (+ contamination caveat + held-out set), blog, featured project.

## Success criteria

- Reproducible Spider-evaluator numbers (accuracy + invalid/timeout rates + per-hardness) for
  each cell of the ablation, with confidence intervals.
- A defensible answer to "size vs. fine-tuning vs. schema representation," not just a size curve.
- Repo + report + blog; featured on the site.

## Licensing

Note licenses explicitly: Spider (data), Qwen2.5-Coder (weights), frontier-model outputs
(terms), any generated held-out data. Repo code is MIT.

## Stack

Python 3.11/3.12 (WSL2) · PyTorch CUDA 12.8 · unsloth · transformers · peft · trl ·
datasets · the official Spider / test-suite evaluator · LLMGateway (frontier baselines) ·
matplotlib.

[Biomedical RAG Agent]: https://github.com/Lawson-Darrow/Biomedical-RAG-Agent
