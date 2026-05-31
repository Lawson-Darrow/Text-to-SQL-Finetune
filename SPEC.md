# Project Spec — Text-to-SQL Fine-Tune (size ladder)

*Status: scope approved 2026-05-31. Project #2 of the LLM/NLP arc (fine-tuning, fresh domain).*

## One-liner

Fine-tune small open **Qwen2.5-Coder** models (1.5B / 3B / 7B) to translate natural-language
questions into SQL, and measure — with **execution accuracy** — where the size ladder
crosses a frontier baseline. The efficiency story: small + open + cheap ≈ big + expensive.

## Why this project

- **Carries forward the "measure everything" identity** from Project #1 ([Biomedical RAG Agent])
  with the cleanest possible eval: *execution accuracy* (run the predicted SQL, check it
  returns the right rows) — objective, no LLM judge, no judge-bias caveat.
- **Fresh, non-medical domain**; shows range (code generation + structured reasoning) beyond
  the retrieval work in #1.
- **Employable + contextualized** against public benchmarks (Spider/BIRD leaderboards).

## Scope

**In:**
- Models: **Qwen2.5-Coder-{1.5B, 3B, 7B}-Instruct** (the size ladder).
- Method: **LoRA / QLoRA SFT** via `unsloth` (OSS, fast on Blackwell); DPO as a stretch.
- Data: **Spider** (primary). Train split → NL+schema → SQL pairs.
- Eval: **execution accuracy** + exact-set-match on the Spider dev split.
- Baselines: base model (no FT) vs fine-tuned vs **frontier via LLMGateway** (gpt-4.1 / claude),
  reusing Project #1's inference infra. Report accuracy, latency, cost.
- Headline artifact: the **accuracy-vs-size curve** with the frontier line overlaid.

**Out (v1):**
- BIRD (harder benchmark) — stretch.
- DPO / preference tuning — stretch.
- Full fine-tuning (LoRA only), serving/deployment.

## Compute & environment

- **Primary:** local **RTX 5090** (Blackwell, 32GB). Colab Pro as fallback.
- **Env caveat:** Blackwell (sm_120) needs CUDA 12.8+ / torch 2.7+ and a **non-3.14 Python**
  (use 3.11/3.12). First milestone after scaffold is a **stack derisk**: confirm a LoRA step
  actually runs on the card before building further.
- OSS tooling preferred throughout (unsloth, transformers, peft, trl, datasets).

## Milestones

1. **Scaffold** — repo, spec, structure. → *current*
2. **Stack derisk** — torch+CUDA on the 5090; one LoRA step executes on-GPU.
3. **Data pipeline** — Spider → prompt/target pairs with schema serialization.
4. **First fine-tune** — LoRA SFT on one size (7B); sanity-check it emits valid SQL.
5. **Eval harness** — execution accuracy + exact-set-match; base vs FT vs frontier.
6. **Size ladder** — fine-tune 1.5B/3B/7B; accuracy-vs-size curve vs frontier; latency/cost.
7. **Writeup** — report + chart, blog post, second featured project on ledarrow.dev.

## Success criteria

- A reproducible **execution-accuracy** number for each model (base, FT ×3 sizes, frontier).
- The size-ladder curve, with a defensible takeaway (e.g. "fine-tuned 7B matches frontier;
  3B gets 90% of the way at a fraction of the cost").
- Repo + report + blog; featured on the site.

## Stack

Python 3.11/3.12 · PyTorch (CUDA 12.8) · unsloth · transformers · peft · trl · datasets ·
sqlite3 (execution eval) · LLMGateway (frontier baselines) · matplotlib (the curve).

[Biomedical RAG Agent]: https://github.com/Lawson-Darrow/Biomedical-RAG-Agent
