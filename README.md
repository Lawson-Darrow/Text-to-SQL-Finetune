# Text-to-SQL Fine-Tune

Fine-tuning small open **Qwen2.5-Coder** models for natural-language → SQL, evaluated
with the **official Spider / test-suite evaluator**, to answer: **what actually moves
accuracy on small text-to-SQL models — model size, fine-tuning, or how you represent the
schema?**

Three ablation axes: schema representation, fine-tuning (base/few-shot/LoRA), and model
size (1.5B/3B/7B). The size ladder is one axis, not the whole thesis.

Project #2 of an LLM/NLP arc. Project #1 — the
[Biomedical RAG Agent](https://github.com/Lawson-Darrow/Biomedical-RAG-Agent) — was about
grounding and retrieval; this one is about fine-tuning and the accuracy/size frontier. The
through-line is rigorous, objective evaluation.

## Approach

```
Spider (NL + schema → SQL) → LoRA/QLoRA SFT (unsloth) on 1.5B/3B/7B
    → generate SQL → execute against the DB → execution accuracy
    → base vs fine-tuned vs frontier (via LLMGateway) → accuracy-vs-size curve
```

## Status

Milestone 3 — data + official evaluator wired; base-model baseline measured before any
training. Base Qwen2.5-Coder-1.5B (zero-shot, `with_keys` schema, 50 dev questions):
**execution accuracy 0.64**, exact-match ~0.44, via the official `test-suite-sql-eval`.
Stack derisk (M2) passed on the RTX 4090 / WSL2. See [SPEC.md](SPEC.md) for milestones.

```bash
# in WSL2: data + evaluator setup, then a baseline run
bash scripts/prepare_data.sh
PYTHONPATH=src python scripts/run_baseline.py --n 50 --schema-style with_keys
```

## Setup (planned)

Training runs locally on an RTX 4090 (24GB, Ada) under WSL2 Ubuntu (Python 3.11–3.12). Eval
uses the official Spider / test-suite evaluator. See SPEC for the full stack.

## License

MIT (added when the repo goes public).
