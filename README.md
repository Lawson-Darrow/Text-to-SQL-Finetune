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

Milestone 1 — scaffold. See [SPEC.md](SPEC.md) for the plan and milestones.

## Setup (planned)

Training runs locally on an RTX 5090 (CUDA 12.8 / torch 2.7+, Python 3.11–3.12). Eval uses
SQLite execution against the Spider databases. See SPEC for the full stack.

## License

MIT (added when the repo goes public).
