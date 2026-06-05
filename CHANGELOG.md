# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); this project uses
[SemVer](https://semver.org/) (pre-1.0: minor = breaking allowed).

## [Unreleased]

## [0.1.0] - 2026-06-05

Initial public release. Fine-tuning small open Qwen2.5-Coder models for
text-to-SQL, evaluated with the official Spider test-suite evaluator.

### Added
- Three ablation axes: schema representation, fine-tuning (base / few-shot /
  LoRA), and model size (1.5B / 3B / 7B).
- LoRA/QLoRA supervised fine-tuning via unsloth; SQL generation executed against
  the Spider databases for execution accuracy.
- Evaluation on full Spider dev (n=1034) with bootstrap 95% CIs, plus frontier
  baselines (Claude, GPT) through an OpenAI-compatible gateway.
- Accuracy-vs-size curve and schema-ablation plots; full writeup in `REPORT.md`.

### Findings
- Model size dominates (0.56 to 0.69 to 0.79, non-overlapping CIs); the open 7B
  is statistically tied with frontier models.
- A one-epoch LoRA on the 1.5B is a significant gain of about 7 points.
- Small evaluation slices were misleading: the fine-tuning effect looked flat at
  n=100 and was clearly positive on full dev, which is why CIs are reported.
