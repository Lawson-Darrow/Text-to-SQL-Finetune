# Contributing

This is a research-grade project; interfaces may change. Issues, ideas, and PRs
are welcome, especially around new ablation axes, additional base models, and the
evaluation harness.

## Dev setup

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows; use bin/activate on *nix
pip install -e ".[dev]"
ruff check .
```

`torch` (CUDA build) and `unsloth` are installed out of band per the GPU/CUDA
setup, not pinned here; see SPEC "Compute & environment". Fine-tuning runs locally
on an RTX 4090 under WSL2. Data prep (`bash scripts/prepare_data.sh`) downloads the
Spider databases and the official test-suite evaluator.

## Bar for changes

- **Lint clean** (`ruff check .`).
- **Report uncertainty.** Headline accuracy numbers come with bootstrap 95% CIs;
  do not draw conclusions from small slices (a flat fine-tuning effect at n=100
  turned clearly positive on full dev). Keep the CIs.
- **No test-set tuning.** Hyperparameters and prompts are chosen on dev/train, not
  on the evaluation set.
- **Keep the axes separate.** Schema representation, fine-tuning, and model size
  are distinct axes; report them as such.
- Execution accuracy via the official evaluator is the metric. Exact-match is
  secondary.

## Scope

See [SPEC.md](SPEC.md) for milestones and `docs/stack_derisk.md` for the verified
stack. New base models and schema-representation variants are welcome.
