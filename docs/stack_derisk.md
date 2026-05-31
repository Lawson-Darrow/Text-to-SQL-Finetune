# Stack derisk — pass/fail matrix (Milestone 2) ✅ PASSED 2026-05-31

The training+eval stack runs on the RTX 4090 (Ada, sm_89, 24GB) under WSL2 Ubuntu.
Env set up with `uv` (no sudo). Re-runnable via `scripts/derisk.sh`.

| # | Check | Pass criteria | Result |
|---|---|---|---|
| 0 | Env | `nvidia-smi` sees the 4090 in WSL2; driver + CUDA noted | ✅ 4090, driver 610.47 |
| 1 | torch sees the GPU | `get_device_capability()` → `(8, 9)` (sm_89) | ✅ torch 2.6.0+cu124, (8,9) |
| 2 | Forward pass | load Qwen2.5-Coder-1.5B bf16, one forward pass | ✅ loaded, no kernel error |
| 3 | LoRA backward | one **LoRA bf16** training step (peft+trl) | ✅ loss 3.74, 1.09M trainable params |
| 4 | Generation | greedy-decode a SQL string from a prompt | ✅ emitted valid SQL |
| 5 | Eval query | official Spider evaluator on one (pred, gold) pair | ⏳ Milestone 3 |
| 6 | QLoRA 4-bit | bitsandbytes 4-bit load on sm_89 | ⏳ deferred (LoRA bf16 is the MVP; Ada is mature) |

Pinned versions (WSL2 venv via `uv`):

```
OS/WSL:       Ubuntu 24.04.1 LTS (WSL2)
NVIDIA driver: 610.47
CUDA:          12.4 (torch wheel) / 13.0 (system nvcc)
python:        3.12.3
torch:         2.6.0+cu124
transformers:  5.9.0
peft:          0.19.1
trl:           1.5.1
accelerate:    1.13.0
triton:        3.2.0
unsloth:       (not yet — add for the training milestone)
bitsandbytes:  (not yet — only if QLoRA is wanted)
```

Note for training code: transformers 5.x deprecates `torch_dtype=` → use `dtype=`.
