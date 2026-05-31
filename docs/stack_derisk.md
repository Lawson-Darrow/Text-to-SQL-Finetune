# Stack derisk — pass/fail matrix (Milestone 2)

Goal: prove the training+eval stack runs on the RTX 5090 (Blackwell, sm_120) **before**
building anything. Run on **WSL2 Ubuntu** (not native Windows). Record pass/fail + versions.

| # | Check | Pass criteria | Result |
|---|---|---|---|
| 0 | Env | `nvidia-smi` sees the 5090; driver + CUDA noted | |
| 1 | torch sees the GPU | `torch.cuda.is_available()`, `get_device_capability()` → `(12, 0)` (sm_120) | |
| 2 | Forward pass | load Qwen2.5-Coder-7B bf16, one forward pass, no kernel error | |
| 3 | LoRA backward | one **LoRA bf16** training step (peft+trl), loss decreases | |
| 4 | Generation | greedy-decode a SQL string from a prompt | |
| 5 | Eval query | official Spider evaluator runs on one (pred, gold) pair against its SQLite DB | |
| 6 | (optional) 4-bit | bitsandbytes 4-bit load on sm_120 — **expected to be the flaky one** | |

Pinned versions (fill in once it works):

```
OS/WSL: 
NVIDIA driver: 
CUDA: 12.8.x
python: 3.11/3.12
torch: 
transformers: 
peft: 
trl: 
unsloth: 
bitsandbytes: 
triton: 
```

If check 6 fails, that's fine — the MVP is LoRA bf16 (checks 1–5). QLoRA is a nice-to-have.
