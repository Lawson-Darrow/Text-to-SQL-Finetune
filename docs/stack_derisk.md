# Stack derisk — pass/fail matrix (Milestone 2)

Goal: prove the training+eval stack runs on the RTX 4090 (Ada, sm_89, 24GB) **before**
building anything. Run on **WSL2 Ubuntu** (already set up). Record pass/fail + versions.

| # | Check | Pass criteria | Result |
|---|---|---|---|
| 0 | Env | `nvidia-smi` sees the 4090 in WSL2; driver + CUDA noted | |
| 1 | torch sees the GPU | `torch.cuda.is_available()`, `get_device_capability()` → `(8, 9)` (sm_89) | |
| 2 | Forward pass | load Qwen2.5-Coder-1.5B bf16 (derisk size), one forward pass, no kernel error | |
| 3 | LoRA backward | one **LoRA bf16** training step (peft+trl), loss decreases | |
| 4 | Generation | greedy-decode a SQL string from a prompt | |
| 5 | Eval query | official Spider evaluator runs on one (pred, gold) pair against its SQLite DB | |
| 6 | QLoRA 4-bit | bitsandbytes 4-bit load on sm_89 — mature on Ada, should pass | |

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
