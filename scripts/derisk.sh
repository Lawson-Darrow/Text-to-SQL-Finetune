#!/usr/bin/env bash
# Milestone 2 stack derisk — run in WSL2 Ubuntu on the RTX 4090.
# Installs the training stack into a WSL-native venv and runs the pass/fail checks
# from docs/stack_derisk.md on Qwen2.5-Coder-1.5B (small, for a fast derisk).
#
#   bash <(sed 's/\r$//' /mnt/c/Users/lawso/Desktop/Projects/Text-to-SQL-Finetune/scripts/derisk.sh)
set -euo pipefail

WORK="$HOME/t2s-derisk"
mkdir -p "$WORK"; cd "$WORK"

echo "=== [0] nvidia-smi ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

echo "=== venv (python3.12) ==="
[ -d .venv ] || python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -q --upgrade pip

echo "=== install torch (CUDA 12.4 build) ==="
pip install -q torch --index-url https://download.pytorch.org/whl/cu124

echo "=== install HF stack ==="
pip install -q transformers peft trl accelerate datasets

echo "=== [1] torch GPU check ==="
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA not available to torch"
print("device:", torch.cuda.get_device_name(0))
cap = torch.cuda.get_device_capability(0)
print("capability:", cap, "(expect (8, 9) for Ada/4090)")
PY

echo "=== [2-4] forward + LoRA bf16 backward + generate on Qwen2.5-Coder-1.5B ==="
python - <<'PY'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="cuda")
model = get_peft_model(model, LoraConfig(
    r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
model.print_trainable_parameters()

ids = tok("SELECT count(*) FROM users", return_tensors="pt").to("cuda")
out = model(**ids, labels=ids["input_ids"])
out.loss.backward()
print("[3] LoRA bf16 backward OK | loss =", round(float(out.loss), 4))

model.eval()
prompt = "-- DB: users(id, name, age)\n-- Q: how many users?\n-- SQL:"
gen = model.generate(**tok(prompt, return_tensors="pt").to("cuda"), max_new_tokens=24, do_sample=False)
print("[4] generation OK:", repr(tok.decode(gen[0], skip_special_tokens=True)[len(prompt):].strip()[:80]))
print("DERISK_OK")
PY
