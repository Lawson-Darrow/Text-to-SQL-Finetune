"""Milestone 5 — LoRA bf16 SFT on Spider train.

Builds (schema+question -> SQL) chat pairs from the Spider train split and fine-tunes a
LoRA adapter. Trains the 1.5B by default (most headroom; fast); adapter saved outside the
repo in WSL. Eval afterward with:
    PYTHONPATH=src python scripts/run_baseline.py --model <name> --adapter <adapter_dir> --n 100

Run (WSL2, derisk venv):
    PYTHONPATH=src python scripts/run_finetune.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Raise the open-file limit: training + peft's model-card save can exhaust the
# default soft limit (1024 on Ubuntu/WSL), which silently breaks adapter saving.
try:
    import resource

    _soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(8192, _hard), _hard))
except Exception:
    pass

from datasets import load_dataset

from text2sql.data import Example, load_schemas
from text2sql.train import build_messages, finetune

DATA = Path(os.environ.get("T2S_DATA", str(Path.home() / "t2s-data/data")))
ADAPTERS = Path(os.environ.get("T2S_ADAPTERS", str(Path.home() / "t2s-data/adapters")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--n-train", type=int, default=0, help="0 = full train split")
    ap.add_argument("--schema-style", default="with_keys")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    ds = load_dataset("xlangai/spider", split="train")
    if args.n_train:
        ds = ds.select(range(args.n_train))
    examples = [Example(r["db_id"], r["question"], r["query"]) for r in ds]
    schemas = load_schemas(DATA / "tables.json")
    rows = build_messages(examples, schemas, args.schema_style)
    print(f"train rows: {len(rows)} | model: {args.model} | schema: {args.schema_style} | epochs: {args.epochs}")

    tag = args.model.split("/")[-1] + "-spider-lora"
    out_dir = ADAPTERS / tag
    adapter = finetune(
        args.model, rows, str(out_dir), epochs=args.epochs, batch_size=args.batch_size
    )
    print("ADAPTER SAVED:", adapter)


if __name__ == "__main__":
    main()
