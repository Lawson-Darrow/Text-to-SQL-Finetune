"""Milestone 6 — size ladder + statistical rigor.

Evaluates base 1.5B/3B/7B and the fine-tuned 1.5B on the FULL Spider dev set, computing
execution accuracy with a 95% bootstrap CI (per-example via the official eval_exec_match).
The headline accuracy-vs-size result, with the fine-tuning effect measured honestly.

Run (WSL2): PYTHONPATH=src python scripts/run_ladder.py
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import torch

from text2sql.data import build_prompt, load_examples, load_schemas, serialize_schema
from text2sql.evaluate import bootstrap_ci, per_example_execution
from text2sql.infer import generate_sql, load_model

DATA = Path(os.environ.get("T2S_DATA", str(Path.home() / "t2s-data/data")))
EVAL = Path(os.environ.get("T2S_EVAL", str(Path.home() / "t2s-data/test-suite-sql-eval")))
ADAPTERS = Path(os.environ.get("T2S_ADAPTERS", str(Path.home() / "t2s-data/adapters")))

FT_15B = str(ADAPTERS / "Qwen2.5-Coder-1.5B-Instruct-spider-lora/adapter")
CONFIGS = [
    {"label": "1.5B base", "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct", "adapter": None, "size": 1.5, "bs": 16},
    {"label": "3B base", "model": "Qwen/Qwen2.5-Coder-3B-Instruct", "adapter": None, "size": 3.0, "bs": 16},
    {"label": "7B base", "model": "Qwen/Qwen2.5-Coder-7B-Instruct", "adapter": None, "size": 7.0, "bs": 8},
    {"label": "1.5B FT", "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct", "adapter": FT_15B, "size": 1.5, "bs": 16},
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="0 = full dev (1034)")
    ap.add_argument("--schema-style", default="with_keys")
    args = ap.parse_args()

    examples = load_examples(DATA / "dev.json")
    if args.n:
        examples = examples[: args.n]
    schemas = load_schemas(DATA / "tables.json")
    prompts = [build_prompt(ex.question, serialize_schema(schemas[ex.db_id], args.schema_style)) for ex in examples]
    gold_pairs = [(ex.query, ex.db_id) for ex in examples]
    print(f"ladder | n={len(examples)} | schema={args.schema_style}")

    results = []
    for cfg in CONFIGS:
        model, tok = load_model(cfg["model"], adapter_path=cfg["adapter"])
        preds = generate_sql(model, tok, prompts, batch_size=cfg["bs"])
        flags = per_example_execution(preds, gold_pairs, DATA / "database", EVAL)
        mean, lo, hi = bootstrap_ci(flags)
        row = {**{k: cfg[k] for k in ("label", "model", "size")},
               "ft": cfg["adapter"] is not None, "exec": round(mean, 4),
               "ci95": [round(lo, 4), round(hi, 4)], "n": len(examples)}
        results.append(row)
        print(f"  {cfg['label']:10s} exec={mean:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
        del model
        gc.collect()
        torch.cuda.empty_cache()

    Path("results").mkdir(exist_ok=True)
    Path("results/ladder.json").write_text(
        json.dumps({"schema_style": args.schema_style, "rows": results}, indent=2), encoding="utf-8"
    )
    print("wrote results/ladder.json")


if __name__ == "__main__":
    main()
