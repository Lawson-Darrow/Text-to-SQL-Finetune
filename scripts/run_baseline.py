"""Milestone 3 — wire the official evaluator and get a BASE-model baseline (no fine-tuning).

Proves the measurement pipeline end-to-end before any training: load Spider dev, serialize
the schema, generate SQL with the base model, score with the official Spider evaluator.

Data + evaluator live in WSL (not the repo); override via env:
  T2S_DATA (default ~/t2s-data/data), T2S_EVAL (default ~/t2s-data/test-suite-sql-eval)

Run (in WSL, derisk venv):
  PYTHONPATH=src python scripts/run_baseline.py --n 50 --schema-style with_keys
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from text2sql.data import build_prompt, load_examples, load_schemas, serialize_schema
from text2sql.evaluate import run_official_eval
from text2sql.infer import generate_sql, load_model

DATA = Path(os.environ.get("T2S_DATA", str(Path.home() / "t2s-data/data")))
EVAL = Path(os.environ.get("T2S_EVAL", str(Path.home() / "t2s-data/test-suite-sql-eval")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="dev examples (first N)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--adapter", default=None, help="optional LoRA adapter path (fine-tuned eval)")
    ap.add_argument("--schema-style", default="with_keys", choices=["minimal", "with_types", "with_keys"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--etype", default="all", choices=["all", "exec", "match"])
    args = ap.parse_args()

    examples = load_examples(DATA / "dev.json")[: args.n]
    schemas = load_schemas(DATA / "tables.json")
    prompts = [
        build_prompt(ex.question, serialize_schema(schemas[ex.db_id], args.schema_style))
        for ex in examples
    ]
    tag = "FT" if args.adapter else "base"
    print(f"dev examples: {len(examples)} | model: {args.model} [{tag}] | schema: {args.schema_style}")

    model, tok = load_model(args.model, adapter_path=args.adapter)
    preds = generate_sql(model, tok, prompts, batch_size=args.batch_size)
    print("sample pred:", repr(preds[0]))

    gold_pairs = [(ex.query, ex.db_id) for ex in examples]
    out, metrics = run_official_eval(
        gold_pairs, preds, DATA / "database", DATA / "tables.json", EVAL,
        work_dir="/tmp/t2s-eval", etype=args.etype, python_bin=sys.executable,
    )
    print(out[-2500:])
    print("PARSED METRICS:", metrics)


if __name__ == "__main__":
    main()
