"""Frontier baseline — GPT/Claude via LLMGateway on the same Spider eval as the open ladder.

Completes the open-vs-frontier comparison: same full dev set, same official evaluator,
same with_keys schema, bootstrap CIs. Writes results/frontier.json.

Run (WSL2): PYTHONPATH=src python scripts/run_frontier.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from text2sql.data import build_prompt, load_examples, load_schemas, serialize_schema
from text2sql.evaluate import bootstrap_ci, per_example_execution
from text2sql.gateway import generate_sql_gateway

DATA = Path(os.environ.get("T2S_DATA", str(Path.home() / "t2s-data/data")))
EVAL = Path(os.environ.get("T2S_EVAL", str(Path.home() / "t2s-data/test-suite-sql-eval")))
DEFAULT_MODELS = ["gpt-4.1-mini", "claude-haiku-4-5"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="0 = full dev (1034)")
    ap.add_argument("--schema-style", default="with_keys")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    examples = load_examples(DATA / "dev.json")
    if args.n:
        examples = examples[: args.n]
    schemas = load_schemas(DATA / "tables.json")
    prompts = [build_prompt(ex.question, serialize_schema(schemas[ex.db_id], args.schema_style)) for ex in examples]
    gold_pairs = [(ex.query, ex.db_id) for ex in examples]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"frontier | n={len(examples)} | models={models}")

    results = []
    for m in models:
        preds = generate_sql_gateway(m, prompts, workers=args.workers)
        flags = per_example_execution(preds, gold_pairs, DATA / "database", EVAL)
        mean, lo, hi = bootstrap_ci(flags)
        results.append({"label": m, "model": m, "frontier": True, "exec": round(mean, 4),
                        "ci95": [round(lo, 4), round(hi, 4)], "n": len(examples)})
        print(f"  {m:18s} exec={mean:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

    Path("results").mkdir(exist_ok=True)
    Path("results/frontier.json").write_text(
        json.dumps({"schema_style": args.schema_style, "rows": results}, indent=2), encoding="utf-8"
    )
    print("wrote results/frontier.json")


if __name__ == "__main__":
    main()
