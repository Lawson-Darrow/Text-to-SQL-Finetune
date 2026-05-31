"""Milestone 4 — schema-representation ablation.

The high-signal axis: hold the model fixed and vary only how the DB schema is serialized
(minimal / with_types / with_keys), measuring execution accuracy via the official evaluator.
Writes results/schema_ablation_<model>.{json,png}.

Run (WSL2, derisk venv):
  PYTHONPATH=src python scripts/run_ablation.py --n 100 --model Qwen/Qwen2.5-Coder-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from text2sql.data import build_prompt, load_examples, load_schemas, serialize_schema
from text2sql.evaluate import run_official_eval
from text2sql.infer import generate_sql, load_model

DATA = Path(os.environ.get("T2S_DATA", str(Path.home() / "t2s-data/data")))
EVAL = Path(os.environ.get("T2S_EVAL", str(Path.home() / "t2s-data/test-suite-sql-eval")))
STYLES = ["minimal", "with_types", "with_keys"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    examples = load_examples(DATA / "dev.json")[: args.n]
    schemas = load_schemas(DATA / "tables.json")
    gold_pairs = [(ex.query, ex.db_id) for ex in examples]
    print(f"ablation | model={args.model} | n={len(examples)} | styles={STYLES}")

    model, tok = load_model(args.model)  # load once, reuse across styles

    results: dict[str, dict] = {}
    for style in STYLES:
        prompts = [
            build_prompt(ex.question, serialize_schema(schemas[ex.db_id], style)) for ex in examples
        ]
        preds = generate_sql(model, tok, prompts, batch_size=args.batch_size)
        _, metrics = run_official_eval(
            gold_pairs, preds, DATA / "database", DATA / "tables.json", EVAL,
            work_dir=f"/tmp/t2s-ablation/{style}", etype="all", python_bin=sys.executable,
        )
        results[style] = metrics
        print(f"  {style:12s} -> {metrics}")

    tag = args.model.split("/")[-1]
    Path("results").mkdir(exist_ok=True)
    payload = {"model": args.model, "n": len(examples), "results": results}
    Path(f"results/schema_ablation_{tag}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    acc = [results[s].get("execution", 0.0) for s in STYLES]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(STYLES, acc, color="#1f6feb")
    for b, v in zip(bars, acc):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center")
    plt.ylim(0, 1)
    plt.ylabel("execution accuracy")
    plt.title(f"Schema serialization ablation\n{tag}, n={len(examples)} Spider dev")
    plt.tight_layout()
    plt.savefig(f"results/schema_ablation_{tag}.png", dpi=120)
    print(f"wrote results/schema_ablation_{tag}.json + .png")


if __name__ == "__main__":
    main()
