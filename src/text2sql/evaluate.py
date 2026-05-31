"""Wrapper around the official Spider / test-suite evaluator (taoyds/test-suite-sql-eval).

Per the plan we do NOT roll our own row-compare as the headline metric — execution accuracy
has too many pitfalls. We write gold/pred files in the evaluator's format and shell out to its
`evaluation.py`, then parse the report (exact-match + execution accuracy, overall + per-hardness).
"""

from __future__ import annotations

import random
import re
import subprocess
import sys
from pathlib import Path


def per_example_execution(
    preds: list[str],
    gold_pairs: list[tuple[str, str]],
    db_dir: str | Path,
    evaluator_dir: str | Path,
) -> list[bool]:
    """Per-example execution correctness via the official `eval_exec_match` (so CIs can be
    bootstrapped). Returns one bool per example."""
    sys.path.insert(0, str(evaluator_dir))
    from exec_eval import eval_exec_match  # noqa: E402

    out: list[bool] = []
    for pred, (gold, db_id) in zip(preds, gold_pairs):
        db = str(Path(db_dir) / db_id / f"{db_id}.sqlite")
        try:
            ok = eval_exec_match(db, pred or "SELECT 1", gold, False, False, False)
        except Exception:  # noqa: BLE001
            ok = 0
        out.append(bool(ok))
    return out


def bootstrap_ci(flags: list[bool], n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """(mean, lo95, hi95) accuracy with a percentile bootstrap CI."""
    n = len(flags)
    if n == 0:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    mean = sum(flags) / n
    boots = sorted(sum(flags[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return (mean, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)])


def write_eval_files(gold_pairs: list[tuple[str, str]], preds: list[str], work_dir: str | Path):
    """gold_pairs: (gold_sql, db_id); preds: predicted_sql (same order)."""
    wd = Path(work_dir)
    wd.mkdir(parents=True, exist_ok=True)
    gold_f, pred_f = wd / "gold.txt", wd / "pred.txt"
    gold_f.write_text(
        "\n".join(f"{' '.join(sql.split())}\t{db}" for sql, db in gold_pairs) + "\n",
        encoding="utf-8",
    )
    # predictions must be one line each and non-empty (blank lines break the parser)
    pred_f.write_text(
        "\n".join((" ".join(p.split()) or "SELECT 1") for p in preds) + "\n", encoding="utf-8"
    )
    return gold_f, pred_f


def parse_report(out: str) -> dict:
    """Best-effort parse of the evaluator's printed table; 'all' is the last column."""
    metrics: dict[str, float] = {}
    # The evaluator prints rows like "execution   <easy> <medium> <hard> <extra> <all>";
    # the 'all' column is the last number. Match both label spellings, take the last value.
    for label, key in [("exact match", "exact_match"), ("execution", "execution")]:
        m = re.search(rf"^{label}\s+([\d.]+(?:\s+[\d.]+)*)", out, re.I | re.M)
        if m:
            metrics[key] = float(m.group(1).split()[-1])
    return metrics


def run_official_eval(
    gold_pairs: list[tuple[str, str]],
    preds: list[str],
    db_dir: str | Path,
    tables_json: str | Path,
    evaluator_dir: str | Path,
    work_dir: str | Path,
    etype: str = "all",
    python_bin: str = "python",
) -> tuple[str, dict]:
    gold_f, pred_f = write_eval_files(gold_pairs, preds, work_dir)
    cmd = [
        python_bin,
        str(Path(evaluator_dir) / "evaluation.py"),
        "--gold", str(gold_f),
        "--pred", str(pred_f),
        "--db", str(db_dir),
        "--table", str(tables_json),
        "--etype", etype,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(evaluator_dir))
    out = res.stdout + ("\n--- stderr ---\n" + res.stderr if res.stderr else "")
    return out, parse_report(out)
