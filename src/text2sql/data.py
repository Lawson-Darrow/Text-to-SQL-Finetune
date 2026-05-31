"""Spider data loading + schema serialization.

Schema serialization is a first-class, ablated component (per the plan): the format we
feed the model is one of the highest-signal levers, so it lives here behind a `style`
switch we can sweep.

Standard Spider layout (after download):
    <root>/tables.json
    <root>/dev.json                  (and train_spider.json)
    <root>/database/<db_id>/<db_id>.sqlite
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Schema:
    db_id: str
    # table name -> list of (column_name, type)
    tables: dict[str, list[tuple[str, str]]]
    primary_keys: list[str]  # "table.col"
    foreign_keys: list[tuple[str, str]]  # ("table.col", "table.col")


@dataclass
class Example:
    db_id: str
    question: str
    query: str  # gold SQL


def load_schemas(tables_json: str | Path) -> dict[str, Schema]:
    """Parse Spider's tables.json into per-db Schema objects."""
    out: dict[str, Schema] = {}
    for db in json.loads(Path(tables_json).read_text(encoding="utf-8")):
        tnames = db["table_names_original"]
        cols = db["column_names_original"]  # [[table_idx, col_name], ...]; [0] is [-1, "*"]
        ctypes = db["column_types"]
        tables: dict[str, list[tuple[str, str]]] = {t: [] for t in tnames}
        col_ref: list[str] = []  # index -> "table.col" (for keys); index 0 is "*"
        for (ti, cname), ctype in zip(cols, ctypes):
            if ti < 0:
                col_ref.append("*")
                continue
            tables[tnames[ti]].append((cname, ctype))
            col_ref.append(f"{tnames[ti]}.{cname}")
        pks = [col_ref[i] for i in db.get("primary_keys", []) if i < len(col_ref)]
        fks = [
            (col_ref[a], col_ref[b])
            for a, b in db.get("foreign_keys", [])
            if a < len(col_ref) and b < len(col_ref)
        ]
        out[db["db_id"]] = Schema(db["db_id"], tables, pks, fks)
    return out


def load_examples(dataset_json: str | Path) -> list[Example]:
    rows = json.loads(Path(dataset_json).read_text(encoding="utf-8"))
    return [Example(r["db_id"], r["question"], r["query"]) for r in rows]


def serialize_schema(schema: Schema, style: str = "with_keys") -> str:
    """Render a schema as prompt text. `style` is the ablation knob:

    - "minimal":   CREATE-like table(col, col, ...)
    - "with_types": table(col type, ...)
    - "with_keys":  with_types + PRIMARY KEY / FOREIGN KEY lines
    """
    lines: list[str] = []
    for table, cols in schema.tables.items():
        if style == "minimal":
            body = ", ".join(c for c, _ in cols)
        else:
            body = ", ".join(f"{c} {t}" for c, t in cols)
        lines.append(f"{table}({body})")

    if style == "with_keys":
        if schema.primary_keys:
            lines.append("-- primary keys: " + ", ".join(schema.primary_keys))
        if schema.foreign_keys:
            fk = ", ".join(f"{a} = {b}" for a, b in schema.foreign_keys)
            lines.append("-- foreign keys: " + fk)
    return "\n".join(lines)


SYSTEM = (
    "You are an expert at translating natural-language questions into a single SQLite "
    "SQL query. Output only the SQL query, no explanation."
)


def build_prompt(question: str, schema_text: str) -> str:
    return (
        f"Database schema:\n{schema_text}\n\n"
        f"Question: {question}\n"
        f"SQL:"
    )
