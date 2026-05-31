"""Combined ablation chart: execution accuracy by schema style, grouped by model.

Reads every results/schema_ablation_*.json and renders the size x schema-representation
figure (the multi-axis artifact). Run: PYTHONPATH=src python scripts/plot_ablation.py
"""

from __future__ import annotations

import glob
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLES = ["minimal", "with_types", "with_keys"]

data: dict[str, dict[str, float]] = {}
for f in sorted(glob.glob("results/schema_ablation_*.json")):
    if "combined" in f:
        continue
    d = json.loads(open(f).read())
    tag = d["model"].split("/")[-1]
    data[tag] = {s: d["results"].get(s, {}).get("execution", 0.0) for s in STYLES}

# smaller models first (by the number before 'B')
models = sorted(data, key=lambda m: float("".join(c for c in m.split("-")[2] if c.isdigit() or c == ".") or 0))
x = np.arange(len(STYLES))
w = 0.8 / max(len(models), 1)

plt.figure(figsize=(7, 4.5))
for i, m in enumerate(models):
    vals = [data[m][s] for s in STYLES]
    bars = plt.bar(x + i * w - (len(models) - 1) * w / 2, vals, w, label=m)
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)

plt.xticks(x, STYLES)
plt.ylim(0, 1.15)
plt.ylabel("execution accuracy")
plt.title("Text-to-SQL: schema representation x model size (Spider dev, n=100)")
plt.legend(loc="upper center", ncol=len(models))
plt.tight_layout()
plt.savefig("results/schema_ablation_combined.png", dpi=120)
print("wrote results/schema_ablation_combined.png for models:", models)
