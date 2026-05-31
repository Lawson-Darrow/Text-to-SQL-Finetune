"""Base vs fine-tuned comparison chart from results/finetune_compare.json.

Run: PYTHONPATH=src python scripts/plot_finetune.py
"""

from __future__ import annotations

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.loads(open("results/finetune_compare.json").read())
metrics = ["execution", "exact_match"]
base = [d["base"][m] for m in metrics]
ft = [d["finetuned"][m] for m in metrics]

x = np.arange(len(metrics))
w = 0.35
plt.figure(figsize=(6.5, 4.5))
b1 = plt.bar(x - w / 2, base, w, label="base", color="#9aa4b2")
b2 = plt.bar(x + w / 2, ft, w, label="fine-tuned (LoRA)", color="#1f6feb")
for bars in (b1, b2):
    for b in bars:
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.2f}", ha="center", fontsize=9)
plt.xticks(x, ["execution acc", "exact-match"])
plt.ylim(0, 1.0)
plt.ylabel("accuracy")
plt.title(f"LoRA SFT: base vs fine-tuned\n{d['model']}, Spider dev n={d['n']}, {d['schema_style']}")
plt.legend()
plt.tight_layout()
plt.savefig("results/finetune_compare.png", dpi=120)
print("wrote results/finetune_compare.png")
