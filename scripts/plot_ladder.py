"""Accuracy-vs-size curve from results/ladder.json (base line w/ CIs + fine-tuned point).

Run: PYTHONPATH=src python scripts/plot_ladder.py
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.loads(open("results/ladder.json").read())
rows = d["rows"]
frontier = []
if os.path.exists("results/frontier.json"):
    frontier = json.loads(open("results/frontier.json").read())["rows"]
base = sorted((r for r in rows if not r["ft"]), key=lambda r: r["size"])
ft = [r for r in rows if r["ft"]]


def err(rs):
    return [[r["exec"] - r["ci95"][0] for r in rs], [r["ci95"][1] - r["exec"] for r in rs]]


plt.figure(figsize=(7, 4.5))
xs = [r["size"] for r in base]
ys = [r["exec"] for r in base]
plt.errorbar(xs, ys, yerr=err(base), marker="o", capsize=4, color="#1f6feb", label="base")
for r in base:
    plt.annotate(f"{r['exec']:.2f}", (r["size"], r["exec"]), textcoords="offset points", xytext=(6, 8), fontsize=8)

if ft:
    fx = [r["size"] for r in ft]
    fy = [r["exec"] for r in ft]
    plt.errorbar(fx, fy, yerr=err(ft), marker="s", linestyle="none", capsize=4, color="#e8590c", label="fine-tuned (LoRA)")
    for r in ft:
        plt.annotate(f"{r['exec']:.2f}", (r["size"], r["exec"]), textcoords="offset points", xytext=(6, -14), fontsize=8, color="#e8590c")

# Frontier models as horizontal reference lines (no meaningful x-position).
fcolors = ["#2f9e44", "#9c36b5", "#1098ad"]
for i, r in enumerate(frontier):
    plt.axhline(r["exec"], color=fcolors[i % len(fcolors)], linestyle="--", linewidth=1.2,
                label=f"{r['label']} (frontier) {r['exec']:.2f}")

plt.xscale("log")
plt.xticks(xs, [str(s) for s in xs])
plt.xlabel("model size (B params, log scale)")
plt.ylabel("execution accuracy")
plt.ylim(0, 1)
plt.title(f"Text-to-SQL: execution accuracy vs size (Spider dev n={base[0]['n']}, {d['schema_style']})")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/ladder.png", dpi=120)
print("wrote results/ladder.png")
