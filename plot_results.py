"""Будує графіки навчання й оцінювання з results/.

    python tools/plot_results.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "docs" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED, GRID = "#1f2328", "#6e7781", "#e6e8eb"
C1, C2, C3 = "#2f6fdb", "#e0663a", "#1a9e77"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
})

metrics = [json.loads(l) for l in (RES / "metrics.jsonl").read_text().splitlines() if l.strip()]
steps = np.array([m["steps"] for m in metrics]) / 1e6
score = np.array([m["mean_score_50"] for m in metrics], dtype=float)
win = np.array([m["win_rate_50"] for m in metrics], dtype=float) * 100
hits = np.array([m["mean_hits_50"] for m in metrics], dtype=float)
ent = np.array([m["entropy"] for m in metrics])
bonus = np.array([m["hit_bonus"] for m in metrics])

eps = np.genfromtxt(RES / "episodes.csv", delimiter=",", names=True)

fig, axes = plt.subplots(2, 2, figsize=(10, 6.4))
ax = axes[0, 0]
ax.scatter(eps["step"] / 1e6, eps["score"], s=3, color=C1, alpha=0.18, linewidths=0)
ax.plot(steps, score, color=C1, lw=2)
ax.axhline(0, color=MUTED, lw=1, ls="--")
ax.set_title("Різниця рахунку за партію (агент − CPU)", color=INK, loc="left")
ax.set_xlabel("кроків агента, млн"); ax.set_ylim(-22, 22)

ax = axes[0, 1]
ax.plot(steps, win, color=C3, lw=2)
ax.set_title("Частка виграних партій, %", color=INK, loc="left")
ax.set_xlabel("кроків агента, млн"); ax.set_ylim(-3, 103)

ax = axes[1, 0]
ax.plot(steps, hits, color=C2, lw=2)
ax.set_title("Відбиттів м'яча агентом за партію", color=INK, loc="left")
ax.set_xlabel("кроків агента, млн")

ax = axes[1, 1]
ax.plot(steps, ent, color=C1, lw=2, label="ентропія політики")
ax2 = ax.twinx()
ax2.plot(steps, bonus, color=C2, lw=2, ls="--", label="бонус за відбиття")
ax2.spines["right"].set_visible(True); ax2.grid(False)
ax.set_title("Ентропія політики і бонус за відбиття", color=INK, loc="left")
ax.set_xlabel("кроків агента, млн")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper right", fontsize=9)

for a in axes.flat:
    a.title.set_fontsize(10.5)
fig.tight_layout()
fig.savefig(OUT / "training_history.png", dpi=170)
print("saved training_history.png")

ev_path = RES / "evaluation_final.json"
if ev_path.exists():
    ev = json.loads(ev_path.read_text())
    diffs = [g["diff"] for g in ev["games"]]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    vals, counts = np.unique(diffs, return_counts=True)
    ax.set_axisbelow(True)
    ax.bar(vals, counts, color=C3, width=0.8)
    ax.set_title(
        f"Фінальний тест: {ev['episodes']} нових партій, перемог {ev['wins']}/{ev['episodes']}",
        color=INK, loc="left", fontsize=10.5)
    ax.set_xlabel("різниця рахунку агент − CPU (максимум +21)")
    ax.set_ylabel("партій")
    ax.set_xticks(range(min(vals.min(), 10), 22, 1))
    fig.tight_layout()
    fig.savefig(OUT / "evaluation_hist.png", dpi=170)
    print("saved evaluation_hist.png")
