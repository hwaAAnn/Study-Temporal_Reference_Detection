"""Mean Delta of human texts and their same-topic AI anchors, per period (experiment 02).

One panel per model pair (post-AI 2024-2021, pre-AI 2021-2018); x = period of the human texts; two
series (human, AI) with 95% bootstrap intervals. The vertical distance between the series is the C-H
gap that mia_contrast.py compares across periods. Colours follow 00's figure (human = slot 1 blue,
AI = slot 2 orange); both series are also labelled directly, so colour is not the only cue.

Example:
  python scripts/plot_period_means.py --scores outputs/mia.jsonl \
      --periods 2019:H19:C19 2022:H22:C22 2024:H24:C24 2025+:B:C --out docs/experiments/figs/02-period-means.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
HUMAN, AI = "#2a78d6", "#eb6834"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--periods", nargs="+", required=True, help="label:human_group:ai_group")
    ap.add_argument("--pairs", nargs="+", default=["2024-2021", "2021-2018"])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = [json.loads(l) for l in Path(args.scores).read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = np.random.default_rng(0)
    periods = [p.split(":") for p in args.periods]

    def mean_ci(v):
        boots = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(args.n_boot)]
        return v.mean(), np.quantile(boots, 0.025), np.quantile(boots, 0.975)

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    fig, axes = plt.subplots(1, len(args.pairs), figsize=(4.4 * len(args.pairs), 3.3), sharey=True, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    x = np.arange(len(periods))
    for ax, pair in zip(axes, args.pairs):
        new, old = pair.split("-")
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.axhline(0, color=INK_2, linewidth=0.8, alpha=0.6)
        ends = []
        for name, color, idx in (("human", HUMAN, 1), ("AI (same topic)", AI, 2)):
            stats = []
            for p in periods:
                v = np.array([np.mean(np.array(r["lp"][new]) - np.array(r["lp"][old]))
                              for r in recs if r["group"] == p[idx]])
                stats.append(mean_ci(v))
            m = np.array([s[0] for s in stats])
            lo = m - np.array([s[1] for s in stats])
            hi = np.array([s[2] for s in stats]) - m
            ax.errorbar(x, m, yerr=[lo, hi], color=color, linewidth=2, marker="o", markersize=5, capsize=3,
                        markeredgecolor=SURFACE, markeredgewidth=1.5)
            ends.append((name, m[-1]))
        # direct labels only where the two series end far enough apart to stay legible
        lo_y, hi_y = ax.get_ylim()
        if abs(ends[0][1] - ends[1][1]) > 0.08 * (hi_y - lo_y):
            for name, y in ends:
                ax.annotate(name, (x[-1], y), xytext=(8, 0), textcoords="offset points", va="center",
                            color=INK_2, fontsize=8)
        ax.set_xticks(x, [p[0] for p in periods])
        ax.set_xlim(-0.3, len(periods) - 1 + 0.9)
        ax.set_title(f"Δ {new} − {old}", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("first posted (human text)")
    axes[0].set_ylabel("mean per-token Δ (nats)")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, linewidth=2, marker="o", markersize=5) for c in (HUMAN, AI)]
    fig.legend(handles, ["human", "AI (same topic)"], loc="upper center", ncol=2, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.0))
    if args.title:
        fig.suptitle(args.title, x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
