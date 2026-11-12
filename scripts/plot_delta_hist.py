"""Delta distributions per group, one panel per model pair (e.g. post-AI 2024-2021, pre-AI 2021-2018).

Step histograms (density) so overlapping groups stay readable; panels share the x axis so the shift
between the post-AI and pre-AI pair can be compared directly. Colours are the first three categorical
slots of the reference palette (they pass the all-pairs colour-vision checks in light mode); the
group-mean table next to the figure in the docs is the non-colour way to read it.

Example:
  python scripts/plot_delta_hist.py --scores outputs/k0.jsonl --pairs 2024-2021 2021-2018 \
      --out docs/experiments/figs/00-delta-hist.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"B": "#2a78d6", "C": "#eb6834", "D": "#1baf7a", "A": "#eda100"}
NAMES = {"A": "A human, pre-2021", "B": "B human, 2025+", "C": "C chat LLM", "D": "D′ pre-ChatGPT LM"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--pairs", nargs="+", default=["2024-2021", "2021-2018"], help="new-old model keys")
    ap.add_argument("--groups", nargs="+", default=["B", "C", "D"])
    ap.add_argument("--names", nargs="*", default=None, help="legend names, one per group (default: built-in)")
    ap.add_argument("--bins", type=int, default=40)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = [json.loads(l) for l in Path(args.scores).read_text(encoding="utf-8").splitlines() if l.strip()]
    names = dict(zip(args.groups, args.names)) if args.names else NAMES
    vals = {}
    for pair in args.pairs:
        new, old = pair.split("-")
        for g in args.groups:
            vals[pair, g] = np.array([np.mean(np.array(r["lp"][new]) - np.array(r["lp"][old]))
                                      for r in recs if r["group"] == g])
    allv = np.concatenate([v for v in vals.values() if len(v)])
    lo, hi = np.quantile(allv, [0.002, 0.998])
    bins = np.linspace(lo, hi, args.bins + 1)

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    fig, axes = plt.subplots(1, len(args.pairs), figsize=(4.4 * len(args.pairs), 3.2), sharex=True, sharey=True,
                             facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, pair in zip(axes, args.pairs):
        new, old = pair.split("-")
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.axvline(0, color=INK_2, linewidth=0.8, alpha=0.6)
        for g in args.groups:
            v = vals[pair, g]
            if not len(v):
                continue
            ax.hist(np.clip(v, lo, hi), bins=bins, density=True, histtype="step", linewidth=1.6,
                    color=SERIES.get(g, INK_2))
            ax.axvline(v.mean(), color=SERIES.get(g, INK_2), linewidth=1.0, linestyle=(0, (3, 2)))
        ax.set_title(f"Δ {new} − {old}", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("mean per-token log-prob difference (nats)")
    axes[0].set_ylabel("density")
    # one legend row above the panels (inside a panel it covers the curves); means are per panel, so the
    # legend names the groups and the dashed lines mark each group's mean
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=SERIES.get(g, INK_2), linewidth=1.6) for g in args.groups]
    labels = [f"{names.get(g, g)} (n={len(vals[args.pairs[0], g])})" for g in args.groups]
    handles.append(Line2D([], [], color=INK_2, linewidth=1.0, linestyle=(0, (3, 2))))
    labels.append("group mean")
    fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
