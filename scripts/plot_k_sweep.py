"""Experiment 03 figure: what the surprisal filter k does to the Delta-based results and to detection.

(a) Each quantity as a share of its k = 0 value: DiD on B/C (filter, and the same number of tokens removed at
    random), polishing slope s_post, polish_24 (U), and the size of the B vs GPT-Neo gap (the knowledge effect).
(b) Detection C vs B, AUROC, Delta 2024-2022 and 2024-2021: filter vs random removal.

Colours follow 01: blue = chat-model vs human, orange = polishing, aqua = human drift, ink grey = GPT-Neo.
Dashed hollow markers = random removal.

Example:
  python scripts/plot_k_sweep.py --eval outputs/k_sweep.json --out docs/experiments/figs/03-k-sweep.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#9a9893", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="k_sweep.py --json output")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = json.loads(Path(args.eval).read_text(encoding="utf-8"))
    ks = np.array(res["ks"])
    P = {q: np.array(v) for q, v in res["point"].items()}
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), facecolor=SURFACE,
                             gridspec_kw=dict(left=0.07, right=0.985, bottom=0.13, top=0.88, wspace=0.26))
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.set_axisbelow(True)
        ax.grid(color=GRID, linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xticks(ks, [f"{k:g}" for k in ks])
        ax.set_xlabel("k: share of tokens removed (the old model's most surprising)")

    def line(ax, y, color, label, dashed=False, marker="o"):
        ax.plot(ks, y, color=color, marker=marker, markersize=5.5, linewidth=1.5, linestyle="--" if dashed else "-",
                markerfacecolor=SURFACE if dashed else color, markeredgewidth=1.2, label=label)

    ax = axes[0]
    ax.axhline(1, color=INK_2, linewidth=0.8, alpha=0.6)
    line(ax, P["DiD"] / P["DiD"][0], BLUE, "DiD, chat-model vs human")
    line(ax, P["DiD_rand"] / P["DiD_rand"][0], BLUE, "DiD, same number removed at random", dashed=True)
    line(ax, P["s_post"] / P["s_post"][0], ORANGE, "polishing slope s_post")
    line(ax, P["U"] / P["U"][0], AQUA, "2024 human drift (polish_24)")
    line(ax, P["D_gap"] / P["D_gap"][0], INK_2, "|human − GPT-Neo gap| (knowledge)", marker="s")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("share of the k = 0 value")
    ax.legend(loc="lower left", frameon=False, fontsize=7.5)
    ax.set_title("(a) Results as k grows", loc="left", fontsize=10, color=INK)

    ax = axes[1]
    D = res["detection"]
    for pname, color, m in (("2024-2022", BLUE, "o"), ("2024-2021", INK_2, "s")):
        line(ax, [v["auroc"] for v in D[f"{pname}_filter"]], color, f"Δ{pname}, filter", marker=m)
        line(ax, [v["auroc"] for v in D[f"{pname}_random"]], color, f"Δ{pname}, random removal", dashed=True, marker=m)
    ax.set_ylabel("AUROC, chat-model vs human abstracts")
    ax.legend(loc="lower center", frameon=False, fontsize=7.5, ncol=2)
    lo = min(min(v["auroc"] for v in D[k_]) for k_ in D)
    ax.set_ylim(lo - 0.025, max(max(v["auroc"] for v in D[k_]) for k_ in D) + 0.008)
    ax.set_title("(b) Detection", loc="left", fontsize=10, color=INK)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
