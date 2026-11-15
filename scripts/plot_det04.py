"""Experiment 04 figure: AUROC of each detector by condition, on RAID abstracts and APT-Eval.

(a) RAID abstracts: chat generators (greedy, sampling, and the four attacked with paraphrase / synonym), and the
    pre-chat base generators (greedy, sampling). 95% bootstrap intervals over sources.
(b) APT-Eval: human texts polished by five LLMs, by polishing degree (pooled over polishers and domains).

Detectors: the time-pair score s_0.75 (blue, filled), Delta (blue, hollow), likelihood under DatedGPT-2024 (ink
grey square), Binoculars (orange diamond), Fast-DetectGPT (aqua triangle).

Example:
  python scripts/plot_det04.py --eval outputs/det04.json --out docs/experiments/figs/04-detection.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#9a9893", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
DETS = [("s075", "time pair s₀.₇₅ (ours)", BLUE, "o", True), ("delta", "Δ 2024−2022 (ours)", BLUE, "o", False),
        ("ll24", "likelihood (DatedGPT-2024)", INK_2, "s", True), ("bino", "Binoculars (Llama-3.2-1B pair)", ORANGE, "D", True),
        ("fdgpt", "Fast-DetectGPT (GPT-Neo-2.7B)", AQUA, "^", True)]
RAID_ROWS = [("RAID chat, greedy", "chat, greedy"), ("RAID chat, sampling", "chat, sampling"),
             ("RAID 4 chat, sampling +paraphrase", "chat + paraphrase"),
             ("RAID 4 chat, sampling +synonym", "chat + synonym"),
             ("RAID pre-chat, greedy", "pre-chat, greedy"), ("RAID pre-chat, sampling", "pre-chat, sampling")]
APT_ROWS = [("APT extreme_minor", "extreme minor"), ("APT minor", "minor"), ("APT slight_major", "slight major"),
            ("APT major", "major")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="det04.py --json output")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = json.loads(Path(args.eval).read_text(encoding="utf-8"))
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4), facecolor=SURFACE,
                             gridspec_kw=dict(left=0.13, right=0.99, bottom=0.12, top=0.80, wspace=0.42,
                                              width_ratios=[1.25, 1]))
    offs = {d: (k - 2) * 0.13 for k, (d, *_) in enumerate(DETS)}
    for ax, rows, title in ((axes[0], RAID_ROWS, "(a) RAID abstracts: AI-written (chat: sampling unless greedy)"),
                            (axes[1], APT_ROWS, "(b) APT-Eval: human text polished by an LLM")):
        ax.set_facecolor(SURFACE)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.axvline(0.5, color=INK_3, linewidth=0.8, linestyle=":")
        for r, (key, _) in enumerate(rows):
            for d, _, color, m, filled in DETS:
                if key in res["pools"]:
                    v = res["pools"][key]["metrics"][d]["auroc"]
                    lo, hi = res["pools"][key]["auroc_ci"][d]
                    ax.plot([lo, hi], [r + offs[d]] * 2, color=color, linewidth=1.1)
                else:
                    v = res["single"][key][d]["auroc"]
                ax.plot(v, r + offs[d], marker=m, markersize=6, color=color, linestyle="none",
                        markerfacecolor=color if filled else SURFACE, markeredgecolor=color, markeredgewidth=1.3)
        lo_x = min(min(ln.get_xdata()) for ln in ax.lines if len(ln.get_xdata()) and ln.get_xdata()[0] != 0.5)
        ax.set_xlim(min(0.45, lo_x - 0.03), 1.02)
        ax.set_yticks(range(len(rows)), [lab for _, lab in rows])
        ax.invert_yaxis()
        ax.set_xlabel("AUROC (AI vs human)")
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    handles = [plt.Line2D([], [], marker=m, color=c, linestyle="none", markersize=6.5,
                          markerfacecolor=c if f else SURFACE, markeredgewidth=1.3) for _, _, c, m, f in DETS]
    fig.legend(handles, [lab for _, lab, *_ in DETS], loc="upper center", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.55, 1.0), columnspacing=1.6)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
