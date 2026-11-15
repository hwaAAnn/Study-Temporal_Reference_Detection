"""Experiment 08 figure: AUROC of Fast-DetectGPT and Binoculars for every (reference r, scorer s) DatedGPT cell.

One sequential blue ramp for AUROC (magnitude); each cell carries its value, so color is never the only
channel. Dashed lines mark the ChatGPT boundary (between 2022 and 2023). Outlined cells: the usual
single-model setting at 2022 and at 2024 (diagonal), and the mixed setting (reference 2022, scorer 2024).

Example:
  python scripts/plot_slots.py --eval outputs/s08_eval.json --out docs/experiments/figs/08-detector-slots.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

SURFACE, INK, INK_2, INK_3 = "#fcfcfb", "#0b0b0b", "#52514e", "#9a9893"
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # blue 100 -> 700
LO, HI = 0.70, 1.00
KEY = {("2022", "2022"): "single 2022", ("2024", "2024"): "single 2024", ("2022", "2024"): "mixed"}


def panel(ax, grid: dict, vint: list[str], title: str, cmap) -> object:
    n = len(vint)
    m = [[grid[f"{r}_{s}"] for s in vint] for r in vint]
    im = ax.imshow(m, cmap=cmap, vmin=LO, vmax=HI, origin="upper")
    for i, r in enumerate(vint):
        for j, s in enumerate(vint):
            v = m[i][j]
            dark = (v - LO) / (HI - LO) > 0.55
            ax.text(j, i, f"{v:.4f}"[1:], ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if dark else INK)
            if (r, s) in KEY:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.8))
    for k in (2.5,):
        ax.axhline(k, color=INK_2, lw=1, ls=(0, (3, 2)))
        ax.axvline(k, color=INK_2, lw=1, ls=(0, (3, 2)))
    ax.set_xticks(range(n), vint, fontsize=9, color=INK_2)
    ax.set_yticks(range(n), vint, fontsize=9, color=INK_2)
    ax.set_xlabel("scorer s (training-data year)", fontsize=9, color=INK_2)
    ax.set_ylabel("reference r (training-data year)", fontsize=9, color=INK_2)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="H19/C19")
    args = ap.parse_args()
    res = json.loads(Path(args.eval).read_text(encoding="utf-8"))
    vint = res["vintages"]
    s = res["sets"][args.set]
    cmap = LinearSegmentedColormap.from_list("blue", RAMP)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.3), facecolor=SURFACE)
    for ax, (det, name) in zip(axes, [("F", "Fast-DetectGPT"), ("Bi", "Binoculars")]):
        ax.set_facecolor(SURFACE)
        im = panel(ax, s["dets"][det]["auroc"], vint, name, cmap)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("AUROC", fontsize=9, color=INK_2)
    cb.ax.tick_params(labelsize=8, colors=INK_2, length=0)
    cb.outline.set_visible(False)
    fig.text(0.01, 0.01, f"{args.set} ({s['n_pairs']} pairs). Dashed: ChatGPT boundary. Outlined: single-model "
             "setting at 2022 and 2024 (diagonal), and reference 2022 + scorer 2024.",
             fontsize=8, color=INK_3)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
