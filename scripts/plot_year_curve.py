"""Experiment 01 figure: when do DatedGPT vintages start to prefer AI-written text?

(a) G(v): how much vintage v prefers a same-title chat-model abstract over the human one (B/C, 2025+); the
    line fitted to the pre-ChatGPT vintages 2013-2022 and its 95% prediction band from their scatter
    (model-to-model noise). No per-point intervals: the same 400 pairs enter every vintage, so the interval
    of a level says nothing about the gap between vintages; the departures' intervals are in (b).
(b) Each curve minus its own 2013-2022 line: chat AI vs human (G), polishing slope on the same text (S),
    drift of 2024 human abstracts toward AI (U), GPT-Neo vs human (D'). 95% bootstrap intervals over texts
    at 2023 and 2024.
(c) Membership fingerprint: FineWeb-Edu documents minus their AI continuations (M), relative to their
    2013-2019 mean, for documents from the 2021 and 2022 crawls. After 2022 the AI continuation side is
    favoured (the shift of a and b), so the drop there is not membership.

Colours follow the entity: slot 1 blue = chat-model text, slot 2 orange = polishing, slot 3 aqua = human
drift; ink grey = GPT-Neo and the web documents (told apart by marker and direct label).

Example:
  python scripts/plot_year_curve.py --eval outputs/yc_eval.json --out docs/experiments/figs/01-year-curve.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#9a9893", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
T975_8 = 2.306


def trend(years, c):
    pre = years <= 2022
    b, a = np.polyfit(years[pre], c[pre], 1)
    res = c[pre] - (a + b * years[pre])
    s = np.sqrt((res ** 2).sum() / (pre.sum() - 2))
    xm, sxx = years[pre].mean(), ((years[pre] - years[pre].mean()) ** 2).sum()
    se = s * np.sqrt(1 + 1 / pre.sum() + (years - xm) ** 2 / sxx)
    return a + b * years, se


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="year_curve.py --json output")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    res = json.loads(Path(args.eval).read_text(encoding="utf-8"))
    years = np.array(res["years"], float)
    C = res["curves"]
    cur = lambda k: np.array(C[k]["curve"])

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    fig = plt.figure(figsize=(13.2, 4.5), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, left=0.055, right=0.99, bottom=0.12, top=0.79, wspace=0.24)
    axes = [fig.add_subplot(gs[k]) for k in range(3)]
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.set_axisbelow(True)
        ax.grid(color=GRID, linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.axvspan(2022.5, 2024.6, color=GRID, alpha=0.35, linewidth=0)
        ax.set_xlim(2012.5, 2024.6)
        ax.set_xticks(range(2013, 2025), [str(y) if y % 2 == 0 else "" for y in range(2013, 2025)])
        ax.set_xlabel("model vintage (training-data crawl year)")
    dot = dict(markeredgecolor=SURFACE, markeredgewidth=1.2)

    # ---- (a) main curve with the pre-ChatGPT line and prediction band ----
    ax = axes[0]
    g = cur("G")
    line, se = trend(years, g)
    ax.fill_between(years, line - T975_8 * se, line + T975_8 * se, color=INK_3, alpha=0.18, linewidth=0)
    ax.plot(years[years <= 2022], line[years <= 2022], color=INK_2, linewidth=1.3)
    ax.plot(years[years >= 2022], line[years >= 2022], color=INK_2, linewidth=1.3, linestyle="--")
    ax.plot(years, g, color=BLUE, marker="o", markersize=6, linewidth=1.6, **dot)
    e = C["G"]
    ax.text(0.03, 0.96, "departure from the line\n"
            f"2023: r = {e['r2023']:+.2f} (z = {e['z2023']:.0f})\n2024: r = {e['r2024']:+.2f} (z = {e['z2024']:.0f})",
            transform=ax.transAxes, ha="left", va="top", fontsize=8, color=INK_2, linespacing=1.4)
    ax.set_ylabel("G(v) = log p(AI) − log p(human) (nats/token)")
    ax.set_title("(a) Same-title chat-model vs human abstracts (2025+)", loc="left", fontsize=10, color=INK)

    # ---- (b) departures from each curve's own pre-ChatGPT line ----
    ax = axes[1]
    ax.axhline(0, color=INK_2, linewidth=0.8, alpha=0.6)
    series = [("G", BLUE, "o", "-", "chat-model vs human (G)"),
              ("S", ORANGE, "o", "-", "polishing, same text (S, per unit edit)"),
              ("U", AQUA, "o", "-", "2024 human drift toward AI (U)"),
              ("G_D", INK_2, "s", "--", "GPT-Neo vs human (D′)")]
    for k, color, m, ls, _ in series:
        c = cur(k)
        line, _ = trend(years, c)
        d = c - line
        ax.plot(years, d, color=color, marker=m, markersize=5.5, linewidth=1.5, linestyle=ls,
                markerfacecolor=color if ls == "-" else SURFACE, markeredgewidth=1.2)
        for v in (2023, 2024):
            lo_, hi_ = C[k][f"r{v}_ci"]
            ax.plot([v, v], [lo_, hi_], color=color, linewidth=1.2)
    ax.set_ylabel("departure from own 2013–2022 line (nats/token)")
    ax.set_title("(b) What shifts after ChatGPT", loc="left", fontsize=10, color=INK)

    # ---- (c) membership fingerprint ----
    ax = axes[2]
    ax.axhline(0, color=INK_2, linewidth=0.8, alpha=0.6)
    for k, crawl, m, face in (("M_21", 2021, "o", INK_2), ("M_22", 2022, "s", SURFACE)):
        c = cur(k)
        c = c - c[(years >= 2013) & (years <= 2019)].mean()
        ax.axvline(crawl, color=INK_3, linewidth=0.8, linestyle=":")
        ax.plot(years, c, color=INK_2, marker=m, markersize=5.5, linewidth=1.4, markerfacecolor=face,
                markeredgewidth=1.2, linestyle="-" if k == "M_21" else "--",
                label=f"{crawl}-crawl documents")
    ax.legend(loc="upper left", frameon=False, fontsize=7.5, handlelength=2.2)
    f = res["f"]
    ax.text(0.02, 0.04, f"2021 peak lost at 2022: f = {f['value']:.2f} [{f['ci'][0]:.2f}, {f['ci'][1]:.2f}]\n"
            "(1 = own year's crawl only, 0 = cumulative)\n"
            "drop after 2022: AI continuation favoured (a, b)",
            transform=ax.transAxes, fontsize=7.5, color=INK_2, linespacing=1.4)
    ax.set_ylabel("doc − its AI continuation, vs 2013–19 mean")
    ax.set_title("(c) Web documents: which vintage saw them", loc="left", fontsize=10, color=INK)

    handles = [Line2D([], [], color=c, marker=m, markersize=6, linewidth=1.5, linestyle=ls,
                      markerfacecolor=c if ls == "-" else SURFACE) for _, c, m, ls, _ in series]
    labels = [lab for *_, lab in series]
    handles.append(Line2D([], [], color=INK_2, linewidth=1.3))
    labels.append("line fitted to 2013–22 (band: 95% prediction, model scatter)")
    handles.append(Line2D([], [], color=GRID, linewidth=8, alpha=0.8))
    labels.append("post-ChatGPT vintages")
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.0),
               handlelength=2.2, columnspacing=1.4)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
