"""Experiment 05 figure: Delta rises with the share of an abstract an LLM changed; how that carries over
to real LLM polishers; and the AI-edited share it implies for 2024 and 2025+ human abstracts.

(a) Local chat models polishing pre-ChatGPT (2022) abstracts: mean shift Delta(polished) - Delta(original)
    in bins of the share of words changed, with 95% paper-bootstrap intervals and the fitted line through
    0; the AI-free pair (2021-2018) as placebo; the same-title AI abstract as the "from scratch" end point.
(b) APT-Eval, the same 51 originals polished by our local models, GPT-4o and three other large LLMs:
    per (polisher, level) means and the fitted lines through 0.
(c) beta = population shift / slope (AI-edited share of the text) for 2024 and 2025+, per calibration,
    with 95% bootstrap intervals; Liang et al. (2024) for reference (their unit is sentences).

Colours follow the entity: slot 1 blue = our local models, slot 2 orange = GPT-4o, slot 3 aqua = other
large LLMs, ink grey = placebo and marker words. In (c) the period is the marker shape, not a colour.

Example:
  python scripts/plot_polish_calibration.py --eval outputs/polish_eval.json \
      --out docs/experiments/figs/05-polish-calibration.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
LOCAL_C, GPT_C, OTHER_C = "#2a78d6", "#eb6834", "#1baf7a"
LOCAL = ["qwen2.5-1.5b-instruct", "llama3.2-1b-instruct"]
OTHERS = ["DeepSeek-V3", "Llama3.1-70B", "Llama3-8B"]
PERIODS = [("24", "2024 human abstracts", "o"), ("25", "2025+ human abstracts", "s")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="polish_calibration.py --json output")
    ap.add_argument("--out", required=True)
    ap.add_argument("--liang", type=float, default=0.175, help="Liang et al. 2024, CS abstracts by Feb 2024")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    res = json.loads(Path(args.eval).read_text(encoding="utf-8"))
    est = res["estimates"]
    val = lambda k: est[k]["value"] if k in est else np.nan
    cis = lambda k: est[k]["ci"] if k in est else [np.nan, np.nan]

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    fig = plt.figure(figsize=(13.2, 4.1), facecolor=SURFACE)
    gs_ab = fig.add_gridspec(1, 2, left=0.055, right=0.6, bottom=0.14, top=0.83, wspace=0.08)
    gs_c = fig.add_gridspec(1, 1, left=0.775, right=0.99, bottom=0.14, top=0.83)  # room for row labels
    axa = fig.add_subplot(gs_ab[0])
    axb = fig.add_subplot(gs_ab[1], sharey=axa)
    axc = fig.add_subplot(gs_c[0])
    for ax in (axa, axb, axc):
        ax.set_facecolor(SURFACE)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    dot = dict(markeredgecolor=SURFACE, markeredgewidth=1.2, linestyle="none")
    xs = np.linspace(0, 1, 50)

    # ---- (a) local polishing of 2022 abstracts ----
    axa.grid(color=GRID, linewidth=0.6)
    axa.axhline(0, color=INK_2, linewidth=0.8, alpha=0.6)
    bins = res["bins_local_H22"]
    e = np.array([b["e"] for b in bins])
    for key, ci_key, color, face, line in (("d_pre", "ci_d_pre", INK_2, SURFACE, "--"), ("d", "ci_d", LOCAL_C, LOCAL_C, "-")):
        m = np.array([b[key] for b in bins])
        lo, hi = m - np.array([b[ci_key][0] for b in bins]), np.array([b[ci_key][1] for b in bins]) - m
        s = val("s_all_d_pre" if key == "d_pre" else "s_all_d_post")
        axa.plot(xs, s * xs, color=color, linewidth=1.5, linestyle=line, alpha=0.7)
        axa.errorbar(e, m, yerr=[lo, hi], marker="o", markersize=6, color=color, markerfacecolor=face,
                     capsize=2.5, elinewidth=1.2, markeredgewidth=1.2, linestyle="none")
    sc = res["scratch"]
    axa.plot([sc["e"]], [sc["d_post"]], marker="D", markersize=7, color=LOCAL_C, **dot)
    left = sc["e"] > 0.5
    axa.annotate("same-title AI abstract\n(written from scratch)", (sc["e"], sc["d_post"]), xytext=(-8 if left else 8, 0),
                 textcoords="offset points", ha="right" if left else "left", va="center", fontsize=8, color=INK_2)
    axa.annotate(f"line through 0: slope {val('s_all_d_post'):.2f}", (1.0, val("s_all_d_post")), xytext=(-4, 8),
                 textcoords="offset points", ha="right", va="bottom", fontsize=8, color=INK_2)
    axa.annotate(f"Δ 2021−2018 (no AI in either model): slope {val('s_all_d_pre'):+.2f}", (0.98, val("s_all_d_pre") * 0.98),
                 xytext=(0, -12), textcoords="offset points", ha="right", fontsize=8, color=INK_2)
    axa.set_xlim(0, 1)
    axa.set_xlabel("share of words changed (1 − word-level similarity)")
    axa.set_ylabel("Δ shift, polished − original (nats/token)")
    axa.set_title("(a) Local models polish 2022 arXiv abstracts", loc="left", fontsize=10, color=INK)

    # ---- (b) APT-Eval: the same 51 originals, real LLM polishers ----
    axb.grid(color=GRID, linewidth=0.6)
    axb.axhline(0, color=INK_2, linewidth=0.8, alpha=0.6)
    for pols, key, color in ((LOCAL, "local", LOCAL_C), (OTHERS, "others", OTHER_C), (["GPT-4o"], "GPT-4o", GPT_C)):
        axb.plot(xs, val(f"apt_s_{key}_d_post") * xs, color=color, linewidth=1.5, alpha=0.7)
        pts = [s for p in pols for s in res["apt"].get(p, {}).values() if isinstance(s, dict) and "e" in s]
        axb.plot([s["e"] for s in pts], [s["d_post"] for s in pts], marker="o", markersize=6, color=color, **dot)
    r_txt = f"R (GPT-4o / local) = {val('R_GPT-4o'):.2f} [{cis('R_GPT-4o')[0]:.2f}, {cis('R_GPT-4o')[1]:.2f}]"
    axb.text(0.02, 0.97, r_txt, transform=axb.transAxes, va="top", fontsize=8, color=INK_2)
    axb.set_xlim(0, 1)
    axb.set_xlabel("share of words changed (1 − word-level similarity)")
    axb.tick_params(labelleft=False)
    axb.set_title("(b) Real LLM polishers (APT-Eval, 51 abstracts)", loc="left", fontsize=10, color=INK)

    # ---- (c) beta per calibration ----
    rows = [("local-model polishing (main)", "beta_{T}_local", LOCAL_C),
            ("− pre-ChatGPT trend", "beta_{T}_local_trend", LOCAL_C),
            ("− LLM-topic titles", "beta_{T}_local_nollm", LOCAL_C),
            ("GPT-4o-like polishing", "beta_{T}_GPT-4o", GPT_C)]
    rows += [(f"{p}-like", f"beta_{{T}}_{p}", OTHER_C) for p in OTHERS]
    rows += [("marker words (local slope)", "betaW_{T}_local", INK_2),
             ("marker words (GPT-4o slope)", "betaW_{T}_GPT-4o", INK_2)]
    lo_all, hi_all = [0.0], [args.liang]
    for k, (_, tmpl, _) in enumerate(rows):
        for T, _, _ in PERIODS:
            c = cis(tmpl.format(T=T))
            lo_all.append(c[0])
            hi_all.append(c[1])
    x_lo = max(-0.3, np.nanmin(lo_all) - 0.03)
    x_hi = min(1.2, np.nanmax(hi_all) + 0.05)  # wider intervals end in an arrow
    axc.grid(axis="x", color=GRID, linewidth=0.6)
    axc.axvline(0, color=INK_2, linewidth=0.8, alpha=0.6)
    if x_hi > 1:
        axc.axvspan(1, x_hi, color=GRID, alpha=0.5, linewidth=0)
        axc.text((1 + x_hi) / 2, len(rows) - 0.5, "β > 1", fontsize=7.5, color=INK_2, ha="center", va="top")
    axc.axvline(args.liang, color=INK_2, linewidth=1, linestyle=":")
    right = (args.liang - x_lo) / (x_hi - x_lo) > 0.55  # label on the side with room
    axc.text(args.liang, -1.45, f" Liang et al. 2024: {args.liang:.1%} \n (sentences, CS, Feb 2024) ",
             fontsize=7.5, color=INK_2, va="bottom", ha="right" if right else "left")
    for k, (label, tmpl, color) in enumerate(rows):
        y = len(rows) - 1 - k
        for j, (T, _, marker) in enumerate(PERIODS):
            key = tmpl.format(T=T)
            v, (lo, hi) = val(key), cis(key)
            yy = y + (0.17 if j == 0 else -0.17)
            clipped_lo, clipped_hi = max(lo, x_lo), min(hi, x_hi)
            axc.plot([clipped_lo, clipped_hi], [yy, yy], color=color, linewidth=1.6, solid_capstyle="round")
            if np.isfinite(v) and x_lo <= v <= x_hi:
                axc.plot([v], [yy], marker=marker, markersize=6.5, color=color, **dot)
            if hi > x_hi:
                axc.annotate("", (x_hi, yy), xytext=(x_hi - 0.04, yy),
                             arrowprops=dict(arrowstyle="->", color=color, linewidth=1.2))
    axc.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], fontsize=8)
    axc.tick_params(axis="y", length=0)
    axc.set_ylim(-1.5, len(rows) - 0.4)
    axc.set_xlim(x_lo, x_hi)
    axc.set_xlabel("β: share of the text AI edited (line through 0)")
    axc.set_title("(c) Implied AI-edited share", loc="left", fontsize=10, color=INK)

    handles = [Line2D([], [], color=LOCAL_C, marker="o", markersize=6, linewidth=1.5),
               Line2D([], [], color=GPT_C, marker="o", markersize=6, linewidth=1.5),
               Line2D([], [], color=OTHER_C, marker="o", markersize=6, linewidth=1.5),
               Line2D([], [], color=INK_2, marker="o", markersize=6, markerfacecolor=SURFACE, linewidth=1.5, linestyle="--")]
    labels = ["our local models (Qwen2.5-1.5B, Llama-3.2-1B)", "GPT-4o", "DeepSeek-V3, Llama3.1-70B, Llama3-8B",
              "placebo: Δ 2021−2018"]
    handles += [Line2D([], [], color=INK_2, marker=m, markersize=6, linestyle="none") for _, _, m in PERIODS]
    labels += [lab for _, lab, _ in PERIODS]
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.0),
               handlelength=2.2, columnspacing=1.4)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
