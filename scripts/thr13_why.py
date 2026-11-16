"""Experiment 13, follow-up: why do time-contrast scores flag post-ChatGPT human abstracts?

Two candidate causes, measured with definitions fixed earlier in the repo (scripts/polish_calibration.py, 02/05):
  knowledge window  the abstract is about LLMs (LLM_TOPIC regex): content the 2024 model knows and the 2022 model doesn't
  AI polishing      rate of the 18 marker stems per 1,000 words (MARK)

A human abstract is "flagged" when its score exceeds the 13 threshold (5% FPR on the H19 human abstracts, s08 scores).
Groups: H19 (2019, control), H24 (2024), B (2025-07~2026-10). Bootstrap over abstracts (one per arXiv id), 2,000 draws,
seed 0; the threshold is held at its point value.

usage: python scripts/thr13_why.py --root <main checkout> [--json out.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from det04 import threshold5  # noqa: E402
from polish_calibration import LLM_TOPIC, marker_rate  # noqa: E402
from thr13 import dg_scores, read  # noqa: E402

DETS = ["F:2022_2024", "Bi:2022_2024", "s09", "ll24", "F:2024_2024", "F:2022_2022"]
GROUPS = ["H19", "H24", "B"]


def texts(root: Path) -> dict[str, str]:
    out = {}
    for line in (root / "data" / "01_all.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("human_group") in ("H19", "H24"):
                out[f"{r['id']}:h"] = r["human"]
    for line in (root / "data" / "12_input.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r["group"] == "B":
                out[r["id"]] = r["text"]
    return out


def boot_ci(fn, n: int, n_boot: int, rng) -> list[float]:
    vals = [fn(rng.integers(0, n, n)) for _ in range(n_boot)]
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    return np.percentile(vals, [2.5, 97.5]).tolist()


def ols_std(y, X):
    """Standardised OLS coefficients (y and every column z-scored)."""
    z = lambda a: (a - a.mean()) / (a.std() or 1.0)  # noqa: E731
    Z = np.column_stack([np.ones(len(y))] + [z(X[:, j]) for j in range(X.shape[1])])
    beta, *_ = np.linalg.lstsq(Z, z(y), rcond=None)
    return beta[1:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="main checkout (data/ and outputs/)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json")
    args = ap.parse_args()
    root = Path(args.root)
    grid = read(root / "outputs" / "s08_grid.jsonl")
    txt = texts(root)
    rows = {g: [] for g in GROUPS}
    for i, r in grid.items():
        if r["label"] == 0 and r["group"] in rows:
            if i not in txt:
                raise SystemExit(f"no text for {i}")
            s = dg_scores(r["cells"])
            rows[r["group"]].append((i, s, txt[i]))
    tau = {d: threshold5(np.array([s[d] for _, s, _ in rows["H19"]])) for d in DETS}
    rng = np.random.default_rng(args.seed)
    res = {"tau": tau, "groups": {}}
    for g in GROUPS:
        R = rows[g]
        n = len(R)
        topic = np.array([bool(LLM_TOPIC.search(t)) for _, _, t in R])
        mark = np.array([marker_rate(t) for _, _, t in R])
        S = {d: np.array([s[d] for _, s, _ in R]) for d in DETS}
        flag = {d: S[d] > tau[d] for d in DETS}
        out = {"n": n, "topic_share": float(topic.mean()),
               "topic_share_ci": boot_ci(lambda ix: topic[ix].mean(), n, args.n_boot, rng),
               "marker_mean": float(mark.mean()), "dets": {}}
        for d in DETS:
            f = flag[d]

            def rate_diff(ix, f=f):
                t, ff = topic[ix], f[ix]
                return ff[t].mean() - ff[~t].mean() if t.any() and (~t).any() else np.nan

            def mark_diff(ix, f=f):
                ff, m = f[ix], mark[ix]
                return m[ff].mean() - m[~ff].mean() if ff.any() and (~ff).any() else np.nan

            def coefs(ix, d=d):
                return ols_std(S[d][ix], np.column_stack([topic[ix].astype(float), mark[ix]]))

            cb = np.array([coefs(rng.integers(0, n, n)) for _ in range(args.n_boot)])
            b = coefs(np.arange(n))
            out["dets"][d] = {
                "fpr": float(f.mean()),
                "fpr_topic": float(f[topic].mean()) if topic.any() else None,
                "fpr_other": float(f[~topic].mean()),
                "fpr_other_ci": boot_ci(lambda ix, f=f: f[ix][~topic[ix]].mean(), n, args.n_boot, rng),
                "rate_diff": float(rate_diff(np.arange(n))),
                "rate_diff_ci": boot_ci(rate_diff, n, args.n_boot, rng),
                "mark_flagged": float(mark[f].mean()) if f.any() else None,
                "mark_unflagged": float(mark[~f].mean()),
                "mark_diff": float(mark_diff(np.arange(n))),
                "mark_diff_ci": boot_ci(mark_diff, n, args.n_boot, rng),
                "beta_topic": float(b[0]), "beta_topic_ci": np.percentile(cb[:, 0], [2.5, 97.5]).tolist(),
                "beta_marker": float(b[1]), "beta_marker_ci": np.percentile(cb[:, 1], [2.5, 97.5]).tolist(),
            }
        res["groups"][g] = out

    print(f"threshold: 5% FPR on {len(rows['H19'])} H19 human abstracts (s08 scores)")
    for g in GROUPS:
        o = res["groups"][g]
        lo, hi = o["topic_share_ci"]
        print(f"\n== {g}: {o['n']} human abstracts; LLM-topic share {o['topic_share']:.3f} [{lo:.3f}, {hi:.3f}];"
              f" marker rate {o['marker_mean']:.2f} per 1,000 words")
        print(f"{'detector':14s} {'FPR':>6s} {'topic':>6s} {'other':>6s} {'[other CI]':>15s} {'topic-other':>12s} {'[CI]':>17s}"
              f" {'mark flg/unf':>13s} {'diff [CI]':>22s} | std beta topic, marker")
        for d in DETS:
            x = o["dets"][d]
            ft = "  -  " if x["fpr_topic"] is None else f"{x['fpr_topic']:.3f}"
            mf = "  - " if x["mark_flagged"] is None else f"{x['mark_flagged']:.2f}"
            print(f"{d:14s} {x['fpr']:6.3f} {ft:>6s} {x['fpr_other']:6.3f} [{x['fpr_other_ci'][0]:.3f}, {x['fpr_other_ci'][1]:.3f}]"
                  f" {x['rate_diff']:+12.3f} [{x['rate_diff_ci'][0]:+.3f}, {x['rate_diff_ci'][1]:+.3f}]"
                  f" {mf:>6s}/{x['mark_unflagged']:.2f} {x['mark_diff']:+6.2f} [{x['mark_diff_ci'][0]:+.2f}, {x['mark_diff_ci'][1]:+.2f}]"
                  f" | {x['beta_topic']:+.2f} [{x['beta_topic_ci'][0]:+.2f}, {x['beta_topic_ci'][1]:+.2f}],"
                  f" {x['beta_marker']:+.2f} [{x['beta_marker_ci'][0]:+.2f}, {x['beta_marker_ci'][1]:+.2f}]")
    h = res["groups"]["H24"]["dets"]["F:2022_2024"]
    k1 = "holds" if h["rate_diff_ci"][0] > 0 else "fails"
    k2 = "holds" if h["fpr_other"] <= 0.15 else "fails"
    m1 = "holds" if h["mark_diff_ci"][0] > 0 else "fails"
    res["verdict"] = {"K1": k1, "K2": k2, "M1": m1}
    print(f"\n=== verdict (H24, F:2022_2024): K1 {k1}; K2 {k2} (FPR among non-LLM-topic {h['fpr_other']:.3f}); M1 {m1}")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
