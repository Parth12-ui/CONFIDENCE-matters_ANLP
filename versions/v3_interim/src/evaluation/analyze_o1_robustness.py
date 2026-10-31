"""
Baseline-aware robustness checks for the paired O1 experiment.

The headline pressure effects (feedback track minus control on the Valid rate)
depend on where the model starts. Llama-3.1-8B-Instruct answers Invalid for most
items at turn 0, so praise (which pushes towards Valid) has far more room to
move than criticism (which pushes towards Invalid). This script reports, at a
fixed turn:

  - conditional push rates: among items whose turn-0 verdict is opposite to the
    direction a track pushes, the share that has moved into that direction,
    minus the same share in the control. This compares praise and criticism on
    the items each one can actually move (H3).
  - log-odds pressure effects on the Valid rate (less sensitive to floors).
  - pressure effects on items the model answered correctly at turn 0 (the usual
    sycophancy setting: abandoning a correct answer).
  - two-sided bootstrap p-values for the H1/H2 Valid-rate tests with Holm
    correction across the four tests.
  - if the stance-free tracks were run: how often affirm / dispute leave the
    model on a different verdict than at turn 0, minus the length-matched filler
    control, overall and on items answered correctly at turn 0.

All intervals are percentile bootstrap intervals over items (paired across
tracks, 2,000 resamples). Turn 0 is identical across tracks (greedy decoding,
same prompt), so turn-0 subsets are the same items on every track.

Usage:
  python src/evaluation/analyze_o1_robustness.py --results_dir results/o1_paired_v3 \
      --turn 2 --tex_out ../interim_report/generated/robust_Main.tex --tag Main
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

N_BOOT = 2000
SEED = 0
PUSH = {"positive": 1.0, "negative": 0.0}  # Valid-verdict value each track pushes towards
STANCE_FREE = ["affirm", "dispute"]  # compared with the "filler" track


def load(results_dir: str, turn: int) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(results_dir, "**", "claims_drift_results.jsonl"), recursive=True))
    rows = [json.loads(l) for p in paths for l in open(p, encoding="utf-8") if l.strip()]
    df = pd.DataFrame(rows).drop_duplicates(subset=["claim_id", "track", "turn"], keep="last")
    df = df[df["turn"].isin([0, turn])]
    df["valid"] = (df["verdict"] == "Valid").astype(float)
    wide = df.pivot_table(index="claim_id", columns=["track", "turn"], values="valid")
    tracks = sorted(set(df["track"]))
    keep = wide.notna().all(axis=1)
    print(f"{keep.sum()} of {len(wide)} items observed at turn {turn} on every track.")
    wide = wide[keep]
    truth = df.groupby("claim_id")["ground_truth_verdict"].first().loc[wide.index]
    out = pd.DataFrame({"is_true": (truth == "Valid").astype(float).values}, index=wide.index)
    out["v0"] = wide[("control", 0)]
    for tr in tracks:
        out[tr] = wide[(tr, turn)]
    return out


def logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1 - p))


def stats(d: pd.DataFrame) -> dict:
    """All point estimates for one (possibly resampled) item table."""
    s = {}
    for tr in PUSH:
        target = PUSH[tr]
        movable = d["v0"] != target
        n = movable.sum()
        moved = (d.loc[movable, tr] == target).mean() if n else np.nan
        moved_ctl = (d.loc[movable, "control"] == target).mean() if n else np.nan
        s[f"{tr}_push"] = 100 * (moved - moved_ctl)
        s[f"{tr}_movable"] = n
        for lab, mask in [("True", d["is_true"] == 1), ("False", d["is_true"] == 0)]:
            m = len(d[mask])
            # +0.5 / +1 (Haldane-Anscombe) keeps 0% and 100% rates finite
            p = (d.loc[mask, tr].sum() + 0.5) / (m + 1)
            pc = (d.loc[mask, "control"].sum() + 0.5) / (m + 1)
            s[f"{tr}_{lab}_logodds"] = float(logit(p) - logit(pc))
            s[f"{tr}_{lab}_valid"] = 100 * (d.loc[mask, tr].mean() - d.loc[mask, "control"].mean())
            # Items answered correctly at turn 0
            corr0 = mask & (d["v0"] == d["is_true"])
            acc = lambda col: (d.loc[corr0, col] == d.loc[corr0, "is_true"]).mean()
            s[f"{tr}_{lab}_corr0_acc"] = 100 * (acc(tr) - acc("control"))
            s[f"{tr}_{lab}_corr0_n"] = int(corr0.sum())
    s["asym_push"] = s["positive_push"] - s["negative_push"]
    if "filler" in d:
        corr0 = d["v0"] == d["is_true"]
        for tr in [t for t in STANCE_FREE if t in d]:
            change = lambda col, m: 100 * (d.loc[m, col] != d.loc[m, "v0"]).mean()
            s[f"{tr}_change"] = change(tr, d.index == d.index) - change("filler", d.index == d.index)
            s[f"{tr}_corr0_change"] = change(tr, corr0) - change("filler", corr0)
    return s


def bootstrap(d: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(d))
    return pd.DataFrame([stats(d.iloc[rng.choice(idx, len(idx))]) for _ in range(N_BOOT)])


def holm(pvals: dict) -> dict:
    order = sorted(pvals, key=pvals.get)
    adj, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * pvals[k]))
        adj[k] = running
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--turn", type=int, default=2)
    ap.add_argument("--tex_out", default=None)
    ap.add_argument("--tag", default="Main")
    args = ap.parse_args()

    d = load(args.results_dir, args.turn)
    point = stats(d)
    boot = bootstrap(d)

    def ci(k):
        lo, hi = np.nanpercentile(boot[k], [2.5, 97.5])
        return point[k], lo, hi

    tests = {
        "H1True": "negative_True_valid", "H1False": "negative_False_valid",
        "H2True": "positive_True_valid", "H2False": "positive_False_valid",
    }
    pvals = {}
    for name, k in tests.items():
        b = boot[k].to_numpy()
        pvals[name] = min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()))
    padj = holm(pvals)

    print(f"\nTurn {args.turn}, {len(d)} items")
    for tr in STANCE_FREE:
        if f"{tr}_change" in point:
            for k in [f"{tr}_change", f"{tr}_corr0_change"]:
                print(f"  {k:28s} {ci(k)[0]:+6.1f}  [{ci(k)[1]:+.1f}, {ci(k)[2]:+.1f}]  (verdict-change rate vs filler)")
    for k in ["positive_push", "negative_push", "asym_push"]:
        print(f"  {k:28s} {ci(k)[0]:+6.1f}  [{ci(k)[1]:+.1f}, {ci(k)[2]:+.1f}]")
    print(f"  movable: praise {point['positive_movable']}, criticism {point['negative_movable']}")
    for tr in PUSH:
        for lab in ["True", "False"]:
            k = f"{tr}_{lab}_logodds"
            print(f"  {k:28s} {ci(k)[0]:+6.2f}  [{ci(k)[1]:+.2f}, {ci(k)[2]:+.2f}]")
    for tr in PUSH:
        for lab in ["True", "False"]:
            k = f"{tr}_{lab}_corr0_acc"
            print(f"  {k:28s} {ci(k)[0]:+6.1f}  [{ci(k)[1]:+.1f}, {ci(k)[2]:+.1f}]  n={point[f'{tr}_{lab}_corr0_n']}")
    for name in tests:
        print(f"  {name}: p={pvals[name]:.4f}  Holm p={padj[name]:.4f}")

    if args.tex_out:
        t = args.tag
        lines = [f"% Generated by analyze_o1_robustness.py ({t}, turn {args.turn}); do not edit by hand."]

        def add(name, val):
            lines.append(f"\\newcommand{{\\{t}{name}}}{{{val}}}")

        def sgn(x, nd=1):
            return f"$+${x:.{nd}f}" if x >= 0 else f"$-${abs(x):.{nd}f}"

        def add_ci(name, k, nd=1):
            v, lo, hi = ci(k)
            add(name, sgn(v, nd))
            add(name + "Lo", sgn(lo, nd))
            add(name + "Hi", sgn(hi, nd))

        add_ci("PushPos", "positive_push")
        add_ci("PushNeg", "negative_push")
        add_ci("PushAsym", "asym_push")
        add("MovablePos", point["positive_movable"])
        add("MovableNeg", point["negative_movable"])
        add_ci("LogOddsPosTrue", "positive_True_logodds", 2)
        add_ci("LogOddsPosFalse", "positive_False_logodds", 2)
        add_ci("LogOddsNegTrue", "negative_True_logodds", 2)
        add_ci("LogOddsNegFalse", "negative_False_logodds", 2)
        add_ci("CorrZeroNegTrueAcc", "negative_True_corr0_acc")
        add_ci("CorrZeroPosFalseAcc", "positive_False_corr0_acc")
        add("CorrZeroTrueN", point["negative_True_corr0_n"])
        for tr, short in [("affirm", "Affirm"), ("dispute", "Dispute")]:
            if f"{tr}_change" in point:
                add_ci(f"Change{short}", f"{tr}_change")
                add_ci(f"CorrZeroChange{short}", f"{tr}_corr0_change")
        add("CorrZeroFalseN", point["positive_False_corr0_n"])
        def pfmt(p):
            return "$<$0.001" if p < 0.001 else f"{p:.3f}"

        for name in tests:
            # LaTeX macro names cannot contain digits: H1True -> HOneTrue
            mname = name.replace("H1", "HOne").replace("H2", "HTwo")
            add(f"P{mname}", pfmt(pvals[name]))
            add(f"PHolm{mname}", pfmt(padj[name]))
        with open(args.tex_out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Wrote {args.tex_out}")


if __name__ == "__main__":
    main()
