#!/usr/bin/env python3
"""
Does the model still represent the correct answer when it gives in? Linear probes on the
hidden states from src/extract_hidden.py.

Two probes per layer and position (pre: before the model's turn-t answer; post: after it),
each a standardised L2 logistic regression fitted on the CONTROL track (all turns):
  - truth probe:   is the claim true?
  - verdict probe: does the model say Valid on this turn?
Folds are claim-twin pairs (5-fold GroupKFold), and every criticism-track state is scored
by the fold model that never saw its pair, so no claim is used both to fit and to test.
The layer is chosen by the truth probe's held-out accuracy on the control track, so the
choice never looks at the criticism track.

On the criticism track, at turns 1-6:
  - caved: true claims answered correctly at turn 0 (unwarranted criticism) whose stated
           verdict is now wrong, i.e. Invalid. Reported: share on which each probe still
           favours the correct answer, against the same share on turns where the model held.
  - same-verdict test: caved turns against turns on FALSE claims where the model also says
           Invalid. Both groups state the same verdict, so a probe that only reads the
           verdict cannot tell them apart (AUROC 0.5); a probe that separates them carries
           the truth beyond the stated verdict. Both probes are scored on the same bootstrap
           resamples, so the truth-minus-verdict AUROC difference gets a paired interval.
Intervals are 95% percentile bootstrap intervals over claim-twin pairs, as in pushback.py.

Usage:
  python3 src/analysis/hidden_probe.py --results_dir results/qwen_math --tag QwenMath \
      [--hidden_dir results/qwen_math/hidden] [--out_dir results/qwen_math/hidden_probe] \
      [--tex_out report/generated/hidden_QwenMath.tex] [--drop_forced] [--reuse_curve]
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pushback import CONTROL, PRESSURE, cluster_boot, load  # noqa: E402

POSITIONS = ("pre", "post")
C = 0.01  # fixed in advance; strong regularisation for ~2,000 examples in 3-4k dimensions
FOLDS = 5
SEED = 0


def load_states(hidden_dir, track):
    z = np.load(os.path.join(hidden_dir, f"{track}.npz"))
    return z["states"], z["claim_id"], z["layers"]


def frame(d, track, cids, n_turns):
    """Label rows aligned with states[i, t]: one row per (conversation i, turn t)."""
    x = d[d["track"] == track].set_index(["claim_id", "turn"])
    rows = []
    for i, cid in enumerate(cids):
        for t in range(n_turns):
            if (cid, t) in x.index:
                r = x.loc[(cid, t)]
                rows.append({"i": i, "turn": t, "claim_id": cid, "pair": r["pair"], "group": r["group"],
                             "true": r["ground_truth_verdict"] == "Valid", "parsed": r["parsed"],
                             "say_valid": r["say_valid"], "say_ok": r["say_ok"]})
    return pd.DataFrame(rows)


def probe():
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def fit_score(Xc, yc, gc, Xn, gn, mask_c=None):
    """Held-out P(label) for control rows (CV) and for criticism rows (model of the fold
    that held out their pair)."""
    mask_c = np.ones(len(yc), bool) if mask_c is None else mask_c
    pc, pn = np.full(len(yc), np.nan), np.full(len(gn), np.nan)
    pairs = np.unique(gc)
    for tr_pairs_idx, te_pairs_idx in GroupKFold(FOLDS).split(pairs, groups=pairs):
        te_pairs = set(pairs[te_pairs_idx])
        te_c = np.isin(gc, list(te_pairs))
        tr = ~te_c & mask_c
        m = probe().fit(Xc[tr], yc[tr])
        pc[te_c] = m.predict_proba(Xc[te_c])[:, 1]
        te_n = np.isin(gn, list(te_pairs))
        if te_n.any():
            pn[te_n] = m.predict_proba(Xn[te_n])[:, 1]
    return pc, pn


def auc(z, col):
    return float(roc_auc_score(z["true"], z[col])) if z["true"].nunique() == 2 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--hidden_dir", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tex_out", default=None)
    ap.add_argument("--drop_forced", action="store_true", help="forced verdicts count as missing")
    ap.add_argument("--reuse_curve", action="store_true",
                    help="take the layer choice from an existing <out_dir>/layer_curve.csv")
    args = ap.parse_args()
    hidden = args.hidden_dir or os.path.join(args.results_dir, "hidden")
    out_dir = args.out_dir or os.path.join(args.results_dir, "hidden_probe")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    d = load(args.results_dir, 6, args.drop_forced)
    Sc, cid_c, layers = load_states(hidden, CONTROL)
    Sn, cid_n, layers_n = load_states(hidden, PRESSURE)
    assert (layers == layers_n).all()
    fc, fn = frame(d, CONTROL, cid_c, Sc.shape[1]), frame(d, PRESSURE, cid_n, Sn.shape[1])
    yc_truth = fc["true"].to_numpy()
    yc_verdict = fc["say_valid"].to_numpy()
    parsed_c = fc["parsed"].to_numpy()
    gc, gn = fc["pair"].to_numpy(), fn["pair"].to_numpy()

    def X(S, f, p, li):
        return S[f["i"].to_numpy(), f["turn"].to_numpy(), p, li].astype(np.float32)

    # 1. Layer curve on the control track (held-out accuracy), both positions
    curve_csv = os.path.join(out_dir, "layer_curve.csv")
    curve = []
    for p, pname in enumerate(POSITIONS if not args.reuse_curve else ()):
        for li, layer in enumerate(layers):
            pc, _ = fit_score(X(Sc, fc, p, li), yc_truth, gc, X(Sn, fn, p, li)[:0], gn[:0])
            pv, _ = fit_score(X(Sc, fc, p, li), yc_verdict, gc, X(Sn, fn, p, li)[:0], gn[:0], parsed_c)
            curve.append({"position": pname, "layer": int(layer),
                          "truth_acc": float(((pc > 0.5) == yc_truth).mean()),
                          "verdict_acc": float(((pv > 0.5) == yc_verdict)[parsed_c].mean())})
            print(f"  {pname} layer {layer:2d}: control truth acc {curve[-1]['truth_acc']:.3f}, "
                  f"verdict acc {curve[-1]['verdict_acc']:.3f}", flush=True)
    if args.reuse_curve:
        curve = pd.read_csv(curve_csv)
    else:
        curve = pd.DataFrame(curve)
        curve.to_csv(curve_csv, index=False)

    # 2. Criticism track at the chosen layer, per position
    out = {"items": int(d["claim_id"].nunique()), "C": C, "folds": FOLDS}
    turns = fn[(fn["turn"] >= 1)].copy()
    for p, pname in enumerate(POSITIONS):
        c = curve[curve["position"] == pname]
        best = c.loc[c["truth_acc"].idxmax()]
        li = int(np.where(layers == best["layer"])[0][0])
        Xc, Xn = X(Sc, fc, p, li), X(Sn, fn, p, li)
        _, pn_truth = fit_score(Xc, yc_truth, gc, Xn, gn)
        _, pn_verdict = fit_score(Xc, yc_verdict, gc, Xn, gn, parsed_c)
        z = fn.assign(p_truth=pn_truth, p_verdict=pn_verdict)
        z = z[(z["turn"] >= 1) & z["parsed"]]
        caved = z[(z["group"] == "unwarranted") & ~z["say_ok"]]
        held = z[(z["group"] == "unwarranted") & z["say_ok"]]
        # Same stated verdict (Invalid), different truth
        same = z[~z["say_valid"] & (((z["group"] == "unwarranted") & ~z["say_ok"]) | ~z["true"])]
        r = {"layer": int(best["layer"]), "control_truth_acc": float(best["truth_acc"]),
             "control_verdict_acc": float(best["verdict_acc"]),
             "caved_turns": int(len(caved)), "held_turns": int(len(held)),
             "same_verdict_turns": int(len(same)), "same_verdict_true": int(same["true"].sum())}
        if len(caved):
            r["caved_truth_ok"] = cluster_boot(caved, lambda q: float((q["p_truth"] > 0.5).mean()), rng)
            r["caved_verdict_ok"] = cluster_boot(caved, lambda q: float((q["p_verdict"] > 0.5).mean()), rng)
        if len(held):
            r["held_truth_ok"] = cluster_boot(held, lambda q: float((q["p_truth"] > 0.5).mean()), rng)
        if len(caved) and (~same["true"]).any():
            r["same_auc_truth"] = cluster_boot(same, lambda q: auc(q, "p_truth"), rng)
            r["same_auc_verdict"] = cluster_boot(same, lambda q: auc(q, "p_verdict"), rng)
            r["same_auc_diff"] = cluster_boot(same, lambda q: auc(q, "p_truth") - auc(q, "p_verdict"), rng)
        out[pname] = r
        print(f"{pname}: layer {r['layer']}, caved {r['caved_turns']} turns; "
              + ", ".join(f"{k} {v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]" for k, v in r.items() if isinstance(v, tuple)))
    json.dump(out, open(os.path.join(out_dir, "probe_summary.json"), "w"), indent=1)

    if args.tex_out:
        T = args.tag
        lines = [f"% generated by src/analysis/hidden_probe.py --tag {T}"]

        def pct(v):
            return f"{100 * v:.1f}"

        for pname in POSITIONS:
            r, P = out[pname], pname.capitalize()
            lines.append(f"\\newcommand{{\\{T}Hid{P}Layer}}{{{r['layer']}}}")
            lines.append(f"\\newcommand{{\\{T}Hid{P}CtrlAcc}}{{{pct(r['control_truth_acc'])}}}")
            lines.append(f"\\newcommand{{\\{T}Hid{P}CavedN}}{{{r['caved_turns']}}}")
            for key, name, fmt in [("caved_truth_ok", "CavedTruth", pct), ("caved_verdict_ok", "CavedVerdict", pct),
                                   ("held_truth_ok", "HeldTruth", pct),
                                   ("same_auc_truth", "AucTruth", lambda v: f"{v:.2f}"),
                                   ("same_auc_verdict", "AucVerdict", lambda v: f"{v:.2f}"),
                                   ("same_auc_diff", "AucDiff", lambda v: f"{v:.2f}")]:
                if key in r:
                    for suf, v in zip(("", "Lo", "Hi"), r[key]):
                        lines.append(f"\\newcommand{{\\{T}Hid{P}{name}{suf}}}{{{fmt(v)}}}")
        os.makedirs(os.path.dirname(args.tex_out) or ".", exist_ok=True)
        open(args.tex_out, "w").write("\n".join(lines) + "\n")
        print(f"wrote {args.tex_out}")


if __name__ == "__main__":
    main()
