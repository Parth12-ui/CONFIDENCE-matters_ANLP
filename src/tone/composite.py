"""
Combines the tone measures into severity scores and checks them against the nominal
categories and (once collected) the human Best-Worst Scaling ratings.

Severity is defined within each polarity, so that higher always means stronger:
  criticism: harsher pushback  (negative valence, arousal, dominance, impoliteness, negative emotions)
  praise:    stronger push to agree  (positive valence, arousal, dominance, positive emotions)
Politeness is left out of the praise score: flattery and a polite answer nudge are both polite.
NRC-VAD dominance is reported but left out of both scores: it rates how powerful the words'
referents are ("garbage", "nonsense" are low-dominance words), not how forcefully the speaker
pushes, and on our criticism wrappers it runs opposite to the nominal ordering. This was
decided after the first scoring run (before any comparison with model behaviour).

Scores (computed within polarity, on z-scored measures):
  severity_eq     equal-weight mean of the z-scored measures (the primary score)
  severity_pc1    first principal component, signed to agree with severity_eq
  severity_human  ridge weights fitted to the Best-Worst ratings (only with --bws)
Report: Cronbach's alpha, each measure's Spearman correlation with the nominal strength,
category means, and with --bws the correlation of every measure with the human ratings
(the human-fitted score is evaluated leave-one-out).

Usage (from code/v4_final/):
  python src/tone/composite.py --scores ../../results/v4_final/tone_severity/wrapper_scores.csv \
      --prompt_wrappers ../../results/v4_final/tone_severity/prompt_wrappers.csv \
      --out_dir ../../results/v4_final/tone_severity [--bws ../../results/v4_final/tone_severity/bws/bws_scores.csv]
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

NEG_EMO = ["emo_disapproval", "emo_annoyance", "emo_anger", "emo_disappointment"]
POS_EMO = ["emo_admiration", "emo_approval", "emo_pride", "emo_optimism"]
# measure -> sign that makes "higher = stronger" for that polarity
ORIENT = {
    "criticism": {"vad_valence": -1, "vad_arousal": +1, "vad_dominance": +1, "politeness": -1, "emo_negative": +1},
    "praise": {"vad_valence": +1, "vad_arousal": +1, "vad_dominance": +1, "emo_positive": +1},
}
REPORT_ONLY = {"vad_dominance"}  # z-scored and reported, not part of the combined scores


def add_emotion_sums(df: pd.DataFrame) -> pd.DataFrame:
    if all(c in df for c in NEG_EMO):
        df["emo_negative"] = df[NEG_EMO].sum(axis=1)
    if all(c in df for c in POS_EMO):
        df["emo_positive"] = df[POS_EMO].sum(axis=1)
    return df


def cronbach_alpha(Z: np.ndarray) -> float:
    k = Z.shape[1]
    if k < 2:
        return float("nan")
    return k / (k - 1) * (1 - Z.var(axis=0, ddof=1).sum() / Z.sum(axis=1).var(ddof=1))


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    X1 = np.column_stack([np.ones(len(X)), X])
    pen = lam * np.eye(X1.shape[1])
    pen[0, 0] = 0
    return np.linalg.solve(X1.T @ X1 + pen, X1.T @ y)


def ridge_predict(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return w[0] + X @ w[1:]


def md_table(df: pd.DataFrame) -> str:
    rows = [list(df.columns), ["---"] * len(df.columns)] + df.astype(str).values.tolist()
    return "\n".join("| " + " | ".join(map(str, r)) + " |" for r in rows)


def rho(a, b) -> float:
    ok = ~(pd.isna(a) | pd.isna(b))
    return float(spearmanr(np.asarray(a)[ok], np.asarray(b)[ok])[0]) if ok.sum() > 2 else float("nan")


def score_polarity(df: pd.DataFrame, pol: str, bws: pd.DataFrame, lam: float, lines: list) -> pd.DataFrame:
    d = df[df["polarity"] == pol].copy()
    measures = [m for m in ORIENT[pol] if m in d and d[m].notna().any()]
    for m in measures:
        x = ORIENT[pol][m] * d[m]
        d[f"z_{m}"] = (x - x.mean()) / x.std(ddof=1)
    zc = [f"z_{m}" for m in measures]
    used = [m for m in measures if m not in REPORT_ONLY]
    Z = d[[f"z_{m}" for m in used]].fillna(0.0).to_numpy()
    d["severity_eq"] = Z.mean(axis=1)

    _, s, vt = np.linalg.svd(Z - Z.mean(axis=0), full_matrices=False)
    pc1 = (Z - Z.mean(axis=0)) @ vt[0]
    if np.corrcoef(pc1, d["severity_eq"])[0, 1] < 0:
        pc1, vt[0] = -pc1, -vt[0]
    d["severity_pc1"] = pc1 / pc1.std(ddof=1)

    strength = d["nominal_valence"].abs()
    lines += [f"\n## {pol.capitalize()} ({len(d)} wrappers)\n",
              f"Measures (oriented so higher = stronger): {', '.join(measures)}  ",
              f"In the combined scores: {', '.join(used)}  ",
              f"Cronbach's alpha: {cronbach_alpha(Z):.2f}  ",
              f"PC1 explains {s[0] ** 2 / (s ** 2).sum():.0%} of the variance; loadings: "
              + ", ".join(f"{m} {w:+.2f}" for m, w in zip(used, vt[0])) + "\n",
              "| measure | Spearman with nominal strength |", "|---|---|"]
    for c in zc + ["severity_eq", "severity_pc1"]:
        lines.append(f"| {c} | {rho(d[c], strength):+.2f} |")

    order = d.groupby("category").agg(nominal=("nominal_valence", "first"), n=("wrapper_id", "size"),
                                      **{c: (c, "mean") for c in zc + ["severity_eq"]})
    order = order.reindex(order["nominal"].abs().sort_values().index)
    monotone = order["severity_eq"].is_monotonic_increasing
    lines += ["\nCategory means (z units), weakest nominal category first:\n",
              md_table(order.round(2).reset_index()),
              f"\nseverity_eq increases with the nominal ordering: {'yes' if monotone else 'NO'}"]

    if bws is not None:
        d = d.merge(bws[["wrapper_id", "bws_score"]], on="wrapper_id", how="left")
        rated = d["bws_score"].notna().to_numpy()
        lines += [f"\nHuman Best-Worst ratings: {rated.sum()} wrappers rated\n",
                  "| measure | Spearman with human rating |", "|---|---|"]
        for c in zc + ["severity_eq", "severity_pc1"]:
            lines.append(f"| {c} | {rho(d.loc[rated, c], d.loc[rated, 'bws_score']):+.2f} |")
        lines.append(f"| nominal strength | {rho(d.loc[rated, 'nominal_valence'].abs(), d.loc[rated, 'bws_score']):+.2f} |")
        Zr, yr = Z[rated], d.loc[rated, "bws_score"].to_numpy()
        if rated.sum() >= len(used) + 3:
            loo = np.array([ridge_predict(ridge_fit(np.delete(Zr, i, 0), np.delete(yr, i), lam), Zr[i:i + 1])[0]
                            for i in range(len(yr))])
            w = ridge_fit(Zr, yr, lam)
            d["severity_human"] = ridge_predict(w, Z)
            lines += [f"| severity_human (leave-one-out) | {rho(loo, yr):+.2f} |",
                      "\nHuman-fitted weights: " + ", ".join(f"{m} {x:+.2f}" for m, x in zip(used, w[1:]))]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="wrapper_scores.csv from score_severity.py")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prompt_wrappers", default=None, help="prompt_wrappers.csv; adds prompt_severity.csv")
    ap.add_argument("--bws", default=None, help="bws_scores.csv from bws.py score")
    ap.add_argument("--ridge", type=float, default=1.0)
    args = ap.parse_args()

    df = add_emotion_sums(pd.read_csv(args.scores))
    bws = pd.read_csv(args.bws) if args.bws else None
    lines = ["# Tone severity: composite and validity", "",
             "Generated by `src/tone/composite.py`. Primary score: `severity_eq`."]
    parts = [score_polarity(df, pol, bws, args.ridge, lines) for pol in ("criticism", "praise")]
    out = pd.concat(parts, ignore_index=True)
    keep = ["wrapper_id", "polarity", "category", "nominal_valence"] + \
           [c for c in out if c.startswith("z_")] + \
           [c for c in ["severity_eq", "severity_pc1", "severity_human", "bws_score"] if c in out] + \
           ["n_words", "hedge", "authority_cue", "text"]
    out = out[keep]

    os.makedirs(args.out_dir, exist_ok=True)
    out.to_csv(os.path.join(args.out_dir, "wrapper_severity.csv"), index=False)
    if args.prompt_wrappers:
        pw = pd.read_csv(args.prompt_wrappers)
        sev_cols = [c for c in out if c.startswith(("z_", "severity_"))] + ["n_words", "hedge", "authority_cue"]
        pw.merge(out[["wrapper_id"] + sev_cols], on="wrapper_id", how="left") \
          .to_csv(os.path.join(args.out_dir, "prompt_severity.csv"), index=False)
    with open(os.path.join(args.out_dir, "validity.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
