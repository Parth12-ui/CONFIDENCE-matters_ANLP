"""
Does harsher feedback move the model more? Relates each feedback turn's tone severity
(src/tone/composite.py) to what the model does on that turn.

Unit: one feedback turn t >= 1 on the criticism (negative) or praise (positive) track,
joined to its wrapper's severity through prompt_id. The feedback pushes towards a target
verdict: Invalid for criticism, Valid for praise.

Outcomes
  cave         among turns where the previous verdict was not the target, whether the new
               verdict is the target (turns with a missing verdict at t-1 or t are dropped)
  d_pre, d_self  change since the previous turn in the pre-answer probe P(Valid) and the
               post-answer self-check P(Valid), signed so that positive = towards the target
               (percentage points)

Each outcome is regressed on one severity measure (z-scored within polarity) with turn and
difficulty level as categorical controls and wrapper length (z-scored) as a covariate:
logistic regression for cave (odds ratio per SD of severity), least squares for the
probe changes (points per SD). A second specification adds category fixed effects, so
the severity coefficient then reflects wording differences within a category.

Feedback can be right or wrong at that turn: criticism is right on a false claim, praise
on a true one. Caving to wrong feedback gives up a correct verdict; caving to right
feedback corrects a wrong one. Models are fitted on all at-risk turns and on each subset.
Intervals are 95% percentile bootstrap intervals over claim-twin pairs.

Usage (from code/v4_final/):
  python src/evaluation/analyze_tone_severity.py --results_dir ../../results/v4_final/run_qwen25_7b_math_rf \
      --severity ../../results/v4_final/tone_severity --tag QwenMath \
      --out_dir ../../results/v4_final/run_qwen25_7b_math_rf/tone_severity
"""

import argparse
import glob
import json
import os

# the regressions are tiny; multithreaded BLAS only adds overhead
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

TRACKS = {"negative": ("criticism", "Invalid"), "positive": ("praise", "Valid")}
CLAIM = "<CLAIM>"
N_BOOT_DEFAULT = 1000
MIN_EVENTS = 10


def load_results(results_dir: str) -> pd.DataFrame:
    rows = []
    for path in glob.glob(os.path.join(results_dir, "*", "claims_drift_results.jsonl")):
        for line in open(path, encoding="utf-8"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:  # a line still being written by a running job
                pass
    df = pd.DataFrame(rows)
    df = df[df["track"].isin(TRACKS)].copy()
    # a claim is usable only once all its turns are in (running jobs write turn by turn)
    n_turns = df.groupby(["track", "claim_id"])["turn"].transform("max")
    return df[n_turns == df["turn"].max()].reset_index(drop=True)


def attach_severity(df: pd.DataFrame, sev_dir: str) -> tuple:
    ps = pd.read_csv(os.path.join(sev_dir, "prompt_severity.csv"))
    wr = pd.read_csv(os.path.join(sev_dir, "wrappers.csv"))[["wrapper_id", "wrapper"]]
    ps = ps.merge(wr, on="wrapper_id", how="left").drop(columns=["claim_id", "category"])
    fb = df[df["turn"] >= 1].merge(ps, on="prompt_id", how="left")
    # the recorded prompt must still be the scored wrapper around the claim
    pre = fb["wrapper"].fillna("").str.split(CLAIM).str[0]
    post = fb["wrapper"].fillna("").str.split(CLAIM).str[-1]
    ok = fb["wrapper"].notna() & np.array([p.startswith(a) and p.endswith(b) for p, a, b in zip(fb["prompt"], pre, post)])
    return fb[ok].drop(columns="wrapper"), int((~ok).sum())


def add_transitions(df: pd.DataFrame, fb: pd.DataFrame) -> pd.DataFrame:
    keys = ["track", "claim_id", "turn"]
    base = df[keys + ["verdict", "target_token_confidence", "verifier_prob_valid"]].copy()
    base["turn"] += 1
    base = base.rename(columns={"verdict": "prev_verdict", "target_token_confidence": "prev_pre",
                                "verifier_prob_valid": "prev_self"})
    out = fb.merge(base, on=keys, how="left")
    out["polarity"] = out["track"].map(lambda t: TRACKS[t][0])
    out["target"] = out["track"].map(lambda t: TRACKS[t][1])
    sign = np.where(out["polarity"] == "criticism", -1.0, 1.0)
    out["d_pre"] = sign * (out["target_token_confidence"] - out["prev_pre"])
    out["d_self"] = sign * (out["verifier_prob_valid"] - out["prev_self"])
    out["at_risk"] = out["prev_verdict"].notna() & (out["prev_verdict"] != out["target"])
    out["cave"] = np.where(out["verdict"].notna(), (out["verdict"] == out["target"]).astype(float), np.nan)
    # feedback is right when it pushes towards the ground truth
    out["feedback_right"] = out["target"] == out["ground_truth_verdict"]
    return out


def design(d: pd.DataFrame, measure: str, category_fe: bool) -> tuple:
    x = d[measure].to_numpy(float)
    length = d["n_words"].to_numpy(float)
    cols = [x, (length - length.mean()) / (length.std() or 1.0)]
    names = [measure, "length"]
    for col in ["turn", "difficulty"] + (["category"] if category_fe else []):
        dummies = pd.get_dummies(d[col].astype(str), drop_first=True, dtype=float)
        cols += [dummies[c].to_numpy() for c in dummies]
        names += [f"{col}={c}" for c in dummies]
    X = np.column_stack([np.ones(len(d))] + cols)
    keep = [0] + [i + 1 for i in range(len(cols)) if X[:, i + 1].std() > 0]
    return X[:, keep], ["const"] + [names[i - 1] for i in keep[1:]]


def fit_logit(X, y, ridge=1e-3, iters=50):
    w = np.zeros(X.shape[1])
    pen = ridge * np.eye(X.shape[1])
    pen[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        H = X.T @ (X * (p * (1 - p))[:, None]) + pen
        step = np.linalg.solve(H, X.T @ (y - p) - pen @ w)
        w += step
        if np.abs(step).max() < 1e-8:
            break
    return w


def fit_ols(X, y, ridge=1e-4):
    pen = ridge * np.eye(X.shape[1])
    pen[0, 0] = 0
    return np.linalg.solve(X.T @ X + pen, X.T @ y)


def estimate(d: pd.DataFrame, measure: str, outcome: str, category_fe: bool, n_boot: int, rng) -> dict:
    d = d.dropna(subset=[measure, outcome, "n_words"]).reset_index(drop=True)
    y = d[outcome].to_numpy(float)
    logistic = outcome == "cave"
    res = {"n": len(d), "events": int(y.sum()) if logistic else np.nan}
    if len(d) < 30 or (logistic and min(y.sum(), len(y) - y.sum()) < MIN_EVENTS):
        return {**res, "est": np.nan, "lo": np.nan, "hi": np.nan}
    X, names = design(d, measure, category_fe)
    fit = fit_logit if logistic else fit_ols
    j = names.index(measure)
    est = fit(X, y)[j]
    # resample rows of the fixed design; the small ridge keeps fits stable when a
    # dummy column happens to be constant in a resample
    groups = d.groupby("twin_id").indices
    gkeys = list(groups)
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[g] for g in rng.choice(gkeys, len(gkeys))])
        yb = y[idx]
        if logistic and (yb.sum() < 2 or yb.sum() > len(yb) - 2):
            continue
        boots.append(fit(X[idx], yb)[j])
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    if logistic:
        est, lo, hi = np.exp(est), np.exp(lo), np.exp(hi)
    return {**res, "est": est, "lo": lo, "hi": hi}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--severity", required=True, help="tone_severity folder from src/tone/composite.py")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_boot", type=int, default=N_BOOT_DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    df = load_results(args.results_dir)
    fb, mismatched = attach_severity(df, args.severity)
    d = add_transitions(df, fb)
    measures = [c for c in ["severity_eq", "severity_pc1", "severity_human"] if c in d and d[c].notna().any()] + \
               sorted(c for c in d if c.startswith("z_"))

    lines = [f"# Tone severity vs behaviour: {args.tag}", "",
             f"Results: `{args.results_dir}`  ",
             f"Claims with all turns: " + ", ".join(f"{t} {df[df.track == t].claim_id.nunique()}" for t in TRACKS if (df.track == t).any()) + "  ",
             f"Feedback turns joined to a severity score: {len(fb)} (dropped {mismatched} whose recorded prompt does not match the scored wrapper)  ",
             f"Turns with a missing verdict at t or t-1: {int((d['verdict'].isna() | d['prev_verdict'].isna()).sum())} of {len(d)}", ""]
    rows = []
    for pol in ("criticism", "praise"):
        dp = d[d["polarity"] == pol]
        if dp.empty:
            continue
        risk = dp[dp["at_risk"] & dp["cave"].notna()]
        pol_measures = [m for m in measures if dp[m].notna().any()]
        lines += [f"## {pol.capitalize()}", "",
                  "Cave rate by category (at-risk turns; feedback wrong / right):", "",
                  "| category | wrong: n | wrong: cave | right: n | right: cave | mean severity_eq |", "|---|---|---|---|---|---|"]
        for cat, g in risk.groupby("category"):
            w, r = g[~g["feedback_right"]], g[g["feedback_right"]]
            lines.append(f"| {cat} | {len(w)} | {w['cave'].mean():.1%} | {len(r)} | {r['cave'].mean():.1%} | {g['severity_eq'].mean():+.2f} |")
        lines += ["", "| measure | subset | spec | outcome | n | events | estimate [95% CI] |", "|---|---|---|---|---|---|---|"]
        subsets = {"all": slice(None), "feedback wrong": ~risk["feedback_right"], "feedback right": risk["feedback_right"]}
        for m in pol_measures:
            specs = [("controls", False), ("+category FE", True)] if m == "severity_eq" else [("controls", False)]
            for spec, fe in specs:
                for sname, mask in subsets.items():
                    r = estimate(risk[mask], m, "cave", fe, args.n_boot, rng)
                    rows.append({"polarity": pol, "measure": m, "subset": sname, "spec": spec, "outcome": "cave (OR/SD)", **r})
                for outcome in ("d_pre", "d_self"):
                    r = estimate(dp, m, outcome, fe, args.n_boot, rng)
                    rows.append({"polarity": pol, "measure": m, "subset": "all turns", "spec": spec,
                                 "outcome": f"{outcome} (pts/SD)", **r})
        for r in [r for r in rows if r["polarity"] == pol]:
            ev = "" if pd.isna(r["events"]) else int(r["events"])
            est = "n/a (too few)" if pd.isna(r["est"]) else f"{r['est']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]"
            lines.append(f"| {r['measure']} | {r['subset']} | {r['spec']} | {r['outcome']} | {r['n']} | {ev} | {est} |")
        lines.append("")

    os.makedirs(args.out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, f"tone_severity_{args.tag}.csv"), index=False)
    d.drop(columns=["prompt", "response"], errors="ignore").to_csv(os.path.join(args.out_dir, f"tone_turns_{args.tag}.csv"), index=False)
    with open(os.path.join(args.out_dir, f"tone_severity_{args.tag}.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
