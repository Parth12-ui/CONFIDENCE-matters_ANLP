"""
Analysis for run_recovery_experiment.py (O3 recovery through praise, O6 point of no return).

For every branch (claim x k x condition) it computes, over the B recovery turns:
  - recovered: the model gives the correct verdict at some recovery turn (only defined
    when it was wrong at turn k)
  - final_correct: correct at the last recovery turn
  - t_r: recovery turns until the first correct verdict (B + 1 if it never recovers)
  - rho: (truth confidence after B turns - truth confidence at turn k) / B, in points per turn

and reports, per branch type, k and condition, the mean with a 95% bootstrap interval over
claims, plus each condition minus the filler control on the same claims (paired). The O6
curve is the recovery rate as a function of k.

Usage:
  python src/evaluation/analyze_recovery.py --results results/o3_recovery_v1/recovery_results.jsonl
"""

import argparse
import os

import numpy as np
import pandas as pd

N_BOOT = 2000
SEED = 0
CONTROL = "filler"
METRICS = ["recovered", "final_correct", "t_r", "rho"]


def per_branch(df: pd.DataFrame) -> pd.DataFrame:
    rec = df[df["phase"] == "recovery"].sort_values("recovery_step")
    budget = int(rec["recovery_step"].max())
    rows = []
    for (cid, btype, k, cond), g in rec.groupby(["claim_id", "branch_type", "k", "condition"]):
        corr = g["corr_t"].tolist()
        wrong_at_k = g["corr_at_k"].iloc[0] == 0
        first = next((i + 1 for i, c in enumerate(corr) if c == 1), None)
        p_k = g["p_valid_at_k"].iloc[0]
        truth = g["ground_truth_verdict"].iloc[0]
        c_k = p_k if truth == "Valid" else 100.0 - p_k
        rows.append({
            "claim_id": cid, "branch_type": btype, "k": int(k), "condition": cond, "wrong_at_k": wrong_at_k,
            "recovered": float(first is not None) if wrong_at_k else np.nan,
            "final_correct": float(corr[-1] == 1) if corr[-1] is not None and not pd.isna(corr[-1]) else np.nan,
            "t_r": float(first if first is not None else budget + 1) if wrong_at_k else np.nan,
            "rho": (g["truth_confidence"].iloc[-1] - c_k) / budget,
        })
    return pd.DataFrame(rows)


def boot_ci(values: np.ndarray, rng: np.random.Generator):
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan, 0
    means = [rng.choice(values, len(values)).mean() for _ in range(N_BOOT)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return values.mean(), lo, hi, len(values)


def summarise(b: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for (btype, k, cond), g in b.groupby(["branch_type", "k", "condition"]):
        row = {"branch_type": btype, "k": k, "condition": cond, "n_claims": len(g), "n_wrong_at_k": int(g["wrong_at_k"].sum())}
        for m in METRICS:
            row[m], row[f"{m}_lo"], row[f"{m}_hi"], _ = boot_ci(g[m].to_numpy(float), rng)
        rows.append(row)
        if cond == CONTROL:
            continue
        # Paired contrast with the filler control on the same claims
        ctl = b[(b.branch_type == btype) & (b.k == k) & (b.condition == CONTROL)].set_index("claim_id")
        j = g.set_index("claim_id").join(ctl, rsuffix="_ctl", how="inner")
        crow = {"branch_type": btype, "k": k, "condition": f"{cond} - {CONTROL}", "n_claims": len(j),
                "n_wrong_at_k": int(j["wrong_at_k"].sum())}
        for m in METRICS:
            crow[m], crow[f"{m}_lo"], crow[f"{m}_hi"], _ = boot_ci((j[m] - j[f"{m}_ctl"]).to_numpy(float), rng)
        rows.append(crow)
    return pd.DataFrame(rows)


def plot_curve(summary: pd.DataFrame, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fixed = summary[(summary.branch_type == "fixed_k") & ~summary.condition.str.contains(" - ")]
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {"positive": "#1b9e77", "encourage": "#66a61e", "filler": "#7a7a7a"}
    for cond, g in fixed.groupby("condition"):
        g = g.sort_values("k")
        ax.plot(g["k"], 100 * g["recovered"], marker="o", label=cond, color=colors.get(cond))
        ax.fill_between(g["k"], 100 * g["recovered_lo"], 100 * g["recovered_hi"], alpha=0.15, color=colors.get(cond))
    ax.set_xlabel("criticism turns before recovery (k)")
    ax.set_ylabel("recovered within B turns (%)")
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Recovery after k criticism turns (claims wrong at k)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--output_dir", default=None)
    args = ap.parse_args()
    out = args.output_dir or os.path.join(os.path.dirname(args.results), "analysis")
    os.makedirs(out, exist_ok=True)

    df = pd.read_json(args.results, lines=True)
    n_base = df[df.phase == "baseline"]["claim_id"].nunique()
    n_run = df[df.phase == "criticism"]["claim_id"].nunique()
    print(f"{n_base} claims at turn 0, {n_run} correct at turn 0 and run through criticism")
    b = per_branch(df)
    b.to_csv(os.path.join(out, "recovery_per_branch.csv"), index=False)
    s = summarise(b)
    s.to_csv(os.path.join(out, "recovery_summary.csv"), index=False)
    plot_curve(s, os.path.join(out, "recovery_curve.png"))
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(s[["branch_type", "k", "condition", "n_claims", "n_wrong_at_k", "recovered", "recovered_lo",
                 "recovered_hi", "rho"]].round(3).to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
