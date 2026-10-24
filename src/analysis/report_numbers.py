"""
LaTeX number macros for the final report that analyze_pushback.py does not write:
praise against the control, tone severity, working vs verdict on caving turns, and how
many verdicts the runner's --force_verdict fallback completed. Also writes the per-level
MATH table.

For each --run TAG DIR (DIR holds the track folders; the tone-severity and derived-answer
outputs are read from DIR/tone_severity/ and DIR/derived/ when present):
  {TAG}PraWrong*, {TAG}PraRight*  accuracy at turn 6, praise vs control, on false claims the
                                  model got right at turn 0 (praise is wrong) and true claims
                                  it got wrong (praise is right); N, Pra, Ctrl, Diff, DiffLo, DiffHi
  {TAG}Cave{Mild,Auth,Reb,Nudge,Own,Flat}  cave rate per category when the feedback is wrong
  {TAG}CritOR*, {TAG}CritORFE*, {TAG}PraOR*  odds ratio of caving per SD of severity
                                  (feedback wrong; FE = with category fixed effects)
  {TAG}CritPre                    pre-answer probe shift per SD of criticism severity (points)
  {TAG}Work*                      caves to wrong criticism: N, working still correct, self-check
                                  still correct, stated confidence <= 10% (MATH runs only)
  {TAG}Forced{Ctrl,Crit,Pra}      share of turns with a forced verdict (runs with the fix)
  {TAG}AccZero, {TAG}Missing      turn-0 accuracy; share of turns without a parseable verdict
A run with forced verdicts also gets {TAG}NF* praise macros with those verdicts treated as
missing. Tone validity macros (Tone*) come from the tone_severity folder.

Usage (from code/v4_final/):
  python src/evaluation/final_macros.py --tone ../../results/v4_final/tone_severity --out <dir>/final_numbers.tex \
      --levels_out <dir>/tab_levels.tex --run QwenMath ../../results/v4_final/run_qwen25_7b_math_rf ...
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

N_BOOT = 2000
CATS = {"mild_doubt_are_you_sure": "Mild", "authority_challenge": "Auth", "strong_rebuttal": "Reb",
        "answer_suggestion": "Nudge", "ownership_ego_investment": "Own", "flattery_sycophancy": "Flat"}


def fmt(x, signed=False, nd=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    s = f"{x:.{nd}f}"
    if signed:
        s = ("$+$" if x >= 0 else "$-$") + s.lstrip("-")
    return s


def load_tracks(run_dir, tracks):
    rows = []
    for t in tracks:
        path = os.path.join(run_dir, t, "claims_drift_results.jsonl")
        if os.path.exists(path):
            rows += [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return pd.DataFrame(rows).drop_duplicates(["claim_id", "track", "turn"], keep="last")


def praise_vs_control(d, rng):
    t0 = d[(d.track == "control") & (d.turn == 0)].set_index("claim_id")
    ok0 = t0.verdict == t0.ground_truth_verdict
    last = d[d.turn == d.turn.max()].set_index(["claim_id", "track"]).verdict.unstack()
    gt, pair = t0.ground_truth_verdict, t0.twin_id
    out = {}
    for name, mask in [("Wrong", (gt == "Invalid") & ok0), ("Right", (gt == "Valid") & ~ok0)]:
        ids = mask[mask].index
        acc = lambda tr, i: (last.loc[i, tr] == gt.loc[i]).mean() * 100
        pairs = pair.loc[ids]
        groups = [pairs.index[pairs == p] for p in pairs.unique()]
        boots = []
        for _ in range(N_BOOT):
            ii = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
            boots.append(acc("positive", ii) - acc("control", ii))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        pa, ca = acc("positive", ids), acc("control", ids)
        out.update({f"Pra{name}N": str(len(ids)), f"Pra{name}Pra": fmt(pa), f"Pra{name}Ctrl": fmt(ca),
                    f"Pra{name}Diff": fmt(pa - ca, True), f"Pra{name}DiffLo": fmt(lo, True), f"Pra{name}DiffHi": fmt(hi, True)})
    return out


def tone_macros(run_dir):
    out = {}
    files = glob.glob(os.path.join(run_dir, "tone_severity", "tone_severity_*.csv"))
    turns = glob.glob(os.path.join(run_dir, "tone_severity", "tone_turns_*.csv"))
    if not files or not turns:
        return out
    est = pd.read_csv(files[0])
    pick = lambda pol, spec, subset, outcome: est[(est.polarity == pol) & (est.measure == "severity_eq") & (est.spec == spec)
                                                  & (est.subset == subset) & est.outcome.str.startswith(outcome)]
    for key, pol, spec in [("CritOR", "criticism", "controls"), ("CritORFE", "criticism", "+category FE"), ("PraOR", "praise", "controls")]:
        r = pick(pol, spec, "feedback wrong", "cave")
        if len(r) and not np.isnan(r.est.iloc[0]):
            out.update({key: fmt(r.est.iloc[0], nd=2), key + "Lo": fmt(r.lo.iloc[0], nd=2), key + "Hi": fmt(r.hi.iloc[0], nd=2)})
        else:
            out.update({key: "--", key + "Lo": "--", key + "Hi": "--"})
    r = pick("criticism", "controls", "all turns", "d_pre")
    if len(r):
        out["CritPre"] = fmt(r.est.iloc[0])
    tt = pd.read_csv(turns[0])
    risk = tt[tt.at_risk & tt.cave.notna() & ~tt.feedback_right]
    for cat, short in CATS.items():
        g = risk[risk.category == cat]
        out[f"Cave{short}"] = fmt(g.cave.mean() * 100) if len(g) else "--"
    return out


def working_macros(run_dir):
    turns = glob.glob(os.path.join(run_dir, "tone_severity", "tone_turns_*.csv"))
    rows = os.path.join(run_dir, "derived", "rows.csv")
    if not turns or not os.path.exists(rows):
        return {}
    tt, dr = pd.read_csv(turns[0]), pd.read_csv(rows)
    m = tt.merge(dr[["claim_id", "track", "turn", "computed_ok"]], on=["claim_id", "track", "turn"], how="left")
    resp = load_tracks(run_dir, ["negative"])[["claim_id", "turn", "verbalized_confidence"]]
    m = m.merge(resp, on=["claim_id", "turn"], how="left", suffixes=("", "_r"))
    conf = m["verbalized_confidence_r"] if "verbalized_confidence_r" in m else m["verbalized_confidence"]
    m["conf"] = conf
    c = m[(m.track == "negative") & m.at_risk & (m.cave == 1) & ~m.feedback_right]
    if c.empty:
        return {}
    orig_valid = c.prev_verdict == "Valid"
    self_ok = np.where(orig_valid, c.verifier_prob_valid > 50, c.verifier_prob_valid < 50)
    return {"WorkN": str(len(c)), "WorkOk": fmt((c.computed_ok == 1).mean() * 100, nd=0),
            "WorkSelf": fmt(self_ok.mean() * 100, nd=0), "WorkLowConf": fmt((c.conf <= 10).mean() * 100, nd=0)}


def forced_macros(d):
    if "verdict_forced" not in d:
        return {}
    out = {}
    for track, short in [("control", "Ctrl"), ("negative", "Crit"), ("positive", "Pra")]:
        g = d[d.track == track]
        if len(g):
            out[f"Forced{short}"] = fmt(g.verdict_forced.fillna(False).astype(bool).mean() * 100, nd=0)
    return out


def tone_validity(tone_dir):
    w = pd.read_csv(os.path.join(tone_dir, "wrapper_severity.csv"))
    out = {"ToneWrappers": str(len(w))}
    for pol, short in [("criticism", "Crit"), ("praise", "Pra")]:
        d = w[w.polarity == pol]
        used = [c for c in d if c.startswith("z_") and c != "z_vad_dominance"]
        Z = d[used].fillna(0).to_numpy()
        k = Z.shape[1]
        alpha = k / (k - 1) * (1 - Z.var(axis=0, ddof=1).sum() / Z.sum(axis=1).var(ddof=1))
        out[f"Tone{short}Alpha"] = fmt(alpha, nd=2)
        out[f"Tone{short}Rho"] = fmt(spearmanr(d.severity_eq, d.nominal_valence.abs())[0], nd=2)
        out[f"Tone{short}N"] = str(len(d))
    dom = w[w.polarity == "criticism"]
    out["ToneDomRho"] = fmt(spearmanr(dom.z_vad_dominance, dom.nominal_valence.abs())[0], True, nd=2)
    return out


def level_table(level_runs, path):
    """Per-level MATH table from analyze_pushback.py --by difficulty outputs."""
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{adjustbox}{max width=\columnwidth}", r"\begin{tabular}{l ccc ccc}", r"\toprule",
             r" & \multicolumn{3}{c}{Qwen2.5-7B} & \multicolumn{3}{c}{Llama-3.1-8B} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Level & Acc$_0$ & Unw.\ crit / ctrl & $D$ crit & Acc$_0$ & Unw.\ crit / ctrl & $D$ crit \\", r"\midrule"]
    tabs = [pd.read_csv(p).set_index("difficulty") for _, p in level_runs]
    for lev in tabs[0].index:
        cells = [lev.replace("Level ", "")]
        for t in tabs:
            r = t.loc[lev]
            cells += [f"{100 * r.turn0_acc:.0f}", f"{100 * r.unwarranted_crit:.0f} / {100 * r.unwarranted_ctrl:.0f} ({int(r.unwarranted_n)})",
                      f"{r.disc_crit:.2f}"]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}",
              r"\caption{MATH by difficulty level (60 items per level). Acc$_0$: turn-0 accuracy (\%). Unw.: accuracy (\%) at turn 6 on true claims answered correctly at turn 0, criticism / control, with $n$ in brackets. $D$ under criticism at turn 6. Llama: forced verdicts kept.}",
              r"\label{tab:levels}", r"\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs=2, action="append", metavar=("TAG", "DIR"), required=True)
    ap.add_argument("--tone", required=True, help="tone_severity folder (wrapper scores)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--levels_out", default=None)
    ap.add_argument("--levels", nargs=2, action="append", metavar=("TAG", "CSV"), default=[],
                    help="by_difficulty.csv from analyze_pushback.py --by difficulty (Qwen first, then Llama)")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    macros = {}
    for tag, run_dir in args.run:
        d = load_tracks(run_dir, ["control", "negative", "positive"])
        vals = {}
        if {"positive", "control"} <= set(d.track):
            vals.update(praise_vs_control(d[d.track.isin(["control", "positive"])], rng))
            if "verdict_forced" in d:
                dn = d.copy()
                dn.loc[dn.verdict_forced.fillna(False).astype(bool), "verdict"] = None
                vals.update({"NF" + k: v for k, v in praise_vs_control(dn[dn.track.isin(["control", "positive"])], rng).items()})
        vals.update(tone_macros(run_dir))
        vals.update(working_macros(run_dir))
        vals.update(forced_macros(d))
        t0 = d[(d.track == "control") & (d.turn == 0)]
        vals["AccZero"] = fmt((t0.verdict == t0.ground_truth_verdict).mean() * 100, nd=0)
        vals["Missing"] = fmt(d.verdict.isna().mean() * 100, nd=1)
        macros.update({tag + k: v for k, v in vals.items()})
    macros.update(tone_validity(args.tone))

    with open(args.out, "w") as f:
        f.write("% Generated by final_macros.py; do not edit by hand.\n")
        for k, v in macros.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    if args.levels_out and args.levels:
        level_table(args.levels, args.levels_out)
    print(f"{len(macros)} macros -> {args.out}")


if __name__ == "__main__":
    main()
