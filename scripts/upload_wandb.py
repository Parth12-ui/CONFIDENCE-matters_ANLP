#!/usr/bin/env python3
"""
Upload the runs in results/ to Weights & Biases: one W&B run per experiment run (config,
per-turn accuracy curves, the pushback summary and the raw per-turn logs as an artifact),
plus one run for the tone-severity scores. Run report/build.sh first so the analysis
folders exist.

Usage (from the repository root, after `wandb login`):
  python3 scripts/upload_wandb.py [--project confidence-matters-anlp] [--entity NAME]
"""

import argparse
import glob
import json
import os

import pandas as pd
import wandb

QWEN = "Qwen/Qwen2.5-7B-Instruct"
LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
RUNS = {
    "qwen_math": dict(model=QWEN, dataset="MATH-500 (300 claims)", max_new_tokens=768, force_verdict=False),
    "qwen_real_analysis": dict(model=QWEN, dataset="real analysis (150 claims)", max_new_tokens=768, force_verdict=False),
    "llama_math": dict(model=LLAMA, dataset="MATH-500 (300 claims)", max_new_tokens=1024, force_verdict=True),
    "llama_real_analysis": dict(model=LLAMA, dataset="real analysis (150 claims)", max_new_tokens=1024, force_verdict=True),
    "llama_math_no_fallback": dict(model=LLAMA, dataset="MATH-500 (300 claims)", max_new_tokens=768, force_verdict=False,
                                   note="superseded: ~40% of verdicts missing without the fallback"),
}
COMMON = dict(quantisation="4-bit", decoding="greedy", prompt_style="reasoning_first", feedback_turns=6)


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "/"))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = v
        elif isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
            out[key + "_lo"], out[key + "_hi"] = v
    return out


def table(path):
    return wandb.Table(dataframe=pd.read_csv(path))


def upload_run(project, entity, name, cfg):
    d = os.path.join("results", name)
    tracks = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(d, "*", "turns.jsonl")))
    run = wandb.init(project=project, entity=entity, name=name, job_type="experiment", reinit="finish_previous",
                     config={**COMMON, **cfg, "tracks": tracks})
    for sub in ["pushback", "pushback_no_forced"]:
        s = os.path.join(d, sub, "pushback_summary.json")
        if os.path.exists(s):
            run.summary.update(flatten(json.load(open(s)), f"{sub}/"))
        a = os.path.join(d, sub, "accuracy_by_turn.csv")
        if os.path.exists(a):
            acc = pd.read_csv(a)
            run.log({f"{sub}/accuracy_by_turn": wandb.Table(dataframe=acc)})
            for (group, track), g in acc.groupby(["group", "track"]):
                for _, r in g.sort_values("turn").iterrows():
                    run.log({f"{sub}/{group}/{track}": r.accuracy * 100, "turn": int(r.turn)})
    for f in glob.glob(os.path.join(d, "pushback", "by_difficulty.csv")) + glob.glob(os.path.join(d, "derived", "*.csv")) \
            + glob.glob(os.path.join(d, "tone", "tone_severity_*.csv")):
        if os.path.basename(f) != "rows.csv":
            run.log({os.path.relpath(f, d).replace(".csv", ""): table(f)})
    art = wandb.Artifact(f"{name}-turns", type="dataset",
                         description=f"Per-turn logs (prompt, response, verdict, probes) for {name}")
    for t in tracks:
        art.add_file(os.path.join(d, t, "turns.jsonl"), name=f"{t}/turns.jsonl")
    run.log_artifact(art)
    url = run.url
    run.finish()
    return url


def upload_tone(project, entity):
    run = wandb.init(project=project, entity=entity, name="tone_severity", job_type="analysis",
                     reinit="finish_previous", config={"wordings": 427})
    for f in ["wrapper_scores.csv", "wrapper_severity.csv", "prompt_severity.csv"]:
        run.log({f.replace(".csv", ""): table(os.path.join("results/tone", f))})
    art = wandb.Artifact("tone-severity", type="dataset", description="Tone severity of every feedback wording")
    art.add_dir("results/tone")
    run.log_artifact(art)
    url = run.url
    run.finish()
    return url


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="confidence-matters-anlp")
    ap.add_argument("--entity", default=None)
    args = ap.parse_args()
    urls = {name: upload_run(args.project, args.entity, name, cfg) for name, cfg in RUNS.items()}
    urls["tone_severity"] = upload_tone(args.project, args.entity)
    for k, v in urls.items():
        print(f"{k}\t{v}")


if __name__ == "__main__":
    main()
