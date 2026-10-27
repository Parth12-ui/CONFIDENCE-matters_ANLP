"""
Feedback wrappers: the part of a feedback prompt that is not the claim itself.

Every feedback prompt in claims_prompts.jsonl is a wrapper around the claim's neutral
question ("I mean, maybe I'm wrong, but This proof attempt is garbage ... <CLAIM>").
Tone is a property of the wrapper, so severity is scored on the wrapper with the claim
removed, once per distinct wrapper, and joined back to result rows through prompt_id.

Usage (from the repository root):
  python src/tone/wrappers.py --prompts data/math/prompts.jsonl data/real_analysis/prompts.jsonl \
      --out results/tone/wrappers.csv
"""

import argparse
import hashlib
import json
import os
import re

import pandas as pd

CLAIM = "<CLAIM>"
# Nominal valences assigned by the authors (same as the runner's CATEGORY_DEFAULT_SCORES)
NOMINAL = {
    "mild_doubt_are_you_sure": -0.40,
    "authority_challenge": -0.65,
    "strong_rebuttal": -0.90,
    "answer_suggestion": +0.40,
    "ownership_ego_investment": +0.55,
    "flattery_sycophancy": +0.85,
}


def polarity(category: str) -> str:
    return "criticism" if NOMINAL[category] < 0 else "praise"


def tone_text(wrapper: str) -> str:
    """The wrapper as scored and shown to raters: the claim slot removed, spacing tidied."""
    text = wrapper.replace(CLAIM, " ")
    return re.sub(r"\s+", " ", text).strip()


def wrapper_id(wrapper: str) -> str:
    return "w" + hashlib.sha1(wrapper.encode("utf-8")).hexdigest()[:10]


def prompt_map(prompts_paths) -> pd.DataFrame:
    """One row per feedback prompt: prompt_id, claim_id, dataset, category, wrapper_id, wrapper."""
    rows = []
    for path in prompts_paths:
        dataset = os.path.basename(os.path.dirname(os.path.abspath(path)))
        records = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        neutral = {r["claim_id"]: r["prompt"] for r in records if r["category"] == "neutral_control"}
        for r in records:
            if r["category"] not in NOMINAL:
                continue
            question = neutral[r["claim_id"]]
            if question not in r["prompt"]:
                raise ValueError(f"{path}: claim text not found in {r['prompt_id']}")
            wrapper = r["prompt"].replace(question, CLAIM)
            rows.append({
                "dataset": dataset,
                "prompt_id": r["prompt_id"],
                "claim_id": r["claim_id"],
                "category": r["category"],
                "wrapper_id": wrapper_id(wrapper),
                "wrapper": wrapper,
            })
    return pd.DataFrame(rows)


def distinct_wrappers(pm: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct wrapper, with how many prompts and datasets use it."""
    g = pm.groupby("wrapper_id")
    out = g.agg(
        category=("category", "first"),
        n_categories=("category", "nunique"),
        wrapper=("wrapper", "first"),
        n_prompts=("prompt_id", "size"),
        datasets=("dataset", lambda s: ";".join(sorted(set(s)))),
    ).reset_index()
    if (out["n_categories"] > 1).any():
        raise ValueError("a wrapper appears under more than one category")
    out = out.drop(columns="n_categories")
    out["polarity"] = out["category"].map(polarity)
    out["nominal_valence"] = out["category"].map(NOMINAL)
    out["text"] = out["wrapper"].map(tone_text)
    return out.sort_values(["polarity", "nominal_valence", "wrapper_id"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", nargs="+", required=True, help="claims_prompts.jsonl files")
    ap.add_argument("--out", required=True, help="wrappers.csv; prompt_wrappers.csv is written beside it")
    args = ap.parse_args()

    pm = prompt_map(args.prompts)
    wr = distinct_wrappers(pm)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    wr.to_csv(args.out, index=False)
    pm.drop(columns="wrapper").to_csv(os.path.join(os.path.dirname(os.path.abspath(args.out)), "prompt_wrappers.csv"), index=False)
    print(f"{len(pm)} feedback prompts, {len(wr)} distinct wrappers")
    print(wr.groupby(["polarity", "category"]).size().to_string())


if __name__ == "__main__":
    main()
