#!/usr/bin/env python3
"""
refine_scorer_data.py

Second pass over the curated scorer dataset (data/curated/full_curated_dataset.jsonl)
that fixes two problems found in the October audit, without re-downloading anything:

1. Noisy sources.
   - final_tonality_ai_praise: positive items were never checked for being feedback
     to an assistant, so about half are product/service reviews. Keep only texts that
     match AI_PRAISE_REGEX (the critique side already required AI_CRITIQUE_REGEX).
   - init_real_human_feedback: OASST replies scored by a word lexicon, so plain
     questions containing "best" or "stupid" get |v| > 0.5. Keep evaluative items only
     when their sign agrees with the critique/praise regexes; keep near-neutral items.

2. Template leakage. Template-built items were grouped by their Alpaca instruction,
   so the same feedback phrase ("trash bro") appears in train and test with different
   instructions. Here every item that contains a phrase from the extract.py /
   preprocess_tonality.py banks is grouped by that phrase, and the split is made at
   group level, so test phrases are never seen in training.

Writes train/val/test/full jsonl (same schema as data/curated) plus summary.json.

Usage:
  python src/data_pipeline/refine_scorer_data.py                 # -> data/curated_v2
  python src/data_pipeline/refine_scorer_data.py --output_dir data/curated_v2 --seed 42
"""

import argparse
import ast
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Imported as a sibling module: the src package __init__ pulls in torch/transformers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess_tonality import (  # noqa: E402
    AI_CRITIQUE_REGEX,
    AI_PRAISE_REGEX,
    EXPANSION_DOUBTS,
    EXPANSION_REBUTTALS,
    REVIEW_REGEX,
)

DEFAULT_INPUT = os.path.join(REPO_ROOT, "data", "curated", "full_curated_dataset.jsonl")
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "curated_v2")
EXTRACT_PY = os.path.join(REPO_ROOT, "src", "data_pipeline", "extract.py")

# extract.py downloads datasets at import time, so its phrase banks are read with ast
EXTRACT_BANKS = [
    "STRONG_REBUTTAL_PREFIXES",
    "MILD_DOUBT_PREFIXES",
    "MILD_POSITIVE_PREFIXES",
    "STRONG_SYCOPHANCY_PREFIXES",
    "SINGLE_WORD_ANCHORS",
    "CONTEXTUAL_ANSWER_REACTIONS",
]
# Sources built from phrase banks; every item of these should match a phrase
TEMPLATE_SOURCES = {
    "synth_rebuttal", "synth_doubt",
    "init_contextual_rebuttal", "init_contextual_doubt",
    "init_contextual_mild_bias", "init_contextual_sycophancy",
    "init_short_strong_rebuttal", "init_short_mild_doubt",
    "init_short_mild_positive", "init_short_strong_sycophancy",
    "init_single_word_anchor", "init_contextual_answer_reaction",
}
NEUTRAL_BAND = 0.15
# AI_PRAISE_REGEX misses short reactions such as "that's a good list thank you";
# these extra cues recover them (reviews are removed by REVIEW_REGEX first)
PRAISE_EXTRA_REGEX = re.compile(
    r"\bthank(s| you)\b"
    r"|\b(that'?s|this is|that was|it'?s) (a |an |some |really |very |so )*"
    r"(great|good|helpful|useful|informative|nice|excellent|awesome|perfect|insightful|clear)\b"
    r"|\b(great|good|nice|excellent|helpful|insightful) (answer|response|explanation|list|tips|summary|idea|ideas|suggestion|suggestions|point)\b",
    re.IGNORECASE,
)
PRAISE_EXTRA_MAX_WORDS = 60
SPLIT_SHARES = {"test": 0.10, "val": 0.10}


def load_phrase_bank() -> list:
    tree = ast.parse(open(EXTRACT_PY, encoding="utf-8").read())
    phrases = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) in EXTRACT_BANKS:
            phrases += [p for p, _ in ast.literal_eval(node.value)]
    phrases += [p for p, _ in EXPANSION_REBUTTALS + EXPANSION_DOUBTS]
    # Longest first, so "completely wrong" does not claim "stop hallucinating, that is completely wrong"
    return sorted({p.lower() for p in phrases}, key=len, reverse=True)


def is_ai_praise(text: str) -> bool:
    if AI_PRAISE_REGEX.search(text):
        return True
    return len(text.split()) <= PRAISE_EXTRA_MAX_WORDS and bool(PRAISE_EXTRA_REGEX.search(text))


def keep_record(r: dict) -> tuple:
    """Returns (keep, reason) for the noisy-source filter."""
    src, text, v = r["source"], r["text"], r["base_score"]
    if src == "final_tonality_ai_praise":
        if REVIEW_REGEX.search(text):
            return False, "praise_review"
        if not is_ai_praise(text):
            return False, "praise_not_about_ai"
    if src == "init_real_human_feedback" and abs(v) >= NEUTRAL_BAND:
        if v < 0 and not AI_CRITIQUE_REGEX.search(text):
            return False, "human_neg_unsupported"
        if v > 0 and not is_ai_praise(text):
            return False, "human_pos_unsupported"
    return True, "kept"


def template_of(text: str, bank: list) -> str:
    low = text.lower()
    for p in bank:
        # Single-word anchors only count as the template when they are the whole text
        if " " not in p and low.strip(" .!?") != p:
            continue
        if p in low:
            return p
    return None


def stratum(v: float) -> str:
    if v <= -0.60:
        return "strong_neg"
    if v <= -0.20:
        return "mild_neg"
    if v < 0.20:
        return "neutral"
    if v < 0.60:
        return "mild_pos"
    return "strong_pos"


def split_groups(groups: dict, seed: int) -> dict:
    """Assigns whole groups to test/val/train, filling each stratum's record share."""
    rng = random.Random(seed)
    by_stratum = defaultdict(list)
    for gid, items in groups.items():
        dom = Counter(stratum(r["base_score"]) for r in items).most_common(1)[0][0]
        by_stratum[dom].append(gid)
    assign = {}
    for st, gids in sorted(by_stratum.items()):
        gids.sort()
        rng.shuffle(gids)
        total = sum(len(groups[g]) for g in gids)
        filled = {"test": 0, "val": 0}
        for g in gids:
            n = len(groups[g])
            for name in ("test", "val"):
                # Take the group only if it does not overshoot the target by more than half its size
                if filled[name] + n / 2 <= SPLIT_SHARES[name] * total:
                    assign[g] = name
                    filled[name] += n
                    break
            else:
                assign[g] = "train"
    return assign


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    print(f"Loaded {len(records)} records from {args.input}")

    reasons = Counter()
    kept = []
    for r in records:
        ok, why = keep_record(r)
        reasons[why] += 1
        if ok:
            kept.append(r)
    print("Filter:", dict(reasons))

    bank = load_phrase_bank()
    groups = defaultdict(list)
    unmatched_template = Counter()
    for r in kept:
        tpl = template_of(r["text"], bank)
        if tpl is not None:
            gid = f"tpl::{tpl}"
        else:
            if r["source"] in TEMPLATE_SOURCES:
                unmatched_template[r["source"]] += 1
            gid = f"text::{r['text'].lower().strip()}"
        r["group_id"] = gid
        groups[gid].append(r)
    n_tpl = sum(1 for g in groups if g.startswith("tpl::"))
    print(f"Groups: {len(groups)} ({n_tpl} templates); template-source items with no phrase match: "
          f"{sum(unmatched_template.values())} {dict(unmatched_template)}")

    assign = split_groups(groups, args.seed)
    splits = {"train": [], "val": [], "test": []}
    for gid, items in groups.items():
        splits[assign[gid]].extend(items)
    rng = random.Random(args.seed)
    for name in splits:
        splits[name].sort(key=lambda r: r["text"])
        rng.shuffle(splits[name])

    # Leakage checks: no shared group, no shared text
    seen = {name: {r["group_id"] for r in items} for name, items in splits.items()}
    texts = {name: {r["text"].lower().strip() for r in items} for name, items in splits.items()}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        if seen[a] & seen[b] or texts[a] & texts[b]:
            raise RuntimeError(f"Leakage between {a} and {b}")
    print("Leakage check passed: no template or text is shared across splits.")

    os.makedirs(args.output_dir, exist_ok=True)

    def save(items, path):
        with open(path, "w", encoding="utf-8") as f:
            for r in items:
                f.write(json.dumps({"text": r["text"], "base_score": r["base_score"],
                                    "source": r["source"], "strata": r["strata"],
                                    "group_id": r["group_id"]}) + "\n")

    for name, items in splits.items():
        save(items, os.path.join(args.output_dir, f"{name}.jsonl"))
    save(splits["train"] + splits["val"] + splits["test"], os.path.join(args.output_dir, "full_curated_dataset.jsonl"))

    summary = {
        "input": os.path.relpath(args.input, REPO_ROOT),
        "seed": args.seed,
        "n_input": len(records),
        "filter": dict(reasons),
        "n_kept": len(kept),
        "n_groups": len(groups),
        "n_template_groups": n_tpl,
        "unmatched_template_items": dict(unmatched_template),
        "splits": {name: {"n": len(items),
                          "templates": sum(1 for g in seen[name] if g.startswith("tpl::")),
                          "strata": dict(Counter(stratum(r["base_score"]) for r in items))}
                   for name, items in splits.items()},
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    for name in ("train", "val", "test"):
        s = summary["splits"][name]
        print(f"  {name:5s}: {s['n']:6d} records, {s['templates']:3d} templates, {s['strata']}")
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
