"""
Best-Worst Scaling (Kiritchenko & Mohammad, ACL 2017) for human severity ratings of a
stratified sample of feedback wrappers.

  make   samples --per_category wrappers from each category (30 criticism + 30 praise by
         default), builds 2N four-item tuples per polarity in which every wrapper appears
         equally often, and writes one sheet per annotator (tuple order and positions shuffled
         per annotator) plus a key that maps sheet letters back to wrapper ids.
  score  reads the filled sheets: each wrapper's score is (#best - #worst) / #appearances,
         in [-1, 1]. Reliability is split-half: annotations are split at random into two
         halves 200 times, the halves' scores are correlated (Spearman) and Spearman-Brown
         corrected.

Annotators fill the `best` and `worst` columns with a letter A-D. Criticism tuples ask
which message pushes back most / least harshly; praise tuples ask which pushes most / least
strongly towards agreeing.

Usage (from code/v4_final/):
  python src/tone/bws.py make --wrappers ../../results/v4_final/tone_severity/wrappers.csv \
      --out_dir ../../results/v4_final/tone_severity/bws --annotators A1 A2 A3
  python src/tone/bws.py score --out_dir ../../results/v4_final/tone_severity/bws
"""

import argparse
import glob
import itertools
import os
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

LETTERS = "ABCD"
QUESTION = {
    "criticism": ("Which message pushes back MOST harshly?", "Which pushes back LEAST harshly?"),
    "praise": ("Which message pushes MOST strongly towards agreeing?", "Which pushes LEAST strongly?"),
}
INSTRUCTIONS = """# Rating feedback messages (Best-Worst Scaling)

Each row shows four messages a user might send to an AI model about a maths claim.
The claim itself is left out; judge only the tone of the message.

- Rows with set = criticism: in `best`, put the letter (A-D) of the message that pushes
  back MOST harshly against the model's answer; in `worst`, the one that pushes back LEAST harshly.
- Rows with set = praise: in `best`, the message that pushes MOST strongly towards
  agreeing (praise, confidence, pressure to confirm); in `worst`, the one that pushes LEAST.

Go with your first impression; there are no right answers, and ties are fine to break at
random. Please work alone and do not discuss ratings until everyone has finished.
Each sheet has 120 rows and takes about 30-40 minutes; breaks are fine. Save the file under the same name when done.
"""


def make_tuples(ids, n_tuples: int, rng, tries: int = 2000) -> list:
    """n_tuples 4-tuples, each id equally often, no id twice in a tuple, few repeated pairs."""
    reps = n_tuples * 4 // len(ids)
    best, best_rep = None, None
    for _ in range(tries):
        pool = np.concatenate([rng.permutation(ids) for _ in range(reps)])
        tuples = [tuple(pool[i:i + 4]) for i in range(0, len(pool), 4)]
        if any(len(set(t)) < 4 for t in tuples):
            continue
        pairs = Counter(frozenset(p) for t in tuples for p in itertools.combinations(t, 2))
        repeated = sum(c - 1 for c in pairs.values())
        if best is None or repeated < best_rep:
            best, best_rep = tuples, repeated
            if repeated == 0:
                break
    if best is None:
        raise RuntimeError("could not build tuples")
    return best


def cmd_make(args):
    rng = np.random.default_rng(args.seed)
    wr = pd.read_csv(args.wrappers)
    sample = wr.groupby("category").sample(n=args.per_category, random_state=args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    sample[["wrapper_id", "polarity", "category", "text"]].to_csv(os.path.join(args.out_dir, "bws_items.csv"), index=False)

    tuples = []
    for pol in ("criticism", "praise"):
        ids = sample.loc[sample["polarity"] == pol, "wrapper_id"].to_numpy()
        n = 2 * len(ids)
        for k, t in enumerate(make_tuples(ids, n, rng)):
            tuples.append({"tuple_id": f"{pol[0]}{k + 1:03d}", "set": pol, "items": t})
    text = dict(zip(sample["wrapper_id"], sample["text"]))

    key_rows = []
    for ann in args.annotators:
        rows = []
        for i in rng.permutation(len(tuples)):
            t = tuples[i]
            items = list(rng.permutation(t["items"]))
            most, least = QUESTION[t["set"]]
            row = {"row": len(rows) + 1, "tuple_id": t["tuple_id"], "set": t["set"], "best_asks": most, "worst_asks": least}
            row.update({L: text[w] for L, w in zip(LETTERS, items)})
            row.update({"best": "", "worst": ""})
            rows.append(row)
            key_rows += [{"annotator": ann, "tuple_id": t["tuple_id"], "letter": L, "wrapper_id": w}
                         for L, w in zip(LETTERS, items)]
        pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, f"sheet_{ann}.csv"), index=False)
    pd.DataFrame(key_rows).to_csv(os.path.join(args.out_dir, "bws_key.csv"), index=False)
    with open(os.path.join(args.out_dir, "INSTRUCTIONS.md"), "w") as f:
        f.write(INSTRUCTIONS)
    print(f"{len(sample)} wrappers, {len(tuples)} tuples per annotator, sheets for {', '.join(args.annotators)} in {args.out_dir}")


def counts_to_scores(ann: pd.DataFrame) -> pd.Series:
    """ann: one row per (annotation, item) with columns wrapper_id, is_best, is_worst."""
    g = ann.groupby("wrapper_id")
    return (g["is_best"].sum() - g["is_worst"].sum()) / g.size()


def cmd_score(args):
    key = pd.read_csv(os.path.join(args.out_dir, "bws_key.csv"))
    items = pd.read_csv(os.path.join(args.out_dir, "bws_items.csv"))
    rows, problems = [], []
    for path in sorted(glob.glob(os.path.join(args.out_dir, "sheet_*.csv"))):
        ann = os.path.basename(path)[len("sheet_"):-len(".csv")]
        sheet = pd.read_csv(path, dtype=str, keep_default_na=False)
        for _, r in sheet.iterrows():
            b, w = r["best"].strip().upper(), r["worst"].strip().upper()
            if not b and not w:
                continue
            if b not in LETTERS or w not in LETTERS or b == w:
                problems.append(f"{ann} row {r['row']}: best={b!r} worst={w!r}")
                continue
            k = key[(key["annotator"] == ann) & (key["tuple_id"] == r["tuple_id"])]
            for _, kr in k.iterrows():
                rows.append({"annotator": ann, "tuple_id": r["tuple_id"], "wrapper_id": kr["wrapper_id"],
                             "is_best": int(kr["letter"] == b), "is_worst": int(kr["letter"] == w)})
    if problems:
        print("Skipped rows with invalid answers:\n  " + "\n  ".join(problems))
    ann = pd.DataFrame(rows)
    if ann.empty:
        raise SystemExit("no filled-in annotations found")
    done = ann.groupby("annotator")["tuple_id"].nunique()
    print("tuples rated per annotator:\n" + done.to_string())

    scores = counts_to_scores(ann).rename("bws_score").reset_index()
    out = items.merge(scores, on="wrapper_id", how="left")
    out.to_csv(os.path.join(args.out_dir, "bws_scores.csv"), index=False)

    rng = np.random.default_rng(args.seed)
    units = ann[["annotator", "tuple_id"]].drop_duplicates().to_numpy()
    lines = []
    for pol in ("criticism", "praise"):
        pu = np.array([u for u in units if u[1][0] == pol[0]])
        rs = []
        for _ in range(200):
            idx = rng.permutation(len(pu))
            halves = [set(map(tuple, pu[idx[: len(pu) // 2]])), set(map(tuple, pu[idx[len(pu) // 2:]]))]
            s = [counts_to_scores(ann[[(a, t) in h for a, t in zip(ann["annotator"], ann["tuple_id"])]]) for h in halves]
            j = s[0].index.intersection(s[1].index)
            rs.append(spearmanr(s[0][j], s[1][j])[0])
        r = float(np.nanmean(rs))
        sb = 2 * r / (1 + r)
        cat = out[out["polarity"] == pol].groupby("category")["bws_score"].mean().sort_values()
        lines += [f"{pol}: split-half Spearman {r:.2f}, Spearman-Brown corrected {sb:.2f}",
                  "  category means: " + ", ".join(f"{c} {v:+.2f}" for c, v in cat.items())]
    print("\n".join(lines))
    with open(os.path.join(args.out_dir, "bws_reliability.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--wrappers", required=True)
    m.add_argument("--out_dir", required=True)
    m.add_argument("--annotators", nargs="+", default=["A1", "A2", "A3"])
    m.add_argument("--per_category", type=int, default=10)
    m.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("score")
    s.add_argument("--out_dir", required=True)
    s.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cmd_make(args) if args.cmd == "make" else cmd_score(args)


if __name__ == "__main__":
    main()
