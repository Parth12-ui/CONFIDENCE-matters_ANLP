"""
Automated tone measures for each distinct feedback wrapper (see wrappers.py).

Measures (all computed on the wrapper text with the claim removed):
  vad_valence, vad_arousal, vad_dominance
      NRC-VAD lexicon (Mohammad, ACL 2018): mean human rating of the matched words.
      Rescaled to [-1, 1] whichever lexicon version is supplied (v1 is 0..1, v2 is -1..1).
  politeness
      P(polite) from a logistic regression on ConvoKit politeness-strategy features,
      trained on the Wikipedia requests of the Stanford Politeness Corpus
      (Danescu-Niculescu-Mizil et al., ACL 2013). pol_* columns hold the strategy markers.
  emo_*
      GoEmotions (Demszky et al., ACL 2020) label probabilities from a classifier
      fine-tuned on that dataset (--goemotions_model).
  n_words, hedge, authority_cue
      Surface covariates: length, and regex flags for hedges and appeals to authority.

Each measure is optional (--skip vad politeness emotions) so the scorer runs with
whatever resources are installed; composite.py uses the columns that are present.

Usage (from the repository root, with the tone venv):
  .venv_tone/bin/python src/tone/score.py --wrappers results/tone/wrappers.csv \
      --vad_path resources/NRC-VAD-Lexicon.txt --out results/tone/wrapper_scores.csv
"""

import argparse
import os
import re

import numpy as np
import pandas as pd

WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
HEDGE = re.compile(
    r"\b(maybe|perhaps|i think|i could be|i might|not sure|i guess|my guess|i'd lean|"
    r"take this with a grain of salt|not to be that person|i mean)\b", re.IGNORECASE)
AUTHORITY = re.compile(
    r"\b(professor|mathematician|instructor|expert|phd|taught|teach|textbook|"
    r"\d+ years|two decades|tenured|experienced)\b", re.IGNORECASE)
EMOTIONS = ["disapproval", "annoyance", "anger", "disappointment", "confusion", "curiosity",
            "admiration", "approval", "gratitude", "optimism", "pride", "neutral"]


def load_vad(path: str) -> dict:
    """word -> (valence, arousal, dominance) on [-1, 1]; reads the v1 or v2 text formats."""
    df = pd.read_csv(path, sep="\t", header=None, comment=None, dtype=str, keep_default_na=False)
    if not _is_number(df.iloc[0, 1]):  # header row (v2 and some v1 copies)
        df = df.iloc[1:]
    df = df.iloc[:, :4]
    df.columns = ["word", "v", "a", "d"]
    vals = df[["v", "a", "d"]].astype(float)
    if vals.min().min() >= 0:  # v1: 0..1
        vals = vals * 2 - 1
    return {w.lower(): tuple(r) for w, r in zip(df["word"], vals.itertuples(index=False))}


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def score_vad(texts, vad: dict) -> pd.DataFrame:
    rows = []
    for t in texts:
        hits = [vad[w] for w in WORD.findall(t.lower()) if w in vad]
        if hits:
            v, a, d = np.mean(hits, axis=0)
        else:
            v = a = d = np.nan
        rows.append({"vad_valence": v, "vad_arousal": a, "vad_dominance": d, "vad_matched": len(hits)})
    return pd.DataFrame(rows)


def score_politeness(texts, cache_dir: str) -> pd.DataFrame:
    """Train the politeness model on the Wikipedia requests, then score our texts."""
    from convokit import Corpus, PolitenessStrategies, Speaker, TextParser, Utterance, download
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    parser = TextParser(verbosity=0)
    ps = PolitenessStrategies(verbose=0)

    local = os.path.join(cache_dir, "wikipedia-politeness-corpus")
    wiki = Corpus(filename=local if os.path.isdir(local) else download("wikipedia-politeness-corpus", data_dir=cache_dir))
    wiki = ps.transform(parser.transform(wiki), markers=False)
    train = [(u.meta["politeness_strategies"], u.meta["Binary"]) for u in wiki.iter_utterances()
             if u.meta.get("Binary") in (1, -1)]
    vec = DictVectorizer()
    X = vec.fit_transform([f for f, _ in train])
    y = np.array([b == 1 for _, b in train])
    clf = LogisticRegression(max_iter=2000, C=1.0)
    acc = cross_val_score(clf, X, y, cv=10).mean()
    clf.fit(X, y)
    print(f"politeness model: {len(train)} Wikipedia requests (polite vs impolite), 10-fold CV accuracy {acc:.3f}")

    spk = Speaker(id="user")
    ours = Corpus(utterances=[Utterance(id=f"u{i}", speaker=spk, text=t) for i, t in enumerate(texts)])
    ours = ps.transform(parser.transform(ours), markers=False)
    feats = [ours.get_utterance(f"u{i}").meta["politeness_strategies"] for i in range(len(texts))]
    out = pd.DataFrame(feats).add_prefix("pol_").reset_index(drop=True)
    out.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", c).strip("_") for c in out.columns]
    out.insert(0, "politeness", clf.predict_proba(vec.transform(feats))[:, 1])
    return out


def score_emotions(texts, model_name: str) -> pd.DataFrame:
    from transformers import pipeline

    clf = pipeline("text-classification", model=model_name, top_k=None, function_to_apply="sigmoid", device=-1)
    rows = []
    for preds in clf(list(texts), batch_size=16, truncation=True):
        p = {d["label"]: d["score"] for d in preds}
        rows.append({f"emo_{e}": p.get(e, np.nan) for e in EMOTIONS})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrappers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vad_path", default="resources/NRC-VAD-Lexicon.txt")
    ap.add_argument("--goemotions_model", default="resources/goemotions",
                    help="local copy or hub id of SamLowe/roberta-base-go_emotions")
    ap.add_argument("--cache_dir", default="resources", help="where ConvoKit stores its corpus")
    ap.add_argument("--skip", nargs="*", default=[], choices=["vad", "politeness", "emotions"])
    args = ap.parse_args()

    wr = pd.read_csv(args.wrappers)
    texts = wr["text"].tolist()
    parts = [wr[["wrapper_id", "polarity", "category", "nominal_valence", "text"]]]

    surface = pd.DataFrame({
        "n_words": [len(WORD.findall(t.lower())) for t in texts],
        "hedge": [int(bool(HEDGE.search(t))) for t in texts],
        "authority_cue": [int(bool(AUTHORITY.search(t))) for t in texts],
    })
    parts.append(surface)
    if "vad" not in args.skip:
        vad = load_vad(args.vad_path)
        print(f"NRC-VAD: {len(vad)} words")
        parts.append(score_vad(texts, vad))
    if "politeness" not in args.skip:
        os.makedirs(args.cache_dir, exist_ok=True)
        parts.append(score_politeness(texts, args.cache_dir))
    if "emotions" not in args.skip:
        parts.append(score_emotions(texts, args.goemotions_model))

    out = pd.concat([p.reset_index(drop=True) for p in parts], axis=1)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False)
    main_cols = [c for c in ["vad_valence", "vad_arousal", "vad_dominance", "politeness",
                             "emo_disapproval", "emo_admiration", "n_words"] if c in out]
    print(out.groupby("category")[main_cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()
