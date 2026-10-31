# How the code evolved

Snapshots of the three earlier stages of the project, taken from the team's original repository.
The current code at the repository root is the fourth stage. The snapshots are kept for
reference; they are not run by anything in the root, and their scripts still assume the old
repository layout.

| Stage | Dates | Authors | What it did | Why it changed |
|---|---|---|---|---|
| [`v1_scorer/`](v1_scorer) | 11–27 Sep 2026 | Dhiraj Chitluri, Gollapalli Phanindra, Parth Dhodapkar, Parth Nitesh Chaudhari | Trained a DeBERTa-v3-large regressor to give every feedback message a valence score | A TF-IDF baseline almost matched it (r = 0.92 vs 0.96) because the same templates were in train and test. On a template-disjoint split it fell to r = 0.69, so the score was replaced by established measures (`src/tone/`) |
| [`v2_pilot/`](v2_pilot) | 27 Sep – 2 Oct 2026 | Dhiraj Chitluri, Parth Dhodapkar, Parth Nitesh Chaudhari | First multi-turn runs: criticism and praise tracks on 92 claims (70 true, 22 false), plus a first reversibility (O2) run | No control track and only 22 false claims, so a change under pressure could not be separated from drift. The O2 run pooled false claims and stopped early |
| [`v3_interim/`](v3_interim) | 2–21 Oct 2026 | Dhiraj Chitluri, Gollapalli Phanindra, Parth Dhodapkar, Parth Nitesh Chaudhari | 75 true real-analysis claims, each paired with a false twin; a no-feedback control track; truth-confidence probes; reversibility and recovery-through-praise experiments | Early stopping meant tracks could only be compared at turn 2. With the verdict asked before any working, Llama accepted only 32% of true claims at turn 0 |
| repository root | 21 Oct 2026 – | Parth Nitesh Chaudhari | Fixed six-turn horizon, reasoning before the verdict, a MATH claim set (levels 1–5), Qwen and Llama, stance-free tracks, tone severity, the derived-answer check and hidden-state probes | Final study, used by `report/` |

## Main changes between stages

**v1 → v2.** The project moved from building a tone scorer to measuring the models. The tracks
reuse the claims and feedback prompts built at the end of v1 (`build_claims_prompts.py`, 28 Sep).

**v2 → v3.**
- Every true claim gets a false twin that is wrong on purpose.
- A control track re-asks the question without feedback, so pressure is measured against drift.
- Confidence is read from the model's probability of "True".
- The runner is a single script, `run_claims_experiment.py`, with one track per run.

**v3 → root.** The `v3_interim` code is the code as it stood on 21 October, when the final stage
started. It therefore already contains the first of the changes below: the stance-free tracks, a
reasoning-first option, and stopping rules switched off by default. The `v3_interim` results were
produced before those changes.
- Every run lasts exactly six feedback turns, with no early stopping.
- The model writes its working before giving a verdict, and the token limit was raised.
- Llama's repetition loops are trimmed and missing verdicts completed (`--force_verdict`).
- A 300-claim MATH-500 set was added, so each claim has a checkable answer.
- Criticism is split into warranted (the claim really is false) and unwarranted.
- Stance-free tracks (dispute, affirm, filler) vary tone without saying whether the answer is right.
- Reversibility (O2) and recovery (O3/O6) were dropped to keep the scope to sustained one-direction feedback.

## What was left out of the snapshots

Snapshots hold the code and the smaller results (summaries, metrics, figures).

**Left out because of size:**
- files over 500 KB: raw per-turn logs, the scorer's training data and tokenizer dumps;
- model weights, which were never in git.

**Personal details removed:**
- a teammate's home directory in one metrics file;
- a teammate's cluster username in one script.

The full history of each file is in the original repository.
