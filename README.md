# Confidence Matters: Tone and LLM Confidence over Many Turns

ANLP (Advanced Natural Language Processing) course project, Monsoon 2026, IIIT Hyderabad. Author: Parth Nitesh Chaudhari.

Language models often give up correct answers when a user pushes back. Most evidence comes
from a single push. This project follows **Qwen2.5-7B-Instruct** and **Llama-3.1-8B-Instruct**
through six turns of sustained criticism, sustained praise or a neutral re-ask control, on
maths claims whose truth is known. After every turn it records the model's verdict, two
confidence probes, its stated confidence and, on MATH, the answer its own working reaches.

## Links

| What | Link |
|---|---|
| Code (this repository) | <https://github.com/Parth12-ui/CONFIDENCE-matters_ANLP> |
| Report | [`report/report.pdf`](report/report.pdf) |
| Slides | [`report/slides/`](report/slides) |
| All experiment logs (Weights & Biases project) | [confidence-matters-anlp](https://wandb.ai/parthniteshchaudhari-international-institute-of-informat/confidence-matters-anlp) |

### Weights & Biases (W&B) runs

| Run | What it contains | Link |
|---|---|---|
| Qwen, MATH | control, criticism, praise, and the stance-free dispute / affirm / filler tracks | [qwen_math](https://wandb.ai/parthniteshchaudhari-international-institute-of-informat/confidence-matters-anlp/runs/85dqtcpt) |
| Qwen, real analysis | control, criticism, praise | [qwen_real_analysis](https://wandb.ai/parthniteshchaudhari-international-institute-of-informat/confidence-matters-anlp/runs/39ccn69r) |
| Llama, MATH | control, criticism, praise, dispute | [llama_math](https://wandb.ai/parthniteshchaudhari-international-institute-of-informat/confidence-matters-anlp/runs/04prhtnw) |
| Llama, real analysis | control, criticism, praise | [llama_real_analysis](https://wandb.ai/parthniteshchaudhari-international-institute-of-informat/confidence-matters-anlp/runs/wtibwzrd) |
| Llama, MATH, no fallback | first Llama MATH run, before the verdict fallback (superseded, kept for the comparison) | [llama_math_no_fallback](https://wandb.ai/parthniteshchaudhari-international-institute-of-informat/confidence-matters-anlp/runs/8hmj2p15) |
| Tone severity | severity scores of all 427 feedback wordings | [tone_severity](https://wandb.ai/parthniteshchaudhari-international-institute-of-informat/confidence-matters-anlp/runs/2aonbehx) |

Each experiment run holds its configuration, per-turn accuracy curves, the pushback summary
and the raw per-turn logs (`turns.jsonl` for every track) as a downloadable artifact. The
hidden-state probe results were produced after the upload and are only in this repository
(`results/qwen_math/hidden_probe/`, `results/llama_math/hidden_probe*/`).

### Hugging Face (HF) models and data

| What | Link | Revision used |
|---|---|---|
| Qwen2.5-7B-Instruct | <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct> | `a09a35458c702b33eeacc393d103063234e8bc28` |
| Llama-3.1-8B-Instruct (reference; gated) | <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct> | `0e9e39f249a16976918f6564b8830bc894c89659` |
| Ungated Llama copy: weights, config, tokenizer | <https://huggingface.co/NousResearch/Meta-Llama-3.1-8B-Instruct> | `d10aef7999a2b5ba950ab3974312feeedbfe0b77` |
| Ungated Llama copy: `tokenizer_config.json` | <https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct> | `a2856192dd7c25b842431f39c179a6c2c2f627d1` |
| MATH-500 dataset | <https://huggingface.co/datasets/HuggingFaceH4/MATH-500> | `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be` |
| GoEmotions emotion classifier (RoBERTa) | <https://huggingface.co/SamLowe/roberta-base-go_emotions> | `d75048347613a25d77de8cf6412eaae9fa7b26be` |

The revisions are the `main` branch of each repository when the runs were made (20–28 October
2026); none of these repositories has changed since. The old name
`meta-llama/Meta-Llama-3.1-8B-Instruct`, used in some scripts, redirects to the Llama link above.

### Other resources

| What | Link |
|---|---|
| NRC-VAD lexicon (valence, arousal, dominance) | <https://saifmohammad.com/WebPages/nrc-vad.html> (research use only, so not included here; `scripts/setup_tone.sh` downloads it) |
| Stanford Politeness Corpus | through [ConvoKit](https://convokit.cornell.edu/) |
| Real-analysis source | S. K. Mapa, *Introduction to Real Analysis* |

## Main findings

| | Qwen2.5-7B | Llama-3.1-8B |
|---|---|---|
| True MATH claims answered correctly, still correct after 6 turns: criticism / control | 76.3% / 93.9% | 14.7% / 68.6% |
| Same, real analysis | 55.6% / 95.2% | 14.3% / 59.5% |
| Discrimination between true and false MATH claims after criticism (control) | 0.67 (0.72) | 0.11 (0.46) |

- **Criticism dominates.** It lowers accuracy on claims the model got right in every setting, more so on harder problems.
- **Praise moves Qwen little but sways Llama.**
- **Harsher criticism moves both models more only when it also says the claim is false.** Disagreement of graded harshness without a claim about the answer does not.
- **The models give in differently.** When Qwen abandons a correct MATH answer, its working still reaches that answer in 73% of cases. Llama's probes and stated confidence move with its verdict, but its hidden states keep a weak trace of the correct answer.

## Layout

```
data/
  math/             300 MATH-500 answer claims (levels 1-5), each true claim with a false twin
  real_analysis/    150 real-analysis claims (75 true + 75 false twins); twins.json gives why each twin is false
src/
  run_experiment.py one model, one track: multi-turn conversations, verdicts, probes
  extract_hidden.py hidden states of the logged conversations (one forward pass each, no generation)
  data/             build the MATH claims and the 13 prompts per claim
  tone/             tone severity of every feedback wording (NRC-VAD, politeness, GoEmotions)
  analysis/         pushback, tone effects, derived answers, stance-free tracks, hidden-state probes,
                    report numbers, plot
scripts/            run on a Slurm cluster or locally, fetch and verify weights, tone setup, W&B upload
results/
  qwen_math/ qwen_real_analysis/ llama_math/ llama_real_analysis/
                    one folder per track (control, negative = criticism, positive = praise,
                    dispute / affirm / filler = stance-free), each with turns.jsonl;
                    pushback/, derived/, tone/, hidden_probe/ hold the analysis outputs
  llama_math_no_fallback/  first Llama MATH run, before the verdict fallback (superseded)
  tone/             severity scores of the 427 feedback wordings, plus blank Best-Worst Scaling sheets
report/             LaTeX sources, generated tables and numbers, build.sh, report.pdf, slides/
versions/           snapshots of the three earlier stages (scorer, pilot, interim) and how the code evolved
```

## Why the experiments are reproducible

Reproducing this project means two different things, and both are covered.

**1. Every number in the report can be rebuilt exactly, on a laptop.**
The raw output of every run is in this repository: `results/<run>/<track>/turns.jsonl` holds,
for every claim and every turn, the exact prompt sent, the model's full response, the parsed
verdict, the stated confidence and the probe probabilities. `bash report/build.sh` reruns
every analysis from these files and rebuilds all tables, figures and number macros in the
report, then the PDF. No GPU and no model is needed, and nothing in the report is typed in
by hand. The same raw logs are also stored as artifacts on the W&B runs above.

**2. The runs themselves are deterministic.** Nothing in the pipeline depends on chance
that is not fixed in the code:

| Source of variation | How it is fixed |
|---|---|
| Which MATH problems are used | `src/data/build_math.py` samples 30 per level with seed 42; the resulting `data/math/claims.json` is committed |
| Which feedback wordings each claim gets | `src/data/build_prompts.py` uses seed 42; both `prompts.jsonl` files are committed |
| The order of the six feedback turns | shuffled per claim with `random.Random(42 + claim_id)` in `src/run_experiment.py`, so every model sees the same order for the same claim |
| The model's responses | greedy decoding (`do_sample=False`): the model always takes its most likely next token, so there is no sampling randomness |
| The model weights | `scripts/verify_model.py` checks a local model folder file by file against the hashes Hugging Face publishes; the revisions used are listed above |
| Run settings | passed as command-line flags (below), stored in the W&B run configuration, and the prompt style and track are written into every `turns.jsonl` record |
| Confidence intervals | bootstrap resampling with a fixed seed (0) in every analysis script; the hidden-state probes use deterministic 5-fold splits by claim-twin pair |
| Earlier design decisions | the code and results of the three earlier stages are kept in `versions/`, with what changed and why |

**What can still differ.** All runs used 4-bit quantisation (bitsandbytes) on an NVIDIA RTX
2080 Ti. Low-precision arithmetic can give very slightly different results on another GPU
type or with other library versions, and in a long conversation one different token can
change the rest of a response. Regenerating the conversations on different hardware may
therefore change a few individual responses, though not, as far as we can tell, the
findings. `requirements.txt` gives minimum versions, not exact ones. The rebuild in point 1
is exact regardless, because it starts from the logged responses.

The hidden-state files (about 1 GB per track) are not stored. `report/build.sh` copies the
probe results from `results/*/hidden_probe*/`; recomputing them needs a GPU for about one hour
per model (see below).

## Setup

```bash
pip install -r requirements.txt
```

Experiments need one GPU with about 11 GB of memory (models are loaded in 4-bit). Analysis and
the report run on CPU; the report needs [tectonic](https://tectonic-typesetting.github.io/).

## Reproducing the report

```bash
bash report/build.sh
```

## Running the experiments

Each track is one run of `src/run_experiment.py`. For example, Qwen on MATH, criticism track,
with the settings used in the report:

```bash
python3 src/run_experiment.py --model_name Qwen/Qwen2.5-7B-Instruct \
    --claims_json_path data/math/claims.json --prompts_path data/math/prompts.jsonl \
    --output_dir results/run_qwen_math/negative --track negative \
    --prompt_style reasoning_first --max_new_tokens 768 --max_turns 6 \
    --confidence_floor 50.0 --load_in_4bit
```

Llama used `--max_new_tokens 1024 --force_verdict`. That option trims repetition loops and
completes a missing verdict line. Tracks are `control`, `negative`, `positive`, `dispute`,
`affirm` and `filler`.

To run several tracks in one Slurm job, use `scripts/run_ada.sbatch`. Its defaults are those of
earlier pilots, so set the report settings explicitly:

```bash
MODEL=Qwen/Qwen2.5-7B-Instruct TAG=qwen_math DATA=data/math \
    TRACKS="control negative positive dispute affirm filler" \
    PROMPT_STYLE=reasoning_first MAX_NEW_TOKENS=768 MAX_TURNS=6 BY=difficulty \
    sbatch scripts/run_ada.sbatch
```

For Llama, use `MODEL=meta-llama/Llama-3.1-8B-Instruct MAX_NEW_TOKENS=1024 FORCE_VERDICT=1`.
Output goes to `results/run_<TAG>/`. (`scripts/run_local.sh` is a pilot script with short
verdict-first settings, not the report settings.)

**Model weights.** `scripts/get_model.sh` downloads them and `scripts/verify_model.py` checks
they match the Hub. `scripts/llama_tokenizer.py` rebuilds Meta's exact Llama tokenizer
configuration from the ungated copies.

**Hidden-state probes** (MATH, one Slurm job per model):

```bash
MODEL=Qwen/Qwen2.5-7B-Instruct RUN=qwen_math TAG=QwenMath sbatch scripts/run_hidden_ada.sbatch
MODEL=meta-llama/Llama-3.1-8B-Instruct RUN=llama_math TAG=LlamaMath sbatch scripts/run_hidden_ada.sbatch
```

**Rebuild the data:**

```bash
python3 src/data/build_math.py --math500 path/to/MATH-500/test.jsonl
python3 src/data/build_prompts.py --claims_path data/math/claims.json --output_path data/math/prompts.jsonl --subject mathematics
python3 src/data/build_prompts.py   # real analysis
```

**Tone severity** (CPU, separate environment):

```bash
bash scripts/setup_tone.sh
python3 src/tone/wrappers.py --prompts data/math/prompts.jsonl data/real_analysis/prompts.jsonl --out results/tone/wrappers.csv
.venv_tone/bin/python src/tone/score.py --wrappers results/tone/wrappers.csv --out results/tone/wrapper_scores.csv
python3 src/tone/composite.py --scores results/tone/wrapper_scores.csv --prompt_wrappers results/tone/prompt_wrappers.csv --out_dir results/tone
```

**Upload the runs to W&B** (after `wandb login` and `bash report/build.sh`):

```bash
python3 scripts/upload_wandb.py
```

## Acknowledgements

This project couldn't have been possible without the support of Dhiraj Chitluri, Gollapalli Phanindra and Parth Dhodapkar.
