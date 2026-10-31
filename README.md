# Confidence Matters: Tone and LLM Confidence over Many Turns

ANLP course project, Monsoon 2026, IIIT Hyderabad. Author: Parth Nitesh Chaudhari.

Language models often give up correct answers when a user pushes back. Most evidence comes
from a single push. This project follows **Qwen2.5-7B-Instruct** and **Llama-3.1-8B-Instruct**
through six turns of sustained criticism, sustained praise or a neutral re-ask control, on
claims with a known truth value. After every turn it records the model's verdict, two
confidence probes, its stated confidence and, on MATH, the answer its own working reaches.

- **Report:** [`report/report.pdf`](report/report.pdf)
- **Experiment logs (Weights & Biases):** [confidence-matters-anlp](https://wandb.ai/parthniteshchaudhari-international-institute-of-informat/confidence-matters-anlp)

## Main findings

| | Qwen2.5-7B | Llama-3.1-8B |
|---|---|---|
| True MATH claims answered correctly, still correct after 6 turns: criticism / control | 76.3% / 93.9% | 14.7% / 68.6% |
| Same, real analysis | 55.6% / 95.2% | 14.3% / 59.5% |
| Discrimination between true and false MATH claims after criticism (control) | 0.67 (0.72) | 0.11 (0.46) |

- **Criticism dominates.** It lowers accuracy on claims the model got right in every setting, more so on harder problems.
- **Praise moves Qwen little but sways Llama.**
- **Harsher criticism moves both models more only when it also says the claim is false.** Disagreement of graded harshness without a claim about the answer does not.
- **The models give in differently.** When Qwen abandons a correct MATH answer, its working still reaches that answer in 73% of cases. Llama's probes and stated confidence move with its verdict.

## Layout

```
data/
  math/             300 MATH-500 answer claims (levels 1-5), each true claim with a false twin
  real_analysis/    150 real-analysis claims (75 true + 75 false twins); twins.json gives why each twin is false
src/
  run_experiment.py one model, one track: multi-turn conversations, verdicts, probes
  data/             build the MATH claims and the 13 prompts per claim
  tone/             tone severity of every feedback wording (NRC-VAD, politeness, GoEmotions)
  analysis/         pushback, tone effects, derived answers, stance-free tracks, report numbers, plot
scripts/            run locally or on a Slurm cluster, fetch and verify weights, tone setup, W&B upload
results/
  qwen_math/ qwen_real_analysis/ llama_math/ llama_real_analysis/
                    one folder per track (control, negative = criticism, positive = praise,
                    dispute / affirm / filler = stance-free), each with turns.jsonl;
                    pushback/, derived/, tone/ hold the analysis outputs
  llama_math_no_fallback/  first Llama MATH run, before the verdict fallback (superseded)
  tone/             severity scores of the 427 feedback wordings, plus blank Best-Worst Scaling sheets
report/             LaTeX sources, generated tables and numbers, build.sh, report.pdf
versions/           snapshots of the three earlier stages (scorer, pilot, interim) and how the code evolved
```

## Setup

```bash
pip install -r requirements.txt
```

Experiments need one GPU with about 11 GB (models are loaded in 4-bit). Analysis and the report run on CPU; the report needs [tectonic](https://tectonic-typesetting.github.io/).

## Reproducing the report

Every table, figure and number in the report is regenerated from the raw `turns.jsonl` files:

```bash
bash report/build.sh
```

## Running the experiments

Each track is one run of `src/run_experiment.py`. For example, Qwen on MATH, criticism track, with the settings used in the report:

```bash
python3 src/run_experiment.py --model_name Qwen/Qwen2.5-7B-Instruct \
    --claims_json_path data/math/claims.json --prompts_path data/math/prompts.jsonl \
    --output_dir results/run_qwen_math/negative --track negative \
    --prompt_style reasoning_first --max_new_tokens 768 --max_turns 6 \
    --confidence_floor 50.0 --load_in_4bit
```

Llama used `--max_new_tokens 1024 --force_verdict`. That option trims repetition loops and completes a missing verdict line. Tracks are `control`, `negative`, `positive`, `dispute`, `affirm` and `filler`.

- **Several tracks in one go:** `scripts/run_local.sh` on any GPU machine, or `scripts/run_ada.sbatch` on a Slurm cluster. Both write to `results/run_<TAG>/`.
- **Model weights:** `scripts/get_model.sh` downloads them and `scripts/verify_model.py` checks they match the Hub. `scripts/llama_tokenizer.py` rebuilds Meta's exact Llama tokenizer config from ungated copies.

Rebuild the data:

```bash
python3 src/data/build_math.py --math500 path/to/MATH-500/test.jsonl
python3 src/data/build_prompts.py --claims_path data/math/claims.json --output_path data/math/prompts.jsonl --subject mathematics
python3 src/data/build_prompts.py   # real analysis
```

Tone severity (CPU, separate environment):

```bash
bash scripts/setup_tone.sh
python3 src/tone/wrappers.py --prompts data/math/prompts.jsonl data/real_analysis/prompts.jsonl --out results/tone/wrappers.csv
.venv_tone/bin/python src/tone/score.py --wrappers results/tone/wrappers.csv --out results/tone/wrapper_scores.csv
python3 src/tone/composite.py --scores results/tone/wrapper_scores.csv --prompt_wrappers results/tone/prompt_wrappers.csv --out_dir results/tone
```

Upload the runs to Weights & Biases (after `wandb login` and `bash report/build.sh`):

```bash
python3 scripts/upload_wandb.py
```

## Models and data

| What | Link |
|---|---|
| Qwen2.5-7B-Instruct | <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct> |
| Llama-3.1-8B-Instruct | <https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct> |
| Ungated Llama copies used to rebuild it | <https://huggingface.co/NousResearch/Meta-Llama-3.1-8B-Instruct>, <https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct> |
| MATH-500 | <https://huggingface.co/datasets/HuggingFaceH4/MATH-500> |
| GoEmotions classifier | <https://huggingface.co/SamLowe/roberta-base-go_emotions> |
| NRC-VAD lexicon | <https://saifmohammad.com/WebPages/nrc-vad.html> (research use; not redistributed) |
| Stanford Politeness Corpus | via [ConvoKit](https://convokit.cornell.edu/) |

The real-analysis claims are drawn from S. K. Mapa, *Introduction to Real Analysis*.

## W&B runs

| Run | Link |
|---|---|
| Qwen, MATH | [qwen_math](https://wandb.ai/parthniteshchaudhari-international-institute-of-informat/confidence-matters-anlp/runs/85dqtcpt) |
| Qwen, real analysis | [qwen_real_analysis](https://wandb.ai/parthniteshchaudhari-international-institute-of-informat/confidence-matters-anlp/runs/39ccn69r) |
| Llama, MATH | [llama_math](https://wandb.ai/parthniteshchaudhari-international-institute-of-informat/confidence-matters-anlp/runs/04prhtnw) |
| Llama, real analysis | [llama_real_analysis](https://wandb.ai/parthniteshchaudhari-international-institute-of-informat/confidence-matters-anlp/runs/wtibwzrd) |
| Llama, MATH, no fallback (superseded) | [llama_math_no_fallback](https://wandb.ai/parthniteshchaudhari-international-institute-of-informat/confidence-matters-anlp/runs/8hmj2p15) |
| Tone severity | [tone_severity](https://wandb.ai/parthniteshchaudhari-international-institute-of-informat/confidence-matters-anlp/runs/2aonbehx) |

Each run holds its configuration, per-turn accuracy curves, the pushback summary and the raw per-turn logs as an artifact.

## Acknowledgements

This project couldn't have been possible without the support of Dhiraj Chitluri, Gollapalli Phanindra and Parth Dhodapkar.
