#!/usr/bin/env bash
# Full O1 (v4: six tracks, fixed horizon) + O2 (true claims) + O3/O6 (recovery) run for
# one model on Ada, sequentially on one GPU (the 'low' QoS allows one GPU per user).
# Each stage writes into its own folder and is skipped if its results already exist,
# so a resubmitted job resumes after the last finished stage.
#
#   MODEL=Qwen/Qwen2.5-7B-Instruct TAG=qwen25_7b sbatch scripts/ada_full_run.sbatch
#   MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct TAG=llama31_8b sbatch scripts/ada_full_run.sbatch
#     (Llama is gated: run `huggingface-cli login` on Ada first)
#
# Optional overrides: TRACKS (O1 tracks to run), DATA (folder with claims_paired.json and
# claims_prompts.jsonl), RUN_O2 / RUN_O3 (1 or 0). The MATH companion set, criticism only:
#   MODEL=Qwen/Qwen2.5-7B-Instruct TAG=qwen25_7b_math DATA=data/claims_math \
#     TRACKS="control negative" RUN_O2=0 RUN_O3=0 sbatch scripts/ada_full_run.sbatch
#SBATCH --job-name=anlp_full
#SBATCH --partition=u22
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-12:00:00
#SBATCH --output=logs/full_%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/anlp_project}"
PY="${PY:-$HOME/anlp_venv/bin/python}"
MODEL="${MODEL:?set MODEL}"
TAG="${TAG:?set TAG}"
OUT="results/full_${TAG}"
TRACKS="${TRACKS:-control positive negative filler affirm dispute}"
DATA="${DATA:-data/claims}"
RUN_O2="${RUN_O2:-1}"
RUN_O3="${RUN_O3:-1}"
# Model weights go to node-local /ssd_scratch (home quota is too small for 8B checkpoints)
export HF_HOME="${HF_HOME:-/ssd_scratch/$USER/hf}"
export PYTHONUNBUFFERED=1
mkdir -p logs "$OUT" "$HF_HOME"
# HF_HOME points at scratch, so read the login token from the usual home location
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
    export HF_TOKEN
fi
if [[ "$MODEL" == meta-llama/* ]] && [ -z "${HF_TOKEN:-}" ]; then
    echo "!!! $MODEL is gated and no Hugging Face token was found in ~/.cache/huggingface/token."
    echo "!!! Log in on Ada, then resubmit:"
    echo "!!!   srun -p u22 -c1 --mem=2G -t 10 --pty ~/anlp_venv/bin/huggingface-cli login"
    exit 1
fi
nvidia-smi -L
echo "Model $MODEL, data $DATA, O1 tracks [$TRACKS], O2=$RUN_O2, O3=$RUN_O3 -> $OUT (HF_HOME=$HF_HOME)"
# Download the weights first, with a stall timeout and retries (a hung download once
# stalled a job for 30 minutes); huggingface_hub resumes partial files.
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
for ATTEMPT in 1 2 3 4 5; do
    timeout 3h "$(dirname "$PY")/huggingface-cli" download "$MODEL" --exclude "*.pth" "original/*" > /dev/null && break
    echo "!!! download attempt $ATTEMPT failed or stalled; retrying"
    sleep 30
done

done_marker() { [ -f "$1/.done" ]; }

# O1 v4: one process per track, sequentially; every item gets 6 feedback turns
for TRACK in $TRACKS; do
    D="$OUT/o1/$TRACK"
    if done_marker "$D"; then echo ">>> O1 $TRACK already done"; continue; fi
    echo ">>> O1 $TRACK ($(date))"
    $PY src/evaluation/run_claims_experiment.py --model_name "$MODEL" --output_dir "$D" \
        --claims_json_path "$DATA/claims_paired.json" --prompts_path "$DATA/claims_prompts.jsonl" \
        --track "$TRACK" --max_new_tokens 128 --confidence_floor 50.0 --load_in_4bit
    touch "$D/.done"
done
$PY src/evaluation/analyze_o1_paired.py --results_dir "$OUT/o1"
if [[ " $TRACKS " == *" control "* && " $TRACKS " == *" negative "* ]]; then
    $PY src/evaluation/analyze_pushback.py --results_dir "$OUT/o1" --max_turn 6 --tag Push \
        --out_dir "$OUT/o1/pushback_t6" | tee "$OUT/o1/pushback_t6.txt"
fi
if [[ " $TRACKS " == *" positive "* ]]; then
    $PY src/evaluation/analyze_o1_robustness.py --results_dir "$OUT/o1" --turn 6 | tee "$OUT/o1/robustness_t6.txt"
fi

# O2: true claims, 8 criticism turns, no early stopping
D="$OUT/o2"
if [ "$RUN_O2" = 1 ] && ! done_marker "$D"; then
    echo ">>> O2 ($(date))"
    $PY src/evaluation/run_reversibility_experiment.py --model_name "$MODEL" --output_dir "$D" \
        --claims_json_path "$DATA/claims_paired.json" --prompts_path "$DATA/claims_prompts.jsonl" \
        --truth_filter Valid --max_turns 8 \
        --max_new_tokens 128 --order shuffled --load_in_4bit --no_plot
    $PY src/evaluation/analyze_reversibility.py --summary_csv "$D/reversibility_summary.csv" \
        --claim_summary_csv "$D/reversibility_claim_summary.csv" --output_dir "$D/figures"
    touch "$D/.done"
fi

# O3 + O6: k = 1,2,4,6 and the first-flip turn, B = 3 recovery turns per condition
D="$OUT/o3"
if [ "$RUN_O3" = 1 ] && ! done_marker "$D"; then
    echo ">>> O3/O6 ($(date))"
    $PY src/evaluation/run_recovery_experiment.py --model_name "$MODEL" --output_dir "$D" \
        --claims_json_path "$DATA/claims_paired.json" --prompts_path "$DATA/claims_prompts.jsonl" \
        --branch_k 1,2,4,6 --budget 3 --max_new_tokens 128 --load_in_4bit
    $PY src/evaluation/analyze_recovery.py --results "$D/recovery_results.jsonl"
    touch "$D/.done"
fi

echo ">>> FULL RUN FINISHED: $OUT ($(date))"
