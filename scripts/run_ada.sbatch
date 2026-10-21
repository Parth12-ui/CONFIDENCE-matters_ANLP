#!/usr/bin/env bash
# O1 tracks for one model and one dataset on Ada, one GPU (the 'low' QoS allows one per
# user), then the pushback analysis. Each track is skipped if it already finished, so a
# resubmitted job resumes. Results go to <repo>/results/run_${TAG}/.
#
# Submit from new_code/:
#   MODEL=Qwen/Qwen2.5-7B-Instruct TAG=qwen25_7b_math DATA=data/claims_math sbatch scripts/ada_run.sbatch
#   MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct TAG=llama31_8b sbatch scripts/ada_run.sbatch
#     (Llama is gated: save a Hugging Face token in ~/.cache/huggingface/token on Ada first)
# Weights: compute nodes download very slowly, so fetch them on the login node first with
# scripts/fetch_model.sh, either into a home directory (pass that folder as MODEL) or into
# /share1, which compute nodes cannot see: then set MODEL_ON_LOGIN=/share1/$USER/models/NAME
# and the job copies it to the node's /ssd_scratch over the internal network.
# /share1 has a 25 GB per-user quota (27 GB hard), so it holds one 7-8B model at a time.
# Optional: TRACKS (default "control negative"; also positive, filler, affirm, dispute),
#           BY (column for a per-group breakdown, e.g. difficulty for the MATH levels)
#SBATCH --job-name=anlp_o1
#SBATCH --partition=u22
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-12:00:00
#SBATCH --output=logs/o1_%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/anlp_project/new_code}"
PY="${PY:-$HOME/anlp_venv/bin/python}"
MODEL_ON_LOGIN="${MODEL_ON_LOGIN:-}"
if [ -n "$MODEL_ON_LOGIN" ]; then
    MODEL="/ssd_scratch/$USER/models/$(basename "$MODEL_ON_LOGIN")"
fi
MODEL="${MODEL:?set MODEL or MODEL_ON_LOGIN}"
TAG="${TAG:?set TAG}"
DATA="${DATA:-data/claims}"
TRACKS="${TRACKS:-control negative}"
BY="${BY:-}"
OUT="../results/run_${TAG}"
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
    exit 1
fi
nvidia-smi -L
if [ -n "$MODEL_ON_LOGIN" ]; then
    echo ">>> copying $MODEL_ON_LOGIN from the login node to $MODEL ($(date))"
    mkdir -p "$MODEL"
    rsync -a --partial "ada-gw1:$MODEL_ON_LOGIN/" "$MODEL/"
    echo ">>> copied ($(date)): $(du -sh "$MODEL" | cut -f1)"
fi
echo "Model $MODEL, data $DATA, tracks [$TRACKS] -> $OUT"
# Download the weights first, with a stall timeout and retries (a hung download once
# stalled a job for 30 minutes); huggingface_hub resumes partial files.
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
# MODEL may also be a local folder (compute-node downloads can be very slow; fetch the
# weights on the login node into a home directory instead).
if [ ! -d "$MODEL" ]; then
    for ATTEMPT in 1 2 3 4 5; do
        timeout 3h "$(dirname "$PY")/huggingface-cli" download "$MODEL" --exclude "*.pth" "original/*" > /dev/null && break
        echo "!!! download attempt $ATTEMPT failed or stalled; retrying"
        sleep 30
    done
fi

for TRACK in $TRACKS; do
    D="$OUT/$TRACK"
    if [ -f "$D/.done" ]; then echo ">>> $TRACK already done"; continue; fi
    echo ">>> $TRACK ($(date))"
    $PY src/evaluation/run_claims_experiment.py --model_name "$MODEL" --output_dir "$D" \
        --claims_json_path "$DATA/claims_paired.json" --prompts_path "$DATA/claims_prompts.jsonl" \
        --track "$TRACK" --max_new_tokens 128 --confidence_floor 50.0 --load_in_4bit
    touch "$D/.done"
done

if [[ " $TRACKS " == *" control "* && " $TRACKS " == *" negative "* ]]; then
    $PY src/evaluation/analyze_pushback.py --results_dir "$OUT" --max_turn 6 --tag Push \
        --out_dir "$OUT/pushback_t6" ${BY:+--by "$BY"} | tee "$OUT/pushback_t6.txt"
fi
echo ">>> FINISHED: $OUT ($(date))"
