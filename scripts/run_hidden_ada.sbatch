#!/usr/bin/env bash
# Hidden-state truth probes for one model's MATH runs on Ada: rebuild the logged control and
# criticism conversations, read their hidden states in one forward pass each (no generation;
# about 1 GPU-hour), then fit the probes on CPU. Same GPU type and 4-bit settings as the
# experiment runs.
#
# Submit from the repository root (one job per model; the 'low' QoS allows one GPU per user):
#   MODEL=Qwen/Qwen2.5-7B-Instruct RUN=qwen_math TAG=QwenMath sbatch scripts/run_hidden_ada.sbatch
#   MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct RUN=llama_math TAG=LlamaMath sbatch scripts/run_hidden_ada.sbatch
# MODEL / MODEL_ON_LOGIN and the Hugging Face token work as in scripts/run_ada.sbatch.
# Optional: LIMIT (first N conversations per track, for a smoke test),
#           KEEP_HIDDEN=1 (copy the state files, about 1 GB per track, back to results/RUN/hidden).
#SBATCH --job-name=anlp_hidden
#SBATCH --partition=u22
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/hidden_%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/CONFIDENCE-matters_ANLP}"
PY="${PY:-$HOME/anlp_venv/bin/python}"
MODEL_ON_LOGIN="${MODEL_ON_LOGIN:-}"
if [ -n "$MODEL_ON_LOGIN" ]; then
    MODEL="/ssd_scratch/$USER/models/$(basename "$MODEL_ON_LOGIN")"
fi
MODEL="${MODEL:?set MODEL or MODEL_ON_LOGIN}"
RUN="${RUN:?set RUN (qwen_math or llama_math)}"
TAG="${TAG:?set TAG (QwenMath or LlamaMath)}"
LIMIT="${LIMIT:-}"
HIDDEN="/ssd_scratch/$USER/hidden/$RUN"
export HF_HOME="${HF_HOME:-/ssd_scratch/$USER/hf}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p logs "$HIDDEN" "$HF_HOME"
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
    export HF_TOKEN
fi
if [[ "$MODEL" == meta-llama/* ]] && [ -z "${HF_TOKEN:-}" ]; then
    echo "!!! $MODEL is gated and no Hugging Face token was found in ~/.cache/huggingface/token."
    exit 1
fi
"$PY" -c "import sklearn" 2>/dev/null || { echo "!!! scikit-learn missing: $PY -m pip install scikit-learn"; exit 1; }
nvidia-smi -L
if [ -n "$MODEL_ON_LOGIN" ]; then
    echo ">>> copying $MODEL_ON_LOGIN from the login node to $MODEL ($(date))"
    mkdir -p "$MODEL"
    rsync -a --partial "ada-gw1:$MODEL_ON_LOGIN/" "$MODEL/"
fi

echo ">>> hidden states: $MODEL on results/$RUN -> $HIDDEN ($(date))"
"$PY" src/extract_hidden.py --model_name "$MODEL" --results_dir "results/$RUN" \
    --tracks control negative --out_dir "$HIDDEN" --load_in_4bit ${LIMIT:+--limit "$LIMIT"}

if [ -n "$LIMIT" ]; then
    echo ">>> smoke test done: $(ls "$HIDDEN"). Resubmit without LIMIT for the probes."
    exit 0
fi
echo ">>> probes ($(date))"
"$PY" src/analysis/hidden_probe.py --results_dir "results/$RUN" --hidden_dir "$HIDDEN" \
    --out_dir "results/$RUN/hidden_probe" --tag "$TAG" --tex_out "results/$RUN/hidden_probe/hidden_$TAG.tex"
if [[ "$RUN" == llama* ]]; then
    "$PY" src/analysis/hidden_probe.py --results_dir "results/$RUN" --hidden_dir "$HIDDEN" --drop_forced \
        --out_dir "results/$RUN/hidden_probe_no_forced" --tag "${TAG}NF" \
        --tex_out "results/$RUN/hidden_probe_no_forced/hidden_${TAG}NF.tex"
fi
if [ -n "${KEEP_HIDDEN:-}" ]; then
    mkdir -p "results/$RUN/hidden" && cp "$HIDDEN"/* "results/$RUN/hidden/"
fi
echo ">>> FINISHED ($(date)): results/$RUN/hidden_probe"
