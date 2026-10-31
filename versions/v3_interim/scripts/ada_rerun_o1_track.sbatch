#!/usr/bin/env bash
# Rerun one O1 track of a finished full run (e.g. after a prompt-bank fix), keep the old
# results under superseded/, and redo the O1 analyses over the updated track set.
#   MODEL=Qwen/Qwen2.5-7B-Instruct TAG=qwen25_7b TRACK=positive sbatch scripts/ada_rerun_o1_track.sbatch
#SBATCH --job-name=anlp_rerun_track
#SBATCH --partition=u22
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/rerun_%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/anlp_project}"
PY="${PY:-$HOME/anlp_venv/bin/python}"
MODEL="${MODEL:?set MODEL}"
TAG="${TAG:?set TAG}"
TRACK="${TRACK:?set TRACK}"
OUT="results/full_${TAG}"
export HF_HOME="${HF_HOME:-/ssd_scratch/$USER/hf}"
export PYTHONUNBUFFERED=1
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
    export HF_TOKEN
fi
mkdir -p logs "$HF_HOME" "$OUT/superseded"
nvidia-smi -L

D="$OUT/o1/$TRACK"
if [ -d "$D" ]; then
    # Outside o1/ so the analyses, which glob o1/**, do not pick it up
    mv "$D" "$OUT/superseded/${TRACK}_$(date +%Y%m%d_%H%M%S)"
fi
echo ">>> O1 $TRACK rerun ($(date))"
$PY src/evaluation/run_claims_experiment.py --model_name "$MODEL" --output_dir "$D" \
    --claims_json_path data/claims/claims_paired.json --track "$TRACK" \
    --max_new_tokens 128 --confidence_floor 50.0 --load_in_4bit
touch "$D/.done"

$PY src/evaluation/analyze_o1_paired.py --results_dir "$OUT/o1"
$PY src/evaluation/analyze_o1_robustness.py --results_dir "$OUT/o1" --turn 6 | tee "$OUT/o1/robustness_t6.txt"
echo ">>> RERUN FINISHED: $D ($(date))"
