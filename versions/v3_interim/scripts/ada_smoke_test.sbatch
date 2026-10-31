#!/usr/bin/env bash
# Smoke test of the O1 / O2 / O3 runners and analyses on Ada with a tiny open chat
# model (no gated weights needed). Checks code paths, not results.
#   sbatch scripts/ada_smoke_test.sbatch
#SBATCH --job-name=anlp_smoke
#SBATCH --partition=u22
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=logs/smoke_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/anlp_project}"
PY="${PY:-$HOME/anlp_venv/bin/python}"
MODEL="${MODEL:-HuggingFaceTB/SmolLM2-135M-Instruct}"
OUT="${OUT:-results/smoke_$(date +%Y%m%d_%H%M%S)}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/ssd_scratch/$USER/hf}"
mkdir -p logs "$OUT" "$HF_HOME"
nvidia-smi -L

echo ">>> O1: all six tracks, 1 claim pair"
$PY src/evaluation/run_claims_experiment.py --model_name "$MODEL" --output_dir "$OUT/o1" \
    --claims_json_path data/claims/claims_paired.json --track full --limit_claims 1 --max_new_tokens 32
echo ">>> O1: reasoning-first prompt style"
$PY src/evaluation/run_claims_experiment.py --model_name "$MODEL" --output_dir "$OUT/o1_reasoning" \
    --claims_json_path data/claims/claims_paired.json --track affirm --limit_claims 1 \
    --prompt_style reasoning_first --max_new_tokens 64
$PY src/evaluation/analyze_o1_paired.py --results_dir "$OUT/o1"
$PY src/evaluation/analyze_o1_robustness.py --results_dir "$OUT/o1" --turn 6

echo ">>> O2: 2 true claims, 3 turns"
$PY src/evaluation/run_reversibility_experiment.py --model_name "$MODEL" --output_dir "$OUT/o2" \
    --limit_claims 2 --max_turns 3 --no_plot --load_in_4bit
$PY src/evaluation/analyze_reversibility.py --summary_csv "$OUT/o2/reversibility_summary.csv" \
    --claim_summary_csv "$OUT/o2/reversibility_claim_summary.csv" --output_dir "$OUT/o2/figures"

echo ">>> O3: 3 claims, k = 1,2, B = 1"
$PY src/evaluation/run_recovery_experiment.py --model_name "$MODEL" --output_dir "$OUT/o3" \
    --limit_claims 3 --branch_k 1,2 --budget 1 --only_initially_correct false --max_new_tokens 32 --load_in_4bit
$PY src/evaluation/analyze_recovery.py --results "$OUT/o3/recovery_results.jsonl"

echo ">>> Unit tests"
$PY -m pytest -q tests

echo ">>> SMOKE TEST PASSED: $OUT"
