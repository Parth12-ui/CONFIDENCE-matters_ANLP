#!/usr/bin/env bash
# Retrain the DeBERTa-v3-large tonality scorer on the filtered, template-grouped split
# (data/curated_v2) and evaluate it on that split's held-out templates and on the
# out-of-template sanity set.
#   sbatch scripts/ada_train_scorer.sbatch
#SBATCH --job-name=anlp_scorer
#SBATCH --partition=u22
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/scorer_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/anlp_project}"
PY="${PY:-$HOME/anlp_venv/bin/python}"
export HF_HOME="${HF_HOME:-/ssd_scratch/$USER/hf}"
export PYTHONUNBUFFERED=1
CKPT="checkpoints/deberta_large_curated_v2"
mkdir -p logs ../../results/v1_scorer/scorer_v2 "$HF_HOME"
nvidia-smi -L

PYTHON_BIN="$PY" DATA_DIR=data/curated_v2 OUTPUT_DIR="$CKPT" bash scripts/run_train_deberta_large.sh

$PY src/evaluation/evaluate_testset.py --model_dir "$CKPT" --test_file data/curated_v2/test.jsonl \
    --output_json ../../results/v1_scorer/scorer_v2/evaluation_metrics_testset.json --save_predictions ../../results/v1_scorer/scorer_v2/test_predictions.jsonl
SCORER_MODEL_DIR="$CKPT" $PY src/evaluation/sanity_eval.py | tee ../../results/v1_scorer/scorer_v2/sanity_eval.txt
echo ">>> SCORER RETRAIN FINISHED"
