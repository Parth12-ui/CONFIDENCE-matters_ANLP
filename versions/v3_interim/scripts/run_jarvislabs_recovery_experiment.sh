#!/usr/bin/env bash
# ==============================================================================
# O3/O6 recovery experiment: k criticism turns, then B recovery turns under
# tone-only encouragement, the O1 praise prompts, or a length-matched neutral
# control, on true claims the model gets right at turn 0. Then the analysis.
#
# Usage:
#   bash scripts/run_jarvislabs_recovery_experiment.sh [MODEL] [OUTPUT_DIR] [LIMIT_CLAIMS] [QUANT]
#     QUANT  4bit (default, matches O1 v3) or bf16
# Environment overrides: BRANCH_K (default 1,2,4,6), BUDGET (default 3)
# ==============================================================================

set -e

MODEL_NAME="${1:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
OUTPUT_DIR="${2:-../../results/v3_interim/o3_recovery_v1}"
LIMIT_CLAIMS="${3:-}"
QUANT="${4:-4bit}"
BRANCH_K="${BRANCH_K:-1,2,4,6}"
BUDGET="${BUDGET:-3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."  # repo root

LIMIT_FLAG=""
if [ -n "$LIMIT_CLAIMS" ]; then
    LIMIT_FLAG="--limit_claims $LIMIT_CLAIMS"
fi
QUANT_FLAG=""
if [ "$QUANT" = "4bit" ]; then
    QUANT_FLAG="--load_in_4bit"
fi

python3 src/evaluation/run_recovery_experiment.py \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --branch_k "$BRANCH_K" \
    --budget "$BUDGET" \
    --max_new_tokens 128 \
    $QUANT_FLAG \
    $LIMIT_FLAG

python3 src/evaluation/analyze_recovery.py --results "$OUTPUT_DIR/recovery_results.jsonl"
