#!/usr/bin/env bash
# ==============================================================================
# O1 Paired Experiment Launcher for JarvisLabs
#
# Runs the positive, negative and neutral re-ask control tracks as three
# concurrent processes on the same GPU (batch size 1 leaves the GPU underused,
# so this gives a near-linear speedup), with stopping rules disabled so every
# claim gets the same 6 follow-up turns. Then runs the paired analysis.
#
# Usage:
#   bash scripts/run_jarvislabs_experiment.sh [MODEL] [OUTPUT_DIR] [LIMIT_PAIRS] [QUANT]
#     MODEL       default meta-llama/Meta-Llama-3.1-8B-Instruct
#     OUTPUT_DIR  default results/o1_paired_v3
#     LIMIT_PAIRS number of claim pairs (original + false twin); empty = all 75
#     QUANT       4bit (default; ~6 GB per process, fits 3 on a 24 GB card)
#                 or bf16 (~16 GB per process; use on 48-80 GB cards)
#
# Environment overrides:
#   TRACKS          tracks to run as parallel processes (default: all six). affirm /
#                   dispute are stance-free praise / criticism of the model's own
#                   verdict, filler is their length-matched neutral control. Six 4-bit
#                   processes need ~36 GB; on smaller cards run TRACKS in two batches.
#   PROMPT_STYLE    verdict_first (default, as in v3) or reasoning_first
#   MAX_NEW_TOKENS  default 128 (verdict_first) / 384 (reasoning_first)
#
# Smoke test (about 15-20 min):
#   bash scripts/run_jarvislabs_experiment.sh meta-llama/Meta-Llama-3.1-8B-Instruct results/smoke_v3 2
# ==============================================================================

set -e

MODEL_NAME="${1:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
OUTPUT_DIR="${2:-../../results/v3_interim/o1_paired_v4}"
LIMIT_PAIRS="${3:-}"
QUANT="${4:-4bit}"
TRACKS="${TRACKS:-positive negative control affirm dispute filler}"
PROMPT_STYLE="${PROMPT_STYLE:-verdict_first}"
if [ "$PROMPT_STYLE" = "reasoning_first" ]; then
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-384}"
else
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."  # repo root

LIMIT_FLAG=""
if [ -n "$LIMIT_PAIRS" ]; then
    LIMIT_FLAG="--limit_claims $LIMIT_PAIRS"
fi
QUANT_FLAG=""
if [ "$QUANT" = "4bit" ]; then
    QUANT_FLAG="--load_in_4bit"
fi

echo "=============================================================================="
echo " O1 PAIRED EXPERIMENT ON JARVISLABS"
echo " Model:       $MODEL_NAME"
echo " Output dir:  $OUTPUT_DIR"
echo " Claim pairs: ${LIMIT_PAIRS:-all 75 (150 items)}"
echo " Precision:   $QUANT"
echo "=============================================================================="

python3 -m pip install -q accelerate bitsandbytes

mkdir -p "$OUTPUT_DIR"
PIDS=()
for TRACK in $TRACKS; do
    TRACK_DIR="$OUTPUT_DIR/$TRACK"
    mkdir -p "$TRACK_DIR"
    echo ">>> Launching $TRACK track -> $TRACK_DIR (log: $TRACK_DIR/run.log)"
    nohup python3 src/evaluation/run_claims_experiment.py \
        --model_name "$MODEL_NAME" \
        --prompts_path "data/claims/claims_prompts.jsonl" \
        --claims_json_path "data/claims/claims_paired.json" \
        --output_dir "$TRACK_DIR" \
        --track "$TRACK" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --prompt_style "$PROMPT_STYLE" \
        --confidence_floor 50.0 \
        $QUANT_FLAG \
        $LIMIT_FLAG > "$TRACK_DIR/run.log" 2>&1 &
    PIDS+=($!)
done

echo ">>> Waiting for all tracks (progress: tail -f $OUTPUT_DIR/*/run.log)"
FAILED=0
for PID in "${PIDS[@]}"; do
    wait "$PID" || FAILED=1
done
if [ "$FAILED" -ne 0 ]; then
    echo "!!! At least one track failed. Check $OUTPUT_DIR/*/run.log. Partial JSONL results are still usable."
fi

echo ">>> Running paired analysis..."
python3 src/evaluation/analyze_o1_paired.py --results_dir "$OUTPUT_DIR"
python3 src/evaluation/analyze_o1_robustness.py --results_dir "$OUTPUT_DIR" --turn 6 | tee "$OUTPUT_DIR/robustness_t6.txt"

ARCHIVE_NAME="jarvislabs_results_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czf "$ARCHIVE_NAME" "$OUTPUT_DIR"

echo "=============================================================================="
echo " DONE. Archive: $ARCHIVE_NAME"
echo " Per-track logs: $OUTPUT_DIR/{positive,negative,control}/run.log"
echo " Analysis:       $OUTPUT_DIR/analysis/"
echo "=============================================================================="
