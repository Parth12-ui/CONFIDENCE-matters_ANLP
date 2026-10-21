#!/usr/bin/env bash
# O1 tracks on any single-GPU machine (e.g. a Jarvislabs instance), one process per track
# in parallel, then the pushback analysis. Results go to <repo>/results/run_${TAG}/.
#
# Run from new_code/:
#   bash scripts/run_o1.sh MODEL TAG [DATA] [QUANT]
#     DATA   data/claims (default) or data/claims_math
#     QUANT  bf16 (default; ~16 GB per 7-8B process, fits three on a 48-80 GB card) or 4bit
#   TRACKS="control negative" (default), BY=difficulty for a per-level breakdown
# Example: bash scripts/run_o1.sh Qwen/Qwen2.5-32B-Instruct qwen25_32b_math data/claims_math 4bit
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
MODEL="${1:?MODEL}"
TAG="${2:?TAG}"
DATA="${3:-data/claims}"
QUANT="${4:-bf16}"
TRACKS="${TRACKS:-control negative}"
BY="${BY:-}"
OUT="../results/run_${TAG}"
QUANT_FLAG=""
[ "$QUANT" = "4bit" ] && QUANT_FLAG="--load_in_4bit"
mkdir -p "$OUT"

PIDS=()
for TRACK in $TRACKS; do
    D="$OUT/$TRACK"
    [ -f "$D/.done" ] && { echo ">>> $TRACK already done"; continue; }
    mkdir -p "$D"
    echo ">>> $TRACK -> $D (log: $D/run.log)"
    ( python3 src/evaluation/run_claims_experiment.py --model_name "$MODEL" --output_dir "$D" \
        --claims_json_path "$DATA/claims_paired.json" --prompts_path "$DATA/claims_prompts.jsonl" \
        --track "$TRACK" --max_new_tokens 128 --confidence_floor 50.0 $QUANT_FLAG \
        > "$D/run.log" 2>&1 && touch "$D/.done" ) &
    PIDS+=($!)
done
FAILED=0
for PID in "${PIDS[@]}"; do wait "$PID" || FAILED=1; done
[ "$FAILED" -ne 0 ] && echo "!!! A track failed; see $OUT/*/run.log"

python3 src/evaluation/analyze_pushback.py --results_dir "$OUT" --max_turn 6 --tag Push \
    --out_dir "$OUT/pushback_t6" ${BY:+--by "$BY"} | tee "$OUT/pushback_t6.txt"
echo ">>> FINISHED: $OUT"
