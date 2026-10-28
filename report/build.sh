#!/usr/bin/env bash
# Rebuild the report from the raw runs in results/: every analysis, table, figure and
# number macro, then the PDF (tectonic, official ACL style).
#   bash report/build.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RES=results; AN=src/analysis; GEN=report/generated
mkdir -p "$GEN"

# Pushback analysis and tone effects per run (MATH runs also by difficulty level)
for run in qwen_math qwen_real_analysis llama_math llama_real_analysis; do
    tag=$(echo "$run" | sed 's/qwen/Qwen/;s/llama/Llama/;s/_math/Math/;s/_real_analysis/RA/')
    BY=""; [[ "$run" == *math ]] && BY="--by difficulty"
    python3 $AN/pushback.py --results_dir $RES/$run --max_turn 6 --tag "$tag" \
        --out_dir $RES/$run/pushback --tex_out $GEN/pushback_$tag.tex $BY > /dev/null
    python3 $AN/tone_effects.py --results_dir $RES/$run --severity $RES/tone \
        --tag "$tag" --out_dir $RES/$run/tone > /dev/null
done
# Llama with forced verdicts treated as missing (robustness)
for run in llama_math llama_real_analysis; do
    tag=$(echo "$run" | sed 's/llama/Llama/;s/_math/Math/;s/_real_analysis/RA/')NF
    python3 $AN/pushback.py --results_dir $RES/$run --max_turn 6 --tag "$tag" --drop_forced \
        --out_dir $RES/$run/pushback_no_forced --tex_out $GEN/pushback_$tag.tex > /dev/null
done
# Answer the model's own working reaches (MATH)
for run in qwen_math llama_math; do
    python3 $AN/derived_answer.py --results_dir $RES/$run --claims data/math/claims.json \
        --out_dir $RES/$run/derived > /dev/null
done
python3 $AN/report_numbers.py --tone $RES/tone --out $GEN/final_numbers.tex --levels_out $GEN/tab_levels.tex \
    --levels Qwen $RES/qwen_math/pushback/by_difficulty.csv \
    --levels Llama $RES/llama_math/pushback/by_difficulty.csv \
    --run QwenMath $RES/qwen_math --run QwenRA $RES/qwen_real_analysis \
    --run LlamaMath $RES/llama_math --run LlamaRA $RES/llama_real_analysis \
    --run LlamaMathOld $RES/llama_math_no_fallback
# Stance-free tracks (MATH): tone without a claim about the answer
python3 $AN/stance_free.py --run QwenMath $RES/qwen_math --run LlamaMath $RES/llama_math \
    --tex_out $GEN/stance_free.tex --table_out $GEN/tab_stancefree.tex > /dev/null
python3 $AN/plot.py --out $GEN/fig_pushback.pdf \
    --panel "Qwen2.5-7B, MATH" $RES/qwen_math/pushback/accuracy_by_turn.csv \
    --panel "Qwen2.5-7B, real analysis" $RES/qwen_real_analysis/pushback/accuracy_by_turn.csv \
    --panel "Llama-3.1-8B, MATH" $RES/llama_math/pushback/accuracy_by_turn.csv \
    --panel "Llama-3.1-8B, real analysis" $RES/llama_real_analysis/pushback/accuracy_by_turn.csv
# Hidden-state probes: fitted on Ada from the saved states (scripts/run_hidden_ada.sbatch)
cp $RES/qwen_math/hidden_probe/hidden_QwenMath.tex $RES/llama_math/hidden_probe/hidden_LlamaMath.tex \
    $RES/llama_math/hidden_probe_no_forced/hidden_LlamaMathNF.tex $GEN/

cd report
tectonic -Z search-path=. report.tex
echo "Built report/report.pdf"
