# v3: interim study (interim report v3)

Authors: Dhiraj Chitluri, Gollapalli Phanindra, Parth Dhodapkar, Parth Nitesh Chaudhari.

- `data/claims/`: 75 real-analysis claims, 75 false twins, built by `src/data_pipeline/`;
  `claims_prompts.jsonl` is not included (over 500 KB) and is rebuilt by `build_claims_prompts.py`
- `src/evaluation/`: O1 runner (`run_claims_experiment.py`) and analyses (paired, robustness,
  report tables), O2 reversibility, O3/O6 recovery; `track_runner.py` is a copy from v2 that the
  O2 runner imports
- `scripts/`: JarvisLabs launchers and the first Ada scripts (written for the pre-split layout)
- `results/o1_paired_v3_llama31_8b/`: O1 on Llama-3.1-8B with early stopping, analysed at turn 2
  (`analysis_t2/`) and turn 6 (`analysis_t6/`); `pushback_t2/`, `pushback_t6/`; raw logs not included
- `tests/`: O2 tests (`python -m pytest tests` from this folder)

`README_pre_split.md` is the repository README as it was on 21 Oct 2026. The final study
continues at the repository root.
