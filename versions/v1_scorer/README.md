# v1: tonality scorer

DeBERTa-v3-large valence scorer for feedback turns, and its data. Not used by the final report's
results; see `../README.md` for why it was replaced. Authors: Dhiraj Chitluri, Gollapalli
Phanindra, Parth Dhodapkar, Parth Nitesh Chaudhari.

- `src/data_pipeline/`: template and source extraction (`extract.py`), curation
  (`preprocess_tonality.py`), the audit's filtered, template-grouped split (`refine_scorer_data.py`)
- `src/training/`, `src/models/`, `src/engine.py`: training and inference
- `src/evaluation/`: test-set metrics, the out-of-template sanity set, benchmark
- `data/`: `curated_v2/summary.json` (the filtered, template-grouped split) and `sanity_eval.jsonl`
  (draft labels); the raw and curated datasets are not included (over 500 KB)
- `checkpoints/`: model configs and training metadata (no weights or tokenizer files)
- `scripts/`: training launchers, including `ada_train_scorer.sbatch`
- `results/`: test-set metrics on the original split, before the audit
