# v2: O1 pilot (interim report v2)

Authors: Dhiraj Chitluri, Parth Dhodapkar, Parth Nitesh Chaudhari.

The original O1 track runners (`track_runner.py`, `run_positive_track.py`,
`run_negative_track.py`, `analyze_results.py`) and `salvage_o1_v2.py`, which cleans the pilot run
for reporting. The claims data they used is in `../v3_interim/data/claims`. Results are in `results/`:
`o1_full_run_v2/` and `o1_v2_salvaged/` (raw and cleaned pilot), `o2_full_run_v2/` (first
reversibility run; pools false claims and stops early, do not report), `archive/` (earlier partial
runs). Raw per-turn logs over 500 KB are not included.
