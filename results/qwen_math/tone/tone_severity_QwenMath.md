# Tone severity vs behaviour: QwenMath

Results: `results/qwen_math`  
Claims with all turns: negative 300, positive 300  
Feedback turns joined to a severity score: 3600 (dropped 0 whose recorded prompt does not match the scored wrapper)  
Turns with a missing verdict at t or t-1: 118 of 3600

## Criticism

Cave rate by category (at-risk turns; feedback wrong / right):

| category | wrong: n | wrong: cave | right: n | right: cave | mean severity_eq |
|---|---|---|---|---|---|
| authority_challenge | 228 | 5.7% | 25 | 40.0% | +0.06 |
| mild_doubt_are_you_sure | 224 | 1.3% | 25 | 16.0% | -1.06 |
| strong_rebuttal | 232 | 25.4% | 35 | 71.4% | +0.91 |

| measure | subset | spec | outcome | n | events | estimate [95% CI] |
|---|---|---|---|---|---|---|
| severity_eq | all | controls | cave (OR/SD) | 769 | 114 | 4.41 [3.20, 6.91] |
| severity_eq | feedback wrong | controls | cave (OR/SD) | 684 | 75 | 6.07 [4.26, 12.91] |
| severity_eq | feedback right | controls | cave (OR/SD) | 85 | 39 | 4.33 [2.80, 95.12] |
| severity_eq | all turns | controls | d_pre (pts/SD) | 1800 |  | 12.34 [10.40, 14.40] |
| severity_eq | all turns | controls | d_self (pts/SD) | 1800 |  | 6.35 [4.72, 8.13] |
| severity_eq | all | +category FE | cave (OR/SD) | 769 | 114 | 1.63 [0.63, 4.21] |
| severity_eq | feedback wrong | +category FE | cave (OR/SD) | 684 | 75 | 2.63 [0.93, 8.16] |
| severity_eq | feedback right | +category FE | cave (OR/SD) | 85 | 39 | 0.81 [0.00, 226.48] |
| severity_eq | all turns | +category FE | d_pre (pts/SD) | 1800 |  | 4.82 [0.27, 9.67] |
| severity_eq | all turns | +category FE | d_self (pts/SD) | 1800 |  | 0.62 [-2.56, 3.66] |
| severity_pc1 | all | controls | cave (OR/SD) | 769 | 114 | 3.31 [2.60, 4.97] |
| severity_pc1 | feedback wrong | controls | cave (OR/SD) | 684 | 75 | 4.26 [3.23, 7.50] |
| severity_pc1 | feedback right | controls | cave (OR/SD) | 85 | 39 | 3.27 [2.28, 37.70] |
| severity_pc1 | all turns | controls | d_pre (pts/SD) | 1800 |  | 9.98 [8.44, 11.47] |
| severity_pc1 | all turns | controls | d_self (pts/SD) | 1800 |  | 5.16 [3.79, 6.56] |
| z_emo_negative | all | controls | cave (OR/SD) | 769 | 114 | 3.27 [2.34, 5.79] |
| z_emo_negative | feedback wrong | controls | cave (OR/SD) | 684 | 75 | 4.70 [2.80, 14.98] |
| z_emo_negative | feedback right | controls | cave (OR/SD) | 85 | 39 | 3.64 [2.60, 22.31] |
| z_emo_negative | all turns | controls | d_pre (pts/SD) | 1800 |  | 10.23 [8.62, 12.00] |
| z_emo_negative | all turns | controls | d_self (pts/SD) | 1800 |  | 4.70 [3.40, 6.04] |
| z_politeness | all | controls | cave (OR/SD) | 769 | 114 | 3.22 [2.48, 4.79] |
| z_politeness | feedback wrong | controls | cave (OR/SD) | 684 | 75 | 3.85 [2.97, 6.15] |
| z_politeness | feedback right | controls | cave (OR/SD) | 85 | 39 | 3.10 [1.82, 56.75] |
| z_politeness | all turns | controls | d_pre (pts/SD) | 1800 |  | 8.10 [6.59, 9.61] |
| z_politeness | all turns | controls | d_self (pts/SD) | 1800 |  | 4.48 [3.20, 6.00] |
| z_vad_arousal | all | controls | cave (OR/SD) | 769 | 114 | 2.01 [1.64, 2.58] |
| z_vad_arousal | feedback wrong | controls | cave (OR/SD) | 684 | 75 | 2.15 [1.81, 2.92] |
| z_vad_arousal | feedback right | controls | cave (OR/SD) | 85 | 39 | 2.49 [1.63, 15.74] |
| z_vad_arousal | all turns | controls | d_pre (pts/SD) | 1800 |  | 8.54 [7.06, 10.16] |
| z_vad_arousal | all turns | controls | d_self (pts/SD) | 1800 |  | 3.67 [2.45, 5.02] |
| z_vad_dominance | all | controls | cave (OR/SD) | 769 | 114 | 0.39 [0.29, 0.49] |
| z_vad_dominance | feedback wrong | controls | cave (OR/SD) | 684 | 75 | 0.33 [0.22, 0.42] |
| z_vad_dominance | feedback right | controls | cave (OR/SD) | 85 | 39 | 0.39 [0.12, 0.58] |
| z_vad_dominance | all turns | controls | d_pre (pts/SD) | 1800 |  | -6.11 [-7.63, -4.72] |
| z_vad_dominance | all turns | controls | d_self (pts/SD) | 1800 |  | -4.61 [-6.02, -3.25] |
| z_vad_valence | all | controls | cave (OR/SD) | 769 | 114 | 2.76 [2.29, 3.63] |
| z_vad_valence | feedback wrong | controls | cave (OR/SD) | 684 | 75 | 3.23 [2.56, 4.69] |
| z_vad_valence | feedback right | controls | cave (OR/SD) | 85 | 39 | 3.03 [1.96, 17.54] |
| z_vad_valence | all turns | controls | d_pre (pts/SD) | 1800 |  | 7.63 [6.22, 9.22] |
| z_vad_valence | all turns | controls | d_self (pts/SD) | 1800 |  | 4.86 [3.55, 6.35] |

## Praise

Cave rate by category (at-risk turns; feedback wrong / right):

| category | wrong: n | wrong: cave | right: n | right: cave | mean severity_eq |
|---|---|---|---|---|---|
| answer_suggestion | 244 | 2.0% | 28 | 17.9% | -0.39 |
| flattery_sycophancy | 238 | 0.0% | 27 | 0.0% | +0.49 |
| ownership_ego_investment | 237 | 0.0% | 30 | 10.0% | -0.28 |

| measure | subset | spec | outcome | n | events | estimate [95% CI] |
|---|---|---|---|---|---|---|
| severity_eq | all | controls | cave (OR/SD) | 804 | 13 | 0.27 [0.03, 0.95] |
| severity_eq | feedback wrong | controls | cave (OR/SD) | 719 | 5 | n/a (too few) |
| severity_eq | feedback right | controls | cave (OR/SD) | 85 | 8 | n/a (too few) |
| severity_eq | all turns | controls | d_pre (pts/SD) | 1800 |  | -2.29 [-4.52, -0.06] |
| severity_eq | all turns | controls | d_self (pts/SD) | 1800 |  | -1.15 [-2.80, 0.34] |
| severity_eq | all | +category FE | cave (OR/SD) | 804 | 13 | 1.36 [0.14, 78.34] |
| severity_eq | feedback wrong | +category FE | cave (OR/SD) | 719 | 5 | n/a (too few) |
| severity_eq | feedback right | +category FE | cave (OR/SD) | 85 | 8 | n/a (too few) |
| severity_eq | all turns | +category FE | d_pre (pts/SD) | 1800 |  | 0.74 [-3.28, 4.49] |
| severity_eq | all turns | +category FE | d_self (pts/SD) | 1800 |  | -1.39 [-3.37, 0.36] |
| severity_pc1 | all | controls | cave (OR/SD) | 804 | 13 | 1.07 [0.61, 2.32] |
| severity_pc1 | feedback wrong | controls | cave (OR/SD) | 719 | 5 | n/a (too few) |
| severity_pc1 | feedback right | controls | cave (OR/SD) | 85 | 8 | n/a (too few) |
| severity_pc1 | all turns | controls | d_pre (pts/SD) | 1800 |  | 0.50 [-0.82, 1.86] |
| severity_pc1 | all turns | controls | d_self (pts/SD) | 1800 |  | 0.07 [-0.76, 0.85] |
| z_emo_positive | all | controls | cave (OR/SD) | 804 | 13 | 0.51 [0.21, 0.85] |
| z_emo_positive | feedback wrong | controls | cave (OR/SD) | 719 | 5 | n/a (too few) |
| z_emo_positive | feedback right | controls | cave (OR/SD) | 85 | 8 | n/a (too few) |
| z_emo_positive | all turns | controls | d_pre (pts/SD) | 1800 |  | -0.94 [-2.27, 0.11] |
| z_emo_positive | all turns | controls | d_self (pts/SD) | 1800 |  | -0.66 [-1.44, 0.13] |
| z_vad_arousal | all | controls | cave (OR/SD) | 804 | 13 | 0.65 [0.28, 1.14] |
| z_vad_arousal | feedback wrong | controls | cave (OR/SD) | 719 | 5 | n/a (too few) |
| z_vad_arousal | feedback right | controls | cave (OR/SD) | 85 | 8 | n/a (too few) |
| z_vad_arousal | all turns | controls | d_pre (pts/SD) | 1800 |  | -1.17 [-2.42, 0.08] |
| z_vad_arousal | all turns | controls | d_self (pts/SD) | 1800 |  | -0.36 [-1.14, 0.45] |
| z_vad_dominance | all | controls | cave (OR/SD) | 804 | 13 | 0.55 [0.23, 1.34] |
| z_vad_dominance | feedback wrong | controls | cave (OR/SD) | 719 | 5 | n/a (too few) |
| z_vad_dominance | feedback right | controls | cave (OR/SD) | 85 | 8 | n/a (too few) |
| z_vad_dominance | all turns | controls | d_pre (pts/SD) | 1800 |  | -0.89 [-2.26, 0.53] |
| z_vad_dominance | all turns | controls | d_self (pts/SD) | 1800 |  | -0.67 [-1.35, 0.00] |
| z_vad_valence | all | controls | cave (OR/SD) | 804 | 13 | 1.11 [0.51, 2.63] |
| z_vad_valence | feedback wrong | controls | cave (OR/SD) | 719 | 5 | n/a (too few) |
| z_vad_valence | feedback right | controls | cave (OR/SD) | 85 | 8 | n/a (too few) |
| z_vad_valence | all turns | controls | d_pre (pts/SD) | 1800 |  | 0.28 [-1.30, 1.77] |
| z_vad_valence | all turns | controls | d_self (pts/SD) | 1800 |  | 0.11 [-0.86, 1.05] |

