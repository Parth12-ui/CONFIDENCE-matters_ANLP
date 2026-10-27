# Tone severity vs behaviour: LlamaMath

Results: `results/llama_math`  
Claims with all turns: negative 300, positive 300  
Feedback turns joined to a severity score: 3600 (dropped 0 whose recorded prompt does not match the scored wrapper)  
Turns with a missing verdict at t or t-1: 4 of 3600

## Criticism

Cave rate by category (at-risk turns; feedback wrong / right):

| category | wrong: n | wrong: cave | right: n | right: cave | mean severity_eq |
|---|---|---|---|---|---|
| authority_challenge | 90 | 81.1% | 55 | 89.1% | +0.07 |
| mild_doubt_are_you_sure | 83 | 67.5% | 42 | 78.6% | -1.00 |
| strong_rebuttal | 107 | 86.9% | 59 | 91.5% | +0.93 |

| measure | subset | spec | outcome | n | events | estimate [95% CI] |
|---|---|---|---|---|---|---|
| severity_eq | all | controls | cave (OR/SD) | 436 | 358 | 1.92 [1.47, 2.77] |
| severity_eq | feedback wrong | controls | cave (OR/SD) | 280 | 222 | 1.82 [1.32, 2.85] |
| severity_eq | feedback right | controls | cave (OR/SD) | 156 | 136 | 2.38 [1.29, 8.63] |
| severity_eq | all turns | controls | d_pre (pts/SD) | 1800 |  | 10.30 [9.01, 11.67] |
| severity_eq | all turns | controls | d_self (pts/SD) | 1800 |  | 8.05 [6.72, 9.35] |
| severity_eq | all | +category FE | cave (OR/SD) | 436 | 358 | 2.65 [1.39, 6.06] |
| severity_eq | feedback wrong | +category FE | cave (OR/SD) | 280 | 222 | 2.43 [1.09, 6.59] |
| severity_eq | feedback right | +category FE | cave (OR/SD) | 156 | 136 | 5.40 [1.54, 113.99] |
| severity_eq | all turns | +category FE | d_pre (pts/SD) | 1800 |  | 4.47 [1.40, 7.72] |
| severity_eq | all turns | +category FE | d_self (pts/SD) | 1800 |  | 4.93 [1.55, 8.28] |
| severity_pc1 | all | controls | cave (OR/SD) | 436 | 358 | 1.70 [1.36, 2.27] |
| severity_pc1 | feedback wrong | controls | cave (OR/SD) | 280 | 222 | 1.63 [1.22, 2.38] |
| severity_pc1 | feedback right | controls | cave (OR/SD) | 156 | 136 | 2.03 [1.31, 5.02] |
| severity_pc1 | all turns | controls | d_pre (pts/SD) | 1800 |  | 8.33 [7.30, 9.44] |
| severity_pc1 | all turns | controls | d_self (pts/SD) | 1800 |  | 6.55 [5.53, 7.54] |
| z_emo_negative | all | controls | cave (OR/SD) | 436 | 358 | 1.58 [1.27, 2.06] |
| z_emo_negative | feedback wrong | controls | cave (OR/SD) | 280 | 222 | 1.56 [1.21, 2.24] |
| z_emo_negative | feedback right | controls | cave (OR/SD) | 156 | 136 | 1.68 [0.86, 3.88] |
| z_emo_negative | all turns | controls | d_pre (pts/SD) | 1800 |  | 9.03 [7.94, 10.19] |
| z_emo_negative | all turns | controls | d_self (pts/SD) | 1800 |  | 6.00 [4.90, 7.09] |
| z_politeness | all | controls | cave (OR/SD) | 436 | 358 | 1.44 [1.15, 1.92] |
| z_politeness | feedback wrong | controls | cave (OR/SD) | 280 | 222 | 1.27 [0.96, 1.79] |
| z_politeness | feedback right | controls | cave (OR/SD) | 156 | 136 | 1.87 [1.10, 5.37] |
| z_politeness | all turns | controls | d_pre (pts/SD) | 1800 |  | 6.19 [5.06, 7.32] |
| z_politeness | all turns | controls | d_self (pts/SD) | 1800 |  | 4.59 [3.31, 5.76] |
| z_vad_arousal | all | controls | cave (OR/SD) | 436 | 358 | 1.55 [1.20, 2.05] |
| z_vad_arousal | feedback wrong | controls | cave (OR/SD) | 280 | 222 | 1.52 [1.13, 2.14] |
| z_vad_arousal | feedback right | controls | cave (OR/SD) | 156 | 136 | 1.84 [1.20, 4.41] |
| z_vad_arousal | all turns | controls | d_pre (pts/SD) | 1800 |  | 6.76 [5.70, 7.91] |
| z_vad_arousal | all turns | controls | d_self (pts/SD) | 1800 |  | 5.77 [4.78, 6.85] |
| z_vad_dominance | all | controls | cave (OR/SD) | 436 | 358 | 0.64 [0.49, 0.79] |
| z_vad_dominance | feedback wrong | controls | cave (OR/SD) | 280 | 222 | 0.65 [0.46, 0.84] |
| z_vad_dominance | feedback right | controls | cave (OR/SD) | 156 | 136 | 0.63 [0.30, 0.99] |
| z_vad_dominance | all turns | controls | d_pre (pts/SD) | 1800 |  | -5.43 [-6.66, -4.29] |
| z_vad_dominance | all turns | controls | d_self (pts/SD) | 1800 |  | -4.64 [-5.77, -3.44] |
| z_vad_valence | all | controls | cave (OR/SD) | 436 | 358 | 1.65 [1.35, 2.14] |
| z_vad_valence | feedback wrong | controls | cave (OR/SD) | 280 | 222 | 1.65 [1.27, 2.35] |
| z_vad_valence | feedback right | controls | cave (OR/SD) | 156 | 136 | 1.68 [1.06, 3.20] |
| z_vad_valence | all turns | controls | d_pre (pts/SD) | 1800 |  | 6.77 [5.69, 7.92] |
| z_vad_valence | all turns | controls | d_self (pts/SD) | 1800 |  | 6.04 [4.99, 7.10] |

## Praise

Cave rate by category (at-risk turns; feedback wrong / right):

| category | wrong: n | wrong: cave | right: n | right: cave | mean severity_eq |
|---|---|---|---|---|---|
| answer_suggestion | 192 | 47.9% | 94 | 67.0% | -0.39 |
| flattery_sycophancy | 162 | 16.7% | 67 | 20.9% | +0.49 |
| ownership_ego_investment | 154 | 19.5% | 76 | 38.2% | -0.32 |

| measure | subset | spec | outcome | n | events | estimate [95% CI] |
|---|---|---|---|---|---|---|
| severity_eq | all | controls | cave (OR/SD) | 745 | 255 | 0.30 [0.19, 0.41] |
| severity_eq | feedback wrong | controls | cave (OR/SD) | 508 | 149 | 0.35 [0.21, 0.53] |
| severity_eq | feedback right | controls | cave (OR/SD) | 237 | 106 | 0.17 [0.06, 0.33] |
| severity_eq | all turns | controls | d_pre (pts/SD) | 1800 |  | -12.02 [-14.18, -9.73] |
| severity_eq | all turns | controls | d_self (pts/SD) | 1800 |  | -7.99 [-10.11, -5.94] |
| severity_eq | all | +category FE | cave (OR/SD) | 745 | 255 | 0.75 [0.45, 1.25] |
| severity_eq | feedback wrong | +category FE | cave (OR/SD) | 508 | 149 | 0.83 [0.44, 1.59] |
| severity_eq | feedback right | +category FE | cave (OR/SD) | 237 | 106 | 0.52 [0.15, 1.35] |
| severity_eq | all turns | +category FE | d_pre (pts/SD) | 1800 |  | -4.31 [-7.12, -1.42] |
| severity_eq | all turns | +category FE | d_self (pts/SD) | 1800 |  | -0.32 [-2.93, 2.60] |
| severity_pc1 | all | controls | cave (OR/SD) | 745 | 255 | 0.91 [0.78, 1.06] |
| severity_pc1 | feedback wrong | controls | cave (OR/SD) | 508 | 149 | 0.87 [0.71, 1.05] |
| severity_pc1 | feedback right | controls | cave (OR/SD) | 237 | 106 | 0.83 [0.54, 1.18] |
| severity_pc1 | all turns | controls | d_pre (pts/SD) | 1800 |  | 1.28 [0.06, 2.51] |
| severity_pc1 | all turns | controls | d_self (pts/SD) | 1800 |  | -0.11 [-1.08, 0.91] |
| z_emo_positive | all | controls | cave (OR/SD) | 745 | 255 | 0.55 [0.45, 0.66] |
| z_emo_positive | feedback wrong | controls | cave (OR/SD) | 508 | 149 | 0.57 [0.45, 0.71] |
| z_emo_positive | feedback right | controls | cave (OR/SD) | 237 | 106 | 0.47 [0.30, 0.62] |
| z_emo_positive | all turns | controls | d_pre (pts/SD) | 1800 |  | -6.01 [-7.11, -4.89] |
| z_emo_positive | all turns | controls | d_self (pts/SD) | 1800 |  | -3.94 [-5.05, -2.87] |
| z_vad_arousal | all | controls | cave (OR/SD) | 745 | 255 | 0.77 [0.65, 0.89] |
| z_vad_arousal | feedback wrong | controls | cave (OR/SD) | 508 | 149 | 0.86 [0.70, 1.02] |
| z_vad_arousal | feedback right | controls | cave (OR/SD) | 237 | 106 | 0.66 [0.44, 0.89] |
| z_vad_arousal | all turns | controls | d_pre (pts/SD) | 1800 |  | -4.58 [-5.81, -3.37] |
| z_vad_arousal | all turns | controls | d_self (pts/SD) | 1800 |  | -2.29 [-3.23, -1.34] |
| z_vad_dominance | all | controls | cave (OR/SD) | 745 | 255 | 0.55 [0.46, 0.63] |
| z_vad_dominance | feedback wrong | controls | cave (OR/SD) | 508 | 149 | 0.54 [0.43, 0.65] |
| z_vad_dominance | feedback right | controls | cave (OR/SD) | 237 | 106 | 0.55 [0.38, 0.71] |
| z_vad_dominance | all turns | controls | d_pre (pts/SD) | 1800 |  | -4.74 [-6.06, -3.43] |
| z_vad_dominance | all turns | controls | d_self (pts/SD) | 1800 |  | -4.12 [-5.14, -3.10] |
| z_vad_valence | all | controls | cave (OR/SD) | 745 | 255 | 0.84 [0.69, 1.01] |
| z_vad_valence | feedback wrong | controls | cave (OR/SD) | 508 | 149 | 0.85 [0.66, 1.06] |
| z_vad_valence | feedback right | controls | cave (OR/SD) | 237 | 106 | 0.68 [0.41, 1.03] |
| z_vad_valence | all turns | controls | d_pre (pts/SD) | 1800 |  | 0.91 [-0.47, 2.32] |
| z_vad_valence | all turns | controls | d_self (pts/SD) | 1800 |  | -0.57 [-1.71, 0.54] |

