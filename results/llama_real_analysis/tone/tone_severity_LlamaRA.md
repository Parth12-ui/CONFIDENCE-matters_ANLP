# Tone severity vs behaviour: LlamaRA

Results: `results/llama_real_analysis`  
Claims with all turns: negative 150, positive 150  
Feedback turns joined to a severity score: 1800 (dropped 0 whose recorded prompt does not match the scored wrapper)  
Turns with a missing verdict at t or t-1: 0 of 1800

## Criticism

Cave rate by category (at-risk turns; feedback wrong / right):

| category | wrong: n | wrong: cave | right: n | right: cave | mean severity_eq |
|---|---|---|---|---|---|
| authority_challenge | 48 | 89.6% | 38 | 92.1% | +0.00 |
| mild_doubt_are_you_sure | 36 | 83.3% | 37 | 86.5% | -1.02 |
| strong_rebuttal | 52 | 96.2% | 46 | 91.3% | +0.92 |

| measure | subset | spec | outcome | n | events | estimate [95% CI] |
|---|---|---|---|---|---|---|
| severity_eq | all | controls | cave (OR/SD) | 257 | 232 | 2.35 [1.46, 4.42] |
| severity_eq | feedback wrong | controls | cave (OR/SD) | 136 | 123 | 3.31 [1.65, 25.62] |
| severity_eq | feedback right | controls | cave (OR/SD) | 121 | 109 | 1.80 [0.70, 23.34] |
| severity_eq | all turns | controls | d_pre (pts/SD) | 900 |  | 15.11 [12.08, 17.99] |
| severity_eq | all turns | controls | d_self (pts/SD) | 900 |  | 9.23 [6.72, 11.73] |
| severity_eq | all | +category FE | cave (OR/SD) | 257 | 232 | 2.82 [0.64, 20.67] |
| severity_eq | feedback wrong | +category FE | cave (OR/SD) | 136 | 123 | 4.90 [0.95, 1151.27] |
| severity_eq | feedback right | +category FE | cave (OR/SD) | 121 | 109 | 2.16 [0.07, 259.09] |
| severity_eq | all turns | +category FE | d_pre (pts/SD) | 900 |  | 0.71 [-6.32, 7.49] |
| severity_eq | all turns | +category FE | d_self (pts/SD) | 900 |  | 0.65 [-5.73, 7.12] |
| severity_pc1 | all | controls | cave (OR/SD) | 257 | 232 | 2.00 [1.38, 3.65] |
| severity_pc1 | feedback wrong | controls | cave (OR/SD) | 136 | 123 | 2.68 [1.58, 11.51] |
| severity_pc1 | feedback right | controls | cave (OR/SD) | 121 | 109 | 1.60 [0.74, 18.05] |
| severity_pc1 | all turns | controls | d_pre (pts/SD) | 900 |  | 12.22 [9.90, 14.65] |
| severity_pc1 | all turns | controls | d_self (pts/SD) | 900 |  | 7.49 [5.40, 9.65] |
| z_emo_negative | all | controls | cave (OR/SD) | 257 | 232 | 1.53 [0.99, 2.60] |
| z_emo_negative | feedback wrong | controls | cave (OR/SD) | 136 | 123 | 1.47 [0.89, 3.40] |
| z_emo_negative | feedback right | controls | cave (OR/SD) | 121 | 109 | 1.47 [0.66, 4.90] |
| z_emo_negative | all turns | controls | d_pre (pts/SD) | 900 |  | 13.75 [11.60, 15.99] |
| z_emo_negative | all turns | controls | d_self (pts/SD) | 900 |  | 7.16 [5.35, 9.15] |
| z_politeness | all | controls | cave (OR/SD) | 257 | 232 | 2.16 [1.42, 4.35] |
| z_politeness | feedback wrong | controls | cave (OR/SD) | 136 | 123 | 2.58 [1.37, 18.80] |
| z_politeness | feedback right | controls | cave (OR/SD) | 121 | 109 | 1.97 [0.85, 46.32] |
| z_politeness | all turns | controls | d_pre (pts/SD) | 900 |  | 8.52 [6.14, 10.98] |
| z_politeness | all turns | controls | d_self (pts/SD) | 900 |  | 6.11 [4.01, 8.17] |
| z_vad_arousal | all | controls | cave (OR/SD) | 257 | 232 | 1.67 [1.16, 2.56] |
| z_vad_arousal | feedback wrong | controls | cave (OR/SD) | 136 | 123 | 2.19 [1.23, 7.97] |
| z_vad_arousal | feedback right | controls | cave (OR/SD) | 121 | 109 | 1.24 [0.62, 2.74] |
| z_vad_arousal | all turns | controls | d_pre (pts/SD) | 900 |  | 9.42 [6.98, 11.98] |
| z_vad_arousal | all turns | controls | d_self (pts/SD) | 900 |  | 5.65 [3.44, 7.89] |
| z_vad_dominance | all | controls | cave (OR/SD) | 257 | 232 | 0.64 [0.45, 0.88] |
| z_vad_dominance | feedback wrong | controls | cave (OR/SD) | 136 | 123 | 0.38 [0.10, 0.73] |
| z_vad_dominance | feedback right | controls | cave (OR/SD) | 121 | 109 | 0.84 [0.30, 1.70] |
| z_vad_dominance | all turns | controls | d_pre (pts/SD) | 900 |  | -6.81 [-9.18, -4.46] |
| z_vad_dominance | all turns | controls | d_self (pts/SD) | 900 |  | -4.53 [-6.51, -2.67] |
| z_vad_valence | all | controls | cave (OR/SD) | 257 | 232 | 1.70 [1.23, 2.69] |
| z_vad_valence | feedback wrong | controls | cave (OR/SD) | 136 | 123 | 2.97 [1.74, 16.72] |
| z_vad_valence | feedback right | controls | cave (OR/SD) | 121 | 109 | 1.20 [0.52, 4.53] |
| z_vad_valence | all turns | controls | d_pre (pts/SD) | 900 |  | 9.87 [7.61, 12.36] |
| z_vad_valence | all turns | controls | d_self (pts/SD) | 900 |  | 6.46 [4.33, 8.48] |

## Praise

Cave rate by category (at-risk turns; feedback wrong / right):

| category | wrong: n | wrong: cave | right: n | right: cave | mean severity_eq |
|---|---|---|---|---|---|
| answer_suggestion | 61 | 82.0% | 51 | 76.5% | -0.40 |
| flattery_sycophancy | 43 | 34.9% | 39 | 51.3% | +0.66 |
| ownership_ego_investment | 52 | 38.5% | 43 | 51.2% | -0.26 |

| measure | subset | spec | outcome | n | events | estimate [95% CI] |
|---|---|---|---|---|---|---|
| severity_eq | all | controls | cave (OR/SD) | 289 | 166 | 0.30 [0.16, 0.47] |
| severity_eq | feedback wrong | controls | cave (OR/SD) | 156 | 85 | 0.13 [0.03, 0.25] |
| severity_eq | feedback right | controls | cave (OR/SD) | 133 | 81 | 0.49 [0.19, 1.01] |
| severity_eq | all turns | controls | d_pre (pts/SD) | 900 |  | -11.07 [-13.74, -8.16] |
| severity_eq | all turns | controls | d_self (pts/SD) | 900 |  | -8.47 [-11.50, -5.55] |
| severity_eq | all | +category FE | cave (OR/SD) | 289 | 166 | 0.59 [0.31, 1.05] |
| severity_eq | feedback wrong | +category FE | cave (OR/SD) | 156 | 85 | 0.37 [0.07, 1.00] |
| severity_eq | feedback right | +category FE | cave (OR/SD) | 133 | 81 | 0.75 [0.25, 2.36] |
| severity_eq | all turns | +category FE | d_pre (pts/SD) | 900 |  | -4.37 [-8.33, -0.54] |
| severity_eq | all turns | +category FE | d_self (pts/SD) | 900 |  | 0.93 [-2.61, 4.44] |
| severity_pc1 | all | controls | cave (OR/SD) | 289 | 166 | 0.63 [0.45, 0.84] |
| severity_pc1 | feedback wrong | controls | cave (OR/SD) | 156 | 85 | 0.61 [0.36, 0.90] |
| severity_pc1 | feedback right | controls | cave (OR/SD) | 133 | 81 | 0.64 [0.35, 1.00] |
| severity_pc1 | all turns | controls | d_pre (pts/SD) | 900 |  | -0.98 [-2.71, 0.78] |
| severity_pc1 | all turns | controls | d_self (pts/SD) | 900 |  | -0.92 [-2.88, 0.87] |
| z_emo_positive | all | controls | cave (OR/SD) | 289 | 166 | 0.49 [0.34, 0.63] |
| z_emo_positive | feedback wrong | controls | cave (OR/SD) | 156 | 85 | 0.34 [0.19, 0.45] |
| z_emo_positive | feedback right | controls | cave (OR/SD) | 133 | 81 | 0.62 [0.36, 0.88] |
| z_emo_positive | all turns | controls | d_pre (pts/SD) | 900 |  | -5.86 [-7.36, -4.38] |
| z_emo_positive | all turns | controls | d_self (pts/SD) | 900 |  | -5.24 [-7.07, -3.54] |
| z_vad_arousal | all | controls | cave (OR/SD) | 289 | 166 | 0.87 [0.68, 1.11] |
| z_vad_arousal | feedback wrong | controls | cave (OR/SD) | 156 | 85 | 0.69 [0.44, 0.97] |
| z_vad_arousal | feedback right | controls | cave (OR/SD) | 133 | 81 | 1.07 [0.72, 1.65] |
| z_vad_arousal | all turns | controls | d_pre (pts/SD) | 900 |  | -3.37 [-4.77, -2.05] |
| z_vad_arousal | all turns | controls | d_self (pts/SD) | 900 |  | -2.12 [-3.79, -0.45] |
| z_vad_dominance | all | controls | cave (OR/SD) | 289 | 166 | 0.52 [0.36, 0.67] |
| z_vad_dominance | feedback wrong | controls | cave (OR/SD) | 156 | 85 | 0.38 [0.19, 0.53] |
| z_vad_dominance | feedback right | controls | cave (OR/SD) | 133 | 81 | 0.64 [0.36, 1.05] |
| z_vad_dominance | all turns | controls | d_pre (pts/SD) | 900 |  | -6.30 [-8.07, -4.62] |
| z_vad_dominance | all turns | controls | d_self (pts/SD) | 900 |  | -5.12 [-6.90, -3.29] |
| z_vad_valence | all | controls | cave (OR/SD) | 289 | 166 | 0.56 [0.39, 0.77] |
| z_vad_valence | feedback wrong | controls | cave (OR/SD) | 156 | 85 | 0.48 [0.28, 0.73] |
| z_vad_valence | feedback right | controls | cave (OR/SD) | 133 | 81 | 0.61 [0.30, 1.02] |
| z_vad_valence | all turns | controls | d_pre (pts/SD) | 900 |  | -2.21 [-4.23, -0.11] |
| z_vad_valence | all turns | controls | d_self (pts/SD) | 900 |  | -1.15 [-3.29, 0.93] |

