# Tone severity vs behaviour: QwenRA

Results: `results/qwen_real_analysis`  
Claims with all turns: negative 150, positive 150  
Feedback turns joined to a severity score: 1800 (dropped 0 whose recorded prompt does not match the scored wrapper)  
Turns with a missing verdict at t or t-1: 115 of 1800

## Criticism

Cave rate by category (at-risk turns; feedback wrong / right):

| category | wrong: n | wrong: cave | right: n | right: cave | mean severity_eq |
|---|---|---|---|---|---|
| authority_challenge | 87 | 17.2% | 27 | 18.5% | +0.04 |
| mild_doubt_are_you_sure | 74 | 6.8% | 25 | 32.0% | -1.03 |
| strong_rebuttal | 83 | 44.6% | 27 | 48.1% | +0.89 |

| measure | subset | spec | outcome | n | events | estimate [95% CI] |
|---|---|---|---|---|---|---|
| severity_eq | all | controls | cave (OR/SD) | 323 | 83 | 2.61 [1.89, 4.07] |
| severity_eq | feedback wrong | controls | cave (OR/SD) | 244 | 57 | 3.67 [2.58, 7.34] |
| severity_eq | feedback right | controls | cave (OR/SD) | 79 | 26 | 1.47 [0.68, 3.63] |
| severity_eq | all turns | controls | d_pre (pts/SD) | 900 |  | 5.55 [3.37, 8.12] |
| severity_eq | all turns | controls | d_self (pts/SD) | 900 |  | 6.69 [3.94, 9.71] |
| severity_eq | all | +category FE | cave (OR/SD) | 323 | 83 | 2.38 [0.91, 6.27] |
| severity_eq | feedback wrong | +category FE | cave (OR/SD) | 244 | 57 | 2.21 [0.48, 8.77] |
| severity_eq | feedback right | +category FE | cave (OR/SD) | 79 | 26 | 5.86 [0.77, 5527.64] |
| severity_eq | all turns | +category FE | d_pre (pts/SD) | 900 |  | -2.51 [-7.18, 2.28] |
| severity_eq | all turns | +category FE | d_self (pts/SD) | 900 |  | -1.22 [-7.89, 5.46] |
| severity_pc1 | all | controls | cave (OR/SD) | 323 | 83 | 2.17 [1.65, 3.09] |
| severity_pc1 | feedback wrong | controls | cave (OR/SD) | 244 | 57 | 2.83 [2.09, 5.03] |
| severity_pc1 | feedback right | controls | cave (OR/SD) | 79 | 26 | 1.38 [0.74, 3.01] |
| severity_pc1 | all turns | controls | d_pre (pts/SD) | 900 |  | 4.48 [2.52, 6.39] |
| severity_pc1 | all turns | controls | d_self (pts/SD) | 900 |  | 5.40 [3.18, 7.75] |
| z_emo_negative | all | controls | cave (OR/SD) | 323 | 83 | 2.04 [1.48, 3.51] |
| z_emo_negative | feedback wrong | controls | cave (OR/SD) | 244 | 57 | 3.21 [2.16, 6.67] |
| z_emo_negative | feedback right | controls | cave (OR/SD) | 79 | 26 | 1.05 [0.51, 2.96] |
| z_emo_negative | all turns | controls | d_pre (pts/SD) | 900 |  | 5.56 [3.59, 7.67] |
| z_emo_negative | all turns | controls | d_self (pts/SD) | 900 |  | 6.10 [3.99, 8.29] |
| z_politeness | all | controls | cave (OR/SD) | 323 | 83 | 2.07 [1.59, 3.13] |
| z_politeness | feedback wrong | controls | cave (OR/SD) | 244 | 57 | 2.68 [1.91, 4.70] |
| z_politeness | feedback right | controls | cave (OR/SD) | 79 | 26 | 1.33 [0.69, 3.43] |
| z_politeness | all turns | controls | d_pre (pts/SD) | 900 |  | 2.92 [1.22, 4.76] |
| z_politeness | all turns | controls | d_self (pts/SD) | 900 |  | 4.74 [2.60, 6.60] |
| z_vad_arousal | all | controls | cave (OR/SD) | 323 | 83 | 1.56 [1.23, 2.07] |
| z_vad_arousal | feedback wrong | controls | cave (OR/SD) | 244 | 57 | 1.78 [1.32, 2.68] |
| z_vad_arousal | feedback right | controls | cave (OR/SD) | 79 | 26 | 1.12 [0.64, 2.07] |
| z_vad_arousal | all turns | controls | d_pre (pts/SD) | 900 |  | 3.14 [1.42, 5.08] |
| z_vad_arousal | all turns | controls | d_self (pts/SD) | 900 |  | 2.66 [0.51, 4.63] |
| z_vad_dominance | all | controls | cave (OR/SD) | 323 | 83 | 0.55 [0.42, 0.69] |
| z_vad_dominance | feedback wrong | controls | cave (OR/SD) | 244 | 57 | 0.50 [0.36, 0.63] |
| z_vad_dominance | feedback right | controls | cave (OR/SD) | 79 | 26 | 0.61 [0.24, 1.21] |
| z_vad_dominance | all turns | controls | d_pre (pts/SD) | 900 |  | -2.74 [-4.38, -1.21] |
| z_vad_dominance | all turns | controls | d_self (pts/SD) | 900 |  | -4.72 [-7.07, -2.67] |
| z_vad_valence | all | controls | cave (OR/SD) | 323 | 83 | 1.91 [1.51, 2.59] |
| z_vad_valence | feedback wrong | controls | cave (OR/SD) | 244 | 57 | 2.05 [1.57, 3.03] |
| z_vad_valence | feedback right | controls | cave (OR/SD) | 79 | 26 | 1.77 [0.89, 4.41] |
| z_vad_valence | all turns | controls | d_pre (pts/SD) | 900 |  | 3.65 [2.13, 5.44] |
| z_vad_valence | all turns | controls | d_self (pts/SD) | 900 |  | 4.90 [2.52, 7.16] |

## Praise

Cave rate by category (at-risk turns; feedback wrong / right):

| category | wrong: n | wrong: cave | right: n | right: cave | mean severity_eq |
|---|---|---|---|---|---|
| answer_suggestion | 100 | 5.0% | 16 | 12.5% | -0.36 |
| flattery_sycophancy | 100 | 1.0% | 15 | 0.0% | +0.54 |
| ownership_ego_investment | 100 | 1.0% | 16 | 6.2% | -0.29 |

| measure | subset | spec | outcome | n | events | estimate [95% CI] |
|---|---|---|---|---|---|---|
| severity_eq | all | controls | cave (OR/SD) | 347 | 10 | 0.46 [0.07, 2.13] |
| severity_eq | feedback wrong | controls | cave (OR/SD) | 300 | 7 | n/a (too few) |
| severity_eq | feedback right | controls | cave (OR/SD) | 47 | 3 | n/a (too few) |
| severity_eq | all turns | controls | d_pre (pts/SD) | 900 |  | -2.54 [-5.45, 0.10] |
| severity_eq | all turns | controls | d_self (pts/SD) | 900 |  | 0.00 [-1.51, 1.48] |
| severity_eq | all | +category FE | cave (OR/SD) | 347 | 10 | 2.76 [0.17, 1740.12] |
| severity_eq | feedback wrong | +category FE | cave (OR/SD) | 300 | 7 | n/a (too few) |
| severity_eq | feedback right | +category FE | cave (OR/SD) | 47 | 3 | n/a (too few) |
| severity_eq | all turns | +category FE | d_pre (pts/SD) | 900 |  | 3.71 [0.30, 7.56] |
| severity_eq | all turns | +category FE | d_self (pts/SD) | 900 |  | 1.55 [-0.81, 4.49] |
| severity_pc1 | all | controls | cave (OR/SD) | 347 | 10 | 0.77 [0.33, 1.35] |
| severity_pc1 | feedback wrong | controls | cave (OR/SD) | 300 | 7 | n/a (too few) |
| severity_pc1 | feedback right | controls | cave (OR/SD) | 47 | 3 | n/a (too few) |
| severity_pc1 | all turns | controls | d_pre (pts/SD) | 900 |  | 0.68 [-0.86, 2.43] |
| severity_pc1 | all turns | controls | d_self (pts/SD) | 900 |  | 0.21 [-1.04, 1.30] |
| z_emo_positive | all | controls | cave (OR/SD) | 347 | 10 | 0.71 [0.32, 1.39] |
| z_emo_positive | feedback wrong | controls | cave (OR/SD) | 300 | 7 | n/a (too few) |
| z_emo_positive | feedback right | controls | cave (OR/SD) | 47 | 3 | n/a (too few) |
| z_emo_positive | all turns | controls | d_pre (pts/SD) | 900 |  | -1.05 [-2.73, 0.51] |
| z_emo_positive | all turns | controls | d_self (pts/SD) | 900 |  | -0.20 [-0.88, 0.52] |
| z_vad_arousal | all | controls | cave (OR/SD) | 347 | 10 | 0.93 [0.42, 1.82] |
| z_vad_arousal | feedback wrong | controls | cave (OR/SD) | 300 | 7 | n/a (too few) |
| z_vad_arousal | feedback right | controls | cave (OR/SD) | 47 | 3 | n/a (too few) |
| z_vad_arousal | all turns | controls | d_pre (pts/SD) | 900 |  | -1.60 [-3.86, 0.27] |
| z_vad_arousal | all turns | controls | d_self (pts/SD) | 900 |  | -0.08 [-0.78, 0.65] |
| z_vad_dominance | all | controls | cave (OR/SD) | 347 | 10 | 0.73 [0.39, 1.58] |
| z_vad_dominance | feedback wrong | controls | cave (OR/SD) | 300 | 7 | n/a (too few) |
| z_vad_dominance | feedback right | controls | cave (OR/SD) | 47 | 3 | n/a (too few) |
| z_vad_dominance | all turns | controls | d_pre (pts/SD) | 900 |  | -0.91 [-2.65, 0.76] |
| z_vad_dominance | all turns | controls | d_self (pts/SD) | 900 |  | 0.47 [-0.39, 1.43] |
| z_vad_valence | all | controls | cave (OR/SD) | 347 | 10 | 0.68 [0.25, 1.54] |
| z_vad_valence | feedback wrong | controls | cave (OR/SD) | 300 | 7 | n/a (too few) |
| z_vad_valence | feedback right | controls | cave (OR/SD) | 47 | 3 | n/a (too few) |
| z_vad_valence | all turns | controls | d_pre (pts/SD) | 900 |  | 0.16 [-1.23, 1.69] |
| z_vad_valence | all turns | controls | d_self (pts/SD) | 900 |  | 0.41 [-1.17, 1.87] |

