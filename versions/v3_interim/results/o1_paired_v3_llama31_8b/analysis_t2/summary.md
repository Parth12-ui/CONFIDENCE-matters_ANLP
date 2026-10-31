# O1 paired analysis

Claims: 150 items (75 true, 75 false). Tracks: control, negative, positive. Overall parse-failure rate: 0.0%.

## Turn 0 vs final turn

| track | truth | turn | accuracy | accuracy_lo | accuracy_hi | valid_rate | truth_confidence | accuracy_n |
|---|---|---|---|---|---|---|---|---|
| control | false claim | 0 | 0.747 | 0.64 | 0.84 | 0.253 | 63.339 | 75 |
| control | false claim | 2 | 0.787 | 0.693 | 0.88 | 0.213 | 60.988 | 75 |
| control | true claim | 0 | 0.32 | 0.213 | 0.427 | 0.32 | 45.056 | 75 |
| control | true claim | 2 | 0.307 | 0.213 | 0.413 | 0.307 | 45.145 | 75 |
| negative | false claim | 0 | 0.747 | 0.64 | 0.84 | 0.253 | 63.339 | 75 |
| negative | false claim | 2 | 0.813 | 0.72 | 0.893 | 0.187 | 70.777 | 75 |
| negative | true claim | 0 | 0.32 | 0.213 | 0.427 | 0.32 | 45.056 | 75 |
| negative | true claim | 2 | 0.187 | 0.107 | 0.28 | 0.187 | 29.225 | 75 |
| positive | false claim | 0 | 0.747 | 0.64 | 0.84 | 0.253 | 63.339 | 75 |
| positive | false claim | 2 | 0.453 | 0.347 | 0.573 | 0.547 | 41.934 | 75 |
| positive | true claim | 0 | 0.32 | 0.213 | 0.427 | 0.32 | 45.056 | 75 |
| positive | true claim | 2 | 0.627 | 0.52 | 0.733 | 0.627 | 62.76 | 75 |

## Summary metrics (Laban et al. 2023; SYCON-Bench)

| track | truth | n_claims | delta_accuracy | delta_accuracy_lo | delta_accuracy_hi | flip_rate | flip_rate_lo | flip_rate_hi | number_of_flips | number_of_flips_lo | number_of_flips_hi | turn_of_flip | turn_of_flip_lo | turn_of_flip_hi | n_initially_correct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| control | false claim | 75 | 0.04 | -0.04 | 0.12 | 0.147 | 0.08 | 0.227 | 0.253 | 0.133 | 0.373 | 2.786 | 2.625 | 2.929 | 56 |
| control | true claim | 75 | -0.013 | -0.093 | 0.08 | 0.147 | 0.067 | 0.227 | 0.173 | 0.08 | 0.267 | 2.5 | 2.167 | 2.833 | 24 |
| negative | false claim | 75 | 0.067 | -0.067 | 0.2 | 0.333 | 0.227 | 0.44 | 1.133 | 0.947 | 1.307 | 1.839 | 1.589 | 2.089 | 56 |
| negative | true claim | 75 | -0.133 | -0.28 | 0.013 | 0.427 | 0.32 | 0.533 | 0.907 | 0.746 | 1.08 | 1.0 | 1.0 | 1.0 | 24 |
| positive | false claim | 75 | -0.293 | -0.44 | -0.147 | 0.533 | 0.413 | 0.653 | 0.693 | 0.56 | 0.827 | 2.018 | 1.804 | 2.232 | 56 |
| positive | true claim | 75 | 0.307 | 0.16 | 0.453 | 0.493 | 0.387 | 0.613 | 0.707 | 0.56 | 0.853 | 2.333 | 1.958 | 2.667 | 24 |

## Pressure track minus control, final turn

| track | control_track | truth | turn | accuracy_diff | accuracy_diff_lo | accuracy_diff_hi | n | valid_rate_diff | valid_rate_diff_lo | valid_rate_diff_hi | truth_confidence_diff | truth_confidence_diff_lo | truth_confidence_diff_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| positive | control | false claim | 2 | -0.333 | -0.467 | -0.187 | 75 | 0.333 | 0.187 | 0.467 | -19.054 | -26.814 | -11.031 |
| positive | control | true claim | 2 | 0.32 | 0.187 | 0.453 | 75 | 0.32 | 0.187 | 0.453 | 17.615 | 10.443 | 24.351 |
| negative | control | false claim | 2 | 0.027 | -0.093 | 0.16 | 75 | -0.027 | -0.16 | 0.093 | 9.789 | 0.126 | 20.178 |
| negative | control | true claim | 2 | -0.12 | -0.267 | 0.027 | 75 | -0.12 | -0.267 | 0.027 | -15.92 | -26.176 | -5.352 |
