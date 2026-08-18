# Beat-Pooling Results (Harmonix)

| pooling | F@0.5 | F@3.0 | PairwiseF | NCE-F | Acc | best val loss | epochs |
|---|---|---|---|---|---|---|---|
| mean | 0.239 | 0.557 | 0.683 | 0.807 | 0.613 | 1.096 | 43 ep (best @31) |
| max | 0.228 | 0.564 | 0.686 | 0.807 | 0.607 | 1.105 | 49 ep (best @35) |
| energy_weighted | 0.230 | 0.549 | 0.675 | 0.803 | 0.609 | 1.120 | 31 ep (best @16) |
| multi_stat | 0.250 | 0.596 | 0.689 | 0.809 | 0.610 | 1.086 | 42 ep (best @19) |
| first | 0.223 | 0.504 | 0.648 | 0.786 | 0.516 | 1.257 | 30 ep (best @13) |
| last | 0.209 | 0.491 | 0.659 | 0.790 | 0.561 | 1.230 | 31 ep (best @19) |
| attention (A2) | 0.251 | 0.610 | 0.694 | 0.811 | 0.618 | 1.127 | 32 ep (best @13) |

Metrics are on the Harmonix val split at the best F-measure@3.0 epoch. F@0.5 / F@3.0 = boundary hit-rate F; PairwiseF / NCE-F = structure labelling; Acc = frame accuracy. `*` = run did not early-stop.

