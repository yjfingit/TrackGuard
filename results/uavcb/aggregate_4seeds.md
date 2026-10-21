# Multi-seed benchmark aggregate

- runs: 4 (seeds: bench_uavcb, ms_uavcb_11, ms_uavcb_23, ms_uavcb_42)
- rates are per-seed proportions reported as mean +/- std; vlm_calls/human are per-run counts as mean +/- std

| method | metric | mean | std | per-seed |
|---|---|---|---|---|
| rules | detect_fail | 0.571 | 0.000 | 0.57, 0.57, 0.57, 0.57 |
| rules | flag_all | 0.571 | 0.000 | 0.57, 0.57, 0.57, 0.57 |
| rules | flag_semantic | 0.000 | 0.000 | 0.00, 0.00, 0.00, 0.00 |
| rules | fpr_fail_ctrl | 0.000 | 0.000 | 0.00, 0.00, 0.00, 0.00 |
| rules | fpr_flag_ctrl | 0.000 | 0.000 | 0.00, 0.00, 0.00, 0.00 |
| rules | vlm_calls | 0.000 | 0.000 | 0.00, 0.00, 0.00, 0.00 |
| rules | human_sequences | 4.000 | 0.000 | 4, 4, 4, 4 |
| direct | detect_fail | 0.143 | 0.117 | 0.00, 0.29, 0.14, 0.14 |
| direct | flag_all | 0.429 | 0.000 | 0.43, 0.43, 0.43, 0.43 |
| direct | flag_semantic | 1.000 | 0.000 | 1.00, 1.00, 1.00, 1.00 |
| direct | fpr_fail_ctrl | 0.000 | 0.000 | 0.00, 0.00, 0.00, 0.00 |
| direct | fpr_flag_ctrl | 0.000 | 0.000 | 0.00, 0.00, 0.00, 0.00 |
| direct | vlm_calls | 159.000 | 0.000 | 159, 159, 159, 159 |
| direct | human_sequences | 3.000 | 0.000 | 3, 3, 3, 3 |
| layered | detect_fail | 0.714 | 0.117 | 0.86, 0.71, 0.57, 0.71 |
| layered | flag_all | 1.000 | 0.000 | 1.00, 1.00, 1.00, 1.00 |
| layered | flag_semantic | 1.000 | 0.000 | 1.00, 1.00, 1.00, 1.00 |
| layered | fpr_fail_ctrl | 0.000 | 0.000 | 0.00, 0.00, 0.00, 0.00 |
| layered | fpr_flag_ctrl | 0.000 | 0.000 | 0.00, 0.00, 0.00, 0.00 |
| layered | vlm_calls | 84.000 | 0.000 | 84, 84, 84, 84 |
| layered | human_sequences | 7.000 | 0.000 | 7, 7, 7, 7 |
