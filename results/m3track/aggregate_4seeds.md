# Multi-seed benchmark aggregate

- runs: 4 (seeds: bench_m3t, ms_m3t_11, ms_m3t_23, ms_m3t_42)
- rates are per-seed proportions reported as mean +/- std; vlm_calls/human are per-run counts as mean +/- std

| method | metric | mean | std | per-seed |
|---|---|---|---|---|
| rules | detect_fail | 0.386 | 0.087 | 0.45, 0.45, 0.36, 0.27 |
| rules | flag_all | 0.409 | 0.052 | 0.45, 0.45, 0.36, 0.36 |
| rules | flag_semantic | 0.107 | 0.071 | 0.14, 0.14, 0.00, 0.14 |
| rules | fpr_fail_ctrl | 0.000 | 0.000 | 0.00, 0.00, 0.00, 0.00 |
| rules | fpr_flag_ctrl | 0.000 | 0.000 | 0.00, 0.00, 0.00, 0.00 |
| rules | vlm_calls | 0.000 | 0.000 | 0.00, 0.00, 0.00, 0.00 |
| rules | human_sequences | 4.500 | 0.577 | 5, 5, 4, 4 |
| direct | detect_fail | 0.318 | 0.052 | 0.36, 0.27, 0.36, 0.27 |
| direct | flag_all | 0.614 | 0.045 | 0.64, 0.64, 0.64, 0.55 |
| direct | flag_semantic | 0.964 | 0.071 | 1.00, 1.00, 1.00, 0.86 |
| direct | fpr_fail_ctrl | 0.000 | 0.000 | 0.00, 0.00, 0.00, 0.00 |
| direct | fpr_flag_ctrl | 0.000 | 0.000 | 0.00, 0.00, 0.00, 0.00 |
| direct | vlm_calls | 250.000 | 4.320 | 250, 244, 252, 254 |
| direct | human_sequences | 6.750 | 0.500 | 7, 7, 7, 6 |
| layered | detect_fail | 0.614 | 0.087 | 0.73, 0.64, 0.55, 0.55 |
| layered | flag_all | 0.977 | 0.045 | 1.00, 1.00, 1.00, 0.91 |
| layered | flag_semantic | 1.000 | 0.000 | 1.00, 1.00, 1.00, 1.00 |
| layered | fpr_fail_ctrl | 0.000 | 0.000 | 0.00, 0.00, 0.00, 0.00 |
| layered | fpr_flag_ctrl | 0.000 | 0.000 | 0.00, 0.00, 0.00, 0.00 |
| layered | vlm_calls | 100.500 | 9.000 | 96, 90, 108, 108 |
| layered | human_sequences | 10.750 | 0.500 | 11, 11, 11, 10 |
