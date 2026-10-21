# Multi-seed benchmark aggregate

- runs: 4 (seeds: bench, ms_luart_11, ms_luart_23, ms_luart_42)
- rates are per-seed proportions reported as mean +/- std; vlm_calls/human are per-run counts as mean +/- std

| method | metric | mean | std | per-seed |
|---|---|---|---|---|
| rules | detect_fail | 0.588 | 0.000 | 0.59, 0.59, 0.59, 0.59 |
| rules | flag_all | 0.603 | 0.029 | 0.59, 0.65, 0.59, 0.59 |
| rules | flag_semantic | 0.036 | 0.071 | 0.00, 0.14, 0.00, 0.00 |
| rules | fpr_fail_ctrl | 0.000 | 0.000 | 0.00, 0.00, 0.00, 0.00 |
| rules | fpr_flag_ctrl | 0.000 | 0.000 | 0.00, 0.00, 0.00, 0.00 |
| rules | vlm_calls | 0.000 | 0.000 | 0.00, 0.00, 0.00, 0.00 |
| rules | human_sequences | 10.250 | 0.500 | 10, 11, 10, 10 |
| direct | detect_fail | 0.250 | 0.056 | 0.29, 0.18, 0.29, 0.24 |
| direct | flag_all | 0.397 | 0.029 | 0.41, 0.41, 0.35, 0.41 |
| direct | flag_semantic | 0.964 | 0.071 | 1.00, 1.00, 0.86, 1.00 |
| direct | fpr_fail_ctrl | 0.000 | 0.000 | 0.00, 0.00, 0.00, 0.00 |
| direct | fpr_flag_ctrl | 0.000 | 0.000 | 0.00, 0.00, 0.00, 0.00 |
| direct | vlm_calls | 379.750 | 2.217 | 378, 383, 379, 379 |
| direct | human_sequences | 6.750 | 0.500 | 7, 7, 6, 7 |
| layered | detect_fail | 0.706 | 0.000 | 0.71, 0.71, 0.71, 0.71 |
| layered | flag_all | 1.000 | 0.000 | 1.00, 1.00, 1.00, 1.00 |
| layered | flag_semantic | 1.000 | 0.000 | 1.00, 1.00, 1.00, 1.00 |
| layered | fpr_fail_ctrl | 0.000 | 0.000 | 0.00, 0.00, 0.00, 0.00 |
| layered | fpr_flag_ctrl | 0.000 | 0.000 | 0.00, 0.00, 0.00, 0.00 |
| layered | vlm_calls | 123.000 | 1.155 | 124, 122, 122, 124 |
| layered | human_sequences | 17.000 | 0.000 | 17, 17, 17, 17 |
