# Error taxonomy

All codes emitted by the audit, grouped by source layer. Severity FAIL =
provable without human judgment; REVIEW = suspicious, needs human look.

## Tier 1: rules (deterministic, file-level)

| code | severity | meaning | typical fix |
|---|---|---|---|
| anno_file_missing | FAIL | `visible.txt`/`infrared.txt` absent | restore annotation file |
| anno_empty | FAIL | annotation file exists but empty | restore from source |
| images_missing | FAIL | modality image directory empty/absent | re-extract archive |
| format_invalid | FAIL | line not 4 comma/space-separated numbers; on per-frame-JSON layouts (UAV-CB) also a missing annotation file (`MISSING_JSON:`) or unparseable JSON (`INVALID_JSON:`) | fix or drop the line |
| frame_count_mismatch | FAIL | image count != annotation line count | re-align frames vs lines |
| bbox_invalid | FAIL | w<=0 or h<=0 | fix the line |
| bbox_out_of_bounds | FAIL | box entirely outside image | fix or drop the line |
| bbox_partial_oob | REVIEW | <50% of box area inside image | often legitimate edge target; verify |
| dual_annotation_copy | FAIL | visible and infrared annotations identical while resolutions differ | re-annotate one modality |
| dual_annotation_copy_same_res | REVIEW | identical annotations, same resolutions | verify whether cross-modal copy occurred |
| cross_sequence_duplicate | FAIL | annotation content identical to another sequence | re-annotate the borrowed sequence |

Real-data anchor points (LUART): `Car_0`/`Car_129` share byte-identical
visible annotations; `Bicycle_26` has 967 IR frames but 966 lines; `Car_238`
has 540 visible frames but 547 lines.

## Tier 2: trajectory (per modality, temporal)

| code | meaning |
|---|---|
| traj_jump | consecutive-frame displacement >6% diag/frame AND IoU<0.05 AND >6 robust-sigma above median |
| traj_frozen | identical box for >max(30, 15% of frames) consecutive frames |
| traj_size_anomaly | >8x area change between neighbors with IoU<0.1 |
| traj_drift | >10% of frames deviate >5% diag from median-filtered trajectory |

These emit suspicious frames (used for tier-3 sampling) and drive REVIEW when
VLM is unavailable or checked too few frames. None is FAIL by itself —
fast motion, static targets and scale changes legitimately occur.

## Tier 3: VLM (semantic, per modality)

| code | meaning |
|---|---|
| vlm_box_mismatch | >=2 sampled frames BAD in one modality and >=50% of that modality's checks -> FAIL |
| vlm_suspect | some BAD/UNCERTAIN but below FAIL threshold -> REVIEW |

Per-frame verdicts GOOD/BAD/UNCERTAIN with reasons are stored in
`report.json` (`vlm.frames`) with the evidence panel path for each.

## Verdict composition

- any tier-1 FAIL issue -> FAIL (skip tiers 2-3)
- vlm_box_mismatch -> FAIL; vlm_suspect -> REVIEW
- trajectory signals with <3 VLM checks, or no VLM -> REVIEW
- otherwise PASS

## Benchmark injection types

Structural (tier-1 detectable): `drop_lines`, `garbage_line`, `oob_box`,
`dual_copy`, `cross_dup`.
Semantic (need tier-2/3): `offset_box` (segment shifted +30% width),
`drift_box` (linear drift to +25% width), `segment_cross_copy` (IR boxes
pasted into visible annotation - unregistered-specific, invisible to rules),
`switch_corner` (box relocated to another image region), `stale_segment`
(box frozen mid-sequence while target moves).
`control` entries receive no corruption and measure false positives.
