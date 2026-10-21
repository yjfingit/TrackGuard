---
name: trackguard
description: Training-free, model-agnostic audit and cleaning of RGB-T single-object tracking dataset annotations, auto-detecting unregistered (non-aligned) layouts such as LUART and registered MOT-style layouts such as M3Track. Use when asked to check, audit, clean, or quality-assess RGB-T/RGB-IR tracking dataset annotations, find annotation errors (missing/malformed/out-of-bounds boxes, dual-modality copy, cross-sequence duplicates, box offset, trajectory drift, frozen/stale boxes), or benchmark data-cleaning pipelines against rules-only or VLM-only baselines. Outputs PASS/FAIL/REVIEW verdicts per sequence with error frames, error codes, and visual evidence panels.
---

# TrackGuard — RGB-T Dataset Audit

Training-free, layered auditing of unregistered RGB-T tracking annotations:
deterministic rules -> per-modality trajectory analysis -> VLM verification
with adaptive sampling and local zoom crops. No training, no fine-tuning,
any OpenAI-compatible VLM works (GPT, Qwen-VL, DeepSeek-VL, MiniMax, GLM-4V,
local vLLM).

## Quick start

```bash
S=~/.qoder/skills/trackguard/scripts   # or <repo>/scripts when used standalone

# 0) no dataset at hand? generate a synthetic unregistered RGB-T dataset (all clean)
python3 $S/make_demo.py --out demo --n 14 --seed 7

# 1) fast structural pass (no VLM, seconds per 100 sequences)
python3 $S/audit.py --root /path/to/LUART --out out_rules --method rules

# 2) full layered audit (rules + trajectory + VLM)
TRACKGUARD_VLM_MODEL=qwen2.5-vl-72b-instruct \
TRACKGUARD_VLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1 \
TRACKGUARD_VLM_API_KEY=sk-xxx \
python3 $S/audit.py --root /path/to/LUART --out out_full

# 3) research benchmark vs rules-only / VLM-only baselines
python3 $S/benchmark.py bench --root /path/to/LUART --out bench_out \
    --clean-file sequences_pass.txt --n 24 --seed 7
```

Dataset layouts are auto-detected per sequence (see
`references/dataset-formats.md`):

- **LUART** (unregistered, primary target): `<root>/<seq>/` with
  `visible.txt`, `infrared.txt` (one `x,y,w,h` per line) and
  `NotAlign/{visible,infrared}/*.jpg`.
- **M3Track** (registered RGB-T, MOT gt): `<root>/{train,test}/<seq>/` with
  `img1V_cut/`, `img1T_cut/` and `gt/gt.txt` (`frame,id,x,y,w,h,...` csv).
  The dominant target id becomes the SOT trajectory; shared-GT and
  MOT-style missing rows are handled natively (no false frame-count or
  dual-copy errors).
- **UAV-CB** (unregistered, LabelMe per-frame JSON): `<root>/<category>/
  {visible,thermal}/<cat>_<mod>_<video>_<idx>.{jpg,json}`. One sequence per
  `<category>_<video>`; modalities carry different resolutions (e.g.
  1920x1080 vs 640x512) so the dual-copy check stays active. Frames are
  sparsely sampled video, so per-frame displacements are naturally huge —
  the trajectory layer flags nothing as jumps and tier 3 carries the load.

## How it works

Three layers; each layer only handles what it can prove:

1. **Rules (tier 1)** — file-level provable errors, verdict FAIL with no human
   judgment needed: annotation file missing/empty, malformed lines,
   frame-count mismatch, non-positive boxes, boxes out of bounds, dual-modality
   annotation copy (identical visible/infrared annotations with different
   resolutions), cross-sequence duplicate annotations.
2. **Trajectory (tier 2)** — per-modality temporal analysis: abrupt motion
   jumps (displacement + IoU with robust sigma), frozen-box runs, >8x size
   anomalies, drift vs median-filtered trajectory. Produces suspicious frames
   that steer tier-3 sampling.
3. **VLM (tier 3)** — semantic verification of sampled frames via a
   model-agnostic OpenAI-compatible client: anchor-frame reference comparison
   (`--vlm-mode ref`) or single-frame judgment (`--vlm-mode solo`). Each query
   image contains ONE modality only; each modality is judged independently.

**Unregistered-RGB-T contract**: cross-modal box coordinates are never
compared. Every check runs per modality; VLM verdicts aggregate per modality
so one corrupted modality cannot be diluted by the healthy one.

**Adaptive sampling** (`--spot-check`, `--max-samples`): suspicious frames
are ranked by severity (jump displacement / drift residual), sampled with
context neighbors, and evenly spaced spot checks are never dropped — so
natural motion jitter on dynamic targets cannot crowd out coverage. Rule-FAIL
sequences skip VLM entirely (cost control).

## Verdicts and outputs

Per sequence: **PASS** clean, **FAIL** provable error, **REVIEW** needs human
look (suspicious signals, partial VLM disagreement, VLM errors). Written to
`--out`:

- `report.md` — human-readable report: summary, error-code distribution,
  cost stats, per-sequence findings with evidence panel paths
- `report.json` — machine-readable full records
- `errors.tsv` — one row per finding (sequence, source layer, code, frames)
- `sequences_{pass,fail,review}.txt` — clean lists for downstream use
- `<seq>/evidence/*.png` — panels: annotated full frame + tight zoom + wide
  context crop per audited frame
- `checkpoint.jsonl` — per-sequence records; rerun with `--resume` to continue

Error codes are documented in `references/error-taxonomy.md`.

## VLM configuration

Env vars (or CLI flags `--vlm-model/--vlm-base-url/--vlm-api-key/--vlm-proxy`):

| Provider | TRACKGUARD_VLM_BASE_URL | example model |
|---|---|---|
| OpenAI | https://api.openai.com/v1 | gpt-4o / gpt-4.1 |
| Qwen-VL (DashScope compat) | https://dashscope.aliyuncs.com/compatible-mode/v1 | qwen2.5-vl-72b-instruct |
| DeepSeek | https://api.deepseek.com/v1 | deepseek-vl2 (vision endpoint) |
| MiniMax | https://api.minimax.chat/v1 | abab6.5s-chat (VL-capable) |
| local vLLM/SGLang | http://localhost:8000/v1 | Qwen2.5-VL-7B-Instruct |

`TRACKGUARD_VLM_PROXY`: URL, `direct` (bypass env proxy), or `env` (default).
Without a configured VLM the audit degrades to rules+trajectory with a
warning — never fails silently.

Model `mock` (`--allow-mock`) runs the full pipeline offline: verdicts come
from `<out>/<seq>/mock_bad_frames.json` oracle files, default GOOD. Use only
for plumbing tests; any reported accuracy must come from a real VLM.

## Benchmark protocol

`benchmark.py` builds a sandbox from audited-clean sequences, injects one
known error per sequence (7 structural + 5 semantic types incl.
cross-modality segment copy, corner switch, stale freeze; 30% control group),
then compares `rules` vs `direct` (VLM on random frames) vs `layered`:

- `detect_fail` — injected errors proven FAIL
- `flag_all` — FAIL or REVIEW (surfaced for human review)
- `flag_semantic` — semantic-only subset (rules should score 0 here)
- `fpr_*_ctrl` — control sequences wrongly flagged
- `vlm_calls` — frames checked (cost proxy), `human_sequences` — review load

Reference runs (mock oracle as the semantic VLM stand-in; re-run with a real
VLM for paper numbers; 4 seeds per dataset, mean±std via `aggregate`):

- LUART test-clean, n=24: rules flag 0.60±0.03 (semantic 0.04±0.07, 0
  calls); direct flag 0.40±0.03 (semantic 0.96±0.07, 380±2 calls);
  layered flag **1.00±0.00** (semantic **1.00±0.00**, proven-FAIL 0.71,
  123±1 calls).
- M3Track (transfer, no code changes), n=16: rules flag 0.41±0.05
  (semantic 0.11±0.07); direct flag 0.61±0.04 (semantic 0.96±0.07,
  250±4 calls); layered flag **0.98±0.04** (semantic **1.00±0.00**,
  proven-FAIL 0.61, 100±9 calls).
- UAV-CB (hard regime: sparse frames, erratic drone motion), n=10: rules
  flag 0.57±0.00 (semantic 0.00); direct flag 0.43±0.00 (semantic 1.00,
  159 calls); layered flag **1.00±0.00** (semantic **1.00±0.00**,
  proven-FAIL 0.71±0.12, 84 calls).
- Control false positives: **0.000 across all 12 runs** (every method,
  every dataset, every seed).

Ablations (LUART sandbox, `--no-traj` and `--max-samples` sweeps; mock
oracle, so numbers isolate pipeline logic):

- Tier-2 trajectory guidance is what catches `switch_corner`: removing it
  (`--no-traj`) drops semantic flag 7/7 → 6/7 on **both** tested seeds —
  in each case exactly the corner-switch sequence silently passes. The
  swapped 35–65% segment aliases with the uniform spot-check grid
  (0/33/67/100% all land outside it); only the motion-discontinuity marks
  at the two corners route the VLM there.
- Budget sweep (`--max-samples`, seed 7): 4 → 84 calls, semantic 6/7;
  6 → 104, 6/7; **8 (default) → 124, 7/7, flag 17/17**; 12 → 162, 7/7.
  ms=8 is the knee; direct needs 378 calls for semantic 7/7 and still
  flags only 7/17 overall.

Validation on real data: full LUART (1453 sequences) reproduces 372/372
manually-confirmed A-tier errors in ~10s and additionally finds errors the
manual audit missed (Car_124 infrared byte-identical to the corrupted
Car_382; Pedestrian_20 infrared frames 418-419 fully below the 512px image).
Full M3Track (98 sequences) finds genuine annotation problems: e.g.
00229librarydark_View1 tracks a target that sits fully outside the 640x512
frame for 1179/1201 frames. Full UAV-CB (10 sequences) reports zero rule
errors — correct, its per-frame JSONs are complete — while its erratic
drone motion lands every sequence in REVIEW until a VLM clears it. A
real-VLM validation batch (vision-capable judge, ref mode, tier-3 prompt
verbatim, 15 frame pairs balanced across all three datasets) scored
15/15: sensitivity 7/7 on injected offset / drift / stale-freeze /
corner-switch / cross-modal-copy errors, specificity 8/8 on clean control
and natural-jitter frames — tier 3 both catches real errors and clears
false trajectory suspicion on dynamic targets.

Subcommands `corrupt` / `run` / `evaluate` allow reusing a sandbox or scoring
existing outputs. `aggregate` computes mean+/-std rates across multiple runs
(seeds) — run `bench` once per seed, then
`aggregate --bench run1,run2,run3` for paper tables. Multiple VLMs: repeat
`run` with different `--vlm-model` into separate output dirs, then `evaluate`.

## Notes

- Rule-FAIL short-circuits tiers 2-3 for that sequence (provable error;
  VLM adds nothing). To still collect VLM evidence, use `--method direct`.
- `traj_frozen` can be a legitimately static target; it yields REVIEW, not
  FAIL. Same for `bbox_partial_oob` (edge targets).
- Sequence names must be unique; category is derived as the `<Class>_<id>`
  prefix and only feeds the VLM prompt wording.

## Resources

- `scripts/audit.py` — main CLI (`--method layered|rules|direct`,
  `--no-traj`, `--max-samples`, `--spot-check`)
- `scripts/benchmark.py` — injection benchmark
  (`corrupt|run|evaluate|bench|aggregate`)
- `scripts/make_demo.py` — synthetic unregistered RGB-T dataset for demos and
  regression checks (all sequences clean by construction)
- `scripts/dataset.py` — discovery/loading (adapt here for new layouts)
- `scripts/rules_layer.py`, `scripts/trajectory_layer.py`,
  `scripts/vlm_layer.py` — the three tiers
- `scripts/vlm_client.py` — model-agnostic VLM client
- `scripts/evidence.py` — evidence panel rendering; `scripts/report.py` — reports
- `references/error-taxonomy.md` — all error codes, severities, fixes
- `references/dataset-formats.md` — LUART layout and porting guide
- `references/vlm-prompts.md` — prompt design and rationale
