# TrackGuard

**Training-free, model-agnostic annotation auditing for RGB-T (visible + thermal/infrared)
single-object tracking datasets — built for *unregistered* (non-aligned) data.**

RGB-T trackers are only as good as their annotations. On unregistered RGB-T data the two
modalities live in different coordinate systems, so the standard single-modality QA tricks
(compare boxes across streams, warp one into the other) simply do not apply. TrackGuard
audits each modality on its own terms and still finds the errors that matter:

missing/malformed annotations, out-of-bounds boxes, dual-modality copies, cross-sequence
copies, box offsets, trajectory drift, frozen/stale segments, corner switches.

Nothing is trained or fine-tuned. Any OpenAI-compatible vision endpoint works
(GPT-4o/Qwen2.5-VL/DeepSeek-VL/MiniMax/GLM-4V/local vLLM), and the whole first two tiers
run with zero VLM calls.

---

## The three tiers

| tier | what it does | cost |
|---|---|---|
| **1. rules** | deterministic format/geometry checks, per modality | O(1) ms/sequence, no model |
| **2. trajectory** | per-modality temporal analysis: jumps, frozen boxes, size anomalies, drift residual; emits *severity-ranked* suspicious frames | no model |
| **3. VLM** | adaptive frame sampling (spot checks + suspicion neighbourhoods), local zoom + context crops, per-frame GOOD/BAD/UNCERTAIN verdicts | 1 call per checked frame |

Verdicts are **PASS / FAIL / REVIEW** per sequence, always with error frames, error codes,
and rendered visual evidence panels. REVIEW exists on purpose: a VLM alone does not get to
convict a sequence, so tier 3 needs corroboration (`>=2` BAD frames and `>=50%` of a
modality's checks) to escalate to FAIL.

### The unregistered contract

The two modalities are audited **independently**. Boxes from `visible.txt` and
`infrared.txt` are never compared numerically, never IoU'd, never warped into each other's
frame. Every VLM query is single-modality (anchor frame + sampled frame of the *same*
modality), so no hidden alignment is assumed anywhere in the pipeline.

---

## Quick start — runs on synthetic data, no downloads

```bash
pip install -r requirements.txt

# 1. generate a clean, unregistered dual-modality dataset (14 sequences)
python3 scripts/make_demo.py --out demo --n 14 --seed 7

# 2. tier 1+2 only: a correct auditor must pass every clean sequence
python3 scripts/audit.py --root demo --out demo_audit --method rules
wc -l < demo_audit/sequences_pass.txt          # -> 14

# 3. inject known errors and compare the three baselines
python3 scripts/benchmark.py bench --root demo --out demo_bench \
    --clean-file demo_audit/sequences_pass.txt --n 12 --seed 7 --vlm-model mock
cat demo_bench/benchmark_report.md
```

On the shipped demo (`results/demo/metrics.tsv`): rules flags 7/8 injected errors
(1/2 semantic), direct-VLM flags 2/8 (2/2 semantic, 192 frame checks), layered TrackGuard
flags **8/8** (**2/2** semantic) with **30** frame checks and **0** false positives on the
control group.

## Auditing a real dataset

```bash
# structure pass first (fast, free, catches the provable errors)
python3 scripts/audit.py --root /path/to/LUART --out out_rules --method rules

# full layered audit with a real VLM
export TRACKGUARD_VLM_MODEL=qwen2.5-vl-72b-instruct
export TRACKGUARD_VLM_BASE_URL=https://<your-endpoint>/compatible-mode/v1
export TRACKGUARD_VLM_API_KEY=sk-...
python3 scripts/audit.py --root /path/to/LUART --out out_full

# subset / smoke test
python3 scripts/audit.py --root /path/to/LUART --out out_smoke --pattern '^(Car|Bus)_' --limit 20
```

Outputs per run: `report.json`, `report.md`, `errors.tsv`,
`sequences_{pass,fail,review}.txt`, and `evidence/` panels per audited frame
(annotated frame + tight zoom + wide context crop, box drawn).
`sequences_pass.txt` is directly reusable as the clean list for `benchmark.py`.

Supported layouts (auto-detected per sequence — see
[references/dataset-formats.md](references/dataset-formats.md)):

- **LUART** — unregistered `<seq>/{visible.txt,infrared.txt}` + `NotAlign/<mod>/images`
- **M3Track** — registered MOT-style `train|test/<seq>/{img1V_cut,img1T_cut,gt/gt.txt}`
- **UAV-CB** — per-frame LabelMe JSONs `<cat>_<mod>_<video>_<idx>.{jpg,json}`
- Anything else: adapt `scripts/dataset.py` only; every other layer consumes the
  `Sequence` interface.

---

## Results

Protocol: take audited-clean sequences, inject **one known error per sequence**
(5 structural + 5 semantic types, 70% corrupted / 30% control), and compare
`rules` (tier 1 only) vs `direct` (VLM on random frames, no rules/trajectory) vs
`layered` (full pipeline). 4 seeds per dataset, mean ± std via
`scripts/benchmark.py aggregate`. Full tables in [`results/`](results/).

| dataset (n) | rules flag | direct flag | **layered flag** | layered semantic | layered VLM calls | direct VLM calls | control FP |
|---|---|---|---|---|---|---|---|
| LUART (24) | 0.60 ± 0.03 | 0.40 ± 0.03 | **1.00 ± 0.00** | **1.00 ± 0.00** | 123 ± 1 | 380 ± 2 | 0.000 |
| M3Track (16) | 0.41 ± 0.05 | 0.61 ± 0.04 | **0.98 ± 0.04** | **1.00 ± 0.00** | 101 ± 9 | 250 ± 4 | 0.000 |
| UAV-CB (10) | 0.57 ± 0.00 | 0.43 ± 0.00 | **1.00 ± 0.00** | **1.00 ± 0.00** | 84 ± 0 | 159 ± 0 | 0.000 |

**Zero control false positives across all 12 runs** (every method, dataset, and seed).
TrackGuard reaches the semantic recall of direct VLM auditing at **~1/3 of the VLM cost**,
while catching the structural errors a VLM-only pass misses entirely (rules-invisible to
`direct`: it flags only 7/17 corrupted sequences at 378 frame checks).

### Ablation: tier 2 is what catches `switch_corner`

| run | semantic flag | flag_all | VLM calls |
|---|---|---|---|
| layered (max_samples 4) | 6/7 | 16/17 | 84 |
| layered (max_samples 6) | 6/7 | 16/17 | 104 |
| **layered (max_samples 8, default)** | **7/7** | **17/17** | **124** |
| layered (max_samples 12) | 7/7 | 17/17 | 162 |
| layered `--no-traj` | 6/7 | 16/17 | 84 |
| direct | 7/7 | 7/17 | 378 |

Removing trajectory guidance (`--no-traj`) drops semantic recall 7/7 → 6/7 **on both tested
seeds**, and in both cases the miss is exactly the corner-switch sequence — silently `PASS`.
Mechanism: the swapped segment occupies frames 35–65%, so the uniform spot-check grid
(0/33/67/100%) lands entirely outside it. **Structured periodic errors alias with uniform
sampling; motion-guided adaptive sampling is the fix.** The budget knee is `max_samples=8`:
12 more frames buys nothing, 4 fewer frames costs a whole error class.
Details: [results/ablation_summary.md](results/ablation_summary.md).

### Prompt validation with a real VLM

A balanced 15-frame-pair set across all three datasets (tier-3 prompt verbatim, `ref` mode)
scored **15/15** — sensitivity 7/7 on injected offset / drift / stale-freeze / corner-switch /
cross-modal-copy, specificity 8/8 on clean controls and natural-jitter frames. Tier 3 both
catches real errors and clears false trajectory suspicion on dynamic targets.
Records: [results/real_vlm_validation.json](results/real_vlm_validation.json).

### Honest scope

- Benchmark tables above use the **mock oracle** VLM (`--vlm-model mock`), which stands in
  for a perfect tier-3 judge so the numbers isolate *pipeline logic* (sampling, aggregation,
  cost) rather than model quality. Real multi-VLM numbers are the remaining step; the
  harness is ready: `benchmark.py run --vlm-model <X>` into per-model dirs, then
  `evaluate` / `aggregate`.
- `vlm_calls` is a frame-count cost proxy, not dollars or latency.
- Real-VLM validation so far is one vision-capable judge (15/15), not a model matrix.
- Datasets are **not** included; you must obtain LUART / M3Track / UAV-CB licenses yourself.

### Real-data sanity checks

- Full LUART (1453 sequences): reproduces 372/372 manually confirmed A-tier errors in ~10 s
  and surfaces additional ones the manual audit missed (byte-identical infrared between
  `Car_124` and the corrupted `Car_382`; `Pedestrian_20` infrared boxes fully below the
  512 px image).
- Full M3Track (98 sequences): genuine problems, e.g. `00229librarydark_View1` tracks a
  target outside the frame for 1179/1201 frames.
- Full UAV-CB (10 sequences): zero rule errors — correct, its JSONs are complete — and its
  erratic drone motion (natural per-frame displacement p50 = 0.54 of the diagonal) lands
  sequences in REVIEW until a VLM clears them, which is the intended behaviour.

---

## Command reference

```
scripts/audit.py      --root R --out O [--method layered|rules|direct] [--pattern RE] [--seqs-file F]
                      [--limit N] [--no-vlm] [--no-traj] [--spot-check K] [--max-samples M]
                      [--direct-frames K] [--vlm-mode ref|solo] [--resume] [--allow-mock]
                      [--vlm-model X --vlm-base-url U --vlm-api-key K | env TRACKGUARD_VLM_*]

scripts/benchmark.py  bench     --root R --out O --clean-file F [--n 24] [--seed 7] [--vlm-model mock]
                    corrupt|run|evaluate|aggregate        (reuse a sandbox, or pool seeds)

scripts/make_demo.py  --out D [--n 14] [--seed 7]         (synthetic unregistered RGB-T data)
```

Ablation flags (`--no-traj`, `--max-samples`, `--spot-check`, `--method rules|direct`) are
first-class: they exist so the contribution of each tier is measurable, not just asserted.

## Repository layout

```
SKILL.md                  agent-skill entry point (workflow + contracts + reference numbers)
scripts/
  audit.py                CLI: tier1 -> tier2 -> tier3
  dataset.py             layout discovery/loading (the ONLY file to touch for new datasets)
  rules_layer.py          tier 1 — deterministic checks
  trajectory_layer.py     tier 2 — per-modality temporal signals, severity ranking
  vlm_layer.py            tier 3 — adaptive sampling, evidence crops, verdict aggregation
  vlm_client.py           model-agnostic OpenAI-compatible client (retries, JPEG payloads)
  evidence.py             panel rendering      report.py — PASS/FAIL/REVIEW outputs
  benchmark.py            injection benchmark + baselines + multi-seed aggregation
  make_demo.py            synthetic dataset generator (reproducible demo)
references/
  error-taxonomy.md       every error code, severity, and recommended fix
  dataset-formats.md      LUART/M3Track/UAV-CB layouts and porting guide
  vlm-prompts.md          prompt design, ref-vs-solo mode, aggregation rules
results/                  metrics/aggregate tables, ablation summary, real-VLM validation
```

## Use as a Qoder agent skill

```bash
mkdir -p ~/.qoder/skills && cp -r . ~/.qoder/skills/trackguard   # or symlink
```

Then invoke `/trackguard --root /path/to/LUART`; `SKILL.md` carries the full workflow the
agent follows.

## Citation

If you use TrackGuard in published work, please cite:

```bibtex
@software{trackguard2026,
  title  = {TrackGuard: Training-Free Layered Annotation Auditing for
            Unregistered RGB-T Tracking Datasets},
  author = {yjfingit},
  year   = {2026},
  url    = {https://github.com/yjfingit/TrackGuard}
}
```

MIT licensed. Contributions welcome — new dataset adapters belong in `scripts/dataset.py`.
