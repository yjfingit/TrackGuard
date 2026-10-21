# Ablation study — tier-2 trajectory guidance and VLM budget

Sandbox: LUART test-clean, n=24, seed 7 (17 corrupt / 7 semantic / 7
control) and the same protocol at seed 11. Reproduce the two sandboxes with
`scripts/benchmark.py corrupt` (see README); commands below use `scripts/audit.py`. Mock VLM oracle
copied from the matching `layered` run into each ablation dir before
`--resume` re-run (oracle is normally written only by `benchmark.py run`).

Commands:

```
# oracle seed (per run dir)
for s in bench/layered/*/; do seq=$(basename $s)
  [ -f "$s/mock_bad_frames.json" ] && mkdir -p <out>/$seq && cp "$s/mock_bad_frames.json" <out>/$seq/
done
python3 scripts/audit.py --root <sandbox_seed7> --out <out> --method layered --allow-mock \
  [--no-traj | --max-samples 4|6|12]
# seed-11 replication
python3 scripts/audit.py --root <sandbox_seed11> --out ablation_seed11/notraj \
  --method layered --no-traj --allow-mock
```

## Seed 7 (<sandbox_seed7>)

| run | detect_FAIL | flag_all | flag_semantic | fp_FAIL_ctrl | vlm_calls | human_seq |
|---|---|---|---|---|---|---|
| rules (baseline) | 10/17 | 10/17 | 0/7 | 0/7 | 0 | 10 |
| direct (baseline) | 5/17 | 7/17 | 7/7 | 0/7 | 378 | 7 |
| direct ms=4 | 4/17 | 6/17 | 6/7 | 0/7 | 189 | 6 |
| layered ms=4 | 12/17 | 16/17 | 6/7 | 0/7 | 84 | 16 |
| layered ms=6 | 15/17 | 16/17 | 6/7 | 0/7 | 104 | 16 |
| **layered ms=8 (default, full)** | 12/17 | **17/17** | **7/7** | 0/7 | 124 | 17 |
| layered ms=12 | 13/17 | 17/17 | 7/7 | 0/7 | 162 | 17 |
| layered `--no-traj` | 12/17 | 16/17 | 6/7 | 0/7 | 84 | 16 |

## Seed 11 replication (`--no-traj` only)

| run | detect_FAIL | flag_all | flag_semantic | vlm_calls |
|---|---|---|---|---|
| layered full | 12/17 | 17/17 | 7/7 | 122 |
| layered `--no-traj` | 12/17 | 16/17 | 6/7 | 84 |
| direct | 3/17 | 7/17 | 7/7 | 383 |
| rules | 10/17 | 11/17 | 1/7 | 0 |

## Findings

1. **Tier-2 is what catches `switch_corner`.** Both seeds: removing the
   trajectory layer drops semantic 7/7 → 6/7, and in both cases the miss is
   exactly the corner-switch sequence (seed 7: Electricbike_281; seed 11:
   Pedestrian_231) — silently PASS. Mechanism: the swap occupies frames
   35–65%, the uniform spot-check grid (0/33/67/100%) lands entirely
   outside it; the trajectory layer marks the two corner discontinuities
   (real displacement, severity-ranked) and the VLM checks those frames.
   Structured periodic errors alias with uniform sampling; motion-guided
   adaptive sampling is the fix.
2. **Budget knee at ms=8 (default).** ms=4/6: 84–104 calls, 6/7 semantic;
   ms=8: 124 calls, 7/7 and flag 17/17; ms=12 adds 38 calls for nothing.
   Direct needs 378 calls for 7/7 semantic and still flags only 7/17
   overall (rules-invisible structural errors are not its job).
3. **FAIL vs REVIEW on corrupt sequences is threshold-noisy by design**
   (FAIL needs >=2 frames and >=50% BAD per modality) — `flag_all` is the
   stable operating metric; layered keeps it at 1.00 while direct sits at
   ~0.41.
4. `--no-traj` is *cheaper* (84 vs 124 calls) because tier-2 suspicion adds
   neighbor frames; the extra 40 calls buy the last semantic sequence. The
   trajectory layer is a recall instrument, not a cost instrument, at LUART
   motion statistics. On cheap-VLM deployments where a silent miss is
   tolerable, `--no-traj --max-samples 4` runs at ~22% of direct's cost
   with 6/7 semantic recall.
