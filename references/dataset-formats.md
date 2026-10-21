# Dataset formats

Layouts are auto-detected per sequence by `dataset.discover_sequences`.

## LUART (unregistered, primary target)

```
<root>/
  <SequenceName>/            e.g. Car_238, Bicycle_26
    visible.txt              one "x,y,w,h" per line, 1920x1080 coords
    infrared.txt             one "x,y,w,h" per line, 640x512 coords
    NotAlign/
      visible/v001.jpg ...   unregistered modality pairs
      infrared/i001.jpg ...
```

- Line i of the annotation corresponds to the i-th image (sorted) of that
  modality. Frames are NOT registered across modalities: visible and
  infrared boxes for the same frame index live in different coordinate
  systems and must never be compared directly.
- `Align/` variant is auto-detected too (`dataset._find_img_dir` probes
  `NotAlign`, `Align`, bare modality dir).
- Sequence category is parsed from the name as the prefix before the final
  `_digits` (e.g. `Electricbike_150` -> `Electricbike`); it only feeds VLM
  prompt wording, never any threshold.

## M3Track (registered RGB-T, MOT ground truth)

```
<root>/
  train/<seq>/  test/<seq>/     e.g. 00029path_View2
    img1V_cut/*.jpg             visible frames
    img1T_cut/*.jpg             thermal frames (same 640x512 resolution)
    gt/gt.txt                   MOT csv: frame,id,x,y,w,h,conf,...
```

- M3Track is multi-object; the audit extracts the **dominant id** (most
  annotated frames) as the SOT trajectory and aligns boxes to frame index.
- One gt file serves both modalities (`Sequence.shared_gt`): the dual-copy
  and cross-modal comparison checks are disabled by design, per-modality
  geometry/trajectory checks stay active.
- MOT rows may legitimately miss frames (target absent), so
  `Sequence.relaxed_count` suppresses the frame-count mismatch check; an
  empty annotation line means "target absent", not a format error.
- Benchmark injections on M3Track use the gt-level subset of error types
  (`oob_box`, `cross_dup`, all semantic types); `drop_lines`/`garbage_line`
  /`dual_copy` are LUART-only because MOT parsing drops unparseable rows and
  the gt is shared.

## UAV-CB (unregistered, LabelMe per-frame JSON)

```
<root>/
  <category>/                   e.g. ground, cloud, building, powerline
    visible/<cat>_visible_<video>_<idx>.jpg + .json
    thermal/<cat>_thermal_<video>_<idx>.jpg + .json
```

- One sequence per `<category>_<video>` (e.g. `ground_1`); frames pair across
  modalities by index. Sequences are small (5 categories x 2 videos = 10).
- Each frame has its own LabelMe JSON; the first `rectangle` shape becomes
  the box, read from corner points (2- or 4-point both handled). An empty
  `shapes` array means "target absent".
- **Missing JSON is an error, not absence** (`MISSING_JSON:<file>` marker ->
  `format_invalid`): in a per-frame-file layout every image must have its
  annotation file; absence is represented by an empty shapes array. Invalid
  JSON similarly becomes `INVALID_JSON:<file>`.
- Modalities carry different resolutions (visible 1920x1080, thermal
  640x512) and different coordinates — genuinely unregistered, so the
  dual-copy and cross-sequence-duplicate checks stay active
  (`Sequence.per_frame_json` marks the layout).
- Frames are sparsely sampled video: single-frame displacements of 30-70%
  of the diagonal are natural, so the 6-sigma jump test correctly flags
  nothing and the drift detector only ranks frames. Expect most sequences
  to land in REVIEW from trajectory signals; tier-3 VLM checks (or humans)
  clear them. Raise `--spot-check` for denser verification.
- Benchmark injections on UAV-CB support the full LUART type set, operating
  on the per-frame JSONs; injected shifts pick the direction with more
  headroom and clamp to stay in-frame so geometry rules cannot shortcut
  detection.

## Porting other RGB-T datasets

Adapt `dataset.py` only; all other layers consume the `Sequence` interface
(`modalities[name] -> .frames, .boxes, .size, .frame_path(idx), .raw_lines`).

Common variants:

- Annotation as `groundtruth.txt` per modality dir: change the anno filename
  mapping in `discover_sequences`.
- LasHeR/GTOT style `<seq>/{v,i}/` image dirs: map `v -> visible`, `i ->
  infrared` in `_find_img_dir`.
- MOT-style `x,y,w,h` with extra columns: adjust `parse_box` (take first 4).
- Single-modality or missing infrared: layers degrade gracefully; the audit
  runs on whatever modalities are discovered.

Split-awareness: LUART train/test membership is not encoded in the layout.
If you need per-split reports, run with `--seqs-file` twice or filter the
output TSVs.

## Image size probing

`ModalityData.size` is read from the first image via PIL and used for
out-of-bounds checks. If images are unavailable, geometry checks are skipped
rather than guessed.
