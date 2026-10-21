#!/usr/bin/env python3
"""Research benchmark: detection rate / false-positive rate / review cost.

Protocol: take audited-clean (PASS) sequences, inject one known error type per
sequence into a sandbox (symlinked images + modified annotations), keep a
control group untouched, then run three methods and score them against the
injection manifest:

  rules    audit --method rules      (tier 1 only)
  direct   audit --method direct     (VLM on K random frames, no rules/traj)
  layered  audit --method layered    (the full skill: rules -> traj -> VLM)

With the mock VLM the "semantic" detections (offset/drift) come from the
oracle file mock_bad_frames.json, which only exercises the pipeline; for
paper numbers re-run with a real VLM (TRACKGUARD_VLM_* env vars).

Subcommands:
  python benchmark.py corrupt --root LUART --out sandbox --n 12 --seed 1 \
      [--clean-file sequences_pass.txt] [--types t1,t2,...]
  python benchmark.py run --sandbox sandbox --out bench_out [--vlm-model mock]
  python benchmark.py evaluate --sandbox sandbox --bench bench_out
  python benchmark.py bench --root LUART --out bench_all --n 12 --seed 1   (all-in-one)
  python benchmark.py aggregate --bench run1,run2,run3 --out agg.md        (mean+/-std)
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT = os.path.join(HERE, "audit.py")

TYPES = ["drop_lines", "garbage_line", "oob_box", "dual_copy",
         "offset_box", "drift_box", "segment_cross_copy", "switch_corner",
         "stale_segment", "cross_dup"]

SEMANTIC_TYPES = ("offset_box", "drift_box", "segment_cross_copy",
                  "switch_corner", "stale_segment")

# injection types meaningful per layout: LUART has separate per-modality
# annotation files; M3Track (shared MOT gt) can only express gt-level edits;
# UAV-CB per-frame JSONs express everything LUART does
LUART_TYPES = ("drop_lines", "garbage_line", "oob_box", "dual_copy",
               "cross_dup") + SEMANTIC_TYPES
M3TRACK_TYPES = ("oob_box", "cross_dup") + SEMANTIC_TYPES
UAVCB_TYPES = LUART_TYPES


def _is_m3track(seq):
    return getattr(seq, "shared_gt", False) and getattr(seq, "relaxed_count", False)


def _is_uavcb(seq):
    return getattr(seq, "per_frame_json", False)


def _plan_types(n_corrupt, layout_types):
    """Type plan: single-shot types cycled, plus one cross_dup pair at the
    end (only when the budget still leaves room for semantic singles)."""
    singles = [t for t in layout_types if t != "cross_dup"]
    fill = n_corrupt - 2 if n_corrupt >= 8 else n_corrupt
    plan = []
    while len(plan) < max(0, fill):
        plan.append(singles[len(plan) % len(singles)])
    if n_corrupt >= 8:
        plan += ["cross_dup", "cross_dup"]
    return plan[:n_corrupt]


def _corrupt_m3track_gt(gt_path, t, W, H):
    """Apply one corruption to the dominant-id rows of a MOT gt file.

    Returns (detail, bad_idx) where bad_idx are 0-based indices into the
    aligned box list (frame - 1) of semantically-bad frames.
    """
    from collections import Counter
    lines = open(gt_path).read().splitlines()
    rows = []  # (line_no, frame, x, y, w, h)
    for i, l in enumerate(lines):
        s = l.replace(",", " ").split()
        if len(s) < 6:
            continue
        try:
            f, tid = int(float(s[0])), int(float(s[1]))
            x, y, w, h = (float(v) for v in s[2:6])
        except ValueError:
            continue
        rows.append((i, f, tid, x, y, w, h))
    tid = Counter(r[2] for r in rows).most_common(1)[0][0]
    rows = [r for r in rows if r[2] == tid]
    n = len(rows)
    bad_idx = []

    def setrow(r, x, y, w, h):
        lines[r[0]] = f"{r[1]},{tid},{x:.0f},{y:.0f},{w:.0f},{h:.0f},1,1,1"

    if t == "oob_box":
        r = rows[n // 2]
        setrow(r, r[3] - 2000, r[4] - 2000, r[5], r[6])
        detail = f"dominant target frame {r[1]} box moved fully out of bounds"
    else:
        if t == "offset_box":
            k1, k2 = int(0.3 * n), int(0.5 * n)
        elif t == "segment_cross_copy":
            k1, k2 = int(0.4 * n), int(0.7 * n)
        elif t == "switch_corner":
            k1, k2 = int(0.35 * n), int(0.65 * n)
        else:  # drift_box / stale_segment
            k1, k2 = int(0.5 * n), n
        r0 = rows[min(k1, n - 1)]
        dx = (350 if r0[3] < W / 2 else -350)
        dy = (220 if r0[4] < H / 2 else -220)
        for k in range(k1, min(k2, n)):
            r = rows[k]
            if t == "offset_box":
                setrow(r, r[3] + 0.3 * W, r[4], r[5], r[6])
            elif t == "drift_box":
                ramp = (k - k1) / max(1, k2 - k1)
                setrow(r, r[3] + ramp * 0.25 * W, r[4], r[5], r[6])
            elif t == "switch_corner":
                setrow(r, max(0.0, min(W - r[5], r[3] + dx)),
                       max(0.0, min(H - r[6], r[4] + dy)), r[5], r[6])
            elif t == "stale_segment":
                setrow(r, r0[3], r0[4], r0[5], r0[6])
            else:  # segment_cross_copy: shift by an IR-vs-RGB scale factor
                setrow(r, r[3] * 0.33, r[4] * 0.4, r[5] * 0.33, r[6] * 0.4)
            if t in SEMANTIC_TYPES:
                bad_idx.append(r[1] - 1)
        detail = f"dominant target (id {tid}) rows {k1}-{min(k2, n)} corrupted ({t})"
    with open(gt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return detail, bad_idx


def _uavcb_json_path(img_dir, frame):
    return os.path.join(img_dir, os.path.splitext(frame)[0] + ".json")


def _uavcb_read_box(img_dir, frame):
    with open(_uavcb_json_path(img_dir, frame)) as f:
        data = json.load(f)
    for s in data.get("shapes", []):
        if s.get("shape_type") == "rectangle" and s.get("points"):
            xs = [p[0] for p in s["points"]]
            ys = [p[1] for p in s["points"]]
            return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
    return None


def _uavcb_write_box(img_dir, frame, box):
    p = _uavcb_json_path(img_dir, frame)
    with open(p) as f:
        data = json.load(f)
    x, y, w, h = box
    rect = next((s for s in data.get("shapes", [])
                 if s.get("shape_type") == "rectangle"), None)
    if rect is None:
        rect = {"label": "target", "points": [], "shape_type": "rectangle",
                "group_id": None, "flags": {}}
        data.setdefault("shapes", []).append(rect)
    rect["points"] = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
    with open(p, "w") as f:
        json.dump(data, f, indent=1)


def _read_boxes(path):
    with open(path) as f:
        lines = f.read().splitlines()
    return lines


def _write_boxes(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _parse(line):
    return [float(v) for v in line.strip().replace(",", " ").split()]


def _fmt(b):
    return ",".join(str(int(round(v))) for v in b)


def cmd_corrupt(args):
    sys.path.insert(0, HERE)
    from dataset import discover_sequences
    seqs = discover_sequences(args.root, seqs_file=args.clean_file)
    if not seqs:
        print("no sequences found", file=sys.stderr)
        return 2
    rng = random.Random(args.seed)
    rng.shuffle(seqs)
    n_corrupt = max(3, int(round(args.n * 0.7)))
    chosen = seqs[:args.n]
    if _is_m3track(seqs[0]):
        layout_types = M3TRACK_TYPES
    elif _is_uavcb(seqs[0]):
        layout_types = UAVCB_TYPES
    else:
        layout_types = LUART_TYPES
    plan = _plan_types(n_corrupt, layout_types)
    os.makedirs(args.sandbox, exist_ok=True)
    manifest = {"generated": time.strftime("%F %T"), "root": args.root,
                "seed": args.seed, "entries": []}

    def link_seq(seq):
        d = os.path.join(args.sandbox, seq.name)
        os.makedirs(d, exist_ok=True)
        if _is_uavcb(seq):
            # sandbox keeps the category dir structure; JSONs are copied so
            # injections never touch the source dataset, images stay symlinks
            for mname, m in seq.modalities.items():
                md = os.path.join(args.sandbox, seq.category, mname)
                os.makedirs(md, exist_ok=True)
                for fn in m.frames:
                    dst = os.path.join(md, fn)
                    if not os.path.exists(dst):
                        os.symlink(os.path.abspath(os.path.join(m.img_dir, fn)), dst)
                    jsrc = _uavcb_json_path(m.img_dir, fn)
                    if os.path.exists(jsrc):
                        shutil.copy(jsrc, os.path.join(md, os.path.basename(jsrc)))
            return
        if _is_m3track(seq):
            for entry in os.listdir(seq.root):
                src = os.path.join(seq.root, entry)
                if entry == "gt":
                    os.makedirs(os.path.join(d, "gt"), exist_ok=True)
                    shutil.copy(os.path.join(src, "gt.txt"),
                                os.path.join(d, "gt", "gt.txt"))
                elif os.path.isdir(src) and not os.path.exists(os.path.join(d, entry)):
                    os.symlink(os.path.abspath(src), os.path.join(d, entry))
            return
        for mname, m in seq.modalities.items():
            if m.img_dir:
                dst = os.path.join(d, os.path.basename(
                    os.path.dirname(m.img_dir)) or ".", mname)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if not os.path.exists(dst):
                    os.symlink(os.path.abspath(m.img_dir), dst)
            if m.anno_path:
                shutil.copy(m.anno_path, os.path.join(d, mname + ".txt"))

    cross_first = None
    for i, seq in enumerate(chosen):
        link_seq(seq)
        if i >= len(plan):
            manifest["entries"].append({"seq": seq.name, "type": "control"})
            continue
        t = plan[i]
        entry = {"seq": seq.name, "type": t, "detail": ""}

        if _is_m3track(seq):
            gt = os.path.join(args.sandbox, seq.name, "gt", "gt.txt")
            m = seq.modalities.get("visible")
            W, H = (m.size if m and m.size else (640, 512))
            if t == "cross_dup":
                if cross_first is None:
                    cross_first = seq.name
                    entry["detail"] = "gt.txt shared with the next cross_dup sequence"
                else:
                    src = os.path.join(args.sandbox, cross_first, "gt", "gt.txt")
                    shutil.copy(src, gt)
                    entry["detail"] = f"gt.txt identical to {cross_first}"
                    cross_first = None
            else:
                detail, bad = _corrupt_m3track_gt(gt, t, W, H)
                entry["detail"] = detail
                if t in SEMANTIC_TYPES:
                    entry["bad_frames"] = bad
            manifest["entries"].append(entry)
            continue

        if _is_uavcb(seq):
            vdir = os.path.join(args.sandbox, seq.category, "visible")
            tdir = os.path.join(args.sandbox, seq.category, "thermal")
            vf = seq.modalities["visible"].frames
            tf = seq.modalities["thermal"].frames if "thermal" in seq.modalities else []
            n = len(vf)
            if t == "drop_lines":
                cut = max(1, n // 20)
                for j in range(n - cut, n):
                    os.remove(_uavcb_json_path(vdir, vf[j]))
                entry["detail"] = f"deleted last {cut} visible JSON files"
            elif t == "garbage_line":
                j = n // 2
                with open(_uavcb_json_path(vdir, vf[j]), "w") as f:
                    f.write("{not json")
                entry["detail"] = f"visible JSON frame {j} replaced with garbage"
            elif t == "oob_box" and tf:
                j = len(tf) // 2
                b = _uavcb_read_box(tdir, tf[j])
                if b:
                    _uavcb_write_box(tdir, tf[j], [b[0] - 2000, b[1] - 2000, b[2], b[3]])
                entry["detail"] = f"thermal frame {j} box moved fully out of bounds"
            elif t == "dual_copy" and tf:
                for j in range(min(n, len(tf))):
                    b = _uavcb_read_box(vdir, vf[j])
                    if b:
                        _uavcb_write_box(tdir, tf[j], b)
                entry["detail"] = "thermal JSON boxes overwritten with visible boxes (copy)"
            elif t == "cross_dup":
                if cross_first is None:
                    cross_first = seq.name
                    entry["detail"] = "visible boxes shared with the next cross_dup sequence"
                else:
                    first_seq = next(s for s in chosen if s.name == cross_first)
                    fdir = os.path.join(args.sandbox, first_seq.category, "visible")
                    ff = first_seq.modalities["visible"].frames
                    for j in range(n):
                        b = _uavcb_read_box(fdir, ff[j % len(ff)])
                        if b:
                            _uavcb_write_box(vdir, vf[j], b)
                    entry["detail"] = f"visible boxes identical to {cross_first}"
                    cross_first = None
            else:
                seg = {"offset_box": (0.3, 0.5), "segment_cross_copy": (0.4, 0.7),
                       "switch_corner": (0.35, 0.65), "stale_segment": (0.4, 0.75),
                       "drift_box": (0.5, 1.0)}[t]
                k1, k2 = int(seg[0] * n), min(int(seg[1] * n), n)
                b0 = _uavcb_read_box(vdir, vf[min(k1, n - 1)]) or [0, 0, 10, 10]
                dx = 350 if b0[0] < 960 else -350
                dy = 220 if b0[1] < 540 else -220
                W, H = 1920, 1080

                def shift_x(b, mag):
                    # shift toward the side with more headroom, clamped so the
                    # box stays fully in-frame: geometry rules must not see it
                    right = max(0.0, W - (b[0] + b[2]))
                    left = max(0.0, b[0])
                    return min(mag, right) if right >= left else -min(mag, left)

                bad = []
                for j in range(k1, k2):
                    b = _uavcb_read_box(vdir, vf[j])
                    if b is None:
                        continue
                    if t == "offset_box":
                        nb = [b[0] + shift_x(b, 0.3 * W), b[1], b[2], b[3]]
                    elif t == "drift_box":
                        ramp = (j - k1) / max(1, k2 - k1)
                        nb = [b[0] + shift_x(b, ramp * 0.25 * W), b[1], b[2], b[3]]
                    elif t == "segment_cross_copy" and j < len(tf):
                        nb = _uavcb_read_box(tdir, tf[j]) or b
                    elif t == "switch_corner":
                        nb = [max(0.0, min(W - b[2], b[0] + dx)),
                              max(0.0, min(H - b[3], b[1] + dy)), b[2], b[3]]
                    else:  # stale_segment
                        nb = b0
                    _uavcb_write_box(vdir, vf[j], nb)
                    if t == "drift_box":
                        if abs(nb[0] - b[0]) > 100:
                            bad.append(j)
                    elif t != "stale_segment" or j > k1:
                        bad.append(j)
                entry["detail"] = f"visible JSON boxes {t} on frames {k1}-{k2}"
                entry["bad_frames"] = bad
            manifest["entries"].append(entry)
            continue

        vis = os.path.join(args.sandbox, seq.name, "visible.txt")
        ir = os.path.join(args.sandbox, seq.name, "infrared.txt")
        vlines = _read_boxes(vis) if os.path.exists(vis) else []
        ilines = _read_boxes(ir) if os.path.exists(ir) else []
        n = len(vlines)

        def rng_frames(p, q):
            return list(range(int(p * n), int(q * n)))

        if t == "drop_lines":
            cut = max(1, len(vlines) // 20)
            _write_boxes(vis, vlines[:-cut])
            entry["detail"] = f"dropped last {cut} visible annotation lines"
        elif t == "garbage_line":
            j = len(vlines) // 2
            vlines[j] = "12,ab,3"
            _write_boxes(vis, vlines)
            entry["detail"] = f"line {j + 1} replaced with garbage"
        elif t == "oob_box":
            j = len(ilines) // 2
            b = _parse(ilines[j])
            ilines[j] = _fmt([b[0] - 2000, b[1] - 2000, b[2], b[3]])
            _write_boxes(ir, ilines)
            entry["detail"] = f"infrared frame {j} box moved fully out of bounds"
        elif t == "dual_copy":
            _write_boxes(ir, vlines)
            entry["detail"] = "infrared.txt overwritten with visible.txt (copy)"
        elif t == "cross_dup":
            if cross_first is None:
                cross_first = seq.name
                entry["detail"] = "visible.txt shared with the next cross_dup sequence"
            else:
                pa = os.path.join(args.sandbox, cross_first, "visible.txt")
                _write_boxes(vis, open(pa).read().splitlines())
                entry["detail"] = f"visible.txt identical to {cross_first}"
                cross_first = None
        elif t == "offset_box":
            for j in range(int(0.3 * n), int(0.5 * n)):
                b = _parse(vlines[j])
                vlines[j] = _fmt([b[0] + 0.3 * 1920, b[1], b[2], b[3]])
            _write_boxes(vis, vlines)
            entry["detail"] = "visible boxes shifted +30% width on frames 30-50%"
            entry["bad_frames"] = rng_frames(0.3, 0.5)
        elif t == "drift_box":
            start = int(0.5 * n)
            for j in range(start, n):
                b = _parse(vlines[j])
                ramp = (j - start) / max(1, n - start)
                vlines[j] = _fmt([b[0] + ramp * 0.25 * 1920, b[1], b[2], b[3]])
            _write_boxes(vis, vlines)
            entry["detail"] = "visible boxes drift linearly up to +25% width"
            entry["bad_frames"] = [j for j in range(start, n)
                                   if (j - start) / max(1, n - start) * 0.25 * 1920 > 100]
        elif t == "segment_cross_copy":
            start, end = int(0.4 * n), int(0.7 * n)
            for j in range(start, min(end, n, len(ilines))):
                vlines[j] = ilines[j]
            _write_boxes(vis, vlines)
            entry["detail"] = ("infrared boxes copied into visible annotation "
                               f"for frames {start}-{end} (unregistered copy)")
            entry["bad_frames"] = list(range(start, min(end, n)))
        elif t == "switch_corner":
            start, end = int(0.35 * n), int(0.65 * n)
            b0 = _parse(vlines[min(start, n - 1)])
            dx = 350 if b0[0] < 960 else -350
            dy = 220 if b0[1] < 540 else -220
            for j in range(start, min(end, n)):
                b = _parse(vlines[j])
                vlines[j] = _fmt([max(0.0, min(1860.0, b[0] + dx)),
                                  max(0.0, min(1040.0, b[1] + dy)), b[2], b[3]])
            _write_boxes(vis, vlines)
            entry["detail"] = (f"visible boxes relocated by "
                               f"({dx},{dy}) px on frames {start}-{end}")
            entry["bad_frames"] = list(range(start, min(end, n)))
        elif t == "stale_segment":
            start, end = int(0.4 * n), int(0.75 * n)
            frozen = vlines[min(start, n - 1)]
            for j in range(start, min(end, n)):
                vlines[j] = frozen
            _write_boxes(vis, vlines)
            entry["detail"] = f"visible boxes frozen on frame {start} for frames {start}-{end}"
            entry["bad_frames"] = list(range(start, min(end, n)))
        manifest["entries"].append(entry)

    with open(os.path.join(args.sandbox, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    n_bad = sum(1 for e in manifest["entries"] if e["type"] != "control")
    n_ctrl = sum(1 for e in manifest["entries"] if e["type"] == "control")
    print(f"sandbox ready: {args.sandbox} ({n_bad} corrupted, {n_ctrl} control)")
    return 0


def _write_oracles(sandbox, bench_out):
    """Write mock_bad_frames.json per sequence so the mock VLM can find the
    semantically-bad frames. bad_frames recorded at injection time take
    precedence; only early ramp frames of a drift where the box still covers
    the target are excluded (threshold 100px)."""
    manifest = json.load(open(os.path.join(sandbox, "manifest.json")))
    for e in manifest["entries"]:
        t = e["type"]
        if t not in SEMANTIC_TYPES:
            continue
        bad = e.get("bad_frames")
        if bad is None:
            vis = os.path.join(sandbox, e["seq"], "visible.txt")
            if not os.path.exists(vis):
                continue  # layout without a single anno file (e.g. per-frame JSON)
            n = len(_read_boxes(vis))
            if t == "offset_box":
                bad = list(range(int(0.3 * n), int(0.5 * n)))
            elif t == "drift_box":
                start = int(0.5 * n)
                bad = [j for j in range(start, n)
                       if (j - start) / max(1, n - start) * 0.25 * 1920 > 100]
            elif t == "segment_cross_copy":
                bad = list(range(int(0.4 * n), int(0.7 * n)))
            elif t == "switch_corner":
                bad = list(range(int(0.35 * n), int(0.65 * n)))
            else:  # stale_segment
                bad = list(range(int(0.4 * n), int(0.75 * n)))
        gt = os.path.join(sandbox, e["seq"], "gt", "gt.txt")
        mods = {"visible": bad, "infrared": bad} if os.path.exists(gt) \
            else {"visible": bad, "infrared": []}
        d = os.path.join(bench_out, e["seq"])
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "mock_bad_frames.json"), "w") as f:
            json.dump(mods, f)


def cmd_run(args):
    def run(method, out, extra):
        cmd = [sys.executable, AUDIT, "--root", args.sandbox, "--out", out,
               "--method", method] + extra
        print(">>", " ".join(cmd), flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"audit failed for method={method}", file=sys.stderr)
        return r.returncode

    os.makedirs(args.out, exist_ok=True)
    vlm_extra = ["--allow-mock"] if args.vlm_model == "mock" else \
                ["--vlm-model", args.vlm_model]
    run("rules", os.path.join(args.out, "rules"), [])
    # the mock oracle is shared by both VLM-based methods so the comparison
    # reflects pipeline logic, not oracle presence
    _write_oracles(args.sandbox, os.path.join(args.out, "direct"))
    run("direct", os.path.join(args.out, "direct"),
        vlm_extra + ["--direct-frames", str(args.direct_frames)])
    _write_oracles(args.sandbox, os.path.join(args.out, "layered"))
    run("layered", os.path.join(args.out, "layered"), vlm_extra)
    return 0


def cmd_evaluate(args):
    manifest = json.load(open(os.path.join(args.sandbox, "manifest.json")))
    entries = manifest["entries"]
    header = ["method", "detect_fail", "flag_all", "flag_semantic",
              "fpr_fail_ctrl", "fpr_flag_ctrl", "vlm_calls", "human_sequences"]
    methods = {}
    for m in ("rules", "direct", "layered"):
        p = os.path.join(args.bench, m, "report.json")
        if not os.path.exists(p):
            continue
        rep = json.load(open(p))
        methods[m] = {r["name"]: r for r in rep["sequences"]}
    rows = []
    for m, recs in methods.items():
        corrupt = [e for e in entries if e["type"] != "control"]
        ctrl = [e for e in entries if e["type"] == "control"]
        sem = [e for e in corrupt if e["type"] in SEMANTIC_TYPES]

        def v(e):
            return recs.get(e["seq"], {}).get("verdict", "MISSING")

        det = sum(1 for e in corrupt if v(e) == "FAIL")
        flag = sum(1 for e in corrupt if v(e) in ("FAIL", "REVIEW"))
        sem_flag = sum(1 for e in sem if v(e) in ("FAIL", "REVIEW"))
        fpr_fail = sum(1 for e in ctrl if v(e) == "FAIL")
        fpr_flag = sum(1 for e in ctrl if v(e) in ("FAIL", "REVIEW"))
        calls = sum((r.get("vlm") or {}).get("n_checked", 0) for r in recs.values())
        human = sum(1 for e in entries if v(e) in ("FAIL", "REVIEW"))
        rows.append([m,
                     f"{det}/{len(corrupt)}" if corrupt else "-",
                     f"{flag}/{len(corrupt)}" if corrupt else "-",
                     f"{sem_flag}/{len(sem)}" if sem else "-",
                     f"{fpr_fail}/{len(ctrl)}" if ctrl else "-",
                     f"{fpr_flag}/{len(ctrl)}" if ctrl else "-",
                     str(calls), str(human)])
    out_md = os.path.join(args.bench, "benchmark_report.md")
    with open(out_md, "w") as f:
        f.write("# TrackGuard: RGB-T Data-Quality Audit Benchmark\n\n"
                f"- generated: {time.strftime('%F %T')}  sandbox: `{args.sandbox}`\n"
                "- methods: rules = tier-1 rules only / direct = VLM on randomly "
                "sampled frames / layered = full TrackGuard pipeline\n"
                "- metrics: detect_fail = injected errors proven FAIL; "
                "flag = FAIL or REVIEW (surfaced for review);\n"
                "  fpr_ctrl = share of control sequences wrongly flagged; "
                "vlm_calls = frames checked (cost proxy);\n"
                "  human_sequences = sequences needing manual handling\n"
                "- VLM backend per report.json config; `mock` validates pipeline "
                "logic only — paper numbers require a real VLM\n\n"
                "| " + " | ".join(header) + " |\n|"
                + "|".join(["---"] * len(header)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(x) for x in r) + " |\n")
        f.write("\n## Injected errors\n\n| sequence | type | detail |\n|---|---|---|\n")
        for e in entries:
            f.write(f"| {e['seq']} | {e['type']} | {e.get('detail', '')} |\n")
    with open(os.path.join(args.bench, "metrics.tsv"), "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    print(f"benchmark report: {out_md}")
    for r in rows:
        print("  " + "  ".join(str(x).ljust(12) for x in r))
    return 0


def cmd_aggregate(args):
    """Aggregate metrics.tsv from multiple benchmark runs (seeds) into
    mean +/- std rates per method. Rates are per-seed, so variance across
    random sandboxes is visible; pooled counts are also reported."""
    import statistics
    bench_dirs = [d for d in args.bench.split(",") if os.path.isdir(
        os.path.join(d, "metrics.tsv")) or os.path.isdir(d)]
    runs = []
    for d in bench_dirs:
        p = os.path.join(d, "metrics.tsv")
        if not os.path.exists(p):
            continue
        per_method = {}
        for line in open(p).read().splitlines()[1:]:
            cols = line.split("\t")
            if len(cols) < 8:
                continue
            per_method[cols[0]] = cols[1:]
        if per_method:
            runs.append((d, per_method))
    if not runs:
        print("no metrics.tsv found in the given dirs", file=sys.stderr)
        return 2

    def rate(v):  # "a/b" -> float or None
        if "/" not in v:
            return None
        a, b = v.split("/")
        return float(a) / float(b) if int(b) else None

    metrics_idx = {"detect_fail": 0, "flag_all": 1, "flag_semantic": 2,
                   "fpr_fail_ctrl": 3, "fpr_flag_ctrl": 4,
                   "vlm_calls": 5, "human_sequences": 6}
    out_lines = ["# Multi-seed benchmark aggregate\n",
                 f"- runs: {len(runs)} (seeds: {', '.join(os.path.basename(d) for d, _ in runs)})",
                 "- rates are per-seed proportions reported as mean +/- std; "
                 "vlm_calls/human are per-run counts as mean +/- std\n",
                 "| method | metric | mean | std | per-seed |",
                 "|---|---|---|---|---|"]
    agg = {}
    for method in ("rules", "direct", "layered"):
        for mname, idx in metrics_idx.items():
            vals = []
            for d, per_method in runs:
                cols = per_method.get(method)
                if not cols:
                    continue
                v = cols[idx]
                vals.append(rate(v) if mname not in ("vlm_calls", "human_sequences")
                            else float(v))
            if not vals:
                continue
            mu = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            agg[(method, mname)] = (mu, sd)
            per = ", ".join(f"{v:.2f}" if v <= 1 else f"{v:.0f}" for v in vals)
            out_lines.append(f"| {method} | {mname} | {mu:.3f} | {sd:.3f} | {per} |")
    out = args.out or os.path.join(os.path.dirname(bench_dirs[0]), "aggregate_report.md")
    with open(out, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"aggregate report: {out} ({len(runs)} runs)")
    for line in out_lines[6:]:
        print("  " + line)
    return 0


def cmd_bench(args):
    args.sandbox = os.path.join(args.out, "sandbox")
    rc = cmd_corrupt(args)
    if rc:
        return rc
    rc = cmd_run(args)
    if rc:
        return rc
    args.bench = args.out
    return cmd_evaluate(args)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("corrupt")
    p.add_argument("--root", required=True)
    p.add_argument("--sandbox", required=True)
    p.add_argument("--clean-file", default=None,
                   help="sequences_pass.txt from a previous full audit")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(fn=cmd_corrupt)

    p = sub.add_parser("run")
    p.add_argument("--sandbox", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--vlm-model", default="mock")
    p.add_argument("--direct-frames", type=int, default=8)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("evaluate")
    p.add_argument("--sandbox", required=True)
    p.add_argument("--bench", required=True)
    p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("aggregate",
                       help="mean+/-std across multiple bench runs (seeds)")
    p.add_argument("--bench", required=True,
                   help="comma-separated bench output dirs (each with metrics.tsv)")
    p.add_argument("--out", default=None, help="output markdown path")
    p.set_defaults(fn=cmd_aggregate)

    p = sub.add_parser("bench")
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--clean-file", default=None)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--vlm-model", default="mock")
    p.add_argument("--direct-frames", type=int, default=8)
    p.set_defaults(fn=cmd_bench)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
