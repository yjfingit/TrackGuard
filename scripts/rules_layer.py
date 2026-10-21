"""Tier-1 deterministic rule checks: file-level provable errors -> FAIL."""


def issue(code, severity, detail, modality=None, frames=None):
    return {"code": code, "severity": severity, "detail": detail,
            "modality": modality, "frames": frames or []}


def check_sequence(seq):
    issues = []
    for mname, m in seq.modalities.items():
        if not m.anno_path:
            issues.append(issue("anno_file_missing", "FAIL",
                                f"{mname}: annotation file not found", mname))
            continue
        if not m.raw_lines and not m.boxes:
            issues.append(issue("anno_empty", "FAIL",
                                f"{mname}: annotation file is empty", mname))
            continue
        if not m.frames:
            issues.append(issue("images_missing", "FAIL",
                                f"{mname}: no images found in '{m.img_dir}'", mname))
            continue

        # empty line = target absent (MOT-style), not a format error
        bad_fmt = [i for i, (l, b) in enumerate(zip(m.raw_lines, m.boxes))
                   if l.strip() and b is None]
        if bad_fmt:
            issues.append(issue("format_invalid", "FAIL",
                                f"{mname}: {len(bad_fmt)} malformed line(s), "
                                f"first at line {bad_fmt[0] + 1}", mname, bad_fmt[:100]))

        if not seq.relaxed_count and len(m.frames) != len(m.raw_lines):
            issues.append(issue("frame_count_mismatch", "FAIL",
                                f"{mname}: images={len(m.frames)} but annotation "
                                f"lines={len(m.raw_lines)}", mname))

        if m.size:
            issues.extend(_check_geometry(mname, m, m.size))

    if not seq.shared_gt:
        issues.extend(_check_dual_copy(seq))
    return issues


def _check_geometry(mname, m, size):
    W, H = size
    issues = []
    bad_wh, full_oob, part_oob = [], [], []
    for i, b in enumerate(m.boxes):
        if b is None:
            continue
        x, y, w, h = b
        if w <= 0 or h <= 0:
            bad_wh.append(i)
            continue
        ix0, iy0 = max(x, 0), max(y, 0)
        ix1, iy1 = min(x + w, W), min(y + h, H)
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        if inter <= 0:
            full_oob.append(i)
        elif inter < 0.5 * w * h:
            part_oob.append(i)
    if bad_wh:
        issues.append(issue("bbox_invalid", "FAIL",
                            f"{mname}: {len(bad_wh)} box(es) with non-positive w/h",
                            mname, bad_wh[:100]))
    if full_oob:
        issues.append(issue("bbox_out_of_bounds", "FAIL",
                            f"{mname}: {len(full_oob)} box(es) entirely outside "
                            f"image ({W}x{H})", mname, full_oob[:100]))
    if part_oob:
        issues.append(issue("bbox_partial_oob", "REVIEW",
                            f"{mname}: {len(part_oob)} box(es) with <50% area inside "
                            f"image (may be edge target)", mname, part_oob[:100]))
    return issues


def _norm(lines):
    return [l.strip() for l in lines if l.strip()]


def _check_dual_copy(seq):
    v = seq.modalities.get("visible")
    i = seq.modalities.get("infrared")
    if not v or not i or not v.raw_bytes or not i.raw_bytes:
        return []
    same_bytes = v.raw_bytes == i.raw_bytes
    same_content = _norm(v.raw_lines) == _norm(i.raw_lines)
    if not (same_bytes or same_content):
        return []
    if v.size and i.size and v.size != i.size:
        return [issue("dual_annotation_copy", "FAIL",
                      "visible and infrared annotations are identical "
                      "while image resolutions differ (copy error)")]
    return [issue("dual_annotation_copy_same_res", "REVIEW",
                  "visible and infrared annotations are identical; "
                  "resolutions match, verify whether cross-modal copy occurred")]


def find_duplicate_annotations(seqs):
    """Dataset-level: identical annotation content shared by different sequences.

    Grouping is by normalized content (strip blank lines) so trailing-newline
    differences don't hide real duplicates.
    """
    groups = {}
    for s in seqs:
        for m in s.modalities.values():
            if not m.raw_lines:
                continue
            key = (m.name, "\n".join(_norm(m.raw_lines)))
            groups.setdefault(key, []).append(s.name)
    flags = {}
    for (mod, _content), names in groups.items():
        if len(set(names)) < 2:
            continue  # same sequence only -> dual_copy domain
        for name in set(names):
            others = [n for n in set(names) if n != name]
            flags.setdefault(name, []).append(
                issue("cross_sequence_duplicate", "FAIL",
                      f"{mod} annotation identical to: {', '.join(sorted(others))}",
                      mod))
    return flags
