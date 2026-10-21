"""Tier-3 VLM verification: adaptive sampling, local-zoom evidence, verdicts.

Unregistered RGB-T rule: every VLM query is single-modality. A query image
never mixes the visible and infrared frames, and reference-based identity
checks use the anchor frame of the SAME modality.
"""
import os

from evidence import render_modality_panel

MODAL_DESC = {"visible": "visible-light RGB", "infrared": "thermal infrared"}

SOLO_PROMPT = """You are auditing annotations of a {category} tracking dataset.
This image is a {modal} frame. The red rectangle marks the annotated target box.
Question: does the red box tightly contain exactly one clear instance of a {category}?
Judge BAD if the box is on empty background, on a different object, contains
multiple candidate objects, or is clearly misplaced. Judge UNCERTAIN only when
the crop is genuinely ambiguous (heavy occlusion, tiny blur).
Answer in JSON only: {{"verdict": "GOOD"|"BAD"|"UNCERTAIN", "reason": "<short>"}}"""

REF_PROMPT = """You are auditing a {category} tracking sequence, image pair from the
same {modal} camera. Image 1 is the reference: the red box marks the target object
at the start of the sequence. Image 2 is a later frame; its red box marks the
current annotation.
Question: does the red box in Image 2 contain the same target object as the
reference, tightly and fully?
Judge BAD if the box in Image 2 is on background, on a different object, or
misses/misplaces the reference target. Judge UNCERTAIN only when genuinely ambiguous.
Answer in JSON only: {{"verdict": "GOOD"|"BAD"|"UNCERTAIN", "reason": "<short>"}}"""


def sample_frames(sus_frames, n_frames, spot_check=4, max_samples=8):
    """Adaptive sampling: suspicious frames (with context) + evenly spaced
    spot checks. When the budget is tight, spot checks are never dropped and
    suspicious frames keep the given (severity-ranked) order, so natural
    motion jitter on dynamic targets cannot crowd out coverage."""
    if n_frames <= 0:
        return []
    spots = set()
    if spot_check > 0:
        for k in range(spot_check):
            spots.add(int(round(k * (n_frames - 1) / max(1, spot_check - 1))))
    spots = {s for s in spots if 0 <= s < n_frames}
    picked = set(spots) | ({0} if n_frames > 1 else {0})
    for f in sus_frames:
        if 0 <= f < n_frames:
            picked.update({max(0, f - 1), f, min(n_frames - 1, f + 1)})
    picked = sorted(picked)
    if len(picked) <= max_samples:
        return picked
    keep = set(spots)
    if n_frames > 1:
        keep.add(0)
    for f in sus_frames:
        if len(keep) >= max_samples:
            break
        for cand in (f, max(0, f - 1), min(n_frames - 1, f + 1)):
            if cand not in keep:
                keep.add(cand)
                if len(keep) >= max_samples:
                    break
    if len(keep) < max_samples:
        rest = [f for f in picked if f not in keep]
        step = max(1, len(rest) // (max_samples - len(keep)))
        keep.update(rest[::step][:max_samples - len(keep)])
    return sorted(keep)


def verify_sequence(seq, client, samples, out_dir, mode="ref"):
    """Run VLM checks on sampled frames. Verdicts are tracked per modality:
    the modalities are never pooled, honoring the unregistered-RGB-T contract."""
    result = {"mode": mode, "frames": {}, "n_bad": 0, "n_uncertain": 0,
              "n_checked": 0, "per_modality": {}, "errors": []}
    if not client or not client.available():
        return result
    ev_dir = os.path.join(out_dir, seq.name, "evidence")
    for mname, m in seq.modalities.items():
        if not m.frames:
            continue
        anchor = next((i for i in range(len(m.boxes)) if m.boxes[i] is not None), None)
        if anchor is None:
            continue
        pm = result["per_modality"].setdefault(
            mname, {"n_checked": 0, "n_bad": 0, "n_uncertain": 0})
        anchor_panel = render_modality_panel(seq, mname, anchor, ev_dir)
        for idx in samples:
            if (idx >= len(m.frames) or idx >= len(m.boxes)
                    or m.boxes[idx] is None or idx == anchor):
                continue
            panel = render_modality_panel(seq, mname, idx, ev_dir)
            if not panel:
                continue
            tpl = REF_PROMPT if mode == "ref" else SOLO_PROMPT
            prompt = tpl.format(category=seq.category, modal=MODAL_DESC.get(mname, mname))
            images = [anchor_panel, panel] if mode == "ref" else [panel]
            key = f"{mname}:{idx}"
            try:
                v = client.judge(prompt, images)
            except Exception as e:
                result["errors"].append(f"{key}: {e}")
                v = {"verdict": "UNCERTAIN", "reason": f"VLM error: {e}"}
            v["panel"] = panel
            result["frames"][key] = v
            result["n_checked"] += 1
            pm["n_checked"] += 1
            if v["verdict"] == "BAD":
                result["n_bad"] += 1
                pm["n_bad"] += 1
            elif v["verdict"] == "UNCERTAIN":
                result["n_uncertain"] += 1
                pm["n_uncertain"] += 1
    return result


def aggregate(rule_issues, traj_signals, vlm_result, vlm_enabled):
    """Map layered findings to PASS / FAIL / REVIEW with error codes.

    Semantic VLM evidence is judged per modality: a single corrupted modality
    must not be diluted by the healthy one.
    """
    codes = []
    if any(i["severity"] == "FAIL" for i in rule_issues):
        for i in rule_issues:
            if i["severity"] == "FAIL":
                codes.append(i["code"])
        return "FAIL", codes
    review_codes = [i["code"] for i in rule_issues if i["severity"] == "REVIEW"]

    if vlm_result and vlm_result.get("n_checked"):
        fail_semantic = False
        suspect = False
        for pm in vlm_result.get("per_modality", {}).values():
            if pm["n_bad"] >= 2 and pm["n_bad"] >= 0.5 * pm["n_checked"]:
                fail_semantic = True
            if pm["n_bad"] or pm["n_uncertain"]:
                suspect = True
        if fail_semantic:
            return "FAIL", review_codes + ["vlm_box_mismatch"]
        if suspect:
            return "REVIEW", review_codes + ["vlm_suspect"]
        if traj_signals and vlm_result["n_checked"] < 3:
            return "REVIEW", review_codes + [s["code"] for s in traj_signals]
        return "PASS", review_codes
    if traj_signals:
        return "REVIEW", review_codes + [s["code"] for s in traj_signals]
    if review_codes:
        return "REVIEW", review_codes
    return "PASS", []
