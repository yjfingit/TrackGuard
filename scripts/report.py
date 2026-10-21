"""Report generation: PASS/FAIL/REVIEW classification, TSV errors, markdown report."""
import json
import os
import time
from collections import Counter

ORDER = {"FAIL": 0, "REVIEW": 1, "PASS": 2}


def _fmt_frames(frames, cap=20):
    if not frames:
        return ""
    s = ",".join(str(f) for f in frames[:cap])
    return s + (f" ...({len(frames)})" if len(frames) > cap else "")


def build_report(records, args, out_dir, vlm_calls=0):
    records = sorted(records, key=lambda r: (ORDER.get(r["verdict"], 3), r["name"]))
    verdicts = Counter(r["verdict"] for r in records)
    code_counts = Counter(c for r in records for c in set(r["codes"]))
    n_ev = sum(len(r["vlm"]["frames"]) for r in records if r.get("vlm"))
    summary = {
        "n_total": len(records),
        "verdicts": dict(verdicts),
        "error_codes": dict(code_counts),
        "vlm_calls": vlm_calls,
        "evidence_frames": n_ev,
        "human_review_sequences": verdicts.get("REVIEW", 0) + verdicts.get("FAIL", 0),
    }
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump({"generated": time.strftime("%F %T"), "config": args,
                   "summary": summary, "sequences": records},
                  f, ensure_ascii=False, indent=1)

    with open(os.path.join(out_dir, "errors.tsv"), "w") as f:
        f.write("sequence\tverdict\tsource\tcode\tmodality\tn_frames\tframes\tdetail\n")
        for r in records:
            for i in r.get("rule_issues", []):
                f.write("\t".join([r["name"], r["verdict"], "rule", i["code"],
                                   i.get("modality") or "",
                                   str(len(i.get("frames", []))),
                                   _fmt_frames(i.get("frames", [])),
                                   i["detail"]]) + "\n")
            for s in r.get("traj_signals", []):
                f.write("\t".join([r["name"], r["verdict"], "traj", s["code"],
                                   s.get("modality") or "",
                                   str(len(s.get("frames", []))),
                                   _fmt_frames(s.get("frames", [])),
                                   s["detail"]]) + "\n")
            v = r.get("vlm")
            if v:
                bad = [k for k, d in v["frames"].items() if d["verdict"] == "BAD"]
                unc = [k for k, d in v["frames"].items() if d["verdict"] == "UNCERTAIN"]
                if bad or unc:
                    f.write("\t".join([r["name"], r["verdict"], "vlm", "vlm_suspect",
                                       "", str(len(bad) + len(unc)),
                                       ",".join(bad + unc),
                                       f"VLM flagged {len(bad)} bad / "
                                       f"{len(unc)} uncertain of {v['n_checked']}"]) + "\n")

    for v in ("pass", "fail", "review"):
        names = [r["name"] for r in records if r["verdict"] == v.upper()]
        with open(os.path.join(out_dir, f"sequences_{v}.txt"), "w") as f:
            f.write("\n".join(names) + ("\n" if names else ""))

    _write_markdown(records, args, out_dir, summary)
    return summary


def _seq_block_md(r):
    lines = [f"### {r['name']}  (category: {r['category']}, "
             f"frames: {r.get('n_frames')})"]
    lines.append(f"- verdict: **{r['verdict']}**  codes: {', '.join(r['codes']) or '-'}")
    for i in r.get("rule_issues", []):
        fr = _fmt_frames(i.get("frames", []))
        lines.append(f"- [{i['severity']}] {i['code']}"
                     + (f" ({i['modality']})" if i.get("modality") else "")
                     + f": {i['detail']}"
                     + (f"  frames: {fr}" if fr else ""))
    for s in r.get("traj_signals", []):
        fr = _fmt_frames(s.get("frames", []))
        lines.append(f"- [traj] {s['code']}"
                     + (f" ({s['modality']})" if s.get("modality") else "")
                     + f": {s['detail']}"
                     + (f"  frames: {fr}" if fr else ""))
    v = r.get("vlm")
    if v and v.get("n_checked"):
        bad = [k for k, d in v["frames"].items() if d["verdict"] == "BAD"]
        unc = [k for k, d in v["frames"].items() if d["verdict"] == "UNCERTAIN"]
        lines.append(f"- [vlm] checked {v['n_checked']} frames: "
                     f"{v['n_bad']} BAD / {v['n_uncertain']} UNCERTAIN "
                     f"(mode={v['mode']})")
        for k in bad + unc:
            d = v["frames"][k]
            lines.append(f"  - {k}: {d['verdict']} — {d['reason']}  evidence: {d['panel']}")
    return "\n".join(lines)


def _write_markdown(records, args, out_dir, summary):
    cfg = args if isinstance(args, dict) else {}
    lines = ["# RGB-T Dataset Audit Report", "",
             f"- generated: {time.strftime('%F %T')}",
             f"- dataset root: `{cfg.get('root')}`",
             f"- method: {cfg.get('method')} (spot_check={cfg.get('spot_check')}, "
             f"vlm_mode={cfg.get('vlm_mode')}, vlm_model={cfg.get('vlm_model') or 'env'})",
             f"- sequences: {summary['n_total']}", "",
             "## Summary", "",
             "| verdict | count |", "|---|---|",
             f"| FAIL (error) | {summary['verdicts'].get('FAIL', 0)} |",
             f"| REVIEW (needs human) | {summary['verdicts'].get('REVIEW', 0)} |",
             f"| PASS (clean) | {summary['verdicts'].get('PASS', 0)} |", "",
             "## Error codes", "", "| code | sequences |", "|---|---|"]
    for c, n in sorted(summary["error_codes"].items(), key=lambda x: -x[1]):
        lines.append(f"| {c} | {n} |")
    lines += ["", "## Cost", "",
              f"- VLM calls: {summary['vlm_calls']}",
              f"- evidence frames rendered: {summary['evidence_frames']}",
              f"- sequences needing human review (FAIL+REVIEW): {summary['human_review_sequences']}",
              ""]
    for verdict, title in (("FAIL", "FAIL sequences (proven errors)"),
                           ("REVIEW", "REVIEW sequences (needs human review)")):
        sel = [r for r in records if r["verdict"] == verdict]
        lines += [f"## {title} ({len(sel)})", ""]
        if sel:
            lines += [ _seq_block_md(r) for r in sel]
        else:
            lines.append("(none)")
        lines.append("")
    passed = [r["name"] for r in records if r["verdict"] == "PASS"]
    lines += [f"## PASS sequences ({len(passed)})", "", ", ".join(passed) or "(none)", ""]
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(lines))
