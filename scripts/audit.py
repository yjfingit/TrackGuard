#!/usr/bin/env python3
"""Layered RGB-T tracking-dataset audit CLI.

Tier 1 deterministic rules -> Tier 2 per-modality trajectory analysis ->
Tier 3 VLM verification (adaptive sampling + local zoom crops).
Unregistered-safe: cross-modal box coordinates are never compared.

Examples:
  python audit.py --root /path/to/LUART --out out_luart
  python audit.py --root LUART --out out --pattern '^(Car|Bus)_' --limit 20
  TRACKGUARD_VLM_MODEL=qwen2.5-vl-72b-instruct \
  TRACKGUARD_VLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1 \
  TRACKGUARD_VLM_API_KEY=sk-xxx python audit.py --root LUART --out out
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import discover_sequences
from rules_layer import check_sequence, find_duplicate_annotations
from trajectory_layer import analyze as traj_analyze
import vlm_layer
from vlm_client import VLMClient
import report as report_mod


def audit_one(seq, args, client, dup_issues):
    rec = {"name": seq.name, "category": seq.category,
           "n_frames": {m: len(mod.frames) for m, mod in seq.modalities.items()},
           "rule_issues": [], "traj_signals": [], "vlm": None, "samples": [],
           "codes": [], "verdict": "PASS"}
    sus_all = []
    if args.method != "direct":
        issues = check_sequence(seq) + dup_issues.get(seq.name, [])
        rec["rule_issues"] = issues
        if args.method == "rules":
            if any(i["severity"] == "FAIL" for i in issues):
                rec["verdict"] = "FAIL"
            elif issues:
                rec["verdict"] = "REVIEW"
            rec["codes"] = sorted(set(i["code"] for i in issues))
            return rec
        if any(i["severity"] == "FAIL" for i in issues):
            rec["verdict"] = "FAIL"
            rec["codes"] = sorted(set(i["code"] for i in issues
                                      if i["severity"] == "FAIL"))
            return rec
        sus_scores = {}
        if args.traj:
            for mname, m in seq.modalities.items():
                if not m.boxes:
                    continue
                signals, sus = traj_analyze(m.boxes, m.size)
                for s in signals:
                    s["modality"] = mname
                rec["traj_signals"] += signals
                for f, sc in sus:
                    if sc > sus_scores.get(f, 0.0):
                        sus_scores[f] = sc
        # severity-ranked across modalities: max score per frame wins
        sus_all = [f for f, _ in sorted(sus_scores.items(),
                                        key=lambda kv: (-kv[1], kv[0]))]

    vlm_enabled = args.vlm and client is not None and client.available()
    if vlm_enabled:
        n = max((len(m.frames) for m in seq.modalities.values()), default=0)
        if args.method == "direct":
            k = min(args.direct_frames, n)
            rng = random.Random(args.seed)
            samples = sorted(rng.sample(range(n), k)) if k > 0 else []
        else:
            samples = vlm_layer.sample_frames(sus_all, n,
                                               args.spot_check, args.max_samples)
        rec["samples"] = samples
        if samples:
            res = vlm_layer.verify_sequence(seq, client, samples,
                                            args.out, mode=args.vlm_mode)
            rec["vlm"] = {
                "mode": res["mode"], "n_checked": res["n_checked"],
                "n_bad": res["n_bad"], "n_uncertain": res["n_uncertain"],
                "per_modality": res["per_modality"],
                "errors": res["errors"][:5],
                "frames": {k: {"verdict": v["verdict"], "reason": v["reason"],
                               "panel": v["panel"]}
                           for k, v in res["frames"].items()}}
    verdict, codes = vlm_layer.aggregate(rec["rule_issues"], rec["traj_signals"],
                                         rec.get("vlm"), vlm_enabled)
    rec["verdict"] = verdict
    rec["codes"] = codes
    return rec


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="dataset root (LUART layout)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--pattern", default=None, help="regex filter on sequence name")
    ap.add_argument("--seqs-file", default=None, help="file with one sequence name per line")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--method", choices=["layered", "rules", "direct"],
                    default="layered")
    ap.add_argument("--no-traj", dest="traj", action="store_false",
                    help="skip tier-2 trajectory analysis (ablation: VLM "
                         "sampling then uses spot checks only)")
    ap.add_argument("--no-vlm", dest="vlm", action="store_false",
                    help="skip tier-3 VLM verification")
    ap.add_argument("--vlm-model", default=None)
    ap.add_argument("--vlm-base-url", default=None)
    ap.add_argument("--vlm-api-key", default=None)
    ap.add_argument("--vlm-proxy", default=None,
                    help='proxy URL, "direct" to bypass, "env" to use env vars')
    ap.add_argument("--vlm-mode", choices=["ref", "solo"], default="ref")
    ap.add_argument("--spot-check", type=int, default=4,
                    help="evenly spaced frames VLM-checked on clean sequences (0=off)")
    ap.add_argument("--max-samples", type=int, default=8)
    ap.add_argument("--direct-frames", type=int, default=8,
                    help="random frames per sequence for --method direct")
    ap.add_argument("--allow-mock", action="store_true",
                    help="permit the offline mock VLM (pipeline test only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="skip sequences already present in checkpoint.jsonl")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    seqs = discover_sequences(args.root, pattern=args.pattern,
                              seqs_file=args.seqs_file, limit=args.limit)
    if not seqs:
        print("no sequences found under --root", file=sys.stderr)
        return 2

    ckpt = os.path.join(args.out, "checkpoint.jsonl")
    records, done = [], set()
    if args.resume and os.path.exists(ckpt):
        with open(ckpt) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    records.append(r)
                    done.add(r["name"])
                except ValueError:
                    pass
    todo = [s for s in seqs if s.name not in done]

    dup_issues = {}
    if args.method != "direct" and todo:
        dup_issues = find_duplicate_annotations(todo)

    client = None
    if args.vlm:
        client = VLMClient(model=args.vlm_model, base_url=args.vlm_base_url,
                           api_key=args.vlm_api_key, proxy=args.vlm_proxy)
        if client.model == "mock" and not args.allow_mock:
            print("[warn] VLM model is 'mock' (offline test mode); disabling VLM. "
                  "Pass --allow-mock to force-enable.", file=sys.stderr)
            client = None
        elif not client.available():
            print("[warn] VLM base_url not configured; running rules+trajectory only. "
                  "Set TRACKGUARD_VLM_BASE_URL / _MODEL / _API_KEY.", file=sys.stderr)
            client = None

    print(f"sequences: {len(todo)} todo / {len(records)} resumed", flush=True)
    t0 = time.time()
    with open(ckpt, "a") as cf:
        for i, seq in enumerate(todo):
            rec = audit_one(seq, args, client, dup_issues)
            records.append(rec)
            cf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            cf.flush()
            if (i + 1) % 10 == 0 or i == len(todo) - 1:
                print(f"[{i + 1}/{len(todo)}] {seq.name} -> {rec['verdict']} "
                      f"({time.time() - t0:.0f}s)", flush=True)

    vlm_calls = client.calls if client else 0
    report_mod.build_report(records, vars(args), args.out, vlm_calls=vlm_calls)
    print(f"done: {len(records)} sequences, report written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
