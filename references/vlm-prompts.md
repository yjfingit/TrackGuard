# VLM prompt design

Two modes, both JSON-only output, temperature 0.

## ref mode (default)

The query contains two images of the SAME modality: the anchor frame (first
valid annotation) and a sampled frame, each with its box drawn. The model
answers whether the sampled box contains the same target as the anchor,
tightly and fully.

Why reference-based: tracking annotations are identity claims ("this is the
same object as frame 0"), not object-class claims. A solo crop cannot detect
box-on-a-different-car; an anchor pair can. Both images stay in one modality,
preserving the unregistered contract.

## solo mode (`--vlm-mode solo`)

One image per query; the model judges whether the box tightly contains
exactly one clear instance of the category. Cheaper, catches empty boxes and
wrong-object boxes, misses identity switches.

## Evidence panels

`evidence.render_panel` composes, per audited frame: the annotated full frame
(resized to <=640px wide) plus a tight zoom (~1.6x box) and a wide context
crop (~5x box), each with the box drawn. The zoom crops are what make small
targets (LUART cars are often 30-100 px in 1920x1080) judgeable by a VLM at
~900px input; the context crop disambiguates which object the box should
cover.

## Verdict grammar

GOOD / BAD / UNCERTAIN, parsed from `{"verdict": ..., "reason": ...}` with a
text fallback. UNCERTAIN is reserved by prompt instruction for genuine
ambiguity (heavy occlusion, tiny blur) and maps to REVIEW, not FAIL — VLM
confidence is deliberately not trusted to convict alone unless >=2 frames and
>=50% of a modality's checks agree on BAD.

## Robustness

- Client retries 429/5xx with exponential backoff; VLM errors per frame
  become UNCERTAIN (never crash the audit).
- Images are re-encoded JPEG <=896px, quality 88 — keeps payloads small for
  batch runs.
- Any OpenAI-compatible `/chat/completions` endpoint works; provider matrix
  is in SKILL.md. Local vLLM serving Qwen2.5-VL is the cheapest loop for
  large-scale runs; frontier APIs (GPT, DeepSeek, MiniMax) for the
  generalization study.
- For multi-model comparison, run `benchmark.py run` once per model into
  separate dirs and `evaluate` them side by side; `report.json` records the
  model name for provenance.

## Prompt validation (real VLM, ref mode, 2026-10)

15 balanced frame pairs (7 injected-error frames across offset / drift /
stale-freeze / corner-switch / cross-modal-copy; 8 clean control and
natural-jitter frames) drawn from the three benchmark sandboxes, judged by
a vision-capable LLM with the tier-3 ref prompt verbatim:
accuracy 15/15, sensitivity 7/7, specificity 8/8. The judge cited zoom-crop
content in every reason (e.g. "box holds only road and grass, target lost"),
confirming the zoom+ctx panel composition carries the discriminative signal.
Records: `rgbt_audit_test/vlm_matrix/vlm_validation.json`. Replace with a
per-model table when running the multi-model study — the same 15 pairs are
reproducible from `vlm_matrix/plan.txt`.
