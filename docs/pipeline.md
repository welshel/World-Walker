# World-Walker pipeline

The CLI accepts exactly one input: `--scene-prompt` or `--input-topdown`. Text mode uses the real Agent2Topdown provider and refuses mock fallback. Both modes use a native 61-frame, 24 FPS pipeline with clean anchors at frames 0 and 60; temporal resampling is disabled.

## Text-mode top-down review

Text mode creates a new Agent2Topdown layout, then FLUX.2 Klein generates four top-down candidates. The pipeline pauses with `WAITING_VISUAL_REVIEW`. Inspect `02_flux_topdown/outputs/seed*.png` and `alignment_contact_sheet.png`, then save `02_flux_topdown/visual_review.json` with:

```json
{
  "APPROVED_FOR_TRIPOSPLAT": true,
  "STRICT_NADIR_VIEW": "PASS",
  "NO_HORIZON": "PASS",
  "NO_OBLIQUE_FACADE": "PASS",
  "CANDIDATE_SHA256": "<sha256 of the selected candidate>"
}
```

Use the selected candidate path and its exact SHA256 in the stage manifest as described by `02_flux_topdown/topdown_used_manifest.json`; the adapter verifies both before it creates `topdown_used.png`. Then resume the same run:

```bash
python -m pipeline.run_pipeline --scene-prompt "..." --config configs/text_input_example.yaml --output outputs/garden --resume
```

A rejected review or hash mismatch prevents TripoSplat from starting.

## Stage adapters and outputs

Every external stage is called through a repository adapter with explicit validation, command construction where applicable, and output checks:

- TripoSplat converts the selected layout into `raw_scene.ply` and records input/output hashes.
- The self-contained camera planner generates a camera trajectory and checks center, near-frustum, and forward-ray clearances, plus motion smoothness.
- The proxy renderer consumes only the current PLY and trajectory, then verifies exactly 61 frames.
- FLUX uses ModelScope-local FLUX.2 Klein 4B Distilled, four steps, and seeds 0–3 for two endpoint edit stages. It audits proxy-to-anchor alignment and blocks FixAnything on `FAIL` by default; an explicit override is available after visual inspection.
- FixAnything uses only the native 61 proxy frames and clean anchors 0/60; it verifies frame count, FPS, and output resolution.
- SeedVR2 runs only when enabled. Its cache fingerprint depends on the FixAnything video, so enabling it does not invalidate upstream stages.

Use `--resume` to reuse hash-validated stage outputs. Use `--force-stage <name>` to rerun a specific stage; prior force-run output is preserved rather than silently overwritten. The final report is written under the chosen output directory.

## FLUX anchor alignment policy

The automatic alignment metric and thresholds are unchanged. GOOD continues. PARTIAL emits a warning and continues. FAIL blocks the default run. A failed audit means the clean anchor differs from the proxy beyond the automatic threshold; it does not by itself predict whether FixAnything inference will fail.

After visually inspecting the saved anchor/proxy comparison, an experienced user may opt in with <code>--allow-anchor-alignment-fail</code>, or set <code>flux.anchor_alignment.allow_fail: true</code>. The pipeline prints the required warning and records <code>ANCHOR_ALIGNMENT_OVERRIDE_USED</code> in stage and final reports. The override participates in the flux_anchor, fixanything, and report resume fingerprints. A completed run with an overridden failure remains marked <code>PARTIAL_ANCHOR_ALIGNMENT_OVERRIDE</code>.
