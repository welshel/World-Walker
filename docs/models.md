# Models and external repositories

| Component | Interface | Configuration |
|---|---|---|
| Agent2Topdown | External repository exposing app.run_pipeline; optional DeepSeek-compatible API | AGENT2TOPDOWN_ROOT, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL |
| TripoSplat | External inference script, snapshot, checkpoint root | TRIPOSPLAT_PYTHON, TRIPOSPLAT_INFER_SCRIPT, TRIPOSPLAT_SNAPSHOT, TRIPOSPLAT_CKPT_ROOT |
| FLUX.2 Klein | black-forest-labs/FLUX.2-klein-4B Distilled, four-step DiffSynth image pipeline | FLUX_MODEL_PATH, FLUX_PYTHON, DIFFSYNTH_ROOT |
| FixAnything / Wan | External runner and compatible checkpoints | FIXANYTHING_PYTHON, FIXANYTHING_RUNNER, FIXANYTHING_MODEL_PATH, FIXANYTHING_LORA_PATH |
| SeedVR2 | Optional external inference repository and checkpoints | SEEDVR2_ROOT, SEEDVR2_INFERENCE_SCRIPT, SEEDVR2_MODEL_PATH, SEEDVR2_VAE_PATH |

Model weights and external repositories are not distributed. Use upstream project documentation for current setup and terms; no unverified download URL is supplied.
