# Natural Language to 3D Video Pipeline

Research prototype for generating camera-controlled 3D scene videos from a natural-language description or a top-down layout image. This is a research prototype, not production-ready software.

## Supported input modes

A. `Natural language → Agent2Topdown → FLUX top-down refinement → TripoSplat → camera-controlled video`

B. `Top-down image → TripoSplat → camera-controlled video`

The shared video pipeline plans a native 61-frame camera path, renders a proxy, creates two FLUX.2 Klein endpoint anchors, and propagates them with FixAnything. SeedVR2 is optional and disabled by default.

```text
Natural language or top-down image
  → Agent2Topdown (text mode only)
  → FLUX.2 Klein top-down refinement (text mode only)
  → TripoSplat Gaussian PLY
  → automatic camera planner
  → native 61-frame proxy
  → FLUX endpoint anchors at frames 0 and 60
  → FixAnything 61-frame video
  → [optional SeedVR2]
```

## Requirements

- Python 3.10 or newer for the orchestration package.
- `ffmpeg` on `PATH`.
- Separate compatible Python environments for TripoSplat, camera/proxy rendering, FLUX, FixAnything, and SeedVR2. Their CUDA and PyTorch requirements differ, so the project does not combine them into one environment.
- External repositories and model weights listed in [docs/installation.md](docs/installation.md). They are not included here.
- A DeepSeek API key for natural-language mode. The real provider is required; mock output is never accepted by the formal CLI.

## Quick start

### 1. Install orchestration

```bash
git clone <your-repository-url>
cd natural-language-3d-video-pipeline
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
cp .env.example .env
```

### 2. Configure external components

Fill `.env` with the stage-specific interpreter, repository, checkpoint and model paths described in [docs/installation.md](docs/installation.md). Set `SCENE_UP_AXIS` explicitly for the input scene. Put `DEEPSEEK_API_KEY` in the local `.env`; do not commit it.

### 3. Validate the chosen mode

```bash
python scripts/check_environment.py --config configs/text_input_example.yaml --mode text --output-dir outputs/environment_check
# Or for direct top-down image input:
python scripts/check_environment.py --config configs/image_input_example.yaml --mode image --output-dir outputs/environment_check
```

The checker returns nonzero when a required component is missing. It does not download weights or start inference. SeedVR2 reports `OPTIONAL_DISABLED` unless explicitly enabled.

### 4A. Natural-language mode

```bash
python -m pipeline.run_pipeline \
  --scene-prompt "A landscaped formal garden with a glass greenhouse, a central stone fountain, wooden benches, symmetrical flower beds, stone paths, green grass, shrubs and surrounding trees." \
  --config configs/text_input_example.yaml \
  --output outputs/garden
```

### 4B. Top-down image mode

```bash
python -m pipeline.run_pipeline \
  --input-topdown /path/to/topdown.png \
  --config configs/image_input_example.yaml \
  --output outputs/garden
```

### 5. Approve generated top-down image in text mode

Text mode stops at `WAITING_VISUAL_REVIEW` after FLUX produces top-down candidates. Inspect the candidate contact sheet and image, then follow [docs/pipeline.md](docs/pipeline.md) to write the review record. Continue the same output with `--resume`. The pipeline will not pass an unreviewed or hash-mismatched top-down to TripoSplat.

### 6. Optional SeedVR2

SeedVR2 is disabled by default. Configure its independent environment and weights first, then run:

```bash
python -m pipeline.run_pipeline \
  --input-topdown /path/to/topdown.png \
  --config configs/image_input_example.yaml \
  --output outputs/garden_hd \
  --enable-seedvr2
```

## Configuration and outputs

- `configs/default.yaml` is the GitHub-safe 61-frame configuration. Machine paths are read from environment variables.
- `configs/model_registry.example.yaml` and `configs/external_repos.example.yaml` document local dependencies. `configs/model_registry.local.yaml` is machine-specific and ignored by Git.
- `.env` and `*.local.yaml` are ignored. Never put credentials in tracked YAML.
- Each run writes inputs, stage outputs, logs, manifests, hashes, reports and comparison media under the selected `--output` directory.
- `--resume` reuses stages only when their configuration and input/output hashes match. Use `--force-stage <stage>` to intentionally rerun a stage.

## Common failure points

- `AGENT2TOPDOWN_PROVIDER_NOT_CONFIGURED`: text mode needs an Agent2Topdown checkout or the bundled backend and a valid DeepSeek API key. No mock fallback is allowed.
- FLUX inference needs ModelScope FLUX.2 Klein 4B Distilled weights, DiffSynth-Studio, a compatible CUDA environment and a visible GPU.
- The camera planner requires an explicit signed scene up axis and a Python environment with NumPy, SciPy, PyTorch and gsplat.
- TripoSplat/FixAnything checkpoint layouts and Git revisions are checked before inference.
- FLUX top-down generation requires a human review step. Endpoint anchor alignment is audited before FixAnything.
- A successful proxy or endpoint metric does not guarantee perceptual quality or full 3D consistency.

## External components and license

This repository contains orchestration and project adapters; it does not distribute third-party model weights. See [third_party/README.md](third_party/README.md) and [release_audit/third_party_licenses.md](release_audit/third_party_licenses.md) for verified provenance and license evidence. The project license has not been selected; see [LICENSE_SELECTION_REQUIRED.md](LICENSE_SELECTION_REQUIRED.md). No specific project license is claimed here.
