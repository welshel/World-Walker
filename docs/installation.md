# World-Walker installation

## A. Orchestration package

Use Python 3.10 or newer. Install the orchestration dependencies in their own environment:

```bash
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
cp .env.example .env
```

Install `ffmpeg` with the operating-system package manager and ensure the executable is on `PATH`. Never commit `.env`.

## B. External repositories

The pipeline invokes stage adapters in this repository. It does not ask users to write custom runner scripts or edit adapter code. Configure the actual checkouts and stage interpreters in `.env`.

- **TripoSplat**: use the upstream VAST-AI-Research repository at the recorded commit in `configs/external_repos.example.yaml`. Configure `TRIPOSPLAT_SNAPSHOT`, `TRIPOSPLAT_CKPT_ROOT`, and `TRIPOSPLAT_PYTHON`.
- **DiffSynth-Studio**: use the recorded Apache-2.0 checkout for FLUX. Configure `DIFFSYNTH_ROOT`; the FLUX environment must import `diffsynth.pipelines.flux2_image`.
- **FixAnything**: configure `FIXANYTHING_ROOT`, `FIXANYTHING_COMMIT`, `FIXANYTHING_DIFFSYNTH_ROOT`, and `FIXANYTHING_PYTHON`. The pinned local checkout is Apache-2.0 according to its inspected `LICENSE`; the registry does not claim an upstream URL that was not verified.
- **Agent2Topdown**: the bundled project backend calls the DeepSeek API. Set `DEEPSEEK_API_KEY` and optionally `DEEPSEEK_BASE_URL` and `AGENT2TOPDOWN_MODEL`. If using a separate compatible checkout, set `AGENT2TOPDOWN_ROOT`. Missing credentials fail before a layout image is generated.
- **SeedVR2**: optional. Keep `seedvr2.enabled: false` unless its separate repository, interpreter and checkpoints are configured. It is not needed for core output.

Do not install these large components into one shared Python environment unless their upstream CUDA/PyTorch requirements are known to be compatible.

## C. Model weights

Weights are not part of this repository.

- **FLUX.2 Klein 4B**: obtain `black-forest-labs/FLUX.2-klein-4B` from ModelScope into a local directory and set `FLUX_MODEL_PATH`. The adapter requires the Distilled model, text encoder safetensors, transformer and VAE safetensors, tokenizer, and model metadata. It runs four inference steps.
- **TripoSplat**: obtain the five checkpoint files and directory layout expected by its upstream inference pipeline, then set `TRIPOSPLAT_CKPT_ROOT`.
- **FixAnything/Wan2.1**: provide the Wan checkpoint tree, `checkpoint_manifest.json`, and FixAnything LoRA; set `FIXANYTHING_MODEL_PATH` and `FIXANYTHING_LORA_PATH`.
- **SeedVR2**: only when enabled, set `SEEDVR2_MODEL_PATH` and `SEEDVR2_VAE_PATH`.

Check each model's current terms and access requirements before downloading or redistributing it. This project does not mirror model files.

## D. API/service configuration

For text mode, set `DEEPSEEK_API_KEY` in `.env` or the process environment. The key is read by the bundled Agent2Topdown backend and is never written to a run report. The API model and base URL use `AGENT2TOPDOWN_MODEL` and `DEEPSEEK_BASE_URL`.

For image mode, no Agent2Topdown API key is needed. Text and image inputs are mutually exclusive CLI arguments.

## E. Stage-specific interpreters and environment variables

Each stage can use its own Python executable:

| Stage | Required environment variables | Interpreter must provide |
|---|---|---|
| Orchestration | activate `.venv` | Python 3.10+, PyYAML, Pillow, ImageIO |
| Agent2Topdown | `AGENT2TOPDOWN_ROOT`, `DEEPSEEK_API_KEY`, `AGENT2TOPDOWN_MODEL` | `requests`, Pillow |
| TripoSplat | `TRIPOSPLAT_PYTHON`, `TRIPOSPLAT_SNAPSHOT`, `TRIPOSPLAT_CKPT_ROOT`, optional `TRIPOSPLAT_PYTHONPATH` | PyTorch and upstream TripoSplat dependencies |
| Camera planner | `CAMERA_PYTHON`, `SCENE_UP_AXIS`, optional `GSPLAT_CUDA_EXTENSION` | NumPy, SciPy, Pillow, PyTorch, gsplat |
| Proxy renderer | `PROXY_PYTHON`, optional `GSPLAT_CUDA_EXTENSION` | NumPy, Pillow, PyTorch, gsplat, ffmpeg available on host |
| FLUX | `FLUX_PYTHON`, `FLUX_MODEL_PATH`, `DIFFSYNTH_ROOT` | PyTorch/CUDA and DiffSynth-Studio FLUX.2 pipeline |
| FixAnything | `FIXANYTHING_PYTHON`, `FIXANYTHING_ROOT`, `FIXANYTHING_DIFFSYNTH_ROOT`, `FIXANYTHING_MODEL_PATH`, `FIXANYTHING_LORA_PATH`, `FIXANYTHING_COMMIT` | PyTorch/CUDA and the pinned FixAnything dependencies |
| SeedVR2 (optional) | `SEEDVR2_PYTHON`, `SEEDVR2_ROOT`, `SEEDVR2_INFERENCE_SCRIPT`, `SEEDVR2_MODEL_PATH`, `SEEDVR2_VAE_PATH` | PyTorch/CUDA and upstream SeedVR2 dependencies |

Set `SCENE_UP_AXIS` explicitly for each scene using one of `+X`, `-X`, `+Y`, `-Y`, `+Z`, `-Z`. The garden examples use `-Y`; do not reuse it for another scene without checking its coordinate convention.

## Environment validation

Run the mode-specific component checker before inference:

```bash
python scripts/check_environment.py --config configs/text_input_example.yaml --mode text --output-dir outputs/environment_check
python scripts/check_environment.py --config configs/image_input_example.yaml --mode image --output-dir outputs/environment_check
```

It checks the core interpreter, ffmpeg, disk/write access, required repositories, stage Python imports, model layouts, provider configuration and pinned Git commit. It does not download models or load weights. Required failures produce a nonzero exit code. SeedVR2 is `OPTIONAL_DISABLED` unless requested with `--enable-seedvr2`.
