#!/usr/bin/env python3
"""Adapter for the previously validated SeedVR2 P_VSR2A inference entrypoint."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from pipeline.utils.subprocess_utils import run_command
from pipeline.utils.video_io import decoded_frame_count


DEFAULTS = {"model_id": "bytedance-community/SeedVR2-3B", "repo_commit": None}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def _file_identity(path: Path, cached: dict[str, Any] | None = None) -> dict[str, Any]:
    path = path.resolve()
    stat = path.stat()
    cached = cached or {}
    if (
        cached.get("path") == str(path)
        and cached.get("size_bytes") == stat.st_size
        and cached.get("mtime_ns") == stat.st_mtime_ns
        and cached.get("sha256")
    ):
        return cached
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
    }


def model_identity(config: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    values = {**DEFAULTS, **config}
    prior = {}
    if manifest_path.is_file():
        try:
            prior = json.loads(manifest_path.read_text()).get("weights", {})
        except (json.JSONDecodeError, OSError):
            prior = {}
    weights = {}
    for key, config_key in (("dit", "weights"), ("vae", "vae_weights")):
        path = Path(values[config_key])
        if not path.is_file():
            raise FileNotFoundError(f"SeedVR2 {key} checkpoint missing: {path}")
        weights[key] = _file_identity(path, prior.get(key))
    script_path = Path(values["script"])
    if not script_path.is_file():
        raise FileNotFoundError(f"SeedVR2 inference script missing: {script_path}")
    script = {
        "path": str(script_path.resolve()),
        "sha256": sha256_file(script_path),
    }
    record = {
        "model_id": values["model_id"],
        "repo": str(Path(values["repo"]).resolve()),
        "repo_commit": values["repo_commit"],
        "weights": weights,
        "inference_script": script,
    }
    atomic_json(manifest_path, record)
    return record


def video_info(path: Path) -> dict[str, Any]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    info = {
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    capture.release()
    info["decoded_frames"] = decoded_frame_count(path)
    return info



def validate_environment(config: dict[str, Any]) -> None:
    required = ("python", "repo", "script", "weights", "vae_weights")
    missing = [key for key in required if not str(config.get(key, "")).strip()]
    if missing:
        raise RuntimeError("SEEDVR2_NOT_CONFIGURED: missing " + ", ".join(missing))
    python = Path(config["python"]).expanduser()
    repo = Path(config["repo"]).expanduser()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise FileNotFoundError("SeedVR2 Python interpreter missing or not executable")
    if not repo.is_dir():
        raise FileNotFoundError("SeedVR2 repository missing")
    for key in ("script", "weights", "vae_weights"):
        if not Path(config[key]).expanduser().is_file():
            raise FileNotFoundError("SeedVR2 configured file missing: " + key)
    expected = str(config.get("repo_commit", "")).strip()
    if expected:
        actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        if actual != expected:
            raise RuntimeError(f"SeedVR2 commit mismatch: expected {expected}, got {actual}")


def build_command(config: dict[str, Any], input_dir: Path, output_dir: Path) -> list[str]:
    repo = Path(config["repo"]).resolve()
    return [
        str(config["python"]), "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
        "--nproc-per-node=1", str(config["script"]), "--video_path", str(input_dir.resolve()),
        "--output_dir", str(output_dir.resolve()), "--seed", str(config.get("seed", 666)),
        "--res_h", str(config.get("res_h", 960)), "--res_w", str(config.get("res_w", 1664)),
        "--sp_size", "1",
    ]

def run(
    config: dict[str, Any],
    input_video: Path,
    input_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    log_path: Path,
    model_manifest_path: Path,
    physical_gpu: str,
) -> dict[str, Any]:
    values = {**DEFAULTS, **config}
    validate_environment(values)
    identity = model_identity(values, model_manifest_path)
    input_video = input_video.resolve()
    if not input_video.is_file():
        raise FileNotFoundError(input_video)
    source_info = video_info(input_video)
    if source_info["decoded_frames"] != 61:
        raise RuntimeError(f"SeedVR2 requires native 61-frame input, got {source_info['decoded_frames']}")
    if (source_info["width"], source_info["height"]) != (832, 480):
        raise RuntimeError(f"SeedVR2 expects FixAnything 832x480 input, got {(source_info['width'], source_info['height'])}")
    if abs(source_info["fps"] - 24.0) > 0.01:
        raise RuntimeError(f"SeedVR2 expects 24 FPS input, got {source_info['fps']}")

    input_dir.mkdir(parents=True, exist_ok=True)
    staged_input = input_dir / "video.mp4"
    if staged_input.exists():
        staged_input.unlink()
    shutil.copy2(input_video, staged_input)
    output_dir.mkdir(parents=True, exist_ok=True)

    repo = Path(values["repo"]).resolve()
    scripts_root = Path(values.get('scripts_root') or (repo / 'scripts')).resolve()
    command = build_command(values, input_dir, output_dir)
    env = {
        "CUDA_VISIBLE_DEVICES": str(physical_gpu),
        "PYTHONPATH": f"{scripts_root}:{repo}",
        "OMP_NUM_THREADS": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "PYTHONUNBUFFERED": "1",
    }
    start = time.time()
    run_command(command, repo, log_path, env)
    elapsed = time.time() - start
    output_video = output_dir / "video.mp4"
    if not output_video.is_file():
        raise RuntimeError(f"SeedVR2 returned success but output is missing: {output_video}")
    result_info = video_info(output_video)
    if result_info["decoded_frames"] != 61:
        raise RuntimeError(f"SeedVR2 output frame-count gate failed: {result_info['decoded_frames']}")
    if abs(result_info["fps"] - source_info["fps"]) > 0.01:
        raise RuntimeError(f"SeedVR2 changed FPS: {source_info['fps']} -> {result_info['fps']}")
    if (result_info["width"], result_info["height"]) != (1664, 960):
        raise RuntimeError(f"SeedVR2 output size mismatch: {result_info['width']}x{result_info['height']}")
    report = {
        "status": "PASS",
        "input_video": str(input_video),
        "input_sha256": sha256_file(input_video),
        "input": source_info,
        "output_video": str(output_video.resolve()),
        "output_sha256": sha256_file(output_video),
        "output": result_info,
        "frame_count_preserved": True,
        "fps_preserved": True,
        "temporal_resampling": False,
        "internal_temporal_padding": "none; 61 frames satisfy (t-1)%4==0 for sp_size=1, so the model receives 61 frames without padding or trimming",
        "scale": 2,
        "seed": values.get("seed", 666),
        "sampling_steps": 1,
        "cfg_scale": 1.0,
        "cfg_rescale": 0.0,
        "sp_size": 1,
        "physical_gpu": str(physical_gpu),
        "elapsed_sec": elapsed,
        "command": command,
        "environment": env,
        "model": identity,
    }
    atomic_json(manifest_path, report)
    return report


if __name__ == "__main__":
    raise SystemExit("This adapter is invoked by pipeline_v1; use run_pipeline.py")



def collect_outputs(output_video: Path) -> dict[str, Any]:
    video = Path(output_video)
    if not video.is_file():
        raise RuntimeError("SeedVR2 output collection failed: output video missing")
    info = video_info(video)
    if info["decoded_frames"] != 61:
        raise RuntimeError("SeedVR2 output is not 61 frames")
    return {"video": video, **info}
