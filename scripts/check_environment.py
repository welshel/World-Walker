#!/usr/bin/env python3
"""Check each stage environment without downloading weights or starting inference."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from pipeline.utils.paths import load_config


def configured_path(value: object) -> Path | None:
    text = str(value or "").strip()
    if not text or (text.startswith("${") and text.endswith("}")):
        return None
    return Path(os.path.expandvars(text)).expanduser()


def provider_check(root: object, api_key_present: bool) -> dict:
    path = configured_path(root)
    if path is None or not (path / "app.py").is_file():
        return {"status": "FAIL", "reason": "AGENT2TOPDOWN_PROVIDER_NOT_CONFIGURED"}
    if not api_key_present:
        return {"status": "FAIL", "reason": "AGENT2TOPDOWN_PROVIDER_NOT_CONFIGURED"}
    return {"status": "PASS", "reason": "real_api_provider_configured"}


def _path_from(section_cfg: dict, key: str, env_key: str | None = None) -> Path | None:
    value = os.environ.get(env_key, "") if env_key else ""
    if not value:
        value = section_cfg.get(key, "")
    return configured_path(value)


def configured_paths(value: object) -> list[Path]:
    text = str(value or "").strip()
    if not text or (text.startswith("${") and text.endswith("}")):
        return []
    return [Path(os.path.expandvars(part)).expanduser() for part in text.split(os.pathsep) if part]


def _probe(interpreter: Path | None, imports: list[str], pythonpath: list[str] | None = None) -> tuple[bool, str]:
    if interpreter is None or not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        return False, "interpreter_missing"
    env = os.environ.copy()
    if pythonpath:
        env["PYTHONPATH"] = os.pathsep.join(str(x) for x in pythonpath if x) + os.pathsep + env.get("PYTHONPATH", "")
    code = "import importlib; " + "; ".join(f"importlib.import_module({name!r})" for name in imports)
    result = subprocess.run([str(interpreter), "-c", code], env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, text=True, timeout=60)
    return result.returncode == 0, "imports_ok" if result.returncode == 0 else "dependency_import_failed"


def _git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _component(status: str, reason: str, **extra) -> dict:
    return {"status": status, "reason": reason, **extra}


def evaluate(config_path: Path, mode: str, output_dir: Path, enable_seedvr2: bool = False) -> dict:
    checks: dict[str, dict] = {}
    try:
        cfg = load_config(config_path)
    except Exception as exc:
        return {"FINAL_ENV_STATUS": "FAIL", "checks": {"CORE": _component("FAIL", "config_parse_failed", error=type(exc).__name__)}}

    core_modules = {name: importlib.util.find_spec(name) is not None for name in ("yaml", "PIL", "imageio")}
    core_ok = sys.version_info >= (3, 10) and all(core_modules.values()) and shutil.which("ffmpeg") is not None
    disk_gib = None
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        probe = output_dir / ".pipeline_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        disk_gib = round(shutil.disk_usage(output_dir).free / (1024 ** 3), 2)
        core_ok = core_ok and disk_gib >= 10
    except OSError:
        core_ok = False
    checks["CORE"] = _component("PASS" if core_ok else "FAIL", "ready" if core_ok else "core_requirement_missing",
                                python=sys.version.split()[0], ffmpeg=bool(shutil.which("ffmpeg")),
                                required_imports=core_modules, free_disk_gib=disk_gib)

    if mode == "text":
        agent_cfg = cfg.get("agent2topdown", {})
        root_value = os.environ.get("AGENT2TOPDOWN_ROOT") or agent_cfg.get("root", "")
        if not str(root_value).strip() or str(root_value).startswith("${"):
            root_value = REPO_ROOT / "pipeline" / "agent2topdown_backend"
        checks["AGENT2TOPDOWN"] = provider_check(root_value, bool(os.environ.get(agent_cfg.get("api_key_env", "DEEPSEEK_API_KEY"))))
    else:
        checks["AGENT2TOPDOWN"] = _component("SKIP", "image_input_mode")

    stages = cfg.get("stages", {})
    tri = stages.get("triposplat", {})
    tri_py = _path_from(tri, "python", "TRIPOSPLAT_PYTHON")
    tri_snapshot = _path_from(tri, "snapshot", "TRIPOSPLAT_SNAPSHOT")
    tri_ckpt = _path_from(tri, "checkpoint_root", "TRIPOSPLAT_CKPT_ROOT")
    tri_required = [
        tri_ckpt / "diffusion_models/triposplat_fp16.safetensors" if tri_ckpt else None,
        tri_ckpt / "vae/triposplat_vae_decoder_fp16.safetensors" if tri_ckpt else None,
        tri_ckpt / "clip_vision/dino_v3_vit_h.safetensors" if tri_ckpt else None,
        tri_ckpt / "vae/flux2-vae.safetensors" if tri_ckpt else None,
        tri_ckpt / "background_removal/birefnet.safetensors" if tri_ckpt else None,
    ]
    tri_ok = bool(tri_py and tri_snapshot and (tri_snapshot / "triposplat.py").is_file() and all(p and p.is_file() for p in tri_required))
    if tri_ok:
        probe_ok, probe_reason = _probe(tri_py, ["torch", "numpy", "PIL", "safetensors"])
        tri_ok = probe_ok
    else:
        probe_reason = "repo_or_checkpoint_missing"
    checks["TRIPOSPLAT"] = _component("PASS" if tri_ok else "FAIL", "ready" if tri_ok else probe_reason,
                                     expected_weight_files=5)

    camera = stages.get("camera", {})
    camera_py = _path_from(camera, "python", "CAMERA_PYTHON")
    up_axis = str(camera.get("up_axis", cfg.get("up_axis", "")))
    camera_ok = up_axis in ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
    camera_probe, camera_reason = _probe(camera_py, ["numpy", "scipy", "PIL", "torch", "gsplat"], [REPO_ROOT])
    camera_ok = camera_ok and camera_probe
    checks["CAMERA_PLANNER"] = _component("PASS" if camera_ok else "FAIL", "ready" if camera_ok else camera_reason if not camera_probe else "signed_scene_up_axis_required")

    proxy = stages.get("proxy", {})
    proxy_py = _path_from(proxy, "python", "PROXY_PYTHON")
    proxy_probe, proxy_reason = _probe(proxy_py, ["numpy", "PIL", "torch", "gsplat"], [REPO_ROOT])
    proxy_ok = proxy_probe and shutil.which("ffmpeg") is not None
    checks["PROXY_RENDERER"] = _component("PASS" if proxy_ok else "FAIL", "ready" if proxy_ok else proxy_reason)

    flux = stages.get("flux_anchor", {})
    flux_py = _path_from(flux, "python", "FLUX_PYTHON")
    flux_model = _path_from(flux, "model_dir", "FLUX_MODEL_PATH")
    flux_pythonpath = configured_paths(os.environ.get("FLUX_PYTHONPATH") or flux.get("pythonpath", ""))
    model_ok = bool(flux_model and (flux_model / "model_index.json").is_file()
                    and list((flux_model / "text_encoder").glob("*.safetensors"))
                    and (flux_model / "transformer/diffusion_pytorch_model.safetensors").is_file()
                    and (flux_model / "vae/diffusion_pytorch_model.safetensors").is_file()
                    and (flux_model / "tokenizer").is_dir())
    flux_probe, flux_reason = _probe(flux_py, ["torch", "diffsynth.pipelines.flux2_image"],
                                    flux_pythonpath or [REPO_ROOT])
    flux_ok = model_ok and flux_probe
    checks["FLUX"] = _component("PASS" if flux_ok else "FAIL", "ready" if flux_ok else "model_or_dependency_missing" if not model_ok else flux_reason,
                                model_id=flux.get("model_id"), model_source="ModelScope-local", distilled_steps=4)

    fix = stages.get("fixanything", {})
    fix_py = _path_from(fix, "python", "FIXANYTHING_PYTHON")
    fix_repo = _path_from(fix, "repo", "FIXANYTHING_ROOT")
    fix_diff = _path_from(fix, "diffsynth_root", "FIXANYTHING_DIFFSYNTH_ROOT")
    fix_model = _path_from(fix, "model_dir", "FIXANYTHING_MODEL_PATH")
    lora = _path_from(fix, "lora_path", "FIXANYTHING_LORA_PATH")
    expected_commit = os.environ.get("FIXANYTHING_COMMIT") or str(fix.get("commit", "")).strip()
    commit_ok = bool(fix_repo and expected_commit and _git_head(fix_repo) == expected_commit)
    fix_files_ok = bool(fix_repo and (fix_repo / "scripts/run_inference.py").is_file()
                         and fix_model and (fix_model / "checkpoint_manifest.json").is_file()
                         and lora and lora.is_file() and fix_diff and fix_diff.exists() and commit_ok)
    if fix_files_ok:
        fix_probe, fix_reason = _probe(fix_py, ["torch"], [fix_repo, fix_diff, REPO_ROOT])
    else:
        fix_probe, fix_reason = False, "repo_checkpoint_lora_or_commit_missing"
    fix_ok = fix_files_ok and fix_probe
    checks["FIXANYTHING"] = _component("PASS" if fix_ok else "FAIL", "ready" if fix_ok else fix_reason)

    if not enable_seedvr2:
        checks["SEEDVR2"] = _component("OPTIONAL_DISABLED", "disabled_by_default")
    else:
        seed = cfg.get("seedvr2", {})
        seed_py = _path_from(seed, "python", "SEEDVR2_PYTHON")
        seed_repo = _path_from(seed, "repo", "SEEDVR2_ROOT")
        seed_script = _path_from(seed, "script", "SEEDVR2_INFERENCE_SCRIPT")
        seed_weights = _path_from(seed, "weights", "SEEDVR2_MODEL_PATH")
        seed_vae = _path_from(seed, "vae_weights", "SEEDVR2_VAE_PATH")
        seed_ok = bool(seed_py and seed_repo and seed_script and seed_weights and seed_vae
                       and seed_script.is_file() and seed_weights.is_file() and seed_vae.is_file())
        if seed_ok:
            seed_probe, seed_reason = _probe(seed_py, ["torch", "cv2"])
            seed_ok = seed_probe
        else:
            seed_reason = "repo_or_checkpoint_missing"
        checks["SEEDVR2"] = _component("PASS" if seed_ok else "FAIL", "ready" if seed_ok else seed_reason)

    required = ["CORE", "TRIPOSPLAT", "CAMERA_PLANNER", "PROXY_RENDERER", "FLUX", "FIXANYTHING"]
    if mode == "text":
        required.append("AGENT2TOPDOWN")
    if enable_seedvr2:
        required.append("SEEDVR2")
    failed = any(checks[key]["status"] != "PASS" for key in required)
    return {"FINAL_ENV_STATUS": "FAIL" if failed else "PASS", "MODE": mode, "checks": checks}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate configured stage-specific environments without running inference.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs/default.yaml"))
    parser.add_argument("--mode", choices=("text", "image"), default="text")
    parser.add_argument("--output-dir", default=str(Path(tempfile.gettempdir()) / "pipeline_environment_check"))
    parser.add_argument("--enable-seedvr2", action="store_true")
    args = parser.parse_args(argv)
    result = evaluate(Path(args.config).expanduser(), args.mode, Path(args.output_dir).expanduser(), args.enable_seedvr2)
    for name, item in result["checks"].items():
        print(f"{name:<20} {item['status']:<18} {item.get('reason', '')}")
    print(f"FINAL_ENV_STATUS = {result['FINAL_ENV_STATUS']}")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["FINAL_ENV_STATUS"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
