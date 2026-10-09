#!/usr/bin/env python3
"""Thin adapter around the audited Agent2Topdown module; no core reimplementation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def resolve_input_mode(image_path: str | None, scene_prompt: str | None) -> tuple[str, str]:
    image_present = image_path is not None and bool(str(image_path).strip())
    prompt_present = scene_prompt is not None and bool(str(scene_prompt).strip())
    if image_present == prompt_present:
        raise ValueError("provide exactly one of --input-topdown/--input or --scene-prompt")
    return ("image", str(image_path)) if image_present else ("text", str(scene_prompt))


def module_code_hash(root: Path) -> str:
    files = sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    requirements = root / "requirements.txt"
    if requirements.is_file():
        files.append(requirements)
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def run(
    prompt: str,
    agent_root: Path,
    output_root: Path,
    generated_output: Path,
    manifest_path: Path,
    mock: bool = False,
) -> dict:
    agent_root = agent_root.resolve()
    output_root = output_root.resolve()
    generated_output = generated_output.resolve()
    manifest_path = manifest_path.resolve()
    started_unix = time.time()
    tick = time.monotonic()
    data = {
        "SCENE_PROMPT": prompt,
        "AGENT2TOPDOWN_ROOT": str(agent_root),
        "ENTRYPOINT": str(agent_root / "app.py"),
        "CODE_SHA256": None,
        "MODEL_OR_SERVICE_INFO": None,
        "SEED": None,
        "LAYOUT_RENDERER_SEED": 17,
        "OUTPUT_PATH": str(generated_output),
        "OUTPUT_SHA256": None,
        "WIDTH": None,
        "HEIGHT": None,
        "IMAGE_MODE": None,
        "RUNTIME_SEC": None,
        "RAW_RUN_DIR": None,
        "INTERMEDIATE_ARTIFACTS": [],
        "MODE": None,
        "STATUS": "FAIL",
        "ERROR": None,
        "STARTED_UNIX": started_unix,
    }
    previous_cwd = Path.cwd()
    try:
        if not (agent_root / "app.py").is_file():
            raise FileNotFoundError(f"Agent2Topdown entrypoint missing: {agent_root / 'app.py'}")
        data["CODE_SHA256"] = module_code_hash(agent_root)
        sys.path.insert(0, str(agent_root))
        os.chdir(agent_root)
        from app import run_pipeline

        raw_root = output_root / "outputs"
        raw_dir, metadata = run_pipeline(prompt, mock=mock, output_root=raw_root)
        raw_dir = Path(raw_dir).resolve()
        source = raw_dir / "layout.png"
        if metadata.get("status") != "success" or not source.is_file():
            raise RuntimeError("Agent2Topdown returned no successful layout.png")
        generated_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, generated_output)
        from PIL import Image

        with Image.open(generated_output) as image:
            width, height = image.size
            image_mode = image.mode
        intermediates = []
        for artifact in sorted(raw_dir.rglob("*")):
            if artifact.is_file():
                intermediates.append(
                    {
                        "path": str(artifact.resolve()),
                        "size_bytes": artifact.stat().st_size,
                        "sha256": sha256(artifact),
                    }
                )
        run_mode = metadata.get("mode")
        service = (
            os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            if run_mode == "deepseek_api"
            else None
        )
        data.update(
            {
                "STATUS": "PASS_MOCK_FALLBACK" if run_mode == "mock" else "PASS",
                "MODE": run_mode,
                "MODEL_OR_SERVICE_INFO": {
                    "model": metadata.get("model"),
                    "service": service,
                    "fallback": "builtin_mock" if run_mode == "mock" else None,
                    "temperature": 0.2 if run_mode == "deepseek_api" else None,
                    "api_key_present": bool(os.environ.get("DEEPSEEK_API_KEY")),
                },
                "OUTPUT_SHA256": sha256(generated_output),
                "WIDTH": width,
                "HEIGHT": height,
                "IMAGE_MODE": image_mode,
                "RUNTIME_SEC": time.monotonic() - tick,
                "RAW_RUN_DIR": str(raw_dir),
                "AGENT_METADATA": metadata,
                "INTERMEDIATE_ARTIFACTS": intermediates,
                "OUTPUT_PATH": str(generated_output),
            }
        )
    except Exception as exc:
        data.update(
            {
                "STATUS": "FAIL",
                "ERROR": f"{type(exc).__name__}: {exc}",
                "RUNTIME_SEC": time.monotonic() - tick,
            }
        )
        raise
    finally:
        os.chdir(previous_cwd)
        atomic_json(manifest_path, data)
    return data


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-prompt", required=True)
    parser.add_argument("--agent-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--generated-output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args(argv)
    result = run(
        args.scene_prompt,
        Path(args.agent_root),
        Path(args.output_root),
        Path(args.generated_output),
        Path(args.manifest),
        args.mock,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

