#!/usr/bin/env python3
"""Run the garden scene planning MVP from a natural-language prompt."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from src.deepseek_client import DeepSeekClient, extract_json_object
from src.flux_prompt_builder import build_flux_prompt
from src.layout_renderer import render_layout
from src.mock_scene import build_mock_scene
from src.utils import load_dotenv, utc_timestamp, write_json
from src.validator import SceneValidationError, validate_scene


def _new_run_dir(root=Path("outputs")):
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = root / f"run_{stamp}"
    suffix = 1
    while candidate.exists():
        candidate = root / f"run_{stamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def run_pipeline(user_prompt, mock=False, output_root=Path("outputs")):
    load_dotenv()
    run_dir = _new_run_dir(output_root)
    (run_dir / "user_input.txt").write_text(user_prompt.strip() + "\n", encoding="utf-8")
    model = "built-in-mock"
    retry_count = 0
    raw_attempts = []
    try:
        if mock:
            scene = build_mock_scene()
            raw = json.dumps(scene, ensure_ascii=False, indent=2)
            raw_attempts.append("[mock fallback]\n" + raw)
        else:
            try:
                client = DeepSeekClient()
            except ValueError as exc:
                raise RuntimeError(f"AGENT2TOPDOWN_PROVIDER_NOT_CONFIGURED: {exc}") from exc
            model = client.model
            repair_context = None
            scene = None
            errors = ""
            for attempt in range(2):
                raw = client.generate(user_prompt, repair_context=repair_context)
                raw_attempts.append(f"[attempt {attempt + 1}]\n" + raw)
                try:
                    scene = extract_json_object(raw)
                    validate_scene(scene)
                    break
                except (ValueError, SceneValidationError) as exc:
                    errors = str(exc)
                    if attempt == 0:
                        retry_count = 1
                        repair_context = raw + "\n--- VALIDATION ERRORS ---\n" + errors
                    else:
                        raise RuntimeError(f"DeepSeek scene remained invalid after one repair attempt. {errors}") from exc

        raw_text = "\n\n".join(raw_attempts)
        (run_dir / "raw_deepseek_response.txt").write_text(raw_text + "\n", encoding="utf-8")
        write_json(run_dir / "scene.json", scene)
        validated = validate_scene(scene)
        write_json(run_dir / "validated_scene.json", validated)
        render_layout(validated, run_dir / "layout.png")
        (run_dir / "flux_prompt.txt").write_text(build_flux_prompt(validated) + "\n", encoding="utf-8")
        metadata = {
            "timestamp": utc_timestamp(),
            "user_prompt": user_prompt,
            "model": model,
            "status": "success",
            "output_dir": str(run_dir),
            "validation_retry_count": retry_count,
            "mode": "mock" if model == "built-in-mock" else "deepseek_api",
        }
        write_json(run_dir / "metadata.json", metadata)
        return run_dir, metadata
    except Exception as exc:
        (run_dir / "raw_deepseek_response.txt").write_text("\n\n".join(raw_attempts) + "\n", encoding="utf-8")
        metadata = {
            "timestamp": utc_timestamp(),
            "user_prompt": user_prompt,
            "model": model,
            "status": "error",
            "output_dir": str(run_dir),
            "validation_retry_count": retry_count,
            "error": str(exc),
        }
        write_json(run_dir / "metadata.json", metadata)
        raise RuntimeError(f"Pipeline failed; raw response and error metadata saved in {run_dir}: {exc}") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate a top-down garden layout and FLUX prompt.")
    parser.add_argument("--prompt", required=True, help="Natural-language description of the garden scene")
    parser.add_argument("--mock", action="store_true", help="Use the built-in scene instead of calling DeepSeek")
    args = parser.parse_args(argv)
    try:
        run_dir, metadata = run_pipeline(args.prompt, mock=args.mock)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Generated {metadata['mode']} garden scene in: {run_dir}")
    print("Files: scene.json, validated_scene.json, layout.png, flux_prompt.txt, metadata.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
