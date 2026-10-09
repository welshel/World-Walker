#!/usr/bin/env python3
"""FLUX.2 Klein garden topdown refinement with an explicit visual-review gate."""
from __future__ import annotations
import json
import shutil
import time
from pathlib import Path

from pipeline.utils.hashing import sha256_file, sha256_path
from pipeline.utils.image_io import image_info, make_transparent_white_border
from pipeline.utils.runtime import atomic_json, gpu_choice, load_json, run_command


def _resolve_candidate(path_value: str, project: Path) -> Path:
    path = Path(path_value)
    return path.resolve() if path.is_absolute() else (project / path).resolve()


def run(cfg: dict, paths: dict, base: Path, project: Path, preferred_gpu: str = "auto", force: bool = False) -> dict:
    source = paths["agent_generated_topdown"]
    candidate_root = paths["flux_topdown_candidates"]
    metadata_path = paths["flux_topdown_metadata"]
    review_path = paths["flux_topdown_review"]
    runner_value = Path(cfg.get("script", paths["flux_topdown_script"])).expanduser()
    runner = (runner_value if runner_value.is_absolute() else project / runner_value).resolve()
    model_value = Path(cfg["model_dir"]).expanduser()
    model_dir = (model_value if model_value.is_absolute() else project / model_value).resolve()
    candidate_root.parent.mkdir(parents=True, exist_ok=True)
    details = {
        "model_id": cfg.get("model_id"),
        "model_source": cfg.get("model_source", "ModelScope local weights"),
        "num_inference_steps": int(cfg.get("num_inference_steps", 4)),
        "seeds": list(cfg.get("seeds", [0, 1, 2, 3])),
        "input_path": str(source),
        "input_sha256": sha256_file(source),
        "candidate_directory": str(candidate_root),
        "review_path": str(review_path),
        "command": None,
        "gpu_physical_index": None,
    }

    valid_metadata = False
    if metadata_path.is_file() and not force:
        try:
            metadata = load_json(metadata_path)
            valid_metadata = (
                metadata.get("FINAL_STATUS") == "PASS"
                and metadata.get("INPUT_SHA256") == details["input_sha256"]
                and metadata.get("SEEDS_COMPLETED") == details["seeds"]
                and all(Path(row.get("path", "")).is_file() for row in metadata.get("per_seed", []))
            )
        except Exception:
            valid_metadata = False
    if force or not valid_metadata:
        if candidate_root.exists():
            backup = candidate_root.with_name(candidate_root.name + f"_pre_force_{int(time.time())}")
            candidate_root.replace(backup)
            details["previous_candidates_preserved_at"] = str(backup)
        candidate_root.mkdir(parents=True, exist_ok=True)
        gpu = gpu_choice(preferred_gpu)
        python = str(cfg["python"])
        pythonpath = str(cfg.get("pythonpath", ""))
        env = {"CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1"}
        if pythonpath:
            env["PYTHONPATH"] = pythonpath
        command = [
            python, str(runner), "--source", str(source), "--out", str(candidate_root),
            "--model-dir", str(model_dir), "--seeds", *[str(s) for s in details["seeds"]],
        ]
        details["gpu_physical_index"] = gpu
        details["command"] = run_command(command, project, base / "logs/stage_flux_topdown.log", env)
        metadata = load_json(metadata_path)
        if metadata.get("FINAL_STATUS") != "PASS" or metadata.get("INPUT_SHA256") != details["input_sha256"]:
            raise RuntimeError("FLUX topdown generation failed or its input hash does not match Agent2Topdown output")

    details.update(
        candidate_outputs=metadata.get("OUTPUTS", []),
        prompt_path=str(candidate_root / "prompt.txt"),
        prompt_sha256=metadata.get("PROMPT_SHA256"),
        candidate_contact_sheet=str(candidate_root / "contact_sheet.png"),
        peak_memory_gib=max((r.get("peak_memory_gib", 0) for r in metadata.get("per_seed", [])), default=None),
    )

    if not review_path.is_file():
        details.update(
            STATUS="WAITING_VISUAL_REVIEW",
            REVIEW_REQUIRED=True,
            INSTRUCTIONS="Inspect outputs/contact_sheet.png and candidate images. Write visual_review.json with a selected seed, strict-nadir PASS, no-horizon PASS, no-oblique-facade PASS, and APPROVED_FOR_TRIPOSPLAT=true before resuming.",
        )
        return details

    review = load_json(review_path)
    seed = int(review.get("SELECTED_SEED", -1))
    if review.get("APPROVED_FOR_TRIPOSPLAT") is not True:
        raise RuntimeError("visual review did not approve this candidate for TripoSplat")
    if review.get("STRICT_NADIR_VIEW") != "PASS":
        raise RuntimeError("strict true-nadir visual gate failed; refusing TripoSplat")
    if review.get("NO_HORIZON") != "PASS" or review.get("NO_OBLIQUE_FACADE") != "PASS":
        raise RuntimeError("topdown view audit failed; refusing TripoSplat")
    selected = next((row for row in metadata.get("per_seed", []) if int(row.get("seed", -1)) == seed), None)
    if selected is None:
        raise RuntimeError(f"review selected seed {seed}, but that seed did not complete")
    candidate = Path(selected["path"]).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    candidate_sha = sha256_file(candidate)
    reviewed_candidate = _resolve_candidate(review.get("CANDIDATE", ""), project)
    if reviewed_candidate != candidate or review.get("CANDIDATE_SHA256") != candidate_sha:
        raise RuntimeError("visual review candidate path/hash does not match the selected FLUX output")
    shutil.copy2(candidate, paths["flux_topdown_candidate"])
    border_record = make_transparent_white_border(
        paths["flux_topdown_candidate"], paths["topdown_used"], int(cfg.get("border_px", 64))
    )
    manifest = {
        "model_id": cfg.get("model_id"),
        "model_source": cfg.get("model_source", "ModelScope local weights"),
        "num_inference_steps": int(cfg.get("num_inference_steps", 4)),
        "selected_seed": seed,
        "agent_layout_path": str(source),
        "agent_layout_sha256": details["input_sha256"],
        "selected_topdown_path": str(paths["flux_topdown_candidate"]),
        "selected_topdown_sha256": candidate_sha,
        "topdown_used_path": str(paths["topdown_used"]),
        "topdown_used_sha256": sha256_file(paths["topdown_used"]),
        "visual_review_path": str(review_path),
        "visual_review_sha256": sha256_file(review_path),
        "strict_nadir_view": review.get("STRICT_NADIR_VIEW"),
        "major_layout_adherence": review.get("MAJOR_LAYOUT_ADHERENCE"),
        "visual_assessment": review.get("VISUAL_ASSESSMENT"),
        "border": border_record,
        "topdown_used_image": image_info(paths["topdown_used"]),
        "created_unix": time.time(),
    }
    atomic_json(paths["flux_topdown_manifest"], manifest)
    details.update(
        STATUS="PASS",
        model_id=manifest["model_id"],
        model_source=manifest["model_source"],
        num_inference_steps=manifest["num_inference_steps"],
        selected_seed=seed,
        strict_nadir_view=manifest["strict_nadir_view"],
        major_layout_adherence=manifest["major_layout_adherence"],
        selected_topdown_sha256=candidate_sha,
        topdown_used_path=str(paths["topdown_used"]),
        topdown_used_sha256=manifest["topdown_used_sha256"],
        topdown_used_mode=manifest["topdown_used_image"]["mode"],
        topdown_used_size=manifest["topdown_used_image"]["size"],
        border=border_record,
    )
    return details
