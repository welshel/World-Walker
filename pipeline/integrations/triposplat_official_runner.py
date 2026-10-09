#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pipeline.integrations.triposplat_tools.core import array_sha256, atomic_json, file_record, gaussian_integrity, sha256


def tensor_record(value: torch.Tensor) -> dict:
    cpu = value.detach().float().cpu().contiguous()
    arr = cpu.numpy()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": bool(torch.isfinite(cpu).all()),
        "min": float(cpu.min()),
        "max": float(cpu.max()),
        "mean": float(cpu.mean()),
        "std": float(cpu.std(unbiased=False)),
        "norm": float(torch.linalg.vector_norm(cpu)),
        "array_sha256": array_sha256(arr),
    }


def save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temp, path)


def preprocessing_trace(path: Path) -> dict:
    with Image.open(path) as source:
        image = source.convert("RGBA") if source.mode == "RGBA" else source.convert("RGB")
        original_size = list(source.size)
        original_mode = source.mode
        real_alpha = source.mode == "RGBA" and source.getchannel("A").getextrema()[0] < 255
        scale = 1024 / min(source.size)
        resized = [max(1, int(round(source.size[0] * scale))), max(1, int(round(source.size[1] * scale)))]
        if real_alpha:
            alpha = np.asarray(image.resize(tuple(resized), Image.Resampling.LANCZOS).getchannel("A"))
            ys, xs = np.nonzero(alpha)
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        else:
            bbox = None
    return {
        "input_path": str(path),
        "input_sha256": sha256(path),
        "original_mode": original_mode,
        "original_resolution": original_size,
        "official_canvas_size": 1024,
        "initial_resize": resized,
        "real_alpha_detected": real_alpha,
        "background_removal_invoked": not real_alpha,
        "alpha_bbox_after_initial_resize_before_erosion": bbox,
        "erode_radius": 1,
        "crop_policy": "official alpha bbox, square, 1.2x half extent, then 1024x1024",
        "final_composite_background": "black RGB",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--ckpt-root", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=420042)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--num-gaussians", type=int, default=262144)
    args = parser.parse_args()

    snapshot = Path(args.snapshot).resolve()
    ckpt = Path(args.ckpt_root).resolve()
    input_path = Path(args.input).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if "p13" in str(ckpt).lower():
        raise RuntimeError("P13 path rejected by official-only guard")

    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    sys.path.insert(0, str(snapshot))
    from triposplat import TripoSplatPipeline

    weights = {
        "DiT": ckpt / "diffusion_models/triposplat_fp16.safetensors",
        "VAE_Gaussian_decoder": ckpt / "vae/triposplat_vae_decoder_fp16.safetensors",
        "DINOv3": ckpt / "clip_vision/dino_v3_vit_h.safetensors",
        "Flux2_VAE_encoder": ckpt / "vae/flux2-vae.safetensors",
        "BiRefNet": ckpt / "background_removal/birefnet.safetensors",
    }
    for path in weights.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    started = time.time()
    pipe = TripoSplatPipeline(
        ckpt_path=str(weights["DiT"]),
        decoder_path=str(weights["VAE_Gaussian_decoder"]),
        dinov3_path=str(weights["DINOv3"]),
        flux2_vae_encoder_path=str(weights["Flux2_VAE_encoder"]),
        rmbg_path=str(weights["BiRefNet"]),
        device="cuda",
    )
    for component in (pipe.dinov3, pipe.vae_encoder, pipe.rmbg, pipe.flow_model, pipe.decoder):
        component.requires_grad_(False).eval()

    trace = preprocessing_trace(input_path)
    prepared = pipe.preprocess_image(input_path, erode_radius=1)
    prepared_path = out / "prepared_condition.png"
    prepared.save(prepared_path)
    generator = torch.Generator(device=pipe._device).manual_seed(args.seed)
    with torch.inference_mode():
        condition = pipe.encode_image(prepared, generator=generator)
        sample = pipe.sample_latent(
            condition,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            shift=args.shift,
            generator=generator,
            show_progress=True,
        )
        latent = sample["latent"]
        camera = sample.get("camera")
        gaussian = pipe.decode_latent(latent, num_gaussians=args.num_gaussians)

    condition_arrays = {key: value.detach().float().cpu().numpy() for key, value in condition.items()}
    latent_arrays = {"latent": latent.detach().float().cpu().numpy()}
    if camera is not None:
        latent_arrays["camera"] = camera.detach().float().cpu().numpy()
    gaussian_arrays = {
        "xyz": gaussian.get_xyz.detach().float().cpu().numpy(),
        "opacity": gaussian.get_opacity.detach().float().cpu().numpy(),
        "scaling": gaussian.get_scaling.detach().float().cpu().numpy(),
        "rotation": (gaussian._rotation + gaussian.rots_bias[None, :]).detach().float().cpu().numpy(),
        "features_dc": gaussian._features_dc.detach().float().cpu().numpy(),
    }
    condition_path = out / "condition_embeddings.npz"
    latent_path = out / "predicted_latent.npz"
    gaussian_path = out / "gaussian_internal.npz"
    save_npz_atomic(condition_path, **condition_arrays)
    save_npz_atomic(latent_path, **latent_arrays)
    save_npz_atomic(gaussian_path, **gaussian_arrays)
    ply_path = out / "raw_official_scene.ply"
    ply_temp = ply_path.with_suffix(".ply.tmp")
    gaussian.save_ply(ply_temp)
    os.replace(ply_temp, ply_path)

    trace["prepared"] = file_record(prepared_path)
    atomic_json(out / "input_preprocessing_trace.json", trace)
    manifest = {
        "format": "p16.p0.official_run.v1",
        "run_id": args.run_id,
        "snapshot": str(snapshot),
        "input": file_record(input_path),
        "weights": {name: str(path) for name, path in weights.items()},
        "NO_P13_ADAPTED_WEIGHT_LOADED": True,
        "seed": args.seed,
        "rng_seeded": ["python.random", "numpy", "torch CPU", "torch CUDA all", "local CUDA Generator"],
        "steps": args.steps,
        "sampler": "official FlowEulerCfgSampler",
        "guidance_scale": args.guidance_scale,
        "shift": args.shift,
        "num_gaussians": args.num_gaussians,
        "prepared": file_record(prepared_path),
        "condition": {key: tensor_record(value) for key, value in condition.items()},
        "condition_file": file_record(condition_path),
        "latent": tensor_record(latent),
        "camera": tensor_record(camera) if camera is not None else None,
        "latent_file": file_record(latent_path),
        "gaussian_array_hashes": {key: array_sha256(value) for key, value in gaussian_arrays.items()},
        "gaussian_integrity": gaussian_integrity(gaussian_arrays),
        "gaussian_file": file_record(gaussian_path),
        "ply": file_record(ply_path),
        "ply_export": "official Gaussian.save_ply default transform (x,-z,y)",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        "elapsed_sec": time.time() - started,
        "training_run": False,
        "optimization_run": False,
        "gt_loaded": False,
    }
    atomic_json(out / "inference_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
