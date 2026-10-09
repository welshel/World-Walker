"""Stable command adapter for the official TripoSplat inference entrypoint."""
from __future__ import annotations
import shutil, sys
from pathlib import Path
from pipeline.utils.subprocess_utils import run_command
from pipeline.utils.hashing import sha256_file
from pipeline.utils.runtime import artifact_record

def validate_environment(config):
    for key in ("python","snapshot","checkpoint_root"):
        value=str(config.get(key,"")).strip()
        if not value: raise RuntimeError("TRIPOSPLAT_NOT_CONFIGURED: missing stages.triposplat."+key)
        path=Path(value).expanduser()
        if key=="python":
            if not path.is_file(): raise FileNotFoundError("TripoSplat Python interpreter missing: "+str(path))
        elif not path.exists(): raise FileNotFoundError("TripoSplat "+key+" missing: "+str(path))
    if not (Path(config["snapshot"])/"triposplat.py").is_file(): raise FileNotFoundError("TripoSplat snapshot has no triposplat.py")
    for rel in ("diffusion_models/triposplat_fp16.safetensors","vae/triposplat_vae_decoder_fp16.safetensors","clip_vision/dino_v3_vit_h.safetensors","vae/flux2-vae.safetensors","background_removal/birefnet.safetensors"):
        if not (Path(config["checkpoint_root"])/rel).is_file(): raise FileNotFoundError("TripoSplat checkpoint missing: "+rel)

def build_command(config,input_image,output_dir):
    runner=Path(__file__).with_name("triposplat_official_runner.py")
    return [str(config["python"]),str(runner),"--snapshot",str(config["snapshot"]),"--ckpt-root",str(config["checkpoint_root"]),"--input",str(input_image),"--out",str(output_dir),"--run-id","release1","--seed",str(config.get("seed",420042)),"--steps",str(config.get("steps",20)),"--guidance-scale",str(config.get("guidance_scale",3.0)),"--shift",str(config.get("shift",3.0)),"--num-gaussians",str(config.get("num_gaussians",262144))]

def run(config,input_image,output_dir,ply_path,manifest_path,log_path,gpu,repo_root):
    validate_environment(config)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    env={"CUDA_VISIBLE_DEVICES":str(gpu),"PYTHONUNBUFFERED":"1"}
    paths=config.get("pythonpath",[])
    if isinstance(paths,str): paths=[x for x in paths.split(":") if x]
    paths=[str(repo_root)]+list(map(str,paths))
    env["PYTHONPATH"]=":".join(paths)
    command=build_command(config,input_image,out)
    run_command(command,repo_root,log_path,env)
    raw=out/"raw_official_scene.ply"
    if not raw.is_file(): raise RuntimeError("TripoSplat returned without raw_official_scene.ply")
    final=Path(ply_path)
    if raw.resolve()!=final.resolve(): shutil.copy2(raw,final)
    manifest=out/"inference_manifest.json"
    if not manifest.is_file(): raise RuntimeError("TripoSplat wrapper did not produce inference_manifest.json")
    import json
    data=json.loads(manifest.read_text())
    if data.get("input",{}).get("sha256")!=sha256_file(input_image): raise RuntimeError("TripoSplat input hash mismatch")
    if not data.get("gaussian_integrity",{}).get("valid"): raise RuntimeError("TripoSplat Gaussian integrity gate failed")
    data["release1_output_ply"]={"path":str(final.resolve()),"sha256":sha256_file(final),"size_bytes":final.stat().st_size}
    manifest.write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n")
    Path(manifest_path).parent.mkdir(parents=True,exist_ok=True)
    if Path(manifest_path).resolve()!=manifest.resolve():shutil.copy2(manifest,manifest_path)
    return {"command":command,"gpu_physical_index":str(gpu),"input_sha256":sha256_file(input_image),"ply_sha256":sha256_file(final),"gaussian_count":data.get("num_gaussians"),"manifest":str(manifest.resolve())}


def collect_outputs(ply_path, manifest_path):
    ply, manifest = Path(ply_path), Path(manifest_path)
    if not ply.is_file() or not manifest.is_file():
        raise RuntimeError("TripoSplat output collection failed: PLY or manifest missing")
    import json
    data = json.loads(manifest.read_text())
    if not data.get("gaussian_integrity", {}).get("valid"):
        raise RuntimeError("TripoSplat output failed its Gaussian integrity gate")
    if data.get("input", {}).get("sha256") != sha256_file(data.get("input", {}).get("path", "")):
        raise RuntimeError("TripoSplat manifest input hash no longer matches")
    return {"ply": ply, "manifest": manifest, "sha256": sha256_file(ply)}
