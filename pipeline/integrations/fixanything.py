"""Adapter and worker for the audited FixAnything/Wan inference API."""
from __future__ import annotations
import argparse, json, os, subprocess, time
from pathlib import Path
from PIL import Image
from pipeline.utils.image_io import resize_cover
from pipeline.utils.hashing import sha256_file
from pipeline.utils.subprocess_utils import run_command

def prepare_inputs(proxy_frames,anchor000,anchor060,input_dir,manifest_path,provenance_path,width=832,height=480):
    src=Path(proxy_frames);dst=Path(input_dir);dst.mkdir(parents=True,exist_ok=True)
    anchors={0:Path(anchor000),60:Path(anchor060)}
    for i,a in anchors.items():
        if not a.is_file():raise FileNotFoundError("required endpoint clean anchor missing: "+str(a))
    proxies=[src/f"{i:03d}.png" for i in range(61)]
    if any(not p.is_file() for p in proxies):raise RuntimeError("FixAnything requires all 61 native proxy frames")
    for old in dst.glob("*.png"):old.unlink()
    records=[]
    for i,p in enumerate(proxies):
        source=anchors.get(i,p)
        with Image.open(source) as image:resized=resize_cover(image.convert("RGB"),width,height);size=list(image.size)
        out=dst/f"{i:03d}.png";resized.save(out)
        records.append({"frame":i,"source":str(source.resolve()),"source_sha256":sha256_file(source),"output":str(out.resolve()),"output_sha256":sha256_file(out),"source_size":size,"output_size":[width,height],"source_type":"FLUX_clean_anchor" if i in anchors else "native_proxy","clean_anchor":i in anchors})
    manifest={"status":"READY_FOR_FIXANYTHING","native_input_frames":61,"native_proxy_frames":61,"output_frames":61,"temporal_resampling":False,"clean_frame_indices":[0,60],"resolution":[width,height],"fps":24,"resize_policy":"cover-center bilinear","frames":records}
    Path(manifest_path).parent.mkdir(parents=True,exist_ok=True);Path(manifest_path).write_text(json.dumps(manifest,indent=2)+"\n")
    prov={"status":"READY_FOR_FIXANYTHING","clean_frame_indices":[0,60],"temporal_resampling":False,"source_proxy_frames":str(src.resolve()),"input_sequence":str(dst.resolve()),"anchor_mapping":[{"global_frame":i,"source":str(anchors[i].resolve()),"source_sha256":sha256_file(anchors[i]),"input_path":records[i]["output"],"input_sha256":records[i]["output_sha256"]} for i in (0,60)]}
    Path(provenance_path).parent.mkdir(parents=True,exist_ok=True);Path(provenance_path).write_text(json.dumps(prov,indent=2)+"\n")
    return manifest,prov

def validate_environment(config):
    for k in ("python","repo","diffsynth_root","model_dir","lora_path"):
        v=str(config.get(k,"")).strip()
        if not v:raise RuntimeError("FIXANYTHING_NOT_CONFIGURED: missing "+k)
        p=Path(v).expanduser()
        if not p.exists():raise FileNotFoundError("FixAnything configuration missing "+k+": "+v)
    expected=str(config.get("commit","")).strip()
    repo=Path(config["repo"])
    if expected:
        actual=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
        if actual!=expected:raise RuntimeError(f"FixAnything commit mismatch: expected {expected}, got {actual}")

def worker(cfg_path):
    cfg=json.loads(Path(cfg_path).read_text());validate_environment(cfg)
    repo=Path(cfg["repo"]).resolve();diff=Path(cfg["diffsynth_root"]).resolve()
    sys_path=[str(repo),str(diff)]
    import sys
    sys.path[:0]=sys_path
    import torch
    from scripts.run_inference import load_pipeline,fix_video
    from fixanything.data import load_frames,save_video
    frames,_=load_frames(cfg["input_dir"],height=480,width=832,num_frames=61)
    if len(frames)!=61:raise RuntimeError(f"FixAnything received {len(frames)} frames, expected 61")
    torch.cuda.reset_peak_memory_stats();t0=time.time()
    pipe=load_pipeline(cfg["lora_path"],cfg["model_dir"],device="cuda",download_source="huggingface")
    generated,_=fix_video(pipe,frames,clean_frame_indices=[0,60],num_frames=61,num_repeat_last=4,height=480,width=832,seed=int(cfg.get("seed",1)),num_inference_steps=int(cfg.get("steps",10)),cfg_scale=float(cfg.get("cfg",5.0)),tiled=bool(cfg.get("tiled",True)))
    if len(generated)!=61:raise RuntimeError(f"FixAnything returned {len(generated)} frames, expected 61")
    out=Path(cfg["output_video"]);out.parent.mkdir(parents=True,exist_ok=True);save_video(generated,str(out),fps=24)
    frames_dir=Path(cfg["output_frames_dir"]);frames_dir.mkdir(parents=True,exist_ok=True)
    for i,im in enumerate(generated):im.save(frames_dir/f"{i:03d}.png")
    model_manifest=Path(cfg["model_dir"])/"checkpoint_manifest.json"
    report={"status":"PASS","model_dir":str(Path(cfg["model_dir"]).resolve()),"official_repo":str(repo),"python":sys.executable,"lora":str(Path(cfg["lora_path"]).resolve()),"checkpoint_manifest":str(model_manifest.resolve()),"output_video":str(out.resolve()),"output_sha256":sha256_file(out),"input_frames":61,"output_frames":len(generated),"fps":24,"resolution":[832,480],"clean_frame_indices":[0,60],"seed":cfg.get("seed",1),"steps":cfg.get("steps",10),"cfg":cfg.get("cfg",5.0),"temporal_resampling":False,"internal_padding":"FixAnything repeat-last=4; returns first 61 model frames","runtime_sec":time.time()-t0,"gpu":torch.cuda.get_device_name(0),"peak_memory_gib":float(torch.cuda.max_memory_allocated()/2**30),"official_commit":cfg.get("commit"),"num_repeat_last":4,"total_seconds":time.time()-t0}
    Path(cfg["report_path"]).write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
    return report


def build_command(python, worker_config):
    return [str(python), "-m", "pipeline.integrations.fixanything", "--worker-config", str(worker_config)]

def run(config,input_dir,output_video,frames_dir,report_path,log_path,repo_root,gpu):
    validate_environment(config)
    worker_cfg={**config,"input_dir":str(input_dir),"output_video":str(output_video),"output_frames_dir":str(frames_dir),"report_path":str(report_path)}
    cfg_path=Path(report_path).with_name("worker_config.json");cfg_path.parent.mkdir(parents=True,exist_ok=True);cfg_path.write_text(json.dumps(worker_cfg,indent=2)+"\n")
    deps=[str(repo_root),str(config["repo"]),str(config["diffsynth_root"])]
    env={"PYTHONPATH":":".join(deps),"CUDA_VISIBLE_DEVICES":str(gpu),"PYTHONUNBUFFERED":"1"}
    run_command(build_command(config["python"], cfg_path),repo_root,log_path,env)
    return json.loads(Path(report_path).read_text())

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--worker-config",required=True);a=ap.parse_args();worker(a.worker_config)
if __name__=="__main__":main()


def collect_outputs(video_path, frames_dir, report_path):
    from pipeline.utils.video_io import decoded_frame_count
    video, frames, report = Path(video_path), Path(frames_dir), Path(report_path)
    if not video.is_file() or not report.is_file() or len(list(frames.glob("*.png"))) != 61:
        raise RuntimeError("FixAnything output collection failed")
    if decoded_frame_count(video) != 61:
        raise RuntimeError("FixAnything output is not 61 frames")
    return {"video": video, "frames": frames, "report": report, "frame_count": 61}
