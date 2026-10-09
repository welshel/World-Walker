"""Adapter for the repository-owned 61-frame Gaussian proxy renderer."""
from __future__ import annotations
from pathlib import Path
from pipeline.utils.subprocess_utils import run_command

def validate_environment(config):
    python=Path(str(config.get("python",""))).expanduser()
    if not python.is_file():raise RuntimeError("PROXY_RENDERER_NOT_CONFIGURED: set PROXY_PYTHON")
    if not __import__("shutil").which("ffmpeg"):raise RuntimeError("ffmpeg is required to encode proxy video")

def build_command(config,ply,trajectory,output_dir):
    size=config.get("resolution",[896,512])
    return [str(config["python"]),"-m","pipeline.camera.proxy_renderer","--ply",str(ply),"--trajectory",str(trajectory),"--output-dir",str(output_dir),"--fps",str(config.get("fps",24)),"--width",str(size[0]),"--height",str(size[1])]

def run(config,ply,trajectory,output_dir,proxy_frames,proxy_video,report_path,log_path,gpu,repo_root):
    validate_environment(config)
    env={"PYTHONPATH":str(repo_root),"CUDA_VISIBLE_DEVICES":str(gpu),"PYTHONUNBUFFERED":"1"}
    ext=str(config.get("gsplat_cuda_extension","")).strip()
    if ext:env["GSPLAT_CUDA_EXTENSION"]=ext
    cmd=build_command(config,ply,trajectory,output_dir);run_command(cmd,repo_root,log_path,env)
    import json,shutil
    out=Path(output_dir); produced=out/"selected_proxy.mp4";frames=out/"frames";manifest=out/"proxy_manifest.json"
    if not produced.is_file() or len(list(frames.glob("*.png")))!=61:raise RuntimeError("proxy renderer did not produce 61 frames and video")
    if Path(proxy_video).resolve()!=produced.resolve():shutil.copy2(produced,proxy_video)
    if Path(proxy_frames).resolve()!=frames.resolve():
        dst=Path(proxy_frames);dst.mkdir(parents=True,exist_ok=True)
        for src in frames.glob("*.png"):shutil.copy2(src,dst/src.name)
    data=json.loads(manifest.read_text());Path(report_path).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(manifest,report_path)
    return {"command":cmd,"gpu_physical_index":str(gpu),"frame_count":data["frame_count"],"fps":data["fps"],"resolution":data["resolution"]}


def collect_outputs(video_path, frame_dir):
    from pipeline.utils.video_io import decoded_frame_count
    video, frames = Path(video_path), Path(frame_dir)
    paths = sorted(frames.glob("*.png")) if frames.is_dir() else []
    if not video.is_file() or len(paths) != 61 or decoded_frame_count(video) != 61:
        raise RuntimeError("proxy output collection failed native 61-frame gate")
    return {"video": video, "frames": frames, "frame_count": 61}
