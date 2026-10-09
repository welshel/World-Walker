"""Adapter for the repository-owned visibility and collision camera planner."""
from __future__ import annotations
import sys
from pathlib import Path
from pipeline.utils.subprocess_utils import run_command

def validate_environment(config):
    python=Path(str(config.get("python",""))).expanduser()
    if not python.is_file():raise RuntimeError("CAMERA_PLANNER_NOT_CONFIGURED: set CAMERA_PYTHON to an interpreter with numpy, scipy, Pillow, torch and gsplat")
    if str(config.get("up_axis","-Y")) not in ("+X","-X","+Y","-Y","+Z","-Z"):raise ValueError("camera up_axis must be an explicit signed axis")

def build_command(config,ply_path,output_dir,repo_root):
    return [str(config["python"]),"-m","pipeline.camera.planner","--ply",str(ply_path),"--output-dir",str(output_dir),"--up-axis",str(config.get("up_axis","-Y"))]

def run(config,ply_path,output_dir,report_path,trajectory_path,log_path,gpu,repo_root):
    validate_environment(config)
    env={"PYTHONPATH":str(repo_root),"CUDA_VISIBLE_DEVICES":str(gpu),"PYTHONUNBUFFERED":"1"}
    extension=str(config.get("gsplat_cuda_extension","")).strip()
    if extension:env["GSPLAT_CUDA_EXTENSION"]=extension
    command=build_command(config,ply_path,output_dir,repo_root)
    run_command(command,repo_root,log_path,env)
    import json,shutil
    report=Path(output_dir)/"report.json";trajectory=Path(output_dir)/"selected_cameras_world_c2w.json"
    if not report.is_file() or not trajectory.is_file():raise RuntimeError("camera planner output is incomplete")
    if Path(report_path).resolve()!=report.resolve():shutil.copy2(report,report_path)
    if Path(trajectory_path).resolve()!=trajectory.resolve():shutil.copy2(trajectory,trajectory_path)
    data=json.loads(report.read_text())
    if data.get("FINAL_STATUS")!="CAMERA_PASS_VISIBILITY_SCORED_NATIVE_61F" or not data.get("collision_free") or not data.get("motion_smooth"):raise RuntimeError("camera planner safety or native-frame gate failed")
    return {"command":command,"gpu_physical_index":str(gpu),"selected_candidate":data["selected_candidate"],"metrics":data}


def collect_outputs(report_path, trajectory_path):
    import json
    report, trajectory = Path(report_path), Path(trajectory_path)
    if not report.is_file() or not trajectory.is_file():
        raise RuntimeError("camera planner output collection failed")
    data = json.loads(report.read_text())
    if data.get("FINAL_STATUS") != "CAMERA_PASS_VISIBILITY_SCORED_NATIVE_61F" or not data.get("collision_free"):
        raise RuntimeError("camera planner output did not pass safety gates")
    return {"report": report, "trajectory": trajectory}
