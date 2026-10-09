from __future__ import annotations
import argparse, hashlib, json, subprocess, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from pipeline.camera.gaussian import read_gaussian_ply, render

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--ply",required=True);ap.add_argument("--trajectory",required=True);ap.add_argument("--output-dir",required=True);ap.add_argument("--fps",type=int,default=24);ap.add_argument("--width",type=int,default=896);ap.add_argument("--height",type=int,default=512);a=ap.parse_args()
    t0=time.monotonic();out=Path(a.output_dir);frames_dir=out/"frames";frames_dir.mkdir(parents=True,exist_ok=True)
    for p in frames_dir.glob("*.png"):p.unlink()
    scene,info=read_gaussian_ply(a.ply);traj=json.loads(Path(a.trajectory).read_text());frames=traj.get("frames",[])
    if len(frames)!=61 or [f.get("frame",i) for i,f in enumerate(frames)]!=list(range(61)):raise RuntimeError("native 61-frame camera JSON required")
    K=np.asarray(traj.get("intrinsics",frames[0]["intrinsics"]),np.float32);rows=[]
    for i,f in enumerate(frames):
        im,_,alpha=render(scene,np.linalg.inv(np.asarray(f["c2w"],np.float32)),K,a.width,a.height);u8=np.uint8(im*255);path=frames_dir/f"{i:03d}.png";Image.fromarray(u8).save(path)
        rows.append({"frame":i,"path":str(path.resolve()),"alpha_coverage":float((alpha>.01).mean()),"black_background_fraction":float((u8.max(2)<8).mean())})
    sheet=Image.new("RGB",(a.width*3,(a.height+24)*3),(20,20,20));d=ImageDraw.Draw(sheet)
    for j,i in enumerate((0,10,20,30,40,50,60)):
        im=Image.open(frames_dir/f"{i:03d}.png");x=(j%3)*a.width;y=(j//3)*(a.height+24);sheet.paste(im,(x,y+24));d.text((x+6,y+5),f"frame {i:03d}",fill="white")
    sheet.save(out/"contact_sheet.png");video=out/"selected_proxy.mp4"
    subprocess.run(["ffmpeg","-y","-loglevel","error","-framerate",str(a.fps),"-i",str(frames_dir/"%03d.png"),"-frames:v","61","-c:v","libx264","-pix_fmt","yuv420p",str(video)],check=True)
    rep={"status":"PASS","ply":info,"input_ply_sha256":hashlib.sha256(Path(a.ply).read_bytes()).hexdigest(),"trajectory":str(Path(a.trajectory).resolve()),"frame_count":61,"fps":a.fps,"resolution":[a.width,a.height],"temporal_resampling":False,"video":str(video.resolve()),"frames":rows,"elapsed_sec":time.monotonic()-t0}
    (out/"proxy_manifest.json").write_text(json.dumps(rep,indent=2)+"\n");print(json.dumps({"status":"PASS","frames":61,"fps":a.fps,"video":str(video)},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
