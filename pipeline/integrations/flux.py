"""Audited two-stage FLUX.2 Klein endpoint generation and alignment adapter."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def audit_alignment(proxy_path,anchor_path):
    import cv2
    proxy=cv2.imread(str(proxy_path),cv2.IMREAD_COLOR);anchor=cv2.imread(str(anchor_path),cv2.IMREAD_COLOR)
    if proxy is None or anchor is None or proxy.shape!=anchor.shape:return {"alignment":"FAIL","reason":"missing_or_mismatched_image"}
    gray_a=cv2.cvtColor(proxy,cv2.COLOR_BGR2GRAY);gray_b=cv2.cvtColor(anchor,cv2.COLOR_BGR2GRAY)
    mask=(gray_a>8).astype(np.uint8)*255
    detector=cv2.ORB_create(nfeatures=2500,scaleFactor=1.2,nlevels=8)
    kp_a,des_a=detector.detectAndCompute(gray_a,mask);kp_b,des_b=detector.detectAndCompute(gray_b,None)
    if des_a is None or des_b is None:return {"alignment":"FAIL","reason":"no_features","proxy_keypoints":len(kp_a),"anchor_keypoints":len(kp_b)}
    matches=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des_a,des_b,k=2)
    good=[m for pair in matches if len(pair)==2 for m,n in [pair] if m.distance<.75*n.distance]
    if len(good)<4:return {"alignment":"FAIL","reason":"too_few_matches","good_matches":len(good)}
    src=np.float32([kp_a[m.queryIdx].pt for m in good]);dst=np.float32([kp_b[m.trainIdx].pt for m in good])
    H,inliers=cv2.findHomography(src,dst,cv2.RANSAC,4.0)
    if H is None:return {"alignment":"FAIL","reason":"homography_failed","good_matches":len(good)}
    mask_i=inliers.ravel().astype(bool);count=int(mask_i.sum());ratio=float(mask_i.mean())
    corners=np.float32([[[0,0],[proxy.shape[1]-1,0],[proxy.shape[1]-1,proxy.shape[0]-1],[0,proxy.shape[0]-1]]])
    warped=cv2.perspectiveTransform(corners,H)[0];base=corners[0]
    corner_shift=float(np.linalg.norm(warped-base,axis=1).mean()/np.hypot(proxy.shape[1],proxy.shape[0]))
    det=float(np.linalg.det(H[:2,:2]));scale=float(np.sqrt(abs(det)))
    verdict="GOOD" if count>=20 and ratio>=.20 and corner_shift<=.04 and .90<=scale<=1.10 else "PARTIAL" if count>=8 and ratio>=.10 and corner_shift<=.12 and .75<=scale<=1.25 else "FAIL"
    return {"alignment":verdict,"good_matches":len(good),"ransac_inliers":count,"inlier_ratio":ratio,"mean_corner_shift_diagonal_fraction":corner_shift,"homography_scale":scale,"camera_preservation":"PASS" if verdict=="GOOD" else "PARTIAL" if verdict=="PARTIAL" else "FAIL","proxy_sha256":sha(proxy_path),"anchor_sha256":sha(anchor_path)}

def worker(request_path):
    req=json.loads(Path(request_path).read_text());out=Path(req["output_dir"]);out.mkdir(parents=True,exist_ok=True)
    import torch
    from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
    model=Path(req["model_path"]);idx=json.loads((model/"model_index.json").read_text())
    if not idx.get("is_distilled",False):raise RuntimeError("FLUX model is not the required distilled checkpoint")
    shards=sorted(str(p) for p in (model/"text_encoder").glob("model-*.safetensors"))
    if len(shards)!=2:raise RuntimeError("FLUX text encoder shards missing")
    configs=[ModelConfig(path=shards),ModelConfig(path=str(model/"transformer/diffusion_pytorch_model.safetensors")),ModelConfig(path=str(model/"vae/diffusion_pytorch_model.safetensors"))]
    tokenizer=ModelConfig(path=str(model/"tokenizer"))
    pipe=Flux2ImagePipeline.from_pretrained(torch_dtype=torch.bfloat16,device="cuda",model_configs=configs,tokenizer_config=tokenizer)
    prompts=[Path(req["foreground_prompt"]).read_text().rstrip("\n"),Path(req["background_prompt"]).read_text().rstrip("\n")]
    report={"MODEL_ID":"black-forest-labs/FLUX.2-klein-4B","MODEL_SOURCE":"ModelScope-local","MODEL_PATH":str(model.resolve()),"DISTILLED":True,"NUM_INFERENCE_STEPS":4,"SEEDS":req["seeds"],"endpoints":{},"generation_contract":{"stage_a_references":["proxy_endpoint","topdown_reference"],"stage_b_references":["stage_a_selected_foreground"],"mask_used":False,"inpaint_pipeline":False,"temporal_resampling":False}}
    for frame in (0,60):
        name=f"{frame:03d}";proxy=Path(req["proxy_frames"])/f"{name}.png";top=Path(req["topdown"])
        a=Image.open(proxy).convert("RGB");b=Image.open(top).convert("RGB");row={"proxy":str(proxy),"proxy_sha256":sha(proxy),"topdown":str(top),"topdown_sha256":sha(top),"stage_a":{},"stage_b":{}}
        for seed in req["seeds"]:
            t=time.time();torch.cuda.reset_peak_memory_stats()
            image=pipe(prompts[0],edit_image=[a,b],edit_image_auto_resize=True,height=512,width=896,seed=int(seed),rand_device="cuda",num_inference_steps=4,cfg_scale=1.)
            if not isinstance(image,Image.Image) or image.size!=(896,512):raise RuntimeError("FLUX Stage A output size/type invalid")
            path=out/name/"foreground"/f"seed{int(seed):02d}.png";path.parent.mkdir(parents=True,exist_ok=True);image.save(path)
            row["stage_a"][str(seed)]={"path":str(path),"sha256":sha(path),"runtime_sec":time.time()-t,"peak_memory_gib":float(torch.cuda.max_memory_allocated()/2**30)}
        base=np.asarray(a).astype(np.int16);support=base.max(2)>8
        scores=[]
        for seed in req["seeds"]:
            path=Path(row["stage_a"][str(seed)]["path"]);x=np.asarray(Image.open(path).convert("RGB")).astype(np.int16)
            scores.append((float(np.abs(x-base).mean()),-float((x.max(2)>8).mean()),int(seed)))
        selected_a=min(scores)[2];fp=Path(row["stage_a"][str(selected_a)]["path"]);fg=Image.open(fp).convert("RGB")
        row["stage_a_selected_seed"]=selected_a;row["stage_a_selection_scores"]=[{"mean_abs_change":x,"negative_support":y,"seed":z} for x,y,z in scores]
        for seed in req["seeds"]:
            t=time.time();torch.cuda.reset_peak_memory_stats()
            image=pipe(prompts[1],edit_image=fg,edit_image_auto_resize=True,height=512,width=896,seed=int(seed),rand_device="cuda",num_inference_steps=4,cfg_scale=1.)
            if not isinstance(image,Image.Image) or image.size!=(896,512):raise RuntimeError("FLUX Stage B output size/type invalid")
            path=out/name/"background"/f"seed{int(seed):02d}.png";path.parent.mkdir(parents=True,exist_ok=True);image.save(path)
            row["stage_b"][str(seed)]={"path":str(path),"sha256":sha(path),"runtime_sec":time.time()-t,"peak_memory_gib":float(torch.cuda.max_memory_allocated()/2**30)}
        fg_arr=np.asarray(fg).astype(np.int16);mask=fg_arr.max(2)>8;rank=[]
        for seed in req["seeds"]:
            path=Path(row["stage_b"][str(seed)]["path"]);x=np.asarray(Image.open(path).convert("RGB")).astype(np.int16)
            rank.append((float((x.max(2)<8).mean()),float(np.abs(x-fg_arr)[mask].mean()) if mask.any() else 999,int(seed)))
        selected_b=min(rank)[2];anchor=Path(req["anchors"])/f"clean_{name}.png";anchor.parent.mkdir(parents=True,exist_ok=True)
        chosen=Path(row["stage_b"][str(selected_b)]["path"]);Image.open(chosen).save(anchor)
        row["stage_b_selected_seed"]=selected_b;row["stage_b_selection_scores"]=[{"black_fraction":x,"foreground_mae":y,"seed":z} for x,y,z in rank]
        row["anchor_path"]=str(anchor);row["anchor_sha256"]=sha(anchor);row["alignment"]=audit_alignment(proxy,anchor);report["endpoints"][name]=row
    verdicts=[report["endpoints"][x]["alignment"]["alignment"] for x in ("000","060")]
    status="GOOD" if all(x=="GOOD" for x in verdicts) else "PARTIAL" if all(x!="FAIL" for x in verdicts) else "FAIL"
    report["ALIGNMENT_STATUS"]=status;report["ANCHOR000_ALIGNMENT"]=report["endpoints"]["000"]["alignment"]["alignment"];report["ANCHOR060_ALIGNMENT"]=report["endpoints"]["060"]["alignment"]["alignment"];report["FINAL_STATUS"]="P_RELEASE1_FLUX_ANCHORS_PASS" if status=="GOOD" else "P_RELEASE1_FLUX_ANCHORS_PARTIAL" if status=="PARTIAL" else "P_RELEASE1_FLUX_ALIGNMENT_FAIL"
    report["GPU"]={"name":torch.cuda.get_device_name(0),"torch":torch.__version__,"cuda":torch.version.cuda,"peak_memory_gib":float(torch.cuda.max_memory_allocated()/2**30)}
    (out/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    summary=Image.new("RGB",(896*3,536*2),(20,20,20));d=ImageDraw.Draw(summary)
    for j,(label,path) in enumerate([("proxy000",Path(req["proxy_frames"])/"000.png"),("clean000",Path(req["anchors"])/"clean_000.png"),("proxy060",Path(req["proxy_frames"])/"060.png"),("clean060",Path(req["anchors"])/"clean_060.png")]):
        im=Image.open(path).convert("RGB").resize((896,512));x=(j%3)*896;y=(j//3)*536;summary.paste(im,(x,y+24));d.text((x+6,y+4),label,fill="white")
    summary.save(out/"alignment_contact_sheet.png")
    print(json.dumps({"FINAL_STATUS":report["FINAL_STATUS"],"alignment":verdicts,"report":str(out/"report.json")},indent=2))
    if status=="FAIL":raise RuntimeError("FLUX endpoint alignment gate failed; FixAnything must not run")
    return report

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--worker-config",required=True);a=ap.parse_args();worker(a.worker_config)
if __name__=="__main__":main()


def validate_environment(config):
    python = Path(str(config.get("python", "")).strip()).expanduser()
    model = Path(str(config.get("model_dir", "")).strip()).expanduser()
    if not python.is_file():
        raise RuntimeError("FLUX_NOT_CONFIGURED: set FLUX_PYTHON")
    if not (model / "model_index.json").is_file():
        raise RuntimeError("FLUX_NOT_CONFIGURED: set FLUX_MODEL_PATH to ModelScope FLUX.2 Klein 4B")
    required = [model / "transformer/diffusion_pytorch_model.safetensors",
                model / "vae/diffusion_pytorch_model.safetensors", model / "tokenizer"]
    if not all(path.exists() for path in required) or not list((model / "text_encoder").glob("*.safetensors")):
        raise RuntimeError("FLUX_MODEL_LAYOUT_INVALID: expected text encoder, transformer, VAE and tokenizer")


def build_command(python, request_path):
    return [str(python), "-m", "pipeline.integrations.flux", "--worker-config", str(request_path)]

def run(config,proxy_frames,topdown,anchor_dir,report_path,alignment_path,log_path,gpu,repo_root,output_dir):
    from pipeline.utils.subprocess_utils import run_command
    validate_environment(config)
    python=str(config.get("python",""));model=str(config.get("model_dir",""));seeds=list(config.get("seeds",[0,1,2,3]))
    request=Path(output_dir)/"worker_request.json";Path(output_dir).mkdir(parents=True,exist_ok=True)
    fg=Path(config.get("foreground_prompt","configs/prompts/flux_foreground.txt"));bg=Path(config.get("background_prompt","configs/prompts/flux_background.txt"))
    if not fg.is_absolute():fg=Path(repo_root)/fg
    if not bg.is_absolute():bg=Path(repo_root)/bg
    data={"model_path":model,"seeds":seeds,"proxy_frames":str(proxy_frames),"topdown":str(topdown),"anchors":str(anchor_dir),"output_dir":str(output_dir),"foreground_prompt":str(fg),"background_prompt":str(bg)}
    request.write_text(json.dumps(data,indent=2)+"\n")
    env={"PYTHONPATH":":".join([str(repo_root),str(config.get("pythonpath",""))]).strip(":") ,"CUDA_VISIBLE_DEVICES":str(gpu),"PYTHONUNBUFFERED":"1"}
    run_command(build_command(python, request),repo_root,log_path,env)
    import shutil
    report=Path(output_dir)/"report.json";alignment=Path(output_dir)/"report.json"
    if Path(report_path).resolve()!=report.resolve():Path(report_path).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(report,report_path)
    if Path(alignment_path).resolve()!=report.resolve():Path(alignment_path).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(report,alignment_path)
    return json.loads(report.read_text())


def collect_outputs(anchor_dir, report_path):
    root, report = Path(anchor_dir), Path(report_path)
    anchors = [root / "anchors" / f"clean_{frame:03d}.png" for frame in (0, 60)]
    if not report.is_file() or any(not path.is_file() for path in anchors):
        raise RuntimeError("FLUX output collection failed: endpoint anchor or report missing")
    data = json.loads(report.read_text())
    if data.get("ANCHOR000_ALIGNMENT") not in ("GOOD", "PARTIAL") or data.get("ANCHOR060_ALIGNMENT") not in ("GOOD", "PARTIAL"):
        raise RuntimeError("FLUX endpoint alignment gate failed")
    return {"anchors": anchors, "report": report}
