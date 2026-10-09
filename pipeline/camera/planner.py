from __future__ import annotations
import argparse, hashlib, json, math, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp
from pipeline.camera.gaussian import read_gaussian_ply, render

N,W,H,FPS=61,896,512,24
F=W/(2*np.tan(np.deg2rad(70)/2)); K=np.array([[F,0,W/2],[0,F,H/2],[0,0,1]],np.float32)
AXES={"+X":[1,0,0],"-X":[-1,0,0],"+Y":[0,1,0],"-Y":[0,-1,0],"+Z":[0,0,1],"-Z":[0,0,-1]}
def sj(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
def align(up):
    from scipy.spatial.transform import Rotation
    u=np.asarray(AXES[up],float);return Rotation.align_vectors(np.array([[0.,-1.,0.]]),u[None])[0].as_matrix()
def look(p,t):
    f=t-p;f=f/np.linalg.norm(f);u=np.array([0.,-1.,0.]);r=np.cross(f,u);r=r/np.linalg.norm(r);d=np.cross(f,r);d=d/np.linalg.norm(d);return np.column_stack((r,d,f))
def metric(p,R):
    tr=np.linalg.norm(np.diff(p,axis=0),axis=1);ang=np.rad2deg(Rotation.from_matrix(np.einsum("nji,njk->nik",R[:-1],R[1:])).magnitude())
    return {"path_length":float(tr.sum()),"endpoint_displacement":float(np.linalg.norm(p[-1]-p[0])),"max_translation_step":float(tr.max()),"p95_translation_step":float(np.percentile(tr,95)),"max_angular_step_deg":float(ang.max()),"p95_angular_step_deg":float(np.percentile(ang,95))}
def groups(xyz,opacity,scales):
    rad=2*scales.max(1);m=(opacity>=.05)&np.isfinite(rad);edges=np.linspace(rad[m].min()-1e-9,rad[m].max()+1e-9,33);out=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        sel=m&(rad>=lo)&(rad<hi)
        if sel.any():out.append((cKDTree(xyz[sel]),float(rad[sel].max())))
    return out
def clearance(x,g):
    a=np.asarray(x).reshape(-1,3);d=np.full(len(a),np.inf)
    for t,r in g:d=np.minimum(d,t.query(a,workers=1)[0]-r)
    return d.reshape(np.asarray(x).shape[:-1])
def audit(p,R,g):
    ts=np.linspace(0,60,241);pp=np.stack([np.interp(ts,np.arange(61),p[:,i]) for i in range(3)],1);rr=Slerp(np.arange(61),Rotation.from_matrix(R))(ts).as_matrix()
    center=clearance(pp,g).min();near=[];zray=[]
    for z in (.01,.02,.03,.04):
        x=z*W/(2*F);y=z*H/(2*F);off=np.array([[u,v,z] for v in np.linspace(-y,y,5) for u in np.linspace(-x,x,5)])
        near.append(clearance(pp[:,None,:]+np.einsum("nij,kj->nki",rr,off),g).min())
    ray=pp[:,None,:]+rr[:,:,2][:,None,:]*np.linspace(0,.06,13)[None,:,None];raymin=clearance(ray,g).min()
    m=metric(p,R);cm=m["max_translation_step"]/8+.075*np.sin(np.deg2rad(m["max_angular_step_deg"]/8));pm=np.hypot(.04*W/(2*F)/4,.04*H/(2*F)/4)
    lower={"center":float(center-cm),"near_frustum":float(min(near)-cm-pm),"forward_ray":float(raymin-cm-.0025)}
    safe=lower["center"]>=.018 and lower["near_frustum"]>=.015 and lower["forward_ray"]>=.018
    smooth=m["max_angular_step_deg"]<2 and m["p95_angular_step_deg"]<1.5 and m["max_translation_step"]<.0155
    return {**m,"continuous_clearance_lower_bound":lower,"center_clearance":lower["center"],"near_frustum_clearance":lower["near_frustum"],"forward_ray_clearance":lower["forward_ray"],"collision_free":bool(safe),"motion_smooth":bool(smooth)}
def score(alpha,depth):
    vis=alpha>.05; valid=vis&np.isfinite(depth);cov=float(vis.mean());near=float(((depth<.30)&valid).sum()/max(1,vis.sum()))
    return {"alpha_coverage":cov,"black_background_fraction":1-cov,"near_fraction":near,"score":cov-1.1*near}
def pose(center,ground,height,radius,deg):
    a=np.deg2rad(deg);p=np.array([center[0]+radius*np.cos(a),ground-height,center[1]+radius*np.sin(a)]);t=np.array([center[0],ground,center[1]]);return p,look(p,t)
def camera_world(p,R,Q):
    M=np.eye(4);M[:3,:3]=Q.T@R;M[:3,3]=Q.T@p;return M
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--ply",required=True);ap.add_argument("--output-dir",required=True);ap.add_argument("--up-axis",choices=sorted(AXES),default="-Y");a=ap.parse_args()
    start=time.monotonic();out=Path(a.output_dir);(out/"angle_scans").mkdir(parents=True,exist_ok=True);(out/"diagnostics").mkdir(exist_ok=True)
    scene,ply=read_gaussian_ply(a.ply);Q=align(a.up_axis);xyz=scene["xyz"]@Q.T;active=scene["opacity"]>.2
    if active.sum()<100:raise RuntimeError("too few active Gaussian points")
    pts=xyz[active];q=np.quantile(pts,[[.02,.5,.98]],axis=0)[0];vertical=np.quantile(pts[:,1],[.01,.5,.98,.99]);center=q[1,[0,2]];ground=float(vertical[2]);major=float(max(q[2,0]-q[0,0],q[2,2]-q[0,2]));height=float(np.clip(max(.12,(vertical[3]-vertical[0])*.08),.12,.24))
    if major<=0:raise RuntimeError("degenerate scene footprint")
    obs=groups(xyz,scene["opacity"],scene["scales"]);radii=sorted(set(round(major/2+pad,3) for pad in (.14,.22,.30)));angles=list(range(0,360,10));windows=[];scans=[];scan_paths=[]
    # Scan 36 views per radius and score five-view sectors before building the trajectory.
    for radius in radii:
        rows=[]
        for deg in angles:
            p,R=pose(center,ground,height,radius,deg);M=camera_world(p,R,Q);im,depth,alpha=render(scene,np.linalg.inv(M),K,W,H);m=score(alpha,depth)
            safe=audit(np.repeat(p[None],61,0),np.repeat(R[None],61,0),obs)
            path=out/"angle_scans"/f"r{radius:.3f}_{deg:03d}.png";Image.fromarray(np.uint8(im*255)).save(path);scan_paths.append(path)
            row={"angle":deg,"radius":radius,**m,"clearance":safe["continuous_clearance_lower_bound"],"safe":safe["collision_free"]};rows.append(row);scans.append(row)
        for i,deg in enumerate(angles):
            block=[rows[(i+j)%len(rows)] for j in range(5)]
            if all(x["safe"] and x["alpha_coverage"]>=.55 and x["near_fraction"]<=.30 and x["clearance"]["center"]>=.02 and x["clearance"]["near_frustum"]>=.02 and x["clearance"]["forward_ray"]>=.02 for x in block):
                windows.append({"radius":radius,"start":deg,"score":float(np.mean([x["score"] for x in block])+.2*min(block[0]["score"],block[-1]["score"]))})
    if not windows:
        rep={"FINAL_STATUS":"CAMERA_FAIL_NO_SAFE_VISIBILITY_ARC","collision_free":False,"candidates":[],"frames":61,"temporal_resampling":False,"scan":scans}
        sj(out/"report.json",rep);raise RuntimeError("no visibility-scored arc passed collision and framing gates")
    windows.sort(key=lambda x:x["score"],reverse=True); tested=[];chosen=None
    for w in windows[:30]:
        width=min(38.,float(np.rad2deg(.0155*60*.98/(1.875*w["radius"]))));t=np.linspace(0,1,61);sm=t**3*(10-15*t+6*t*t);ang=w["start"]+(40-width)/2+width*sm
        arr=[pose(center,ground,height,w["radius"],float(d)) for d in ang];p=np.stack([x[0] for x in arr]);R=np.stack([x[1] for x in arr]);au=audit(p,R,obs)
        rec={"id":f"R{w['radius']:.3f}_A{w['start']:03d}","score":w["score"],"radius":w["radius"],"angular_span_deg":width,**au};tested.append(rec)
        if au["collision_free"] and au["motion_smooth"]:chosen=(w,rec,p,R);break
    if chosen is None:
        sj(out/"report.json",{"FINAL_STATUS":"CAMERA_FAIL_CONTINUOUS_SAFETY","collision_free":False,"candidates":tested,"frames":61,"temporal_resampling":False});raise RuntimeError("no smooth native 61-frame arc passed continuous safety")
    w,c,p,R=chosen;frames=[]
    for i in range(61):
        M=camera_world(p[i],R[i],Q);frames.append({"frame":i,"c2w":M.tolist(),"w2c":np.linalg.inv(M).tolist(),"position":M[:3,3].tolist(),"forward":M[:3,2].tolist(),"up":(-M[:3,1]).tolist(),"image_size":[W,H],"intrinsics":K.tolist()})
    traj={"frame_count":61,"fps":24,"image_size":[W,H],"intrinsics":K.tolist(),"temporal_resampling":False,"up_axis":a.up_axis,"selected_candidate":c["id"],"frames":frames,"safety":c}
    sj(out/"selected_cameras_world_c2w.json",traj)
    # A small diagnostic strip covers the winning sector scan and is reproducible from this PLY.
    sheet=Image.new("RGB",(300*6,198*2),(20,20,20));draw=ImageDraw.Draw(sheet)
    chosen_angles=[(w["start"]+10*j)%360 for j in range(5)]
    chosen_paths=[out/"angle_scans"/f"r{w['radius']:.3f}_{d:03d}.png" for d in chosen_angles]
    for i,path in enumerate(chosen_paths):
        x=(i%6)*300;y=(i//6)*198;im=Image.open(path).convert("RGB").resize((300,172));sheet.paste(im,(x,y+26));draw.text((x+4,y+4),f"scan {chosen_angles[i]:03d}",fill="white")
    sheet.save(out/"diagnostics/angle_scan_contact_sheet.png")
    final={"FINAL_STATUS":"CAMERA_PASS_VISIBILITY_SCORED_NATIVE_61F","selected_candidate":c["id"],"candidates":tested,"frames":61,"fps":24,"resolution":[896,512],"up_axis":a.up_axis,"ground_plane_canonical_y":ground,"ground_confidence":"moderate","camera_height_above_ground":height,"path_length":c["path_length"],"endpoint_displacement":c["endpoint_displacement"],"parallax_proxy":c["endpoint_displacement"],"collision_free":c["collision_free"],"motion_smooth":c["motion_smooth"],"continuous_clearance_lower_bound":c["continuous_clearance_lower_bound"],"trajectory":str((out/"selected_cameras_world_c2w.json").resolve()),"scan_contact_sheet":str((out/"diagnostics/angle_scan_contact_sheet.png").resolve()),"input_ply_sha256":hashlib.sha256(Path(a.ply).read_bytes()).hexdigest(),"elapsed_sec":time.monotonic()-start}
    sj(out/"report.json",final);print(json.dumps(final,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
