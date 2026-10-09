from __future__ import annotations
import importlib.util, os, sys
from pathlib import Path
import numpy as np
TYPES={"char":"i1","int8":"i1","uchar":"u1","uint8":"u1","short":"<i2","int16":"<i2","ushort":"<u2","uint16":"<u2","int":"<i4","int32":"<i4","uint":"<u4","uint32":"<u4","float":"<f4","float32":"<f4","double":"<f8","float64":"<f8"}
C0=0.28209479177387814

def read_gaussian_ply(path):
    path=Path(path)
    with path.open("rb") as f:
        header=[]
        while True:
            line=f.readline()
            if not line: raise ValueError("truncated PLY header")
            header.append(line.decode("ascii").strip())
            if header[-1]=="end_header": break
        fmt=next((x.split()[1] for x in header if x.startswith("format ")),None)
        count=int(next(x.split()[2] for x in header if x.startswith("element vertex ")))
        props=[]; in_vertex=False
        for line in header:
            if line.startswith("element "): in_vertex=line.startswith("element vertex ")
            elif in_vertex and line.startswith("property "):
                p=line.split()
                if p[1]=="list": raise ValueError("list vertex properties unsupported")
                if p[1] not in TYPES: raise ValueError("unsupported PLY scalar type: "+p[1])
                props.append((p[2],TYPES[p[1]]))
        names=[x[0] for x in props]
        required={"x","y","z","opacity","scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3","f_dc_0","f_dc_1","f_dc_2"}
        if missing:=sorted(required-set(names)): raise ValueError("unsupported Gaussian PLY, missing "+str(missing))
        if fmt=="binary_little_endian":
            a=np.fromfile(f,dtype=np.dtype(props),count=count)
            if len(a)!=count: raise ValueError("truncated PLY body")
            rows={n:a[n] for n in names}
        elif fmt=="ascii":
            a=np.loadtxt(f,max_rows=count,dtype=np.float64)
            if a.ndim==1:a=a[None,:]
            if len(a)!=count or a.shape[1]!=len(names):raise ValueError("bad ASCII PLY body")
            rows={n:a[:,i] for i,n in enumerate(names)}
        else: raise ValueError("unsupported Gaussian PLY format: "+str(fmt))
    xyz=np.column_stack([rows[k] for k in ("x","y","z")]).astype(np.float32)
    scales=np.exp(np.clip(np.column_stack([rows["scale_"+str(i)] for i in range(3)]),-20,20)).astype(np.float32)
    opacity=(1/(1+np.exp(-np.clip(np.nan_to_num(rows["opacity"],nan=-20),-80,80)))).astype(np.float32)
    rgb=np.clip(.5+C0*np.column_stack([rows["f_dc_"+str(i)] for i in range(3)]),0,1).astype(np.float32)
    quat=np.column_stack([rows["rot_"+str(i)] for i in range(4)]).astype(np.float32)
    quat/=np.maximum(np.linalg.norm(quat,axis=1,keepdims=True),1e-8)
    valid=np.isfinite(xyz).all(1)&np.isfinite(scales).all(1)&np.isfinite(opacity)
    scene={"xyz":xyz[valid],"scales":scales[valid],"opacity":opacity[valid],"rgb":rgb[valid],"quat":quat[valid]}
    if len(scene["xyz"])==0:raise ValueError("no finite Gaussians")
    return scene,{"path":str(path.resolve()),"format":fmt,"vertex_count_header":count,"finite_gaussians":len(scene["xyz"]),"size_bytes":path.stat().st_size}

def render(scene,w2c,K,width=896,height=512,device="cuda"):
    import torch, gsplat
    if not hasattr(gsplat,"csrc"):
        ext=os.environ.get("GSPLAT_CUDA_EXTENSION","")
        if not ext or not Path(ext).is_file():raise RuntimeError("gsplat CUDA extension missing; set GSPLAT_CUDA_EXTENSION")
        spec=importlib.util.spec_from_file_location("gsplat_cuda",ext)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);sys.modules["gsplat.csrc"]=mod;gsplat.csrc=mod
    from gsplat import rasterization
    t=lambda x:torch.as_tensor(x,dtype=torch.float32,device=device).contiguous()
    with torch.inference_mode():
        rgbd,alpha,_=rasterization(t(scene["xyz"]),t(scene["quat"]),t(scene["scales"]),t(scene["opacity"]),t(scene["rgb"]),t(np.asarray(w2c,np.float32))[None],t(np.asarray(K,np.float32))[None],width,height,near_plane=.1,eps2d=1e-8,sh_degree=None,packed=True,render_mode="RGB+ED")
    x=rgbd[0].cpu().numpy();a=alpha[0,...,0].cpu().numpy()
    return np.clip(x[...,:3],0,1),x[...,3].astype(np.float32),a.astype(np.float32)
