from __future__ import annotations
import hashlib, json, os
from pathlib import Path
import numpy as np

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def array_sha256(value):
    a=np.ascontiguousarray(np.asarray(value));h=hashlib.sha256()
    h.update(str(a.dtype).encode("ascii"));h.update(b"\0");h.update(json.dumps(list(a.shape),separators=(",",":")).encode("ascii"));h.update(b"\0");h.update(a.tobytes(order="C"))
    return h.hexdigest()

def atomic_json(path,payload):
    target=Path(path);target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_suffix(target.suffix+".tmp")
    tmp.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n");os.replace(tmp,target)

def file_record(path):
    target=Path(path);st=target.stat()
    return {"path":str(target.resolve()),"size_bytes":st.st_size,"mtime_unix":st.st_mtime,"sha256":sha256(target)}

def gaussian_integrity(arrays):
    required=("xyz","opacity","scaling","rotation","features_dc")
    missing=[k for k in required if k not in arrays]
    if missing:return {"valid":False,"missing":missing}
    finite={k:bool(np.isfinite(np.asarray(arrays[k])).all()) for k in required}
    scale=np.asarray(arrays["scaling"]);quat=np.asarray(arrays["rotation"]);norm=np.linalg.norm(quat,axis=-1)
    bad_scale=int(np.count_nonzero(scale<=0));bad_quat=int(np.count_nonzero((~np.isfinite(norm))|(norm<1e-8)))
    return {"valid":bool(all(finite.values()) and bad_scale==0 and bad_quat==0),"missing":[],"finite":finite,"nonpositive_scale_count":bad_scale,"invalid_quaternion_count":bad_quat,"quaternion_norm_min":float(np.nanmin(norm)),"quaternion_norm_max":float(np.nanmax(norm))}
