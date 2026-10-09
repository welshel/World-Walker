import hashlib
from pathlib import Path

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def sha256_path(path):
    p=Path(path)
    if p.is_file(): return sha256_file(p)
    if p.is_dir():
        h=hashlib.sha256()
        for q in sorted(x for x in p.rglob('*') if x.is_file()):
            h.update(q.relative_to(p).as_posix().encode()); h.update(bytes.fromhex(sha256_file(q)))
        return h.hexdigest()
    return 'MISSING'
