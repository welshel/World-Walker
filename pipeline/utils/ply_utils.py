from pathlib import Path

def ply_header(path):
    with open(path,'rb') as f:
        lines=[]; count=None
        while True:
            line=f.readline()
            if not line: raise ValueError('truncated PLY header')
            lines.append(line.decode('ascii','replace').strip())
            if lines[-1].startswith('element vertex '): count=int(lines[-1].split()[-1])
            if lines[-1]=='end_header': break
    if count is None or count<=0: raise ValueError('PLY has no vertices')
    return {'vertex_count':count,'header':lines}
