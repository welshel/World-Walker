import os,subprocess
from pathlib import Path

def run_command(command,cwd,log,env=None):
    log=Path(log); log.parent.mkdir(parents=True,exist_ok=True)
    e=os.environ.copy(); e.update(env or {})
    with log.open('w') as f:
        p=subprocess.run([str(x) for x in command],cwd=str(cwd),env=e,stdout=f,stderr=subprocess.STDOUT)
    if p.returncode: raise RuntimeError(f'command failed ({p.returncode}); see {log}')
    return [str(x) for x in command]
