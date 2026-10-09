from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipeline.utils.runtime import stage_cli
if __name__=='__main__': stage_cli('triposplat')
