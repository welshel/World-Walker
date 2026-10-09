import contextlib, io, json, os, unittest
from pathlib import Path
from unittest.mock import patch
from pipeline.run_pipeline import build_parser, main, resolve_project_root
from pipeline.utils.paths import load_config
from pipeline.utils.runtime import stage_config
ROOT=Path(__file__).resolve().parents[1]
class ReleaseContractTests(unittest.TestCase):
 def test_mutually_exclusive_inputs(self):
  p=build_parser()
  with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): p.parse_args(["--output","o"])
  with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): p.parse_args(["--output","o","--scene-prompt","x","--input-topdown","y"])
 def test_defaults(self):
  c=load_config(ROOT/"configs/default.yaml")
  self.assertEqual(c["input"]["mode"],"auto"); self.assertFalse(c["seedvr2"]["enabled"])
  self.assertEqual(c["total_frames"],61); self.assertFalse(c["temporal_resampling"])
  self.assertEqual(c["stages"]["fixanything"]["clean_frame_indices"],[0,60])
 def test_config_relative_root(self): self.assertEqual(resolve_project_root(ROOT/"configs/default.yaml",{"project_root":".."}),ROOT)
 def test_seedvr2_enable_does_not_change_fixanything_fingerprint(self):
  c=load_config(ROOT/"configs/default.yaml"); d=json.loads(json.dumps(c)); d["seedvr2"]["enabled"]=True
  self.assertEqual(stage_config(c,"fixanything"),stage_config(d,"fixanything"))
 def test_dry_run_both_modes(self):
  for args,mode in [(["--scene-prompt","garden"],"text"),(["--input-topdown","missing.png"],"image")]:
   out=io.StringIO()
   with contextlib.redirect_stdout(out): self.assertEqual(main(args+["--config",str(ROOT/"configs/default.yaml"),"--output","out","--dry-run"]),0)
   data=json.loads(out.getvalue()); self.assertEqual(data["INPUT_MODE"],mode); self.assertFalse(data["MODEL_EXECUTION"])
if __name__=="__main__": unittest.main()
