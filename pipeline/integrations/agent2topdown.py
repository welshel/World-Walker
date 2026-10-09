"""Strict adapter for the configured real Agent2Topdown provider."""
from __future__ import annotations
import os
import sys
from pathlib import Path
from pipeline.integrations.agent2topdown_adapter import run as _run
from pipeline.integrations.agent2topdown_adapter import resolve_input_mode

def validate_environment(root: str | Path, api_key_env: str = "DEEPSEEK_API_KEY") -> None:
    root=Path(root).expanduser()
    if not (root/"app.py").is_file():
        raise RuntimeError("AGENT2TOPDOWN_PROVIDER_NOT_CONFIGURED: set AGENT2TOPDOWN_ROOT to an Agent2Topdown checkout containing app.py")
    if not os.environ.get(api_key_env):
        raise RuntimeError("AGENT2TOPDOWN_PROVIDER_NOT_CONFIGURED: set DEEPSEEK_API_KEY for the real Agent2Topdown provider")

def run(prompt, root, output_root, generated_output, manifest_path):
    validate_environment(root)
    result=_run(prompt,Path(root),Path(output_root),Path(generated_output),Path(manifest_path),mock=False)
    if result.get("MODE")=="mock" or result.get("STATUS")=="PASS_MOCK_FALLBACK":
        raise RuntimeError("AGENT2TOPDOWN_MOCK_FALLBACK_FORBIDDEN")
    if result.get("STATUS")!="PASS":
        raise RuntimeError("AGENT2TOPDOWN_REAL_PROVIDER_FAILED: "+str(result.get("ERROR")))
    return result


def build_command(prompt, root, output_root, generated_output, manifest_path):
    adapter = Path(__file__).with_name("agent2topdown_adapter.py")
    return [sys.executable, str(adapter), "--scene-prompt", str(prompt), "--agent-root", str(root),
            "--output-root", str(output_root), "--generated-output", str(generated_output),
            "--manifest", str(manifest_path)]


def collect_outputs(generated_output, manifest_path):
    generated, manifest = Path(generated_output), Path(manifest_path)
    if not generated.is_file() or not manifest.is_file():
        raise RuntimeError("Agent2Topdown output collection failed: layout or manifest missing")
    import json
    record = json.loads(manifest.read_text())
    if record.get("STATUS") != "PASS" or record.get("MODE") != "deepseek_api":
        raise RuntimeError("Agent2Topdown output is not proven to come from the real provider")
    return {"topdown": generated, "manifest": manifest, "sha256": record.get("OUTPUT_SHA256")}
