import unittest
from pathlib import Path
from pipeline.utils.paths import load_config
from pipeline import run_pipeline
from unittest.mock import patch
import json
from scripts.check_environment import evaluate, provider_check
from pipeline.integrations import agent2topdown, camera_planner, flux, fixanything, proxy_renderer, seedvr2, triposplat
import tempfile

ROOT = Path(__file__).resolve().parents[1]


class ReleaseOneConfigTests(unittest.TestCase):
    def test_default_config_has_direct_adapter_settings_and_no_runner_templates(self):
        config = load_config(ROOT / "configs/default.yaml")
        expected = {
            "triposplat": {"python", "snapshot", "checkpoint_root"},
            "camera": {"python", "up_axis"},
            "proxy": {"python", "fps", "resolution"},
            "flux_anchor": {"python", "model_dir", "foreground_prompt", "background_prompt"},
            "fixanything": {"python", "repo", "diffsynth_root", "model_dir", "lora_path", "commit"},
        }
        for stage, keys in expected.items():
            self.assertTrue(keys.issubset(config["stages"][stage]), stage)
            self.assertNotIn("command", config["stages"][stage], stage)
        self.assertFalse(config["seedvr2"]["enabled"])
        self.assertEqual(config["stages"]["fixanything"]["clean_frame_indices"], [0, 60])
        self.assertEqual(config["total_frames"], 61)
        self.assertFalse(config["temporal_resampling"])

    def test_pipeline_writes_fail_closed_e2e_record_on_stage_error(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with patch.object(run_pipeline, "run_one", side_effect=RuntimeError("provider unavailable")):
                code = run_pipeline.main(["--scene-prompt", "garden", "--config", str(ROOT / "configs/default.yaml"), "--output", str(output)])
            self.assertEqual(code, 1)
            failure = json.loads((output / "e2e_failure.json").read_text())
            self.assertEqual(failure["FINAL_E2E_STATUS"], "FAIL")
            self.assertFalse(failure["MOCK_FALLBACK_USED"])
            self.assertEqual(failure["DOWNSTREAM_STAGES"].get("flux_topdown"), "NOT_RUN")

    def test_external_stage_adapters_expose_validation_and_command_builders(self):
        for adapter in (agent2topdown, triposplat, camera_planner, proxy_renderer, flux, fixanything, seedvr2):
            self.assertTrue(callable(adapter.run))
            self.assertTrue(callable(adapter.validate_environment))
            self.assertTrue(callable(adapter.build_command))
            self.assertTrue(callable(adapter.collect_outputs))

    def test_camera_builder_keeps_negative_axis_as_one_cli_argument(self):
        command = camera_planner.build_command(
            {"python": "python", "up_axis": "-Y"},
            Path("scene.ply"), Path("camera_out"), ROOT,
        )
        self.assertIn("--up-axis=-Y", command)
        self.assertNotIn("-Y", command)

    def test_real_agent_provider_requires_configured_root_and_key(self):
        self.assertEqual(provider_check("${AGENT2TOPDOWN_ROOT}", False)["reason"], "AGENT2TOPDOWN_PROVIDER_NOT_CONFIGURED")
        self.assertEqual(provider_check(ROOT / "pipeline/agent2topdown_backend", False)["status"], "FAIL")
        self.assertEqual(provider_check(ROOT / "pipeline/agent2topdown_backend", True)["status"], "PASS")

    def test_unconfigured_environment_returns_component_statuses(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("pipeline.utils.paths._load_local_env"), patch.dict("os.environ", {}, clear=True):
                result = evaluate(ROOT / "configs/default.yaml", "text", Path(directory))
        self.assertEqual(result["FINAL_ENV_STATUS"], "FAIL")
        self.assertEqual(result["checks"]["SEEDVR2"]["status"], "OPTIONAL_DISABLED")
        self.assertEqual(result["checks"]["AGENT2TOPDOWN"]["reason"], "AGENT2TOPDOWN_PROVIDER_NOT_CONFIGURED")

    def test_local_model_registry_is_ignored(self):
        self.assertIn("*.local.yaml", (ROOT / ".gitignore").read_text())


if __name__ == "__main__":
    unittest.main()
