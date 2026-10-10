import contextlib
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.run_pipeline import _finalize_report, build_parser
from pipeline.utils.paths import load_config
from pipeline.utils.runtime import anchor_alignment_policy, legacy_stage_config, run_one, stage_config


class AnchorAlignmentPolicyTests(unittest.TestCase):
    def test_fail_is_blocked_by_default(self):
        with self.assertRaisesRegex(RuntimeError, "FixAnything blocked by anchor alignment FAIL"):
            anchor_alignment_policy({"ANCHOR000_ALIGNMENT": "FAIL", "ANCHOR060_ALIGNMENT": "GOOD"})

    def test_explicit_override_allows_fail_and_records_report_field(self):
        result = anchor_alignment_policy(
            {"ANCHOR000_ALIGNMENT": "FAIL", "ANCHOR060_ALIGNMENT": "GOOD"},
            allow_fail=True,
        )
        self.assertTrue(result["ANCHOR_ALIGNMENT_OVERRIDE_USED"])
        self.assertEqual(result["ANCHOR000_ALIGNMENT"], "FAIL")
        self.assertEqual(
            result["warnings"],
            ["WARNING:\nFLUX clean anchor failed automatic alignment audit.\nContinuing because explicit override was enabled."],
        )

    def test_partial_warns_and_continues_without_override(self):
        result = anchor_alignment_policy(
            {"ANCHOR000_ALIGNMENT": "PARTIAL", "ANCHOR060_ALIGNMENT": "GOOD"}
        )
        self.assertFalse(result["ANCHOR_ALIGNMENT_OVERRIDE_USED"])
        self.assertEqual(len(result["warnings"]), 1)
        self.assertIn("PARTIAL", result["warnings"][0])

    def test_good_continues_without_warning_or_override(self):
        result = anchor_alignment_policy(
            {"ANCHOR000_ALIGNMENT": "GOOD", "ANCHOR060_ALIGNMENT": "GOOD"}
        )
        self.assertFalse(result["ANCHOR_ALIGNMENT_OVERRIDE_USED"])
        self.assertEqual(result["warnings"], [])

    def test_alignment_policy_participates_in_resume_fingerprint(self):
        config = load_config(ROOT / "configs/default.yaml")
        config["_anchor_alignment_allow_fail"] = False
        for stage in ("flux_anchor", "fixanything", "report"):
            blocked_hash = stage_config(config, stage)
            config["_anchor_alignment_allow_fail"] = True
            allowed_hash = stage_config(config, stage)
            self.assertNotEqual(blocked_hash, allowed_hash, stage)
            config["_anchor_alignment_allow_fail"] = False

    def test_cli_override_is_opt_in(self):
        parser = build_parser()
        args = parser.parse_args(["--input-topdown", "layout.png", "--output", "out"])
        self.assertFalse(args.allow_anchor_alignment_fail)
        args = parser.parse_args([
            "--input-topdown", "layout.png", "--output", "out",
            "--allow-anchor-alignment-fail",
        ])
        self.assertTrue(args.allow_anchor_alignment_fail)

    def _write_existing_flux_artifacts(self, temp, config_allow_fail=False):
        root = Path(temp)
        output = root / "run"
        anchors = output / "anchors"
        anchors.mkdir(parents=True)
        proxy_frames = output / "proxy_frames"
        proxy_frames.mkdir()
        (proxy_frames / "000.png").write_bytes(b"proxy")
        topdown = root / "topdown.png"
        topdown.write_bytes(b"layout")
        (anchors / "clean_000.png").write_bytes(b"anchor000")
        (anchors / "clean_060.png").write_bytes(b"anchor060")
        alignment = output / "alignment.json"
        alignment.write_text(json.dumps({
            "ANCHOR000_ALIGNMENT": "FAIL",
            "ANCHOR060_ALIGNMENT": "GOOD",
            "ALIGNMENT_STATUS": "FAIL",
        }))
        anchor_report = output / "anchor_report.json"
        anchor_report.write_text(json.dumps({
            "MODEL_ID": "black-forest-labs/FLUX.2-klein-4B",
            "MODEL_SOURCE": "ModelScope local weights",
            "NUM_INFERENCE_STEPS": 4,
            "endpoints": {},
        }))
        config = {
            "name": "World-Walker",
            "project_root": str(root),
            "input": {"mode": "image", "default_topdown": str(topdown)},
            "total_frames": 61,
            "flux": {"anchor_alignment": {"allow_fail": config_allow_fail}},
            "paths": {
                "input_copy": str(output / "input.png"),
                "input_manifest": str(output / "input_manifest.json"),
                "triposplat_dir": str(output / "triposplat"),
                "ply": str(output / "triposplat/scene.ply"),
                "triposplat_manifest": str(output / "triposplat/manifest.json"),
                "preprocessing_trace": str(output / "triposplat/trace.json"),
                "camera_dir": str(output / "camera"),
                "trajectory": str(output / "camera/trajectory.json"),
                "camera_report": str(output / "camera/report.json"),
                "proxy_dir": str(output / "proxy"),
                "proxy_frames": str(proxy_frames),
                "proxy_video": str(output / "proxy/video.mp4"),
                "proxy_report": str(output / "proxy/report.json"),
                "proxy_contact": str(output / "proxy/contact.png"),
                "topdown": str(topdown),
                "anchor_dir": str(anchors),
                "anchor000": str(anchors / "clean_000.png"),
                "anchor060": str(anchors / "clean_060.png"),
                "alignment": str(alignment),
                "anchor_report": str(anchor_report),
                "fa_input": str(output / "fixanything/input"),
                "fa_input_manifest": str(output / "fixanything/manifest.json"),
                "anchor_provenance": str(output / "fixanything/anchor_provenance.json"),
                "fa_video": str(output / "fixanything/output.mp4"),
                "fa_frames": str(output / "fixanything/frames"),
                "fa_report": str(output / "fixanything/report.json"),
                "final_report": str(output / "report.json"),
                "summary": str(output / "summary.md"),
                "comparison_sheet": str(output / "comparison.png"),
                "comparison_video": str(output / "comparison.mp4"),
                "model_registry": str(output / "model_registry.json"),
            },
        }
        config_path = root / "config.yaml"
        config_path.write_text(json.dumps(config))
        return config_path, output

    def test_run_one_blocks_fail_by_default(self):
        with tempfile.TemporaryDirectory() as temp:
            config, output = self._write_existing_flux_artifacts(temp)
            with self.assertRaisesRegex(RuntimeError, "anchor alignment FAIL"):
                run_one("flux_anchor", config, output, input_mode="image")

    def test_run_one_explicit_override_records_and_warns(self):
        with tempfile.TemporaryDirectory() as temp:
            config, output = self._write_existing_flux_artifacts(temp)
            error = io.StringIO()
            with contextlib.redirect_stderr(error):
                result = run_one(
                    "flux_anchor", config, output, input_mode="image",
                    allow_anchor_alignment_fail=True,
                )
            self.assertTrue(result["details"]["ANCHOR_ALIGNMENT_OVERRIDE_USED"])
            self.assertIn("Continuing because explicit override was enabled.", error.getvalue())
            manifest = json.loads((output / "audit/stages/flux_anchor/stage_manifest.json").read_text())
            self.assertTrue(manifest["details"]["ANCHOR_ALIGNMENT_OVERRIDE_USED"])



    def _prepare_fixanything_outputs(self, config, output):
        config_data = json.loads(Path(config).read_text())
        paths = config_data["paths"]
        input_dir = Path(paths["fa_input"])
        frames_dir = Path(paths["fa_frames"])
        input_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for index in range(61):
            (input_dir / f"{index:03d}.png").write_bytes(b"proxy-frame")
            (frames_dir / f"{index:03d}.png").write_bytes(b"generated-frame")
            frames.append({"clean_anchor": index in (0, 60)})
        Path(paths["fa_input_manifest"]).write_text(json.dumps({
            "frames": frames, "temporal_resampling": False,
        }))
        Path(paths["anchor_provenance"]).write_text("{}")
        Path(paths["fa_video"]).write_bytes(b"generated-video")
        Path(paths["fa_report"]).write_text(json.dumps({
            "status": "SUCCESS", "input_frames": 61, "output_frames": 61,
            "resolution": [832, 480], "seed": 1, "steps": 10, "cfg": 5.0,
            "gpu": "test", "runtime_sec": 0.0,
        }))

    def test_fixanything_stage_still_blocks_fail_without_override(self):
        with tempfile.TemporaryDirectory() as temp:
            config, output = self._write_existing_flux_artifacts(temp)
            self._prepare_fixanything_outputs(config, output)
            with patch("pipeline.utils.runtime.decoded_frame_count", return_value=61):
                with self.assertRaisesRegex(RuntimeError, "anchor alignment FAIL"):
                    run_one("fixanything", config, output, input_mode="image")

    def test_fixanything_stage_enters_only_with_explicit_override(self):
        with tempfile.TemporaryDirectory() as temp:
            config, output = self._write_existing_flux_artifacts(temp)
            self._prepare_fixanything_outputs(config, output)
            warning = io.StringIO()
            with contextlib.redirect_stderr(warning):
                with patch("pipeline.utils.runtime.decoded_frame_count", return_value=61):
                    result = run_one(
                        "fixanything", config, output, input_mode="image",
                        allow_anchor_alignment_fail=True,
                    )
            self.assertEqual(result["status"], "PASS")
            self.assertTrue(result["details"]["ANCHOR_ALIGNMENT_OVERRIDE_USED"])
            self.assertIn("Continuing because explicit override was enabled.", warning.getvalue())
            manifest = json.loads((output / "audit/stages/fixanything/stage_manifest.json").read_text())
            self.assertTrue(manifest["details"]["ANCHOR_ALIGNMENT_OVERRIDE_USED"])

    def test_final_report_records_override_and_partial_status(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "run"
            output.mkdir()
            topdown = output / "input.png"
            topdown.write_bytes(b"layout")
            fa_video = output / "fixanything.mp4"
            fa_video.write_bytes(b"video")
            (output / "alignment.json").write_text(json.dumps({
                "ANCHOR000_ALIGNMENT": "FAIL",
                "ANCHOR060_ALIGNMENT": "GOOD",
            }))
            (output / "fix_report.json").write_text(json.dumps({
                "status": "SUCCESS", "output_frames": 61, "fps": 24, "resolution": [832, 480],
            }))
            default_config = load_config(ROOT / "configs/default.yaml")
            paths = {
                key: str(output / (key.replace("_", "-") + ".artifact"))
                for key in default_config["paths"]
            }
            paths.update({
                "input_copy": str(topdown),
                "model_registry": str(output / "model_registry.json"),
                "fa_video": str(fa_video),
                "fa_report": str(output / "fix_report.json"),
                "alignment": str(output / "alignment.json"),
            })
            config = {
                "name": "World-Walker",
                "project_root": str(root),
                "input": {"mode": "image", "default_topdown": str(topdown)},
                "paths": paths,
                "flux": {"anchor_alignment": {"allow_fail": False}},
                "seedvr2": {"enabled": False},
            }
            config_path = root / "config.yaml"
            config_path.write_text(json.dumps(config))
            report = output / "report.json"
            report.write_text("{}")
            result = _finalize_report(
                config_path, output, "image", None, False, [],
                allow_anchor_alignment_fail=True,
            )
            self.assertTrue(result["ANCHOR_ALIGNMENT_OVERRIDE_USED"])
            self.assertEqual(result["FINAL_STATUS"], "P_PIPE1_PARTIAL_ANCHOR_ALIGNMENT_OVERRIDE")
            self.assertEqual(result["READY_FOR_REUSABLE_PIPELINE"], "PARTIAL")
            self.assertTrue(any("Continuing because explicit override was enabled." in w for w in result["WARNINGS"]))


    def test_legacy_resume_manifest_cannot_bypass_policy_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp:
            config, output = self._write_existing_flux_artifacts(temp)
            run_one(
                "flux_anchor", config, output, input_mode="image",
                allow_anchor_alignment_fail=True,
            )
            manifest_path = output / "audit/stages/flux_anchor/stage_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            legacy_config = load_config(config)
            manifest["config_sha256"] = legacy_stage_config(legacy_config, "flux_anchor")
            manifest_path.write_text(json.dumps(manifest))
            resumed = run_one(
                "flux_anchor", config, output, resume=True, input_mode="image",
                allow_anchor_alignment_fail=True,
            )
            self.assertNotEqual(resumed["status"], "CACHED")
            self.assertTrue(resumed["details"]["ANCHOR_ALIGNMENT_OVERRIDE_USED"])

    def test_config_allow_fail_is_honored(self):
        with tempfile.TemporaryDirectory() as temp:
            config, output = self._write_existing_flux_artifacts(temp, config_allow_fail=True)
            result = run_one("flux_anchor", config, output, input_mode="image")
            self.assertTrue(result["details"]["ANCHOR_ALIGNMENT_OVERRIDE_USED"])


if __name__ == "__main__":
    unittest.main()
