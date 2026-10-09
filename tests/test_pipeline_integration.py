import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from pipeline.run_pipeline import build_parser, stage_id, select_final_output, should_finalize_report
from pipeline.utils.runtime import stage_config, stage_inputs, triposplat_env, text_mode_topdown_source
from pipeline.utils.image_io import make_transparent_white_border
from pipeline.integrations.agent2topdown_adapter import resolve_input_mode
from pipeline.integrations.flux_topdown_adapter import run as run_flux_topdown


class Pipe1IntegrationTests(unittest.TestCase):
    def test_cli_requires_exactly_one_input_mode(self):
        parser = build_parser()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--output", "/tmp/out"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--output", "/tmp/out", "--input-topdown", "a.png", "--scene-prompt", "garden"])
        args = parser.parse_args(["--output", "/tmp/out", "--input-topdown", "a.png"])
        self.assertEqual(args.input_topdown, "a.png")
        args = parser.parse_args(["--output", "/tmp/out", "--scene-prompt", "garden"])
        self.assertEqual(args.scene_prompt, "garden")
        args = parser.parse_args(["--output", "/tmp/out", "--input", "legacy.png"])
        self.assertEqual(args.input_topdown, "legacy.png")

    def test_input_mode_resolver_rejects_missing_or_ambiguous_values(self):
        with self.assertRaises(ValueError):
            resolve_input_mode(None, None)
        with self.assertRaises(ValueError):
            resolve_input_mode("a.png", "garden")
        self.assertEqual(resolve_input_mode("a.png", None), ("image", "a.png"))
        self.assertEqual(resolve_input_mode(None, "garden"), ("text", "garden"))

    def test_seedvr2_toggle_does_not_invalidate_upstream_stage_config(self):
        base = {
            "total_frames": 61,
            "stages": {"triposplat": {"seed": 42}, "fixanything": {"steps": 10}},
            "seedvr2": {"enabled": False, "scale": 2, "seed": 666},
            "paths": {"fa_video": "{output}/generated.mp4"},
        }
        enabled = {**base, "seedvr2": {"enabled": True, "scale": 2, "seed": 666}}
        changed_model_settings = {**base, "seedvr2": {"enabled": False, "scale": 2, "seed": 777}}
        self.assertEqual(stage_config(base, "triposplat"), stage_config(enabled, "triposplat"))
        self.assertEqual(stage_config(base, "fixanything"), stage_config(enabled, "fixanything"))
        self.assertEqual(stage_config(base, "seedvr2"), stage_config(enabled, "seedvr2"))
        self.assertNotEqual(stage_config(base, "seedvr2"), stage_config(changed_model_settings, "seedvr2"))

    def test_seedvr2_fingerprint_depends_only_on_fixanything_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "generated_61f.mp4"
            video.write_bytes(b"fixanything-result")
            hashes = stage_inputs("seedvr2", {"fa_video": video}, video)
            self.assertEqual(list(hashes), [str(video)])

    def test_final_output_falls_back_to_fixanything_when_seedvr2_is_off_or_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fix = root / "fix.mp4"
            seed = root / "seed.mp4"
            fix.write_bytes(b"fix")
            seed.write_bytes(b"seed")
            self.assertEqual(select_final_output(False, "SKIPPED_DISABLED", seed, fix), str(fix))
            self.assertEqual(select_final_output(True, "FAIL", seed, fix), str(fix))
            self.assertEqual(select_final_output(True, "PASS", seed, fix), str(seed))

    def test_triposplat_env_adds_the_package_parent_to_pythonpath(self):
        cfg = {"stages": {"triposplat": {"pythonpath": "/verified/code:/pvs/scripts"}}}
        env = triposplat_env(cfg, "2")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2")
        self.assertIn("/verified/code", env["PYTHONPATH"])
        self.assertIn("/pvs/scripts", env["PYTHONPATH"])

    def test_legacy_numeric_stage_ids_and_new_names(self):
        self.assertEqual(stage_id("0"), "input")
        self.assertEqual(stage_id("6"), "report")
        self.assertEqual(stage_id("7"), "seedvr2")
        self.assertEqual(stage_id("agent2topdown"), "agent2topdown")
        self.assertEqual(stage_id("topdown"), "flux_topdown")
        self.assertEqual(stage_id("flux_topdown"), "flux_topdown")


    def test_text_mode_uses_reviewed_flux_topdown_as_downstream_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent = root / "generated_topdown.png"
            refined = root / "topdown_used.png"
            agent.write_bytes(b"agent-layout")
            refined.write_bytes(b"reviewed-flux-topdown")
            review = root / "visual_review.json"
            manifest = root / "topdown_used_manifest.json"
            digest = __import__("hashlib").sha256(refined.read_bytes()).hexdigest()
            review.write_text(__import__("json").dumps({"APPROVED_FOR_TRIPOSPLAT": True, "STRICT_NADIR_VIEW": "PASS", "NO_HORIZON": "PASS", "NO_OBLIQUE_FACADE": "PASS", "CANDIDATE_SHA256": digest}))
            review_digest = __import__("hashlib").sha256(review.read_bytes()).hexdigest()
            manifest.write_text(__import__("json").dumps({"topdown_used_sha256": digest, "selected_topdown_sha256": digest, "visual_review_sha256": review_digest}))
            self.assertEqual(text_mode_topdown_source(agent, refined, review, manifest), refined)
            review.write_text('{"APPROVED_FOR_TRIPOSPLAT":false,"STRICT_NADIR_VIEW":"FAIL"}')
            self.assertEqual(text_mode_topdown_source(agent, refined, review, manifest), agent)
            review.write_text('{"APPROVED_FOR_TRIPOSPLAT":true,"STRICT_NADIR_VIEW":"PASS","NO_HORIZON":"PASS","NO_OBLIQUE_FACADE":"PASS"}')
            refined.write_bytes(b"tampered")
            self.assertEqual(text_mode_topdown_source(agent, refined, review, manifest), agent)
            refined.unlink()
            self.assertEqual(text_mode_topdown_source(agent, refined, review, manifest), agent)

    def test_rgba_topdown_has_transparent_white_border_and_opaque_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "topdown.png"
            target = root / "topdown_used.png"
            from PIL import Image
            Image.new("RGB", (8, 6), (20, 40, 60)).save(source)
            record = make_transparent_white_border(source, target, border_px=2)
            with Image.open(target) as image:
                self.assertEqual(image.mode, "RGBA")
                self.assertEqual(image.size, (12, 10))
                self.assertEqual(image.getpixel((0, 0)), (255, 255, 255, 0))
                self.assertEqual(image.getpixel((2, 2)), (20, 40, 60, 255))
            self.assertEqual(record["border_alpha"], 0)
            self.assertEqual(record["source_alpha"], 255)


    def test_flux_topdown_stage_waits_for_visual_review_before_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "agent_layout.png"
            source.write_bytes(b"layout")
            candidates = root / "02_flux_topdown" / "outputs"
            candidates.mkdir(parents=True)
            candidate = candidates / "seed00.png"
            candidate.write_bytes(b"flux-candidate")
            metadata = {
                "FINAL_STATUS": "PASS",
                "INPUT_SHA256": __import__("hashlib").sha256(b"layout").hexdigest(),
                "SEEDS_COMPLETED": [0],
                "OUTPUTS": [str(candidate)],
                "per_seed": [{"seed": 0, "path": str(candidate), "peak_memory_gib": 1.0}],
            }
            (candidates / "metadata.json").write_text(__import__("json").dumps(metadata))
            paths = {
                "agent_generated_topdown": source,
                "flux_topdown_candidates": candidates,
                "flux_topdown_metadata": candidates / "metadata.json",
                "flux_topdown_review": root / "02_flux_topdown" / "visual_review.json",
                "flux_topdown_script": root / "missing_runner.py",
                "flux_topdown_candidate": root / "02_flux_topdown" / "topdown.png",
                "topdown_used": root / "02_flux_topdown" / "topdown_used.png",
                "flux_topdown_manifest": root / "02_flux_topdown" / "topdown_used_manifest.json",
            }
            result = run_flux_topdown({"seeds": [0], "model_id": "black-forest-labs/FLUX.2-klein-4B", "model_dir": str(root / "model")},
                                      paths, root, root, "auto", False)
            self.assertEqual(result["STATUS"], "WAITING_VISUAL_REVIEW")
            self.assertFalse(paths["topdown_used"].exists())


    def test_pipeline_finalizes_report_only_after_report_stage_runs(self):
        self.assertFalse(should_finalize_report([{"stage": "flux_topdown", "status": "WAITING_VISUAL_REVIEW"}]))
        self.assertTrue(should_finalize_report([{"stage": "report", "status": "PASS"}]))


if __name__ == "__main__":
    unittest.main()

