import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline" / "agent2topdown_backend"))
import app


class AgentProviderFailClosedTests(unittest.TestCase):
    @patch.object(app, "load_dotenv", return_value=None)
    @patch.object(app, "DeepSeekClient", side_effect=ValueError("DEEPSEEK_API_KEY is not configured"))
    def test_missing_real_provider_fails_without_mock_layout(self, _client, _dotenv):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "AGENT2TOPDOWN_PROVIDER_NOT_CONFIGURED"):
                app.run_pipeline("A formal garden", mock=False, output_root=output)
            run_dirs = [p for p in output.iterdir() if p.is_dir()]
            self.assertEqual(len(run_dirs), 1)
            self.assertFalse((run_dirs[0] / "layout.png").exists())
            metadata = (run_dirs[0] / "metadata.json").read_text()
            self.assertIn("error", metadata)
            self.assertNotIn('"mode": "mock"', metadata)


if __name__ == "__main__":
    unittest.main()
