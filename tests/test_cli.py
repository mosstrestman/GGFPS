import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ggfps_paper.cli import main


class CliTests(unittest.TestCase):
    def test_root_simple_demo_command_runs(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main(
                [
                    "simple-demo",
                    "--n-points",
                    "200",
                    "--labeled-size",
                    "100",
                    "--training-set-size",
                    "40",
                    "--num-folds",
                    "4",
                    "--seed",
                    "7",
                ]
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["schedule"], "ascending")
        self.assertGreaterEqual(payload["test_mae"], 0.0)

    def test_root_demo_command_runs(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main(
                [
                    "demo",
                    "--n-points",
                    "250",
                    "--labeled-size",
                    "120",
                    "--training-set-size",
                    "50",
                    "--num-folds",
                    "4",
                    "--bootstraps",
                    "1",
                    "--seed",
                    "9",
                ]
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["n_records"], 1)
        self.assertIn("on_the_fly", payload["sampling_modes"])


if __name__ == "__main__":
    unittest.main()
