import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TrainScriptTests(unittest.TestCase):
    def test_import_does_not_start_training(self):
        result = subprocess.run(
            [sys.executable, "-c", "import train"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")

    def test_save_plots_writes_three_pngs(self):
        with tempfile.TemporaryDirectory() as output_dir:
            script = """
import sys
from train import save_plots

y_true = ["pas_interessant", "notifiable", "interessant"] * 2
y_pred = ["pas_interessant", "pas_interessant", "interessant",
          "pas_interessant", "notifiable", "notifiable"]
save_plots(y_true, y_pred, sys.argv[1])
"""
            result = subprocess.run(
                [sys.executable, "-c", script, output_dir],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            for name in (
                "confusion_matrix.png",
                "class_metrics.png",
                "class_counts.png",
            ):
                self.assertGreater((Path(output_dir) / name).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
