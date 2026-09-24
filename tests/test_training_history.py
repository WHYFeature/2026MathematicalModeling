"""Check missing-data semantics for the plotter, without training or fake results."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from problem2.training_history import load_history, prepare_history, write_json_atomic


class TrainingHistoryTests(unittest.TestCase):
    def test_legacy_validation_history_has_no_invented_training_values(self):
        rows, best = prepare_history({"best_epoch": 2, "history": [
            {"epoch": 1, "accuracy": .5, "macro_f1": .4, "mae": .8},
            {"epoch": 2, "accuracy": .6, "macro_f1": .5, "mae": .7},
        ]})
        self.assertEqual(best, 2)
        self.assertEqual(rows[0]["train"], {})
        self.assertNotIn("loss", rows[0]["valid"])
        self.assertEqual(rows[1]["valid"]["accuracy"], .6)

    def test_no_history_cannot_be_plotted(self):
        with tempfile.TemporaryDirectory() as directory:
            write_json_atomic(Path(directory) / "metrics.json", {"history": None, "test": {"accuracy": .718}})
            with self.assertRaisesRegex(ValueError, "No recorded epoch history"):
                load_history(directory)

    def test_invalid_or_duplicate_epochs_are_rejected(self):
        for history in ([{"epoch": 1, "accuracy": .5}] * 2,
                        [{"epoch": 1, "accuracy": float("nan")}],
                        [{"epoch": 0, "accuracy": .5}]):
            with self.subTest(history=history), self.assertRaises(ValueError):
                prepare_history({"history": history})


if __name__ == "__main__":
    unittest.main()
