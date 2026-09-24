"""Synthetic replay checks. No BERT training or real experiment changes."""
import argparse
import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_bert_fusion_export as fixture_module
from test_bert_fusion_export import FakeTokenizer, TinyFusion
import problem2_bert_fusion_train as trainer
import problem2_recover_history as recovery
import problem2_report as report
from problem2.training_history import file_sha256, load_history, write_json_atomic


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.ExportTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        with patch.object(trainer, "BertAVFusion", TinyFusion), patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()), patch("problem2.training_history.draw_training_curves"), contextlib.redirect_stdout(io.StringIO()):
            trainer.main(self.fixture.command("--epochs", "2"))
        target = self.fixture.output
        # Simulate a legacy export while retaining all actual predictions.
        metadata = json.loads((target / "metrics.json").read_text(encoding="utf-8"))
        self.reference_best_epoch = metadata["best_epoch"]
        metadata.update(history=None, best_epoch=None, training_config=None)
        write_json_atomic(target / "metrics.json", metadata)
        (target / "training_history.json").unlink()
        (target / "training_history.csv").unlink()
        self.before = {name: file_sha256(target / name) for name in recovery.TARGET_FILES}
        self.args = argparse.Namespace(target_root=target, data_root=self.fixture.data_root,
                                       text_model=self.fixture.root / "local_bert", epochs=2,
                                       batch_size=2, lr=1.5e-5, seed=42, balanced=False)

    def test_exact_replay_attaches_history_and_preserves_original_results(self):
        with patch.object(trainer, "BertAVFusion", TinyFusion), patch.object(recovery, "draw_training_curves", return_value={}), patch.object(trainer, "write_predictions", side_effect=AssertionError("Replay must not export predictions")), patch.object(trainer, "load_missing_records", side_effect=AssertionError("Replay must not evaluate attachment 3")), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(recovery.recover(self.args), 0)
        restored = load_history(self.fixture.output)
        self.assertEqual(restored["provenance"]["kind"], "verified_replay")
        self.assertTrue(restored["provenance"]["exact_state_match"])
        self.assertEqual(restored["best_epoch"], self.reference_best_epoch)
        self.assertEqual(len(restored["history"]), 2)
        self.assertEqual(self.before, {name: file_sha256(self.fixture.output / name) for name in recovery.TARGET_FILES})
        metadata = json.loads((self.fixture.output / "metrics.json").read_text(encoding="utf-8"))
        self.assertIsNone(metadata["history"])
        _, summary = report.analyze(self.fixture.output)
        self.assertTrue(summary["training_history_available"])
        self.assertIsNone(summary["best_epoch"])
        self.assertEqual(summary["training_best_epoch"], self.reference_best_epoch)
        self.assertEqual(summary["test"]["accuracy"], metadata["test"]["accuracy"])
        # Exercise the report text using placeholder image bytes, without
        # manufacturing or publishing experiment figures.
        report_output = self.fixture.root / "report_check"
        (report_output / "figures").mkdir(parents=True)
        for name, _ in (*report.FIGURES, report.TRAINING_FIGURE):
            (report_output / "figures" / f"{name}.png").write_bytes(b"test-only")
        summary["training_curve_info"] = {"epochs_recorded": 2, "best_epoch": self.reference_best_epoch, "has_train_metrics": True}
        report.write_report(None, summary, report_output)
        report_text = (report_output / "stage_report.md").read_text(encoding="utf-8")
        self.assertIn("经权重校验的训练复现过程", report_text)
        self.assertIn("不声称找回了当时的原始日志", report_text)
        # Repeat calls verify and plot; they never run another training job.
        with patch.object(trainer, "main", side_effect=AssertionError("Must reuse verified replay")), patch.object(recovery, "draw_training_curves", return_value={}), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(recovery.recover(self.args), 0)
        # A stale proof must not survive modification of the reference.
        with (self.fixture.output / "metrics.json").open("a", encoding="utf-8") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "Target file changed"):
            load_history(self.fixture.output)

    def test_different_reference_cannot_receive_another_runs_history(self):
        checkpoint = self.fixture.output / "bert_av_fusion.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state["classifier.bias"][0] += 0.01
        torch.save(state, checkpoint)
        before = {name: file_sha256(self.fixture.output / name) for name in recovery.TARGET_FILES}
        with patch.object(trainer, "BertAVFusion", TinyFusion), patch.object(recovery, "draw_training_curves", side_effect=AssertionError("Must not draw an unverified curve")), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(recovery.recover(self.args), 2)
        self.assertFalse((self.fixture.output / "history_recovery" / "training_history.json").exists())
        self.assertFalse((self.fixture.output / "training_curves").exists())
        self.assertEqual(before, {name: file_sha256(self.fixture.output / name) for name in recovery.TARGET_FILES})
        attempts = list((self.fixture.output / "history_recovery").glob("replay_*"))
        self.assertEqual(len(attempts), 1)
        status = json.loads((attempts[0] / "verification.json").read_text(encoding="utf-8"))
        self.assertEqual(status["status"], "unverified")
        self.assertFalse(status["exact_state_match"])

    def test_comparison_requires_exact_values_shapes_and_dtypes(self):
        reference = {"weight": torch.tensor([1.0, 2.0])}
        self.assertTrue(recovery.compare_state(reference, {"weight": reference["weight"].clone()})["exact_state_match"])
        for value in (torch.tensor([1.0, 2.000001]), torch.tensor([[1.0, 2.0]]), reference["weight"].double()):
            with self.subTest(value=value):
                self.assertFalse(recovery.compare_state(reference, {"weight": value})["exact_state_match"])

    def test_extension_preserves_reference_when_a_later_epoch_has_higher_f1(self):
        with patch.object(trainer, "BertAVFusion", TinyFusion), patch.object(recovery, "draw_training_curves", return_value={}), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(recovery.recover(self.args), 0)
        previous = load_history(self.fixture.output)
        real_evaluate = trainer.evaluate
        evaluations = 0

        def later_improvement(*args, **kwargs):
            nonlocal evaluations
            evaluations += 1
            metrics = real_evaluate(*args, **kwargs)
            epoch = (evaluations + 1) // 2
            if evaluations % 2 == 0 and epoch > 2:
                metrics["macro_f1"] = .90 + .01 * epoch
            return metrics

        self.args.epochs = 4
        with patch.object(trainer, "BertAVFusion", TinyFusion), patch.object(trainer, "evaluate", side_effect=later_improvement), patch.object(recovery, "draw_training_curves", return_value={}), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(recovery.recover(self.args), 0)
        extended = load_history(self.fixture.output)
        self.assertEqual(len(extended["history"]), 4)
        self.assertEqual(extended["history"][:2], previous["history"])
        self.assertEqual(extended["best_epoch"], self.reference_best_epoch)
        self.assertEqual(extended["provenance"]["replay_selected_epoch"], 4)
        self.assertTrue(extended["provenance"]["extended_analysis"])
        self.assertEqual(extended["provenance"]["reference_selection_epochs"], 2)
        self.assertEqual(extended["provenance"]["extension_start_epoch"], 3)
        self.assertEqual(self.before, {name: file_sha256(self.fixture.output / name) for name in recovery.TARGET_FILES})
        saved = Path(extended["provenance"]["attempt_root"]) / "previous_verified_history.json"
        self.assertEqual(json.loads(saved.read_text(encoding="utf-8")), previous)

    def test_failed_extension_keeps_previously_verified_history(self):
        with patch.object(trainer, "BertAVFusion", TinyFusion), patch.object(recovery, "draw_training_curves", return_value={}), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(recovery.recover(self.args), 0)
        saved = self.fixture.output / "history_recovery" / "training_history.json"
        before_history = saved.read_bytes()
        real_evaluate = trainer.evaluate

        def different_prefix(*args, **kwargs):
            metrics = real_evaluate(*args, **kwargs)
            metrics["loss"] += .001
            return metrics

        self.args.epochs = 4
        with patch.object(trainer, "BertAVFusion", TinyFusion), patch.object(trainer, "evaluate", side_effect=different_prefix), patch.object(recovery, "draw_training_curves", side_effect=AssertionError("Do not replace verified curves")), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(recovery.recover(self.args), 2)
        self.assertEqual(saved.read_bytes(), before_history)
        self.assertEqual(self.before, {name: file_sha256(self.fixture.output / name) for name in recovery.TARGET_FILES})


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
