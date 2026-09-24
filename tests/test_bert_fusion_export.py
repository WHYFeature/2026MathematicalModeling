"""Small offline checks for BERT AV export; no pretrained model or training job."""
import contextlib
import hashlib
import io
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import problem2_bert_fusion_train as pipeline


class TinyFusion(torch.nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.bert = torch.nn.Linear(1, 4)
        self.classifier = torch.nn.Linear(4, 3)
        self.regressor = torch.nn.Linear(4, 1)
        if len(args) >= 2:
            self.to(args[1])

    def forward(self, ids, mask, types, audio, audio_mask, vision, vision_mask):
        hidden = self.bert(mask.float().mean(1, keepdim=True))
        return self.classifier(hidden), self.regressor(hidden).squeeze(-1)


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        assert text == "test sentence"
        return {"input_ids": [101, 3231, 102] + [0] * 47, "attention_mask": [1] * 3 + [0] * 47}


class ExportTests(unittest.TestCase):
    def setUp(self):
        cpu_only = patch.object(pipeline.torch.cuda, "is_available", return_value=False)
        cpu_only.start()
        self.addCleanup(cpu_only.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data_root = self.root / "DATA"
        self.output = self.root / "output"
        self.output.mkdir()
        generator = np.random.default_rng(7)
        self.payload = {}
        for name, count in (("train", 6), ("valid", 3), ("test", 3)):
            text = np.zeros((count, 3, 50), dtype=np.int64)
            text[:, 0, :3] = [101, 3231, 102]
            text[:, 1, :3] = 1
            self.payload[name] = {
                "id": [f"{name}_{i}" for i in range(count)],
                "text_bert": text,
                "text": generator.normal(size=(count, 50, 768)).astype(np.float32),
                "audio": generator.normal(size=(count, 50, 74)).astype(np.float32),
                "vision": generator.normal(size=(count, 50, 35)).astype(np.float32),
                "classification_labels": np.arange(count) % 3,
                "regression_labels": (np.arange(count) % 3 - 1).astype(np.float32),
            }
        standard = self.data_root / "attachment_2_standard_features" / "aligned_50.pkl"
        standard.parent.mkdir(parents=True)
        with standard.open("wb") as handle:
            pickle.dump(self.payload, handle)
        for variant, steps in (("aligned", 50), ("unaligned", 500)):
            directory = self.data_root / "attachment_3_missing_modality" / variant
            directory.mkdir(parents=True)
            for index in range(2):
                data = {"audio": np.zeros((1, steps, 74), np.float32), "vision": np.zeros((1, steps, 35), np.float32)}
                data["audio"][0, :2] = 4.0
                if variant == "aligned":
                    data["text_bert"] = self.payload["train"]["text_bert"][:1].astype(np.float32).copy()
                    data["text_bert"][0, :, 1] = 0  # Real internal text gap.
                else:
                    data["raw_text"] = np.asarray(["test sentence"])
                with (directory / f"sample_{index + 1:02d}.pkl").open("wb") as handle:
                    pickle.dump({"test": data}, handle)

    def command(self, *args):
        return ["--data-root", str(self.data_root), "--text-model", str(self.root / "local_bert"), "--output", str(self.output), "--batch-size", "2", *args]

    def test_export_only_preserves_checkpoint_and_provenance(self):
        checkpoint = self.output / "bert_av_fusion.pt"
        torch.save(TinyFusion().state_dict(), checkpoint)
        before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        previous = {"best_epoch": 4, "best_valid_f1": 0.6, "history": [{"epoch": 4}], "training_config": {"batch_size": 20}}
        (self.output / "metrics.json").write_text(json.dumps(previous), encoding="utf-8")
        with patch.object(pipeline, "BertAVFusion", TinyFusion), patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()), patch.object(pipeline.torch.optim, "AdamW", side_effect=AssertionError("Export must not create an optimizer")), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pipeline.main(self.command("--export-only")), 0)
        self.assertEqual(before, hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        result = json.loads((self.output / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(result["best_epoch"], 4)
        self.assertEqual(result["history"], previous["history"])
        self.assertEqual(result["training_config"], previous["training_config"])
        for split, count in (("valid", 3), ("test", 3), ("missing_aligned", 2), ("missing_unaligned", 2)):
            with (self.output / f"problem2_{split}_predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
                rows = list(pipeline.csv.DictReader(handle))
            self.assertEqual(len(rows), count)
            self.assertEqual(len({row["id"] for row in rows}), count)
            self.assertEqual("true_class" in rows[0], not split.startswith("missing"))
            for row in rows:
                probabilities = [float(row[key]) for key in ("p_negative", "p_neutral", "p_positive")]
                self.assertAlmostEqual(sum(probabilities), 1.0, places=6)
                self.assertEqual(int(row["predicted_class"]), int(np.argmax(probabilities)))
            if split in ("valid", "test"):
                self.assertEqual([row["id"] for row in rows], self.payload[split]["id"])
                self.assertEqual(sum(map(sum, result[split]["confusion_matrix"])), count)
        # Re-export loads the saved statistics, without fitting again.
        with patch.object(pipeline, "BertAVFusion", TinyFusion), patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()), patch.object(pipeline, "fit_normalizer", side_effect=AssertionError("Must reuse saved statistics")), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pipeline.main(self.command("--export-only")), 0)

    def test_masks_and_valid_only_pooling(self):
        stats = {m: {"mean": np.ones(d, np.float32), "std": np.ones(d, np.float32)} for m, d in (("audio", 74), ("vision", 35))}
        root = self.data_root / "attachment_3_missing_modality"
        aligned = pipeline.load_missing_records(root / "aligned", "unused", stats)
        self.assertEqual(aligned[0]["mask"][1], 0)
        np.testing.assert_array_equal(aligned[0]["audio"][0], 3.0)
        np.testing.assert_array_equal(aligned[0]["audio"][2:], 0.0)
        self.assertFalse(aligned[0]["vision_mask"].any())
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()):
            unaligned = pipeline.load_missing_records(root / "unaligned", "unused", stats)
        self.assertEqual(unaligned[0]["audio_mask"].sum(), 1)
        np.testing.assert_array_equal(unaligned[0]["audio"][0], 3.0)
        np.testing.assert_array_equal(unaligned[0]["audio"][1:], 0.0)
        self.assertEqual(unaligned[0]["mask"].sum(), 3)

    def test_legacy_metadata_is_not_invented(self):
        torch.save(TinyFusion().state_dict(), self.output / "bert_av_fusion.pt")
        with patch.object(pipeline, "BertAVFusion", TinyFusion), patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()), contextlib.redirect_stdout(io.StringIO()):
            pipeline.main(self.command("--export-only"))
        result = json.loads((self.output / "metrics.json").read_text(encoding="utf-8"))
        self.assertIsNone(result["best_epoch"])
        self.assertIsNone(result["history"])
        self.assertIsNone(result["training_config"])

    def test_missing_checkpoint_fails_without_creating_results(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            pipeline.main(self.command("--export-only"))
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_training_path_still_exports_results(self):
        # One tiny synthetic epoch exercises the branch; no BERT training.
        with patch.object(pipeline, "BertAVFusion", TinyFusion), patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()), patch("problem2.training_history.draw_training_curves"), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pipeline.main(self.command("--epochs", "1")), 0)
        result = json.loads((self.output / "metrics.json").read_text(encoding="utf-8"))
        self.assertFalse(result["export_only"])
        self.assertEqual(result["best_epoch"], 1)
        self.assertEqual(len(result["history"]), 1)
        self.assertEqual(len(list(self.output.glob("problem2_*predictions.csv"))), 4)
        self.assertTrue((self.output / "training_history.csv").is_file())
        recorded = json.loads((self.output / "training_history.json").read_text(encoding="utf-8"))
        self.assertEqual(recorded["history"], result["history"])
        for split in ("train", "valid"):
            for key in ("loss", "accuracy", "macro_f1", "mae"):
                self.assertTrue(np.isfinite(recorded["history"][0][split][key]))

    def test_history_keeps_non_improving_epochs_when_export_fails(self):
        measured = {"loss": 1.0, "accuracy": 0.5, "macro_f1": 0.4, "mae": 0.6}
        with patch.object(pipeline, "BertAVFusion", TinyFusion), patch.object(pipeline, "evaluate", return_value=measured), patch.object(pipeline, "load_missing_records", side_effect=RuntimeError("export interrupted")), patch("problem2.training_history.draw_training_curves"), contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "export interrupted"):
            pipeline.main(self.command("--epochs", "3"))
        recorded = json.loads((self.output / "training_history.json").read_text(encoding="utf-8"))
        self.assertEqual(recorded["best_epoch"], 1)
        self.assertEqual([r["epoch"] for r in recorded["history"]], [1, 2, 3])
        metadata = json.loads((self.output / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(recorded["history"], metadata["history"])
        with (self.output / "training_history.csv").open(encoding="utf-8-sig", newline="") as handle:
            self.assertEqual(len(list(pipeline.csv.DictReader(handle))), 3)

    def test_evaluation_loss_is_independent_of_batch_partition(self):
        standard = pipeline.load_standard(self.data_root / "attachment_2_standard_features" / "aligned_50.pkl")
        normalized = pipeline.apply_normalizer(standard["train"], pipeline.fit_normalizer(standard["train"]))
        dataset = pipeline.FusionDataset(normalized, self.payload["train"])
        model = TinyFusion()
        loss_functions = (torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 5.0])), torch.nn.SmoothL1Loss(beta=.5))
        results = [pipeline.evaluate(model, pipeline.DataLoader(dataset, batch_size=size), torch.device("cpu"), loss_functions=loss_functions) for size in (1, 4, 6)]
        for key in ("loss", "classification_loss", "regression_loss", "macro_f1", "accuracy", "mae"):
            for result in results[1:]:
                self.assertAlmostEqual(result[key], results[0][key], places=6)

    def test_training_refuses_to_overwrite_existing_model(self):
        checkpoint = self.output / "bert_av_fusion.pt"
        torch.save(TinyFusion().state_dict(), checkpoint)
        before = checkpoint.read_bytes()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            pipeline.main(self.command("--epochs", "1"))
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(before, checkpoint.read_bytes())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
