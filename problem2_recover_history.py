#!/usr/bin/env python3
"""Replay training and attach curves only after exactly matching a saved model.

The target checkpoint, original metrics, and prediction CSVs are read-only.
Measured replay history is explicitly distinguished from lost original logs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

import problem2_bert_fusion_train as trainer
from problem2.training_history import (
    draw_training_curves, file_sha256, load_history, save_training_history,
    validate_recovered_history, write_json_atomic,
)

TARGET_FILES = (
    "bert_av_fusion.pt", "metrics.json", "normalization.json",
    "problem2_valid_predictions.csv", "problem2_test_predictions.csv",
    "problem2_missing_aligned_predictions.csv", "problem2_missing_unaligned_predictions.csv",
)


def compare_state(reference, candidate):
    """Compare all named parameters and buffers with zero numerical tolerance."""
    same_names = set(reference) == set(candidate)
    different = []
    for name in sorted(set(reference) | set(candidate)):
        if name not in reference or name not in candidate:
            different.append(name)
            continue
        expected = reference[name]
        actual = candidate[name].detach().cpu()
        if expected.dtype != actual.dtype or expected.shape != actual.shape or not torch.equal(expected, actual):
            different.append(name)
    return {
        "exact_state_match": same_names and not different,
        "tensor_count": len(reference), "different_tensor_count": len(different),
        "first_different_tensors": different[:5],
        "comparison": "same keys, tensor shapes, dtypes, and torch.equal values; no tolerance",
    }


def assert_target_unchanged(target, before):
    for name, expected in before.items():
        if file_sha256(target / name) != expected:
            raise RuntimeError(f"Target file changed during verification: {name}")


def normalization_equal(reference_path, candidate_path):
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    return (reference.keys() == candidate.keys() and all(
        reference[modality].keys() == candidate[modality].keys() and all(
            np.array_equal(np.asarray(value), np.asarray(candidate[modality][key]))
            for key, value in values.items()
        ) for modality, values in reference.items()
    ))


def recover(args):
    target = args.target_root.resolve()
    for name in TARGET_FILES:
        if not (target / name).is_file():
            raise FileNotFoundError(f"Required target artifact missing: {target / name}")
    recovery = target / "history_recovery"
    previous_history = None
    reference_selection_epochs = min(5, args.epochs)
    if (recovery / "training_history.json").is_file():
        previous_history = load_history(target)
        validate_recovered_history(target, previous_history)
        config = previous_history["training_config"]
        for key in ("batch_size", "lr", "seed", "balanced"):
            if config[key] != getattr(args, key):
                raise ValueError(f"Cannot extend the verified run with a different {key}")
        reference_selection_epochs = previous_history["provenance"].get(
            "reference_selection_epochs", len(previous_history["history"]),
        )
        if previous_history["history"][-1]["epoch"] >= args.epochs:
            figures = draw_training_curves(previous_history, target / "training_curves")
            print(json.dumps({"status": "already_verified", **figures}, ensure_ascii=False))
            return 0
        print(f"Verified history has {len(previous_history['history'])} epochs; replaying from epoch 1 through {args.epochs}. "
              f"The reference model is selected within the original {reference_selection_epochs}-epoch window.", flush=True)

    before = {name: file_sha256(target / name) for name in TARGET_FILES}
    reference = torch.load(target / "bert_av_fusion.pt", map_location="cpu", weights_only=True)
    recovery.mkdir(parents=True, exist_ok=True)
    # Every attempt is kept for audit. Existing results cannot be overwritten.
    attempt = Path(tempfile.mkdtemp(prefix="replay_", dir=recovery))
    if previous_history is not None:
        write_json_atomic(attempt / "previous_verified_history.json", previous_history)
    checks = []

    def observer(model, history, config):
        result = {"epoch": history[-1]["epoch"], **compare_state(reference, model.state_dict())}
        checks.append(result)
        write_json_atomic(attempt / "checkpoint_checks.json", checks)
        label = "EXACT MATCH" if result["exact_state_match"] else f"different tensors: {result['different_tensor_count']}"
        print(f"Reference check at epoch {result['epoch']}: {label}", flush=True)

    command = [
        "--data-root", str(args.data_root), "--text-model", str(args.text_model),
        "--output", str(attempt), "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size), "--lr", str(args.lr), "--seed", str(args.seed),
    ]
    if args.balanced:
        command.append("--balanced")
    trainer.main(command, epoch_observer=observer, training_only=True)
    payload = load_history(attempt)
    matched = [check["epoch"] for check in checks if check["exact_state_match"]]
    same_stats = normalization_equal(target / "normalization.json", attempt / "normalization.json")
    # Longer training can pick a different global optimum. Keep the reference
    # selection window fixed, and identify its exact epoch in the new run.
    reference_rows = [row for row in payload["history"] if row["epoch"] <= reference_selection_epochs]
    reference_epoch = max(reference_rows, key=lambda row: row["valid"]["macro_f1"])["epoch"]
    reference_matches = reference_epoch in matched
    previous_prefix_equal = (previous_history is None or
                             payload["history"][:len(previous_history["history"])] == previous_history["history"])
    del reference
    assert_target_unchanged(target, before)

    status = {
        "kind": "verified_replay", "created_utc": datetime.now(timezone.utc).isoformat(),
        "attempt_root": str(attempt), "matched_epochs": matched,
        "replay_selected_epoch": payload["best_epoch"],
        "reference_selected_epoch": reference_epoch,
        "reference_selection_epochs": reference_selection_epochs,
        "extended_analysis": args.epochs > reference_selection_epochs,
        "extension_start_epoch": reference_selection_epochs + 1 if args.epochs > reference_selection_epochs else None,
        "previous_verified_prefix_equal": previous_prefix_equal,
        "exact_state_match": reference_matches,
        "normalization_equal": same_stats, "target_files_sha256": before,
        "epoch_checks": checks,
        "runtime": {"torch": torch.__version__, "numpy": np.__version__,
                    "cuda": torch.version.cuda,
                    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
        "notes": (
            "Newly measured replay history, not the lost original log. The checkpoint selected within the original "
            "reference window must match exactly. Later epochs are an extended-training analysis; "
            "their globally best checkpoint does not replace the reference. "
            "Original target weights, metrics, and predictions are retained. "
            "Test predictions/labels are not used for epoch selection or a parameter search."
        ),
    }
    if not (reference_matches and same_stats and previous_prefix_equal):
        status["status"] = "unverified"
        write_json_atomic(attempt / "verification.json", status)
        print(json.dumps({"status": "unverified", "attempt_root": str(attempt),
                          "matched_epochs": matched, "reference_selected_epoch": reference_epoch,
                          "message": "Exact reference reproduction failed. Existing verified history and model remain unchanged."}, ensure_ascii=False, indent=2))
        return 2

    status["status"] = "verified"
    # Additional source hashes identify this replay's data and implementation.
    status["replay_sources_sha256"] = {
        "aligned_50.pkl": file_sha256(args.data_root / "attachment_2_standard_features" / "aligned_50.pkl"),
        "problem2_bert_fusion_train.py": file_sha256(Path(trainer.__file__)),
        "bert_model.py": file_sha256(Path(trainer.__file__).parent / "problem2" / "bert_model.py"),
    }
    write_json_atomic(attempt / "verification.json", status)
    save_training_history(recovery, payload["history"], reference_epoch,
                          payload["training_config"], provenance=status)
    verified = load_history(target)
    validate_recovered_history(target, verified)
    figures = draw_training_curves(verified, target / "training_curves")
    assert_target_unchanged(target, before)
    print(json.dumps({"status": "verified", "matched_epoch": reference_epoch,
                      "extended_training_best_epoch": payload["best_epoch"],
                      "history": str(recovery / "training_history.json"), **figures}, ensure_ascii=False, indent=2))
    return 0


def main(argv=None):
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-root", type=Path, default=project / "problem2_outputs_bert_av_reproduce")
    parser.add_argument("--data-root", type=Path, default=project / "DATA")
    parser.add_argument("--text-model", type=Path, default=project / "models" / "bert-base-uncased")
    # The original selection window stays at five epochs. Later epochs only
    # extend the diagnostic trajectory; they do not replace the target model.
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1.5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balanced", action="store_true")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or not np.isfinite(args.lr):
        parser.error("epochs, batch-size, and lr must be positive")
    return recover(args)


if __name__ == "__main__":
    raise SystemExit(main())
