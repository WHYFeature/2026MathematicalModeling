#!/usr/bin/env python3
"""Fine-tune BERT with audio/vision summaries and export Problem 2 results."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from problem2.bert_model import BertAVFusion
from problem2.data import apply_normalizer, fit_normalizer, load_standard, pool_to_50, save_stats
from problem2.training_history import save_training_history, write_json_atomic


CLASS_NAMES = ("Negative", "Neutral", "Positive")


class FusionDataset(Dataset):
    """Attachment-2 dataset with tokenized text and normalized AV features."""

    def __init__(self, split, raw):
        text = np.asarray(raw["text_bert"], dtype=np.int64)
        self.ids = torch.from_numpy(text[:, 0])
        self.mask = torch.from_numpy(text[:, 1])
        self.types = torch.from_numpy(text[:, 2])
        self.audio = torch.from_numpy(split["x"]["audio"])
        self.vision = torch.from_numpy(split["x"]["vision"])
        self.audio_mask = torch.from_numpy(split["mask"]["audio"].astype(np.bool_))
        self.vision_mask = torch.from_numpy(split["mask"]["vision"].astype(np.bool_))
        self.y = torch.from_numpy(split["classification"].astype(np.int64))
        self.r = torch.from_numpy(split["regression"].astype(np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return (
            self.ids[index], self.mask[index], self.types[index],
            self.audio[index], self.audio_mask[index],
            self.vision[index], self.vision_mask[index],
            self.y[index], self.r[index],
        )


class MissingFusionDataset(Dataset):
    """Attachment-3 inference-only dataset."""

    def __init__(self, records):
        self.ids = torch.from_numpy(np.stack([r["ids"] for r in records]).astype(np.int64))
        self.mask = torch.from_numpy(np.stack([r["mask"] for r in records]).astype(np.int64))
        self.types = torch.from_numpy(np.stack([r["types"] for r in records]).astype(np.int64))
        self.audio = torch.from_numpy(np.stack([r["audio"] for r in records]).astype(np.float32))
        self.vision = torch.from_numpy(np.stack([r["vision"] for r in records]).astype(np.float32))
        self.audio_mask = torch.from_numpy(np.stack([r["audio_mask"] for r in records]).astype(np.bool_))
        self.vision_mask = torch.from_numpy(np.stack([r["vision_mask"] for r in records]).astype(np.bool_))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return (
            self.ids[index], self.mask[index], self.types[index],
            self.audio[index], self.audio_mask[index],
            self.vision[index], self.vision_mask[index],
        )


def metrics_from_prediction(prediction):
    y = np.asarray(prediction["true_classification"], dtype=np.int64)
    p = np.asarray(prediction["classification"], dtype=np.int64)
    cm = np.zeros((3, 3), dtype=np.int64)
    for actual, predicted in zip(y, p):
        cm[int(actual), int(predicted)] += 1
    f1 = []
    for k in range(3):
        tp = cm[k, k]
        precision = tp / max(int(cm[:, k].sum()), 1)
        recall = tp / max(int(cm[k, :].sum()), 1)
        f1.append(2 * precision * recall / max(precision + recall, 1e-12))
    y_reg = np.asarray(prediction["true_regression"], dtype=np.float64)
    p_reg = np.asarray(prediction["regression"], dtype=np.float64)
    pearson = float(np.corrcoef(y_reg, p_reg)[0, 1]) if np.std(y_reg) > 1e-12 and np.std(p_reg) > 1e-12 else 0.0
    return {
        "accuracy": float(np.mean(y == p)),
        "macro_f1": float(np.mean(f1)),
        "f1_by_class": {CLASS_NAMES[i]: float(f1[i]) for i in range(3)},
        "confusion_matrix": cm.tolist(),
        "mae": float(np.mean(np.abs(y_reg - p_reg))),
        "rmse": float(np.sqrt(np.mean((y_reg - p_reg) ** 2))),
        "pearson": pearson,
    }


@torch.inference_mode()
def predict_loader(model, loader, device, with_truth, loss_functions=None):
    if loss_functions is not None and not with_truth:
        raise ValueError("Loss evaluation requires ground-truth labels")
    model.eval()
    probabilities, classifications, regressions = [], [], []
    true_classification, true_regression = [], []
    classification_sum = regression_sum = classification_denominator = 0.0
    sample_count = 0
    for batch in loader:
        if with_truth:
            ids, mask, types, audio, audio_mask, vision, vision_mask, y, regression = batch
        else:
            ids, mask, types, audio, audio_mask, vision, vision_mask = batch
        logits, out_reg = model(
            ids.to(device), mask.to(device), types.to(device),
            audio.to(device), audio_mask.to(device),
            vision.to(device), vision_mask.to(device),
        )
        probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())
        classifications.append(logits.argmax(dim=-1).cpu().numpy())
        regressions.append(out_reg.cpu().numpy())
        if loss_functions is not None:
            cls_loss, reg_loss = loss_functions
            target = y.to(device)
            classification_sum += float(nn.functional.cross_entropy(
                logits, target, weight=cls_loss.weight, reduction="sum",
                label_smoothing=cls_loss.label_smoothing,
            ).item())
            classification_denominator += (
                len(y) if cls_loss.weight is None else float(cls_loss.weight[target].sum().item())
            )
            regression_sum += float(nn.functional.smooth_l1_loss(
                out_reg, regression.to(device), beta=reg_loss.beta, reduction="sum",
            ).item())
            sample_count += len(y)
        if with_truth:
            true_classification.append(y.numpy())
            true_regression.append(regression.numpy())
    result = {
        "probabilities": np.concatenate(probabilities),
        "classification": np.concatenate(classifications),
        "regression": np.concatenate(regressions),
    }
    if with_truth:
        result["true_classification"] = np.concatenate(true_classification)
        result["true_regression"] = np.concatenate(true_regression)
    if loss_functions is not None:
        result["loss_metrics"] = {
            "classification_loss": classification_sum / classification_denominator,
            "regression_loss": regression_sum / sample_count,
        }
        result["loss_metrics"]["loss"] = (
            result["loss_metrics"]["classification_loss"] + 0.15 * result["loss_metrics"]["regression_loss"]
        )
    return result


def evaluate(model, loader, device, loss_functions=None):
    prediction = predict_loader(model, loader, device, with_truth=True, loss_functions=loss_functions)
    return {**metrics_from_prediction(prediction), **prediction.get("loss_metrics", {})}


def write_predictions(path, ids, prediction, include_truth=False):
    if len(ids) != len(prediction["classification"]):
        raise ValueError("Sample IDs and predictions have different lengths")
    for key in ("probabilities", "regression"):
        if not np.isfinite(prediction[key]).all():
            raise ValueError(f"Non-finite {key}; refusing to export invalid predictions")
    fields = [
        "id", "predicted_class", "predicted_label", "regression_prediction",
        "p_negative", "p_neutral", "p_positive",
    ]
    if include_truth:
        fields.extend(["true_class", "true_label", "true_regression"])
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, sample_id in enumerate(ids):
            predicted = int(prediction["classification"][i])
            row = {
                "id": str(sample_id),
                "predicted_class": predicted,
                "predicted_label": CLASS_NAMES[predicted],
                "regression_prediction": float(prediction["regression"][i]),
                "p_negative": float(prediction["probabilities"][i, 0]),
                "p_neutral": float(prediction["probabilities"][i, 1]),
                "p_positive": float(prediction["probabilities"][i, 2]),
            }
            if include_truth:
                actual = int(prediction["true_classification"][i])
                row.update({
                    "true_class": actual,
                    "true_label": CLASS_NAMES[actual],
                    "true_regression": float(prediction["true_regression"][i]),
                })
            writer.writerow(row)


def normalize_feature(x, mask, stats, modality):
    x = ((x - stats[modality]["mean"]) / stats[modality]["std"]).astype(np.float32)
    x[~mask] = 0.0
    return x


def load_missing_records(directory, text_model, stats):
    """Read attachment-3 files and prepare BERT/AV inputs for inference."""
    from transformers import AutoTokenizer

    directory = Path(directory)
    paths = sorted(directory.glob("*.pkl"))
    if not paths:
        raise FileNotFoundError(f"No pickle files found in {directory}")
    tokenizer = None
    records = []
    for path in paths:
        with path.open("rb") as handle:
            item = pickle.load(handle)
        data = item.get("test", item)
        if "text_bert" in data:
            text = np.asarray(data["text_bert"])
            if text.ndim == 3 and text.shape[0] == 1:
                text = text[0]
            if text.shape != (3, 50):
                raise ValueError(f"Unexpected text_bert shape in {path}: {text.shape}")
            if not np.isfinite(text).all() or not np.equal(text, np.rint(text)).all():
                raise ValueError(f"Non-integer or non-finite BERT input in {path}")
            text = np.rint(text).astype(np.int64)
            ids, attention, token_type = text[0], text[1], text[2]
            if (ids < 0).any() or not np.isin(attention, [0, 1]).all() or not np.isin(token_type, [0, 1]).all():
                raise ValueError(f"Invalid BERT IDs or masks in {path}")
        elif "raw_text" in data:
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(text_model, local_files_only=True)
            raw_text = str(np.asarray(data["raw_text"]).reshape(-1)[0])
            encoded = tokenizer(raw_text, max_length=50, truncation=True, padding="max_length")
            ids = np.asarray(encoded["input_ids"], dtype=np.int64)
            attention = np.asarray(encoded["attention_mask"], dtype=np.int64)
            token_type = np.asarray(encoded.get("token_type_ids", [0] * 50), dtype=np.int64)
        else:
            raise ValueError(f"Missing text_bert/raw_text in {path}")

        for modality, dimension in (("audio", 74), ("vision", 35)):
            value = np.asarray(data[modality])
            if value.ndim == 3 and value.shape[0] == 1:
                value = value[0]
            if value.ndim != 2 or value.shape[1] != dimension or value.shape[0] == 0 or not np.isfinite(value).all():
                raise ValueError(f"Invalid {modality} features in {path}: {value.shape}")
        audio, audio_mask = pool_to_50(data["audio"])
        vision, vision_mask = pool_to_50(data["vision"])
        audio = normalize_feature(audio, audio_mask, stats, "audio")
        vision = normalize_feature(vision, vision_mask, stats, "vision")
        records.append({
            "id": path.stem,
            "ids": ids,
            "mask": attention,
            "types": token_type,
            "audio": audio,
            "audio_mask": audio_mask,
            "vision": vision,
            "vision_mask": vision_mask,
        })
    return records


def main(argv=None, *, epoch_observer=None, training_only=False):
    parser = argparse.ArgumentParser(description="Fine-tune BERT with audio/vision fusion")
    parser.add_argument("--data-root", type=Path, default=Path("D:/E_math/DATA"))
    parser.add_argument("--text-model", type=Path, default=Path("D:/E_math/models/bert-base-uncased"))
    parser.add_argument("--output", type=Path, default=Path("D:/E_math/problem2_bert_fusion_outputs"))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1.5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--export-only", action="store_true", help="Load OUTPUT/bert_av_fusion.pt and export predictions without training")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or (not args.export_only and args.epochs < 1):
        parser.error("batch-size must be positive; training requires epochs >= 1")
    checkpoint_path = args.output / "bert_av_fusion.pt"
    metrics_path = args.output / "metrics.json"
    if args.export_only and not checkpoint_path.is_file():
        parser.error(f"No existing checkpoint for --export-only: {checkpoint_path}")
    if not args.export_only and any((args.output / name).exists() for name in (
        "bert_av_fusion.pt", "metrics.json", "training_history.json", "training_history.csv",
    )):
        parser.error("Training output already contains results. Choose a new --output directory; use --export-only to export the saved model.")
    previous = {}
    if args.export_only and metrics_path.is_file():
        previous = json.loads(metrics_path.read_text(encoding="utf-8"))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)

    standard_path = args.data_root / "attachment_2_standard_features" / "aligned_50.pkl"
    with standard_path.open("rb") as handle:
        raw = pickle.load(handle)
    standard = load_standard(standard_path)
    normalization_path = args.output / "normalization.json"
    if args.export_only and normalization_path.is_file():
        saved_stats = json.loads(normalization_path.read_text(encoding="utf-8"))
        stats = {m: {k: np.asarray(v, dtype=np.float32) for k, v in values.items()} for m, values in saved_stats.items()}
        normalization_source = "saved training statistics"
    else:
        # Legacy checkpoints did not save normalization. Rebuild the identical
        # statistics from the original training split, never valid/test data.
        stats = fit_normalizer(standard["train"])
        save_stats(normalization_path, stats)
        normalization_source = "recomputed from attachment-2 training split"
    normalized = {split: apply_normalizer(standard[split], stats) for split in ("train", "valid", "test")}

    train_loader = None if args.export_only else DataLoader(FusionDataset(normalized["train"], raw["train"]), batch_size=args.batch_size, shuffle=True, pin_memory=device.type == "cuda")
    # A dedicated generator keeps the additional evaluation iterator from
    # consuming the training RNG used for dropout and shuffled batches.
    train_eval_loader = None if args.export_only else DataLoader(
        train_loader.dataset, batch_size=args.batch_size * 2, shuffle=False,
        pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(args.seed),
    )
    valid_loader = DataLoader(FusionDataset(normalized["valid"], raw["valid"]), batch_size=args.batch_size * 2, shuffle=False, pin_memory=device.type == "cuda")
    test_loader = DataLoader(FusionDataset(normalized["test"], raw["test"]), batch_size=args.batch_size * 2, shuffle=False, pin_memory=device.type == "cuda")

    model = BertAVFusion(str(args.text_model), device)
    counts = np.bincount(normalized["train"]["classification"], minlength=3).astype(np.float32)
    weights = torch.tensor(1.0 / np.sqrt(counts), device=device)
    weights = weights / weights.mean()
    cls_loss = nn.CrossEntropyLoss(weight=weights if args.balanced else None)
    reg_loss = nn.SmoothL1Loss(beta=0.5)
    optimizer = None if args.export_only else torch.optim.AdamW([
        {"params": model.bert.parameters(), "lr": args.lr},
        {"params": [p for name, p in model.named_parameters() if not name.startswith("bert.")], "lr": args.lr * 5},
    ], weight_decay=0.01)

    best_valid_f1 = previous.get("best_valid_f1", -1.0)
    best_epoch = previous.get("best_epoch")
    history = previous.get("history") if args.export_only else []
    training_config = previous.get("training_config") if args.export_only else {
        "epochs": args.epochs, "batch_size": args.batch_size,
        "lr": args.lr, "head_lr": args.lr * 5, "seed": args.seed,
        "balanced": args.balanced, "weight_decay": 0.01,
        "regression_weight": 0.15, "smooth_l1_beta": 0.5,
        "selection_metric": "valid_macro_f1",
        "train_metric_mode": "full training split, eval mode, after each epoch",
    }
    epochs = () if args.export_only else range(1, args.epochs + 1)
    if args.export_only:
        print(f"Export only: loading {checkpoint_path}; no training.", flush=True)
    for epoch in epochs:
        model.train()
        for ids, mask, types, audio, audio_mask, vision, vision_mask, y, regression in train_loader:
            logits, predicted_regression = model(
                ids.to(device), mask.to(device), types.to(device),
                audio.to(device), audio_mask.to(device),
                vision.to(device), vision_mask.to(device),
            )
            loss = cls_loss(logits, y.to(device)) + 0.15 * reg_loss(predicted_regression, regression.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        train_metrics = evaluate(model, train_eval_loader, device, loss_functions=(cls_loss, reg_loss))
        valid_metrics = evaluate(model, valid_loader, device, loss_functions=(cls_loss, reg_loss))
        # Retain legacy top-level validation metrics while adding both splits.
        history.append({"epoch": epoch, **valid_metrics, "train": train_metrics,
                        "valid": valid_metrics, "lr": optimizer.param_groups[0]["lr"],
                        "head_lr": optimizer.param_groups[1]["lr"]})
        print(f"epoch {epoch:02d} train_loss={train_metrics['loss']:.4f} valid_loss={valid_metrics['loss']:.4f} "
              f"train_acc={train_metrics['accuracy']:.4f} valid_acc={valid_metrics['accuracy']:.4f} "
              f"train_f1={train_metrics['macro_f1']:.4f} valid_f1={valid_metrics['macro_f1']:.4f} "
              f"valid_mae={valid_metrics['mae']:.4f}", flush=True)
        if valid_metrics["macro_f1"] > best_valid_f1:
            best_valid_f1 = valid_metrics["macro_f1"]
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)
        # Save every completed epoch, including epochs that do not improve.
        # Export failures or a later interruption cannot discard this history.
        save_training_history(args.output, history, best_epoch, training_config)
        write_json_atomic(metrics_path, {
            "best_valid_f1": best_valid_f1, "best_epoch": best_epoch,
            "history": history, "training_config": training_config,
            "balanced_loss": bool(args.balanced),
        })
        if epoch_observer is not None:
            epoch_observer(model, history, training_config)

    # A replay verifier can measure the training trajectory without exporting
    # predictions or selecting anything using test-set results.
    if training_only:
        return 0

    if not args.export_only:
        try:
            from problem2.training_history import draw_training_curves
            draw_training_curves({"history": history, "best_epoch": best_epoch}, args.output / "training_curves")
            print(f"Training curves: {args.output / 'training_curves'}", flush=True)
        except (ImportError, RuntimeError, OSError, ValueError) as error:
            print(f"Training history saved; plotting failed: {error}. "
                  "Run problem2_training_curves.py --input-root with this output directory after fixing the plotting dependency.", flush=True)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    valid_prediction = predict_loader(model, valid_loader, device, with_truth=True)
    valid_metrics = metrics_from_prediction(valid_prediction)
    valid_csv = args.output / "problem2_valid_predictions.csv"
    write_predictions(valid_csv, normalized["valid"]["id"], valid_prediction, include_truth=True)
    test_prediction = predict_loader(model, test_loader, device, with_truth=True)
    test_metrics = metrics_from_prediction(test_prediction)
    test_csv = args.output / "problem2_test_predictions.csv"
    write_predictions(test_csv, normalized["test"]["id"], test_prediction, include_truth=True)

    missing_summary = {}
    for variant in ("aligned", "unaligned"):
        records = load_missing_records(args.data_root / "attachment_3_missing_modality" / variant, args.text_model, stats)
        missing_loader = DataLoader(MissingFusionDataset(records), batch_size=max(1, args.batch_size * 2), shuffle=False, pin_memory=device.type == "cuda")
        prediction = predict_loader(model, missing_loader, device, with_truth=False)
        output_csv = args.output / f"problem2_missing_{variant}_predictions.csv"
        write_predictions(output_csv, [record["id"] for record in records], prediction, include_truth=False)
        missing_summary[variant] = {
            "count": len(records),
            "path": str(output_csv),
            "mean_audio_mask_ratio": float(np.mean([record["audio_mask"].mean() for record in records])),
            "mean_vision_mask_ratio": float(np.mean([record["vision_mask"].mean() for record in records])),
            "mean_text_mask_ratio": float(np.mean([record["mask"].mean() for record in records])),
            "predicted_class_counts": np.bincount(prediction["classification"], minlength=3).tolist(),
        }
        print(f"Exported {variant}: {len(records)} samples -> {output_csv}", flush=True)

    metadata = {
        **previous,
        "schema_version": 2,
        "device": str(device),
        "best_valid_f1": float(best_valid_f1) if best_valid_f1 >= 0 else valid_metrics["macro_f1"],
        "best_epoch": best_epoch,
        "balanced_loss": previous.get("balanced_loss") if args.export_only else bool(args.balanced),
        "training_config": training_config,
        "export_only": args.export_only,
        "export_batch_size": args.batch_size * 2,
        "normalization_source": normalization_source,
        "class_names": list(CLASS_NAMES),
        "train_counts": counts.tolist(),
        "valid": valid_metrics,
        "test": test_metrics,
        "history": history,
        "history_note": "Legacy checkpoints cannot recover unrecorded epochs/history." if history is None else None,
        "output": str(args.output.resolve()),
        "files": {
            "model": str(checkpoint_path.resolve()),
            "normalization": str(normalization_path.resolve()),
            "valid_predictions": str(valid_csv.resolve()), "test_predictions": str(test_csv.resolve()),
        },
        "missing_modality": missing_summary,
        "missing_preprocessing": {
            "text": "Provided text_bert mask preserved; raw_text tokenized to at most 50 positions using the local BERT tokenizer.",
            "audio_vision": "All-zero source rows masked before train-statistics normalization; unaligned sequences pooled into 50 bins using valid rows only.",
            "evaluation": "Attachment 3 has no labels here; these files contain predictions, not measured accuracy.",
        },
    }
    write_json_atomic(metrics_path, metadata)
    print(json.dumps({"device": str(device), "best_epoch": best_epoch, "test": test_metrics, "output": str(args.output.resolve())}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
