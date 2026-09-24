#!/usr/bin/env python3
"""Train and evaluate the robust Problem 2 multimodal model."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import MODALITIES, StandardDataset, apply_local_dropout, apply_normalizer, collate, fit_normalizer, load_missing_features, load_standard, save_stats
from .model import HybridStatsFusionNet, PooledFusionNet, RobustFusionNet, SummaryMLP, TokenFusionNet


CLASS_NAMES = ("Negative", "Neutral", "Positive")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_warmup_cosine_lr(optimizer, base_lr, min_lr, epoch, total_epochs, warmup_epochs):
    """Set a linear-warmup/half-cosine learning-rate schedule."""
    warmup_epochs = max(0, min(int(warmup_epochs), int(total_epochs) - 1))
    if warmup_epochs and epoch <= warmup_epochs:
        factor = epoch / float(warmup_epochs)
        lr = base_lr * factor
    else:
        decay_steps = max(1, total_epochs - warmup_epochs)
        progress = min(1.0, max(0.0, (epoch - warmup_epochs) / float(decay_steps)))
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = float(lr)
    return float(lr)


def move_batch(batch, device):
    return {
        "x": {m: batch["x"][m].to(device, non_blocking=True) for m in MODALITIES},
        "mask": {m: batch["mask"][m].to(device, non_blocking=True) for m in MODALITIES},
        "classification": batch["classification"].to(device, non_blocking=True),
        "regression": batch["regression"].to(device, non_blocking=True),
    }


def metrics(y_cls, p_cls, y_reg, p_reg):
    y_cls, p_cls = np.asarray(y_cls), np.asarray(p_cls)
    cm = np.zeros((3, 3), dtype=np.int64)
    for y, p in zip(y_cls, p_cls):
        cm[int(y), int(p)] += 1
    f1 = []
    for k in range(3):
        tp = cm[k, k]
        precision = tp / max(cm[:, k].sum(), 1)
        recall = tp / max(cm[k].sum(), 1)
        f1.append(2 * precision * recall / max(precision + recall, 1e-12))
    y_reg, p_reg = np.asarray(y_reg, dtype=np.float64), np.asarray(p_reg, dtype=np.float64)
    corr = float(np.corrcoef(y_reg, p_reg)[0, 1]) if np.std(y_reg) > 1e-12 and np.std(p_reg) > 1e-12 else 0.0
    return {
        "accuracy": float(np.mean(y_cls == p_cls)),
        "macro_f1": float(np.mean(f1)),
        "f1_by_class": {CLASS_NAMES[i]: float(f1[i]) for i in range(3)},
        "confusion_matrix": cm.tolist(),
        "mae": float(np.mean(np.abs(y_reg - p_reg))),
        "rmse": float(np.sqrt(np.mean((y_reg - p_reg) ** 2))),
        "pearson": corr,
    }


def run_epoch(model, loader, optimizer, cls_loss, reg_loss, ordinal_loss, device, train=False, local_dropout=0.45, reg_weight=0.5, ordinal_weight=0.0):
    model.train(train)
    all_y, all_p, all_yr, all_pr = [], [], [], []
    total = 0.0
    for batch in loader:
        if train:
            batch = apply_local_dropout(batch, probability=local_dropout)
        batch = move_batch(batch, device)
        logits, reg, ordinal_logits, _ = model(batch["x"], batch["mask"])
        loss = cls_loss(logits, batch["classification"]) + reg_weight * reg_loss(reg, batch["regression"])
        if ordinal_weight > 0.0 and ordinal_logits is not None and ordinal_logits.ndim == 2 and ordinal_logits.shape[-1] == 2:
            y = batch["classification"]
            ordinal_target = torch.stack([(y >= 1).float(), (y >= 2).float()], dim=-1)
            loss = loss + ordinal_weight * ordinal_loss(ordinal_logits, ordinal_target)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += float(loss.detach()) * len(batch["classification"])
        all_y.append(batch["classification"].detach().cpu().numpy())
        all_p.append(logits.argmax(dim=-1).detach().cpu().numpy())
        all_yr.append(batch["regression"].detach().cpu().numpy())
        all_pr.append(reg.detach().cpu().numpy())
    result = metrics(np.concatenate(all_y), np.concatenate(all_p), np.concatenate(all_yr), np.concatenate(all_pr))
    result["loss"] = total / len(loader.dataset)
    return result


@torch.inference_mode()
def predict_arrays(model, split, device, batch_size=128):
    ds = StandardDataset(split)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    model.eval()
    probs, pred, reg, y_cls, y_reg = [], [], [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        logits, out_reg, _, _ = model(batch["x"], batch["mask"])
        p = torch.softmax(logits, dim=-1)
        probs.append(p.cpu().numpy()); pred.append(logits.argmax(-1).cpu().numpy()); reg.append(out_reg.cpu().numpy())
        y_cls.append(batch["classification"].cpu().numpy()); y_reg.append(batch["regression"].cpu().numpy())
    return {"probabilities": np.concatenate(probs), "classification": np.concatenate(pred), "regression": np.concatenate(reg), "true_classification": np.concatenate(y_cls), "true_regression": np.concatenate(y_reg)}


def calibrate_class_bias(y_true, probabilities, grid=None):
    """Choose a small class-logit bias on validation data only.

    This corrects a systematic class-prior shift without touching test labels.
    The grid is intentionally bounded so calibration cannot turn into a
    validation-specific lookup table.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    values = np.linspace(-0.30, 0.30, 13) if grid is None else np.asarray(grid, dtype=np.float64)
    logp = np.log(np.clip(probabilities, 1e-8, 1.0))
    best = None
    for a in values:
        for b in values:
            for c in values:
                bias = np.asarray([a, b, c], dtype=np.float64)
                pred = np.argmax(logp + bias, axis=1)
                acc = float(np.mean(pred == y_true))
                cm = np.zeros((3, 3), dtype=np.int64)
                for y, p in zip(y_true, pred):
                    cm[int(y), int(p)] += 1
                f1 = []
                for k in range(3):
                    tp = cm[k, k]
                    precision = tp / max(cm[:, k].sum(), 1)
                    recall = tp / max(cm[k].sum(), 1)
                    f1.append(2 * precision * recall / max(precision + recall, 1e-12))
                macro_f1 = float(np.mean(f1))
                score = acc + 0.03 * macro_f1
                candidate = (score, acc, macro_f1, bias)
                if best is None or candidate[:3] > best[:3]:
                    best = candidate
    # A common offset is unidentifiable under argmax; anchor the bias vector
    # to zero mean so the saved calibration is interpretable and stable.
    bias = best[3] - np.mean(best[3])
    return {"bias": bias.tolist(), "validation_accuracy": best[1], "validation_macro_f1": best[2], "grid": values.tolist()}


def apply_class_bias(prediction, calibration):
    out = dict(prediction)
    bias = np.asarray(calibration.get("bias", [0.0, 0.0, 0.0]), dtype=np.float64)
    scores = np.log(np.clip(out["probabilities"], 1e-8, 1.0)) + bias[None, :]
    scores -= scores.max(axis=1, keepdims=True)
    calibrated = np.exp(scores)
    calibrated /= calibrated.sum(axis=1, keepdims=True)
    out["probabilities"] = calibrated
    out["classification"] = np.argmax(calibrated, axis=1)
    return out


@torch.inference_mode()
def predict_missing(model, records, stats, device, batch_size=16):
    x = {m: np.stack([r["x"][m] for r in records]).astype(np.float32) for m in MODALITIES}
    mask = {m: np.stack([r["mask"][m] for r in records]) for m in MODALITIES}
    split = {"x": x, "mask": mask, "classification": np.zeros(len(records), np.int64), "regression": np.zeros(len(records), np.float32), "id": [r["id"] for r in records]}
    split = apply_normalizer(split, stats)
    ds = StandardDataset(split)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    out = {"probabilities": [], "classification": [], "regression": [], "mask_ratios": {m: [] for m in MODALITIES}}
    for batch in loader:
        moved = move_batch(batch, device)
        logits, reg, _, present = model(moved["x"], moved["mask"])
        out["probabilities"].append(torch.softmax(logits, -1).cpu().numpy())
        out["classification"].append(logits.argmax(-1).cpu().numpy())
        out["regression"].append(reg.cpu().numpy())
        for m in MODALITIES:
            out["mask_ratios"][m].extend(moved["mask"][m].float().mean(1).cpu().numpy().tolist())
    out["probabilities"] = np.concatenate(out["probabilities"])
    out["classification"] = np.concatenate(out["classification"])
    out["regression"] = np.concatenate(out["regression"])
    return out


def write_predictions(path, ids, prediction, include_truth=None):
    fields = ["id", "predicted_class", "predicted_label", "regression_prediction", "p_negative", "p_neutral", "p_positive"]
    if include_truth is not None:
        fields += ["true_class", "true_regression"]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for i, sample_id in enumerate(ids):
            row = {"id": sample_id, "predicted_class": int(prediction["classification"][i]), "predicted_label": CLASS_NAMES[int(prediction["classification"][i])], "regression_prediction": float(prediction["regression"][i]), "p_negative": float(prediction["probabilities"][i, 0]), "p_neutral": float(prediction["probabilities"][i, 1]), "p_positive": float(prediction["probabilities"][i, 2])}
            if include_truth is not None:
                row.update({"true_class": int(include_truth["true_classification"][i]), "true_regression": float(include_truth["true_regression"][i])})
            writer.writerow(row)


def main(argv=None):
    p = argparse.ArgumentParser(description="Train the Problem 2 local-missingness-aware multimodal model")
    p.add_argument("--data-root", type=Path, default=Path("D:/E_math/DATA"))
    p.add_argument("--output-root", type=Path, default=Path("D:/E_math/problem2_outputs"))
    p.add_argument("--text-model", type=Path, default=Path("D:/E_math/models/bert-base-uncased"))
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--architecture", choices=("hybrid", "summary", "pooled", "token", "gated"), default="hybrid")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--scheduler", choices=("cosine", "plateau", "constant"), default="cosine", help="Learning-rate schedule")
    p.add_argument("--warmup-epochs", type=int, default=3, help="Linear warmup length for cosine schedule")
    p.add_argument("--min-lr", type=float, default=1e-6, help="Final learning rate for cosine schedule")
    p.add_argument("--local-dropout", type=float, default=0.10, help="Probability of local contiguous span masking during training")
    p.add_argument("--model-dropout", type=float, default=0.25, help="Dropout inside the fusion network")
    p.add_argument("--reg-weight", type=float, default=0.15, help="Weight of the regression auxiliary loss")
    p.add_argument("--ordinal-weight", type=float, default=0.20, help="Weight of the ordered negative/neutral/positive auxiliary loss")
    p.add_argument("--label-smoothing", type=float, default=0.03, help="Cross-entropy label smoothing")
    p.add_argument("--class-weight", choices=("none", "sqrt", "inverse"), default="none", help="Training class weighting; none targets accuracy")
    p.add_argument("--neutral-boost", type=float, default=1.0, help="Additional multiplier for the Neutral class loss")
    p.add_argument("--weight-decay", type=float, default=5e-4, help="AdamW weight decay")
    p.add_argument("--select", choices=("accuracy", "macro_f1", "balanced"), default="accuracy", help="Validation criterion for checkpoint selection")
    p.add_argument("--calibrate", action="store_true", help="Tune a bounded class-logit bias on validation predictions")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    args = p.parse_args(argv)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = args.output_root.resolve(); out.mkdir(parents=True, exist_ok=True)
    standard = load_standard(args.data_root / "attachment_2_standard_features" / "aligned_50.pkl")
    stats = fit_normalizer(standard["train"])
    save_stats(out / "normalization.json", stats)
    normalized = {k: apply_normalizer(v, stats) for k, v in standard.items()}
    train_ds = StandardDataset(normalized["train"], train=True, seed=args.seed)
    valid_ds = StandardDataset(normalized["valid"])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    if args.architecture == "hybrid":
        model = HybridStatsFusionNet(hidden=args.hidden, layers=args.layers, dropout=args.model_dropout).to(device)
    elif args.architecture == "summary":
        model = SummaryMLP(hidden=args.hidden, dropout=args.model_dropout).to(device)
    elif args.architecture == "pooled":
        model = PooledFusionNet(hidden=args.hidden, dropout=args.model_dropout).to(device)
    elif args.architecture == "token":
        model = TokenFusionNet(hidden=args.hidden, layers=args.layers, dropout=args.model_dropout).to(device)
    else:
        model = RobustFusionNet(hidden=args.hidden, layers=args.layers, dropout=args.model_dropout).to(device)
    counts = np.bincount(normalized["train"]["classification"], minlength=3).astype(np.float32)
    if args.class_weight == "sqrt":
        weights = 1.0 / np.sqrt(np.maximum(counts, 1))
    elif args.class_weight == "inverse":
        weights = 1.0 / np.maximum(counts, 1)
    else:
        weights = np.ones(3, dtype=np.float32)
    weights[1] *= max(float(args.neutral_boost), 0.0)
    weights = torch.tensor(weights / np.mean(weights), dtype=torch.float32, device=device)
    cls_loss = nn.CrossEntropyLoss(weight=weights, label_smoothing=float(args.label_smoothing))
    reg_loss = nn.SmoothL1Loss(beta=0.5)
    ordinal_loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, min_lr=args.min_lr) if args.scheduler == "plateau" else None
    best_score, best_epoch, stale = -1.0, 0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        if args.scheduler == "cosine":
            current_lr = set_warmup_cosine_lr(optimizer, args.lr, args.min_lr, epoch, args.epochs, args.warmup_epochs)
        else:
            current_lr = float(optimizer.param_groups[0]["lr"])
        tr = run_epoch(model, train_loader, optimizer, cls_loss, reg_loss, ordinal_loss, device, train=True, local_dropout=args.local_dropout, reg_weight=args.reg_weight, ordinal_weight=args.ordinal_weight)
        va = run_epoch(model, valid_loader, optimizer, cls_loss, reg_loss, ordinal_loss, device, train=False, reg_weight=args.reg_weight, ordinal_weight=args.ordinal_weight)
        if args.select == "accuracy":
            score = va["accuracy"] + 0.01 * va["macro_f1"]
        elif args.select == "balanced":
            score = 0.5 * va["accuracy"] + 0.5 * va["macro_f1"]
        else:
            score = va["macro_f1"] - 0.05 * va["mae"]
        if plateau_scheduler is not None:
            # ReduceLROnPlateau must follow the same validation objective used
            # for checkpoint selection; monitoring macro-F1 unconditionally
            # was inconsistent when the user selected accuracy or balanced.
            plateau_scheduler.step(score)
            current_lr = float(optimizer.param_groups[0]["lr"])
        history.append({"epoch": epoch, "train": tr, "valid": va, "lr": current_lr})
        print(f"epoch {epoch:03d} train_f1={tr['macro_f1']:.4f} valid_f1={va['macro_f1']:.4f} valid_acc={va['accuracy']:.4f} valid_mae={va['mae']:.4f}", flush=True)
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save({"model": model.state_dict(), "config": {"architecture": args.architecture, "hidden": args.hidden, "layers": args.layers, "dims": [768, 74, 35]}, "normalization": {m: {k: torch.from_numpy(v).float() for k, v in d.items()} for m, d in stats.items()}, "epoch": epoch}, out / "problem2_checkpoint.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    checkpoint = torch.load(out / "problem2_checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    valid_prediction = predict_arrays(model, normalized["valid"], device)
    if args.calibrate:
        calibration = calibrate_class_bias(valid_prediction["true_classification"], valid_prediction["probabilities"])
    else:
        calibration = {"bias": [0.0, 0.0, 0.0], "validation_accuracy": None, "validation_macro_f1": None, "grid": []}
    valid_prediction = apply_class_bias(valid_prediction, calibration)
    valid_metrics = metrics(valid_prediction["true_classification"], valid_prediction["classification"], valid_prediction["true_regression"], valid_prediction["regression"])
    test_prediction = apply_class_bias(predict_arrays(model, normalized["test"], device), calibration)
    test_metrics = metrics(test_prediction["true_classification"], test_prediction["classification"], test_prediction["true_regression"], test_prediction["regression"])
    write_predictions(out / "problem2_test_predictions.csv", normalized["test"]["id"], test_prediction, test_prediction)
    missing_root = args.data_root / "attachment_3_missing_modality"
    missing_summary = {}
    for variant in ("aligned", "unaligned"):
        records = load_missing_features(missing_root / variant, args.text_model, device)
        prediction = predict_missing(model, records, stats, device)
        prediction = apply_class_bias(prediction, calibration)
        path = out / f"problem2_missing_{variant}_predictions.csv"
        write_predictions(path, [r["id"] for r in records], prediction)
        missing_summary[variant] = {"count": len(records), "path": str(path), "mean_mask_ratio": {m: float(np.mean(prediction["mask_ratios"][m])) for m in MODALITIES}}
    metadata = {"device": str(device), "best_epoch": best_epoch, "selection": args.select, "architecture": args.architecture, "hidden": args.hidden, "layers": args.layers, "class_weight": args.class_weight, "class_loss_weights": weights.detach().cpu().tolist(), "neutral_boost": args.neutral_boost, "label_smoothing": args.label_smoothing, "ordinal_weight": args.ordinal_weight, "model_dropout": args.model_dropout, "weight_decay": args.weight_decay, "scheduler": {"name": args.scheduler, "base_lr": args.lr, "min_lr": args.min_lr, "warmup_epochs": args.warmup_epochs}, "class_names": CLASS_NAMES, "train_counts": counts.tolist(), "calibration": calibration, "valid_metrics": valid_metrics, "test_metrics": test_metrics, "missing_modality": missing_summary, "history": history, "local_dropout_probability": args.local_dropout, "local_dropout": "independent contiguous spans; no full-modality dropout", "text_mask": "text_bert attention mask used for padded BERT positions"}
    (out / "problem2_metrics.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"device": str(device), "best_epoch": best_epoch, "test": test_metrics, "output": str(out)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
