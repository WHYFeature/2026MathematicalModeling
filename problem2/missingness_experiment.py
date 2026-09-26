"""Controlled missing-modality experiments for Problem 2.

This is an independent experiment driver.  It never writes to the existing
Problem 2 checkpoints or prediction files.  The same 50-bin representation,
train-only normalization, model, and evaluation code are used for every
missingness condition.  Missingness is injected into masks (and then into the
normalized feature tensor) so padding is not counted as an artificial loss.

Run through ``python -m problem2.missingness_experiment`` from D:\\E_math.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from problem2.data import (
    MODALITIES,
    StandardDataset,
    apply_local_dropout,
    apply_normalizer,
    collate,
    fit_normalizer,
    load_missing_features,
    load_standard,
    save_stats,
)
from problem2.model import HybridStatsFusionNet
from problem2.train import (
    CLASS_NAMES,
    metrics,
    move_batch,
    predict_missing,
    run_epoch,
    seed_everything,
)


@dataclass(frozen=True)
class Condition:
    condition_id: str
    group: str
    missing_modalities: tuple[str, ...]
    missing_rate: float
    position: str
    duration: str


def _parse_seeds(value: str) -> list[int]:
    out = []
    for token in value.split(","):
        token = token.strip()
        if token:
            out.append(int(token))
    if not out:
        raise argparse.ArgumentTypeError("--seeds must contain at least one integer")
    return list(dict.fromkeys(out))


def _normalise_modality_name(modality: str) -> str:
    return {"text": "text", "audio": "audio", "vision": "vision"}[modality]


def _condition_name(group: str, modalities: Iterable[str], rate: float, position: str, duration: str) -> str:
    mods = "_".join(modalities) if modalities else "complete"
    rate_text = f"{rate:.2f}".replace(".", "p")
    return f"{group}__{mods}__r{rate_text}__{position}__{duration}"


def build_conditions() -> list[Condition]:
    """Return the complete, rate, position, duration, and ablation grid.

    The 0.30 position/duration panels answer where and how long a local gap is;
    the rate panel answers how performance changes as the gap grows.  Pair and
    three-modality conditions are included in the ablation panel.
    """
    conditions: list[Condition] = []

    def add(group, mods, rate, position="random", duration="medium"):
        mods = tuple(mods)
        conditions.append(Condition(
            _condition_name(group, mods, rate, position, duration),
            group, mods, float(rate), position, duration,
        ))

    add("baseline", (), 0.0, "none", "none")
    rates = (0.10, 0.20, 0.30, 0.40)
    sets = (("text",), ("audio",), ("vision",),
            ("text", "audio"), ("text", "vision"), ("audio", "vision"),
            ("text", "audio", "vision"))
    for mods in sets:
        for rate in rates:
            add("rate", mods, rate)
    for mod in MODALITIES:
        for pos in ("prefix", "middle", "suffix", "random"):
            add("position", (mod,), 0.30, pos, "medium")
        for duration in ("short", "medium", "long"):
            add("duration", (mod,), 0.30, "random", duration)
    # Full modality ablation: a removed modality is unavailable at every
    # originally valid time bin.  The 30% local gaps above remain the
    # controlled partial-missingness conditions.
    add("ablation", (), 0.0, "none", "none")
    for mods in sets:
        add("ablation", mods, 1.0, "random", "long")
    # Avoid accidental duplicate IDs if the grid is edited later.
    unique = {}
    for c in conditions:
        unique[c.condition_id] = c
    return list(unique.values())


def _pool_split_to_50(split: dict) -> dict:
    """Pool a non-50 temporal representation without loading another pickle.

    ``unaligned_50.pkl`` is normally already 50 bins.  This fallback makes the
    feature-version switch explicit and safe if a future file has a different
    number of rows.  The supplied validity mask is used, so padded rows do not
    become valid after pooling.
    """
    result = {k: v for k, v in split.items() if k not in ("x", "mask")}
    result["x"], result["mask"] = {}, {}
    for m in MODALITIES:
        x = np.asarray(split["x"][m], dtype=np.float32)
        mask = np.asarray(split["mask"][m], dtype=bool)
        if x.shape[1] == 50:
            result["x"][m], result["mask"][m] = x, mask
            continue
        out = np.zeros((x.shape[0], 50, x.shape[2]), dtype=np.float32)
        out_mask = np.zeros((x.shape[0], 50), dtype=bool)
        edges = np.linspace(0, x.shape[1], 51).round().astype(int)
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            if hi <= lo:
                continue
            good = mask[:, lo:hi]
            count = good.sum(axis=1)
            valid_rows = count > 0
            if np.any(valid_rows):
                values = x[:, lo:hi]
                out[valid_rows, i] = (values * good[..., None]).sum(axis=1)[valid_rows] / count[valid_rows, None]
                out_mask[valid_rows, i] = True
        result["x"][m], result["mask"][m] = out, out_mask
    return result


def load_feature_version(path: Path) -> dict:
    data = load_standard(path)
    return {name: _pool_split_to_50(split) for name, split in data.items()}


def _select_positions(valid: np.ndarray, count: int, position: str, duration: str, rng: np.random.Generator) -> np.ndarray:
    available = np.flatnonzero(valid)
    if count <= 0 or available.size == 0:
        return np.empty(0, dtype=np.int64)
    count = min(int(count), int(available.size))
    if count == available.size:
        return available
    if position == "prefix":
        return available[:count]
    if position == "suffix":
        return available[-count:]
    if position == "middle":
        start = max(0, (available.size - count) // 2)
        return available[start:start + count]
    if duration == "long":
        # One contiguous block in valid-coordinate space.
        start = int(rng.integers(0, available.size - count + 1))
        return available[start:start + count]
    if duration == "medium":
        block = max(1, int(round(count * 0.55)))
    else:  # short: several small blocks separated by available positions
        block = max(1, int(round(count * 0.20)))
    chosen: list[int] = []
    # Sample starts without replacement where possible, then fill to exactly
    # the requested count.  The sorting makes the resulting mask deterministic.
    starts = list(range(max(1, available.size - block + 1)))
    rng.shuffle(starts)
    for start in starts:
        candidate = available[start:start + block]
        for value in candidate.tolist():
            if value not in chosen:
                chosen.append(value)
                if len(chosen) >= count:
                    return np.asarray(sorted(chosen[:count]), dtype=np.int64)
        if len(chosen) >= count:
            break
    if len(chosen) < count:
        rest = np.asarray([x for x in available.tolist() if x not in set(chosen)], dtype=np.int64)
        chosen.extend(rest[:count - len(chosen)].tolist())
    return np.asarray(sorted(chosen[:count]), dtype=np.int64)


def apply_condition(split: dict, condition: Condition, seed: int) -> tuple[dict, dict]:
    """Mask a split and return (masked_split, aggregate mask statistics)."""
    x = {m: np.asarray(split["x"][m], dtype=np.float32).copy() for m in MODALITIES}
    original = {m: np.asarray(split["mask"][m], dtype=bool) for m in MODALITIES}
    masks = {m: original[m].copy() for m in MODALITIES}
    rng = np.random.default_rng(seed)
    for row in range(len(split["classification"])):
        for modality in condition.missing_modalities:
            base = original[modality][row]
            valid_count = int(base.sum())
            requested = int(round(condition.missing_rate * valid_count))
            selected = _select_positions(base, requested, condition.position, condition.duration, rng)
            masks[modality][row, selected] = False
    for m in MODALITIES:
        x[m][~masks[m]] = 0.0
    base_total = float(sum(original[m].sum() for m in MODALITIES))
    after_total = float(sum(masks[m].sum() for m in MODALITIES))
    stats = {
        "effective_missing_rate": (base_total - after_total) / max(base_total, 1.0),
        "base_valid_bins": int(base_total),
        "remaining_valid_bins": int(after_total),
    }
    for m in MODALITIES:
        before = float(original[m].sum())
        after = float(masks[m].sum())
        stats[f"effective_missing_rate_{m}"] = (before - after) / max(before, 1.0)
        stats[f"remaining_ratio_{m}"] = after / max(before, 1.0)
    result = {
        "x": x,
        "mask": masks,
        "classification": np.asarray(split["classification"], dtype=np.int64),
        "regression": np.asarray(split["regression"], dtype=np.float32),
        "id": list(split["id"]),
    }
    return result, stats


def evaluate_split(model, split: dict, device: torch.device, batch_size: int) -> dict:
    ds = StandardDataset(split)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate,
                        pin_memory=device.type == "cuda")
    model.eval()
    ys, ps, yrs, prs = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            logits, reg, _, _ = model(batch["x"], batch["mask"])
            ys.append(batch["classification"].cpu().numpy())
            ps.append(logits.argmax(-1).cpu().numpy())
            yrs.append(batch["regression"].cpu().numpy())
            prs.append(reg.cpu().numpy())
    return metrics(np.concatenate(ys), np.concatenate(ps), np.concatenate(yrs), np.concatenate(prs))


def _make_model(hidden: int, layers: int, dropout: float, device: torch.device) -> nn.Module:
    return HybridStatsFusionNet(hidden=hidden, layers=layers, dropout=dropout).to(device)


def train_one(seed: int, normalized: dict, args, out: Path, device: torch.device) -> tuple[nn.Module, dict, dict]:
    seed_everything(seed)
    seed_dir = out / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    train_ds = StandardDataset(normalized["train"], train=True, seed=seed)
    valid_ds = StandardDataset(normalized["valid"])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                               pin_memory=device.type == "cuda")
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate,
                              pin_memory=device.type == "cuda")
    model = _make_model(args.hidden, args.layers, args.model_dropout, device)
    counts = np.bincount(normalized["train"]["classification"], minlength=3).astype(np.float32)
    weights = np.ones(3, dtype=np.float32)
    if args.class_weight == "sqrt":
        weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    elif args.class_weight == "inverse":
        weights = 1.0 / np.maximum(counts, 1.0)
    weights[1] *= args.neutral_boost
    weights = torch.tensor(weights / np.mean(weights), dtype=torch.float32, device=device)
    cls_loss = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    reg_loss = nn.SmoothL1Loss(beta=0.5)
    ordinal_loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.min_lr)
    best_score, best_epoch, stale = -float("inf"), 0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        # Linear warm-up before cosine annealing.  The helper is deliberately
        # local so this experiment is independent of the old formal checkpoint.
        if args.warmup_epochs > 0 and epoch <= args.warmup_epochs:
            lr_now = args.lr * epoch / args.warmup_epochs
            for group in optimizer.param_groups:
                group["lr"] = lr_now
        train_metrics = run_epoch(model, train_loader, optimizer, cls_loss, reg_loss, ordinal_loss, device,
                                   train=True, local_dropout=args.train_local_dropout,
                                   reg_weight=args.reg_weight, ordinal_weight=args.ordinal_weight)
        valid_metrics = run_epoch(model, valid_loader, optimizer, cls_loss, reg_loss, ordinal_loss, device,
                                  train=False, reg_weight=args.reg_weight, ordinal_weight=args.ordinal_weight)
        score = valid_metrics["accuracy"] + 0.01 * valid_metrics["macro_f1"]
        if epoch > args.warmup_epochs:
            scheduler.step()
        lr_now = float(optimizer.param_groups[0]["lr"])
        item = {"epoch": epoch, "lr": lr_now, "train": train_metrics, "valid": valid_metrics}
        history.append(item)
        print(f"seed={seed} epoch {epoch:03d} train_acc={train_metrics['accuracy']:.4f} valid_acc={valid_metrics['accuracy']:.4f} valid_f1={valid_metrics['macro_f1']:.4f}", flush=True)
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "seed": seed,
                        "config": {"hidden": args.hidden, "layers": args.layers,
                                   "model_dropout": args.model_dropout}}, seed_dir / "checkpoint.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    ckpt = torch.load(seed_dir / "checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    history_payload = {"seed": seed, "best_epoch": int(best_epoch), "history": history}
    (seed_dir / "training_history.json").write_text(json.dumps(history_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (seed_dir / "training_history.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "lr", "train_loss", "valid_loss", "train_accuracy", "valid_accuracy", "train_macro_f1", "valid_macro_f1", "valid_mae"])
        for h in history:
            writer.writerow([h["epoch"], h["lr"], h["train"]["loss"], h["valid"]["loss"], h["train"]["accuracy"], h["valid"]["accuracy"], h["train"]["macro_f1"], h["valid"]["macro_f1"], h["valid"]["mae"]])
    return model, history_payload, {"train_counts": counts.tolist(), "class_loss_weights": weights.detach().cpu().tolist()}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_outputs(out: Path, condition_rows: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for the experiment figures; install it in .venv_problem1") from exc
    plt.rcParams.update({"font.family": "Times New Roman", "axes.unicode_minus": False, "font.size": 10})
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    # Average across seeds, using test rows for the final reported curves.
    rows = [r for r in condition_rows if r["split"] == "test"]
    colors = {"text": "#355C7D", "audio": "#C06C84", "vision": "#6C9A8B", "text_audio": "#8E6C88", "text_vision": "#D08C60", "audio_vision": "#5B7DB1", "text_audio_vision": "#555555"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for mods in (("text",), ("audio",), ("vision",), ("text", "audio"), ("text", "vision"), ("audio", "vision"), ("text", "audio", "vision")):
        label = "_".join(mods)
        rr = [r for r in rows if r["group"] == "rate" and tuple(r["missing_modalities"].split("+")) == mods]
        if not rr:
            continue
        by_rate = {}
        for r in rr:
            by_rate.setdefault(float(r["requested_rate"]), []).append(r)
        xs = sorted(by_rate)
        acc = [np.mean([float(x["accuracy"]) for x in by_rate[v]]) for v in xs]
        f1 = [np.mean([float(x["macro_f1"]) for x in by_rate[v]]) for v in xs]
        axes[0].plot(xs, acc, marker="o", label=label, color=colors.get(label, None))
        axes[1].plot(xs, f1, marker="o", label=label, color=colors.get(label, None))
    axes[0].set(xlabel="Requested missing rate", ylabel="Accuracy")
    axes[1].set(xlabel="Requested missing rate", ylabel="Macro-F1")
    for ax in axes:
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(fig_dir / "missingness_curves.png", dpi=300)
    fig.savefig(fig_dir / "missingness_curves.pdf")
    plt.close(fig)

    # Position x duration heatmap for each single modality.
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), constrained_layout=True)
    for ax, mod in zip(axes, MODALITIES):
        matrix = np.full((4, 3), np.nan)
        for i, pos in enumerate(("prefix", "middle", "suffix", "random")):
            for j, dur in enumerate(("short", "medium", "long")):
                rr = [r for r in rows if r["group"] in ("position", "duration") and r["missing_modalities"] == mod and r["position"] == pos and r["duration"] == dur]
                if rr:
                    matrix[i, j] = np.mean([float(x["accuracy"]) for x in rr])
                else:
                    # Position and duration panels are complementary; for a
                    # missing cell use the position-only or duration-only row.
                    rr = [r for r in rows if r["group"] == "position" and r["missing_modalities"] == mod and r["position"] == pos]
                    if not rr:
                        rr = [r for r in rows if r["group"] == "duration" and r["missing_modalities"] == mod and r["duration"] == dur]
                    if rr:
                        matrix[i, j] = np.mean([float(x["accuracy"]) for x in rr])
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set(title=mod.capitalize(), xticks=range(3), xticklabels=["Short", "Medium", "Long"], yticks=range(4), yticklabels=["Prefix", "Middle", "Suffix", "Random"])
        for i in range(4):
            for j in range(3):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", color="white" if matrix[i, j] < 0.6 else "black", fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.8, label="Accuracy")
    fig.savefig(fig_dir / "position_duration_heatmap.png", dpi=300)
    fig.savefig(fig_dir / "position_duration_heatmap.pdf")
    plt.close(fig)

    # Modality ablation bar chart.
    ab = [r for r in rows if r["group"] == "ablation"]
    labels, values = [], []
    for r in ab:
        labels.append(r["missing_modalities"].replace("+", " + ") if r["missing_modalities"] else "Complete")
        values.append(float(r["accuracy"]))
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    if values:
        bars = ax.bar(labels, values, color="#4C78A8")
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=2, fontsize=8)
    ax.set_ylim(0, 1); ax.set_ylabel("Accuracy"); ax.grid(axis="y", alpha=0.25)
    fig.savefig(fig_dir / "modality_ablation.png", dpi=300)
    fig.savefig(fig_dir / "modality_ablation.pdf")
    plt.close(fig)


def _plot_training_curves(out: Path, histories: list[dict]) -> None:
    """Save measured train/validation curves for every seed and a panel."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for the experiment figures; install it in .venv_problem1") from exc
    plt.rcParams.update({"font.family": "Times New Roman", "axes.unicode_minus": False, "font.size": 10})
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    for payload in histories:
        epochs = [h["epoch"] for h in payload["history"]]
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
        axes[0].plot(epochs, [h["train"]["loss"] for h in payload["history"]], label="Train", color="#355C7D")
        axes[0].plot(epochs, [h["valid"]["loss"] for h in payload["history"]], label="Validation", color="#C06C84")
        axes[0].set(xlabel="Epoch", ylabel="Loss")
        axes[1].plot(epochs, [h["train"]["accuracy"] for h in payload["history"]], label="Train", color="#355C7D")
        axes[1].plot(epochs, [h["valid"]["accuracy"] for h in payload["history"]], label="Validation", color="#C06C84")
        axes[1].set(xlabel="Epoch", ylabel="Accuracy")
        axes[2].plot(epochs, [h["train"]["macro_f1"] for h in payload["history"]], label="Train", color="#355C7D")
        axes[2].plot(epochs, [h["valid"]["macro_f1"] for h in payload["history"]], label="Validation", color="#C06C84")
        axes[2].set(xlabel="Epoch", ylabel="Macro-F1")
        for ax in axes:
            ax.grid(alpha=0.25); ax.legend(frameon=False)
        fig.savefig(fig_dir / f"training_curves_seed_{payload['seed']}.png", dpi=300)
        fig.savefig(fig_dir / f"training_curves_seed_{payload['seed']}.pdf")
        plt.close(fig)
    if len(histories) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
        for payload in histories:
            epochs = [h["epoch"] for h in payload["history"]]
            axes[0].plot(epochs, [h["valid"]["accuracy"] for h in payload["history"]], label=f"Seed {payload['seed']}")
            axes[1].plot(epochs, [h["valid"]["macro_f1"] for h in payload["history"]], label=f"Seed {payload['seed']}")
        axes[0].set(xlabel="Epoch", ylabel="Validation accuracy")
        axes[1].set(xlabel="Epoch", ylabel="Validation Macro-F1")
        for ax in axes:
            ax.grid(alpha=0.25); ax.legend(frameon=False)
        fig.savefig(fig_dir / "training_curves_all_seeds.png", dpi=300)
        fig.savefig(fig_dir / "training_curves_all_seeds.pdf")
        plt.close(fig)


def export_attachment3(model, stats, args, device, out: Path) -> dict:
    """Predict aligned and unaligned attachment-3 files separately."""
    root = args.data_root / "attachment_3_missing_modality"
    rows, summaries = [], {}
    for variant in ("aligned", "unaligned"):
        records = load_missing_features(root / variant, args.text_model, device)
        prediction = predict_missing(model, records, stats, device, batch_size=args.batch_size)
        for i, rec in enumerate(records):
            row = {
                "variant": variant, "id": rec["id"],
                "predicted_class": int(prediction["classification"][i]),
                "predicted_label": CLASS_NAMES[int(prediction["classification"][i])],
                "regression_prediction": float(prediction["regression"][i]),
                "p_negative": float(prediction["probabilities"][i, 0]),
                "p_neutral": float(prediction["probabilities"][i, 1]),
                "p_positive": float(prediction["probabilities"][i, 2]),
            }
            for m in MODALITIES:
                row[f"valid_ratio_{m}"] = float(prediction["mask_ratios"][m][i])
            rows.append(row)
        summaries[variant] = {"count": len(records), "mean_valid_ratio": {m: float(np.mean(prediction["mask_ratios"][m])) for m in MODALITIES}, "class_counts": np.bincount(prediction["classification"], minlength=3).tolist()}
    _write_csv(out / "attachment3_predictions.csv", rows)
    (out / "attachment3_summary.json").write_text(json.dumps({"warning": "Attachment 3 has no labels; these are predictions and mask ratios, not accuracy.", "variants": summaries}, ensure_ascii=False, indent=2), encoding="utf-8")
    return summaries


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run controlled Problem 2 missingness and ablation experiments")
    parser.add_argument("--data-root", type=Path, default=Path("D:/E_math/DATA"))
    parser.add_argument("--text-model", type=Path, default=Path("D:/E_math/models/bert-base-uncased"))
    parser.add_argument("--output-root", type=Path, default=Path("D:/E_math/problem2_missingness_experiment"))
    parser.add_argument("--feature-version", choices=("aligned", "unaligned"), default="aligned")
    parser.add_argument("--seeds", type=_parse_seeds, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--model-dropout", type=float, default=0.25)
    parser.add_argument("--train-local-dropout", type=float, default=0.35)
    parser.add_argument("--reg-weight", type=float, default=0.15)
    parser.add_argument("--ordinal-weight", type=float, default=0.20)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--class-weight", choices=("none", "sqrt", "inverse"), default="none")
    parser.add_argument("--neutral-boost", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-attachment3", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    out = args.output_root.resolve()
    if out.exists() and any(out.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"Output directory is non-empty: {out}. Choose a new directory or pass --overwrite.")
        # Keep old output recoverable instead of deleting it.
        backup = out.with_name(out.name + "_backup_" + str(random.randint(1000, 9999)))
        shutil.move(str(out), str(backup))
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_path = args.data_root / "attachment_2_standard_features" / f"{args.feature_version}_50.pkl"
    standard = load_feature_version(feature_path)
    stats = fit_normalizer(standard["train"])
    save_stats(out / "normalization.json", stats)
    normalized = {k: apply_normalizer(v, stats) for k, v in standard.items()}
    conditions = build_conditions()
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config.update({"device": str(device), "feature_path": str(feature_path), "condition_count": len(conditions), "conditions": [asdict(c) for c in conditions], "protocol": {"normalization": "train-only", "test_labels_used_for_selection": False, "attachment3_labels_used": False, "temporal_bins": 50, "effective_rate_definition": "introduced missing valid bins / original valid bins"}})
    (out / "experiment_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    all_rows, seed_summaries, histories = [], [], []
    first_model = first_history = first_stats = None
    for seed in args.seeds:
        model, history, train_info = train_one(seed, normalized, args, out, device)
        histories.append(history)
        if first_model is None:
            first_model, first_history, first_stats = model, history, stats
        seed_valid_base = evaluate_split(model, normalized["valid"], device, args.batch_size * 2)
        seed_test_base = evaluate_split(model, normalized["test"], device, args.batch_size * 2)
        seed_summaries.append({"seed": seed, "best_epoch": history["best_epoch"], "valid_accuracy": seed_valid_base["accuracy"], "valid_macro_f1": seed_valid_base["macro_f1"], "test_accuracy": seed_test_base["accuracy"], "test_macro_f1": seed_test_base["macro_f1"]})
        for condition_index, condition in enumerate(conditions):
            for split_name in ("valid", "test"):
                masked, mask_stats = apply_condition(normalized[split_name], condition, seed * 10000 + condition_index)
                result = evaluate_split(model, masked, device, args.batch_size * 2)
                row = {"seed": seed, "split": split_name, "condition_id": condition.condition_id, "group": condition.group,
                       "missing_modalities": "+".join(condition.missing_modalities), "requested_rate": condition.missing_rate,
                       "position": condition.position, "duration": condition.duration, **mask_stats,
                       "accuracy": result["accuracy"], "macro_f1": result["macro_f1"], "mae": result["mae"], "rmse": result["rmse"], "pearson": result["pearson"],
                       "f1_negative": result["f1_by_class"]["Negative"], "f1_neutral": result["f1_by_class"]["Neutral"], "f1_positive": result["f1_by_class"]["Positive"], "valid_sample_count": len(masked["classification"])}
                all_rows.append(row)
        print(f"seed={seed} complete test_accuracy={seed_test_base['accuracy']:.4f} test_macro_f1={seed_test_base['macro_f1']:.4f}", flush=True)
    _write_csv(out / "condition_metrics.csv", all_rows)
    _write_csv(out / "seed_summary.csv", seed_summaries)
    # Mean and standard deviation across seeds for each split/condition.
    summary_rows = []
    keys = sorted({(r["split"], r["condition_id"]) for r in all_rows})
    metric_names = ("accuracy", "macro_f1", "mae", "rmse", "pearson", "effective_missing_rate")
    for split_name, condition_id in keys:
        rr = [r for r in all_rows if r["split"] == split_name and r["condition_id"] == condition_id]
        base = rr[0]
        row = {"split": split_name, "condition_id": condition_id, "group": base["group"], "missing_modalities": base["missing_modalities"], "requested_rate": base["requested_rate"], "position": base["position"], "duration": base["duration"]}
        for metric in metric_names:
            values = np.asarray([float(x[metric]) for x in rr])
            row[f"{metric}_mean"] = float(values.mean()); row[f"{metric}_std"] = float(values.std(ddof=0))
        summary_rows.append(row)
    _write_csv(out / "ablation_summary.csv", summary_rows)
    if first_model is not None and not args.skip_attachment3:
        export_attachment3(first_model, first_stats, args, device, out)
    _plot_outputs(out, all_rows)
    _plot_training_curves(out, histories)
    report = [
        "# Problem 2 controlled missingness experiment",
        "",
        f"Feature version: `{args.feature_version}` (50 temporal bins)",
        f"Seeds: `{','.join(str(x) for x in args.seeds)}`; conditions: `{len(conditions)}`",
        "",
        "The test set is evaluated only after each seed has selected its checkpoint on validation accuracy. Missingness is applied to original valid bins; padding is excluded from the effective rate.",
        "",
        "Attachment 3 has no labels. Its output is a prediction distribution and valid-ratio diagnostic, never an accuracy estimate.",
        "",
        "See `condition_metrics.csv` for per-seed rows, `ablation_summary.csv` for mean/std summaries, and `figures/` for the figures.",
    ]
    (out / "experiment_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    metadata = {"device": str(device), "feature_version": args.feature_version, "seed_summary": seed_summaries, "condition_count": len(conditions), "attachment3_note": "No attachment-3 accuracy is reported because labels are unavailable.", "files": ["condition_metrics.csv", "seed_summary.csv", "ablation_summary.csv", "attachment3_predictions.csv", "figures/missingness_curves.png", "figures/position_duration_heatmap.png", "figures/modality_ablation.png"]}
    (out / "experiment_summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "device": str(device), "seeds": args.seeds, "conditions": len(conditions), "test_complete": [{"seed": x["seed"], "accuracy": x["test_accuracy"], "macro_f1": x["test_macro_f1"]} for x in seed_summaries]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
