"""Problem 2 controlled missingness experiments on the BERT-AV protocol.

The experiment keeps the formal aligned attachment-2 representation fixed and
compares the original BERT-AV fusion with an availability-aware BERT-AV model,
with and without local contiguous missingness augmentation.  Attachment 3 is
only used after training for unlabeled aligned/unaligned predictions.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from problem2.bert_model import BertAVFusion, BertAVRobustFusion
from problem2.data import MODALITIES, fit_normalizer, pool_to_50, save_stats
from problem2_bert_fusion_train import load_missing_records, metrics_from_prediction


CLASS_NAMES = ("Negative", "Neutral", "Positive")


@dataclass(frozen=True)
class Condition:
    condition_id: str
    group: str
    missing_modalities: tuple[str, ...]
    missing_rate: float
    position: str
    duration: str


def parse_seeds(value: str) -> list[int]:
    seeds = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("--seeds must include an integer")
    return list(dict.fromkeys(seeds))


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BertAVDataset(Dataset):
    def __init__(self, split):
        self.ids = torch.from_numpy(np.asarray(split["ids"], dtype=np.int64))
        self.mask = torch.from_numpy(np.asarray(split["text_mask"], dtype=np.int64))
        self.types = torch.from_numpy(np.asarray(split["token_type_ids"], dtype=np.int64))
        self.audio = torch.from_numpy(np.asarray(split["audio"], dtype=np.float32))
        self.vision = torch.from_numpy(np.asarray(split["vision"], dtype=np.float32))
        self.audio_mask = torch.from_numpy(np.asarray(split["audio_mask"], dtype=np.bool_))
        self.vision_mask = torch.from_numpy(np.asarray(split["vision_mask"], dtype=np.bool_))
        self.y = torch.from_numpy(np.asarray(split["classification"], dtype=np.int64))
        self.r = torch.from_numpy(np.asarray(split["regression"], dtype=np.float32))

    def __len__(self): return len(self.y)

    def __getitem__(self, index):
        return (self.ids[index].clone(), self.mask[index].clone(), self.types[index].clone(),
                self.audio[index].clone(), self.audio_mask[index].clone(),
                self.vision[index].clone(), self.vision_mask[index].clone(),
                self.y[index], self.r[index])


def unpack(batch, device):
    return tuple(x.to(device, non_blocking=True) for x in batch)


def load_aligned(path: Path) -> tuple[dict, dict]:
    """Load aligned_50.pkl once and construct the BERT-AV split view."""
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    view = {}
    for name in ("train", "valid", "test"):
        split = raw[name]
        text_bert = np.asarray(split["text_bert"], dtype=np.int64)
        if text_bert.ndim != 3 or text_bert.shape[1:] != (3, 50):
            raise ValueError(f"Unexpected text_bert shape in {name}: {text_bert.shape}")
        audio = np.asarray(split["audio"], dtype=np.float32)
        vision = np.asarray(split["vision"], dtype=np.float32)
        view[name] = {
            "ids": text_bert[:, 0], "text_mask": text_bert[:, 1].astype(bool), "token_type_ids": text_bert[:, 2],
            "audio": audio, "vision": vision,
            "audio_mask": ~np.all(np.isclose(audio, 0.0), axis=-1),
            "vision_mask": ~np.all(np.isclose(vision, 0.0), axis=-1),
            "classification": np.asarray(split["classification_labels"], dtype=np.int64),
            "regression": np.asarray(split["regression_labels"], dtype=np.float32),
            "id": [str(x) for x in split["id"]],
        }
    return raw, view


def normalize_view(view, stats):
    out = {}
    for split_name, split in view.items():
        item = dict(split)
        for m in ("audio", "vision"):
            x = ((split[m] - stats[m]["mean"]) / stats[m]["std"]).astype(np.float32)
            x[~split[f"{m}_mask"]] = 0.0
            item[m] = x
        out[split_name] = item
    return out


def condition_name(group, mods, rate, pos, duration):
    label = "+".join(mods) if mods else "complete"
    return f"{group}__{label}__r{rate:.2f}__{pos}__{duration}".replace(".", "p")


def build_conditions() -> list[Condition]:
    conditions = []

    def add(group, mods, rate, pos="random", duration="medium"):
        mods = tuple(mods)
        conditions.append(Condition(condition_name(group, mods, rate, pos, duration), group, mods, float(rate), pos, duration))

    add("baseline", (), 0.0, "none", "none")
    sets = (("text",), ("audio",), ("vision",), ("text", "audio"), ("text", "vision"), ("audio", "vision"), ("text", "audio", "vision"))
    for mods in sets:
        for rate in (0.10, 0.20, 0.30, 0.40): add("rate", mods, rate)
    for mod in MODALITIES:
        for pos in ("prefix", "middle", "suffix", "random"): add("position", (mod,), 0.30, pos, "medium")
        for duration in ("short", "medium", "long"): add("duration", (mod,), 0.30, "random", duration)
    add("ablation", (), 0.0, "none", "none")
    for mods in sets: add("ablation", mods, 1.0, "random", "long")
    return list({c.condition_id: c for c in conditions}.values())


def choose_positions(valid: np.ndarray, count: int, position: str, duration: str, rng: np.random.Generator) -> np.ndarray:
    available = np.flatnonzero(valid)
    count = min(max(int(count), 0), len(available))
    if count == 0: return np.empty(0, dtype=np.int64)
    if count == len(available): return available
    if position == "prefix": return available[:count]
    if position == "suffix": return available[-count:]
    if position == "middle":
        start = (len(available) - count) // 2
        return available[start:start + count]
    if duration == "long":
        start = int(rng.integers(0, len(available) - count + 1))
        return available[start:start + count]
    block = max(1, int(round(count * (0.20 if duration == "short" else 0.55))))
    starts = np.arange(max(1, len(available) - block + 1)); rng.shuffle(starts)
    chosen = []
    for start in starts:
        for x in available[start:start + block].tolist():
            if x not in chosen: chosen.append(x)
            if len(chosen) == count: return np.asarray(sorted(chosen), dtype=np.int64)
    return np.asarray(sorted(chosen[:count]), dtype=np.int64)


def apply_condition(split, condition: Condition, seed: int):
    result = {k: (v.copy() if isinstance(v, np.ndarray) else list(v) if isinstance(v, list) else v) for k, v in split.items()}
    rng = np.random.default_rng(seed)
    original = {"text": split["text_mask"].copy(), "audio": split["audio_mask"].copy(), "vision": split["vision_mask"].copy()}
    for row in range(len(split["classification"])):
        for mod in condition.missing_modalities:
            selected = choose_positions(original[mod][row], round(condition.missing_rate * original[mod][row].sum()), condition.position, condition.duration, rng)
            result[f"{mod}_mask"][row, selected] = False
    result["text_mask"] = result["text_mask"].astype(bool)
    for mod in ("audio", "vision"):
        result[mod][~result[f"{mod}_mask"]] = 0.0
    result["ids"] = result["ids"].copy(); result["token_type_ids"] = result["token_type_ids"].copy()
    result["ids"][~result["text_mask"]] = 0
    result["token_type_ids"][~result["text_mask"]] = 0
    base = float(sum(original[m].sum() for m in MODALITIES)); remaining = float(sum(result[f"{m}_mask"].sum() for m in MODALITIES))
    info = {"effective_missing_rate": (base - remaining) / max(base, 1.0), "base_valid_bins": int(base), "remaining_valid_bins": int(remaining)}
    for mod in MODALITIES:
        before = float(original[mod].sum()); after = float(result[f"{mod}_mask"].sum())
        info[f"effective_missing_rate_{mod}"] = (before - after) / max(before, 1.0)
        info[f"remaining_ratio_{mod}"] = after / max(before, 1.0)
    return result, info


def apply_train_local_dropout(batch, probability, max_blocks, max_length):
    ids, mask, types, audio, audio_mask, vision, vision_mask, y, reg = batch
    ids, mask, types = ids.clone(), mask.clone(), types.clone()
    audio, audio_mask = audio.clone(), audio_mask.clone()
    vision, vision_mask = vision.clone(), vision_mask.clone()
    for b in range(ids.shape[0]):
        for m, msk, values in (("text", mask, ids), ("audio", audio_mask, audio), ("vision", vision_mask, vision)):
            if torch.rand(()) > probability: continue
            valid = torch.nonzero(msk[b], as_tuple=False).flatten()
            if valid.numel() <= 1: continue
            blocks = int(torch.randint(1, max_blocks + 1, ()).item())
            for _ in range(blocks):
                length = int(torch.randint(2, max_length + 1, ()).item())
                length = min(length, max(1, valid.numel() - 1))
                start = int(torch.randint(0, valid.numel() - length + 1, ()).item())
                positions = valid[start:start + length]
                msk[b, positions] = False
                if m == "text":
                    values[b, positions] = 0
                else:
                    values[b, positions] = 0.0
    return ids, mask, types, audio, audio_mask, vision, vision_mask, y, reg


def make_model(kind, text_model, device, dropout):
    cls = BertAVFusion if kind == "baseline" else BertAVRobustFusion
    return cls(text_model, device, dropout=dropout)


def evaluate(model, loader, device, loss_fn=None, train=False, local_dropout=0.0, max_blocks=2, max_length=10):
    model.train(train)
    all_prob, all_pred, all_reg, all_y, all_yr = [], [], [], [], []
    total = 0.0
    with torch.set_grad_enabled(train):
        for batch in loader:
            if train and local_dropout > 0: batch = apply_train_local_dropout(batch, local_dropout, max_blocks, max_length)
            ids, mask, types, audio, audio_mask, vision, vision_mask, y, reg = unpack(batch, device)
            logits, out_reg = model(ids, mask, types, audio, audio_mask, vision, vision_mask)
            loss = loss_fn(logits, y) if loss_fn is not None else torch.zeros((), device=device)
            if loss_fn is not None: loss = loss + 0.15 * nn.functional.smooth_l1_loss(out_reg, reg, beta=0.5)
            if train:
                model.optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); model.optimizer.step()
            total += float(loss.detach()) * len(y)
            all_prob.append(torch.softmax(logits, -1).detach().cpu().numpy()); all_pred.append(logits.argmax(-1).detach().cpu().numpy()); all_reg.append(out_reg.detach().cpu().numpy()); all_y.append(y.cpu().numpy()); all_yr.append(reg.cpu().numpy())
    prediction = {"probabilities": np.concatenate(all_prob), "classification": np.concatenate(all_pred), "regression": np.concatenate(all_reg), "true_classification": np.concatenate(all_y), "true_regression": np.concatenate(all_yr)}
    result = metrics_from_prediction(prediction)
    result["loss"] = total / len(loader.dataset)
    return result, prediction


def train_model(kind, seed, normalized, args, model_dir, device):
    seed_everything(seed)
    model_dir.mkdir(parents=True, exist_ok=True)
    model = make_model(kind, str(args.text_model), device, args.dropout)
    train_loader = DataLoader(BertAVDataset(normalized["train"]), batch_size=args.batch_size, shuffle=True, pin_memory=device.type == "cuda")
    valid_loader = DataLoader(BertAVDataset(normalized["valid"]), batch_size=args.batch_size * 2, shuffle=False, pin_memory=device.type == "cuda")
    counts = np.bincount(normalized["train"]["classification"], minlength=3).astype(np.float32)
    cls_weight = None
    if args.balanced: cls_weight = torch.tensor((1.0 / np.sqrt(np.maximum(counts, 1))) / np.mean(1.0 / np.sqrt(np.maximum(counts, 1))), device=device, dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=cls_weight)
    model.optimizer = torch.optim.AdamW([{"params": model.bert.parameters(), "lr": args.lr}, {"params": [p for n, p in model.named_parameters() if not n.startswith("bert.")], "lr": args.lr * 5}], weight_decay=args.weight_decay)
    best, best_epoch, stale, history = -float("inf"), 0, 0, []
    for epoch in range(1, args.epochs + 1):
        train_metrics, _ = evaluate(model, train_loader, device, loss_fn, train=True, local_dropout=args.train_dropout if kind == "robust" else 0.0, max_blocks=args.max_blocks, max_length=args.max_length)
        valid_metrics, _ = evaluate(model, valid_loader, device, loss_fn, train=False)
        history.append({"epoch": epoch, "train": train_metrics, "valid": valid_metrics, "lr": model.optimizer.param_groups[0]["lr"]})
        print(f"model={kind} seed={seed} epoch={epoch:02d} train_acc={train_metrics['accuracy']:.4f} valid_acc={valid_metrics['accuracy']:.4f} valid_f1={valid_metrics['macro_f1']:.4f}", flush=True)
        score = valid_metrics["macro_f1"]
        if score > best:
            best, best_epoch, stale = score, epoch, 0
            torch.save(model.state_dict(), model_dir / "checkpoint.pt")
        else:
            stale += 1
            if stale >= args.patience: break
    model.load_state_dict(torch.load(model_dir / "checkpoint.pt", map_location=device, weights_only=True))
    payload = {"model": kind, "seed": seed, "best_epoch": best_epoch, "history": history, "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    (model_dir / "training_history.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (model_dir / "training_history.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f); writer.writerow(["epoch", "lr", "train_loss", "valid_loss", "train_accuracy", "valid_accuracy", "train_macro_f1", "valid_macro_f1", "valid_mae"])
        for row in history: writer.writerow([row["epoch"], row["lr"], row["train"]["loss"], row["valid"]["loss"], row["train"]["accuracy"], row["valid"]["accuracy"], row["train"]["macro_f1"], row["valid"]["macro_f1"], row["valid"]["mae"]])
    return model, payload


def predict(model, split, device, batch_size):
    loader = DataLoader(BertAVDataset(split), batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda")
    return evaluate(model, loader, device, loss_fn=None, train=False)[1]


def write_csv(path, rows):
    if not rows: return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def plot_figures(out, rows, histories):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Times New Roman", "axes.unicode_minus": False, "font.size": 10})
    figure_dir = out / "figures"; figure_dir.mkdir(exist_ok=True)
    test_rows = [r for r in rows if r["split"] == "test"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for model, color in (("baseline", "#355C7D"), ("robust_noaug", "#D08C60"), ("robust", "#C06C84")):
        for mod in ("text", "audio", "vision"):
            rr = [r for r in test_rows if r["model"] == model and r["group"] == "rate" and r["missing_modalities"] == mod]
            if not rr: continue
            rr = sorted(rr, key=lambda x: float(x["requested_rate"]))
            axes[0].plot([float(x["requested_rate"]) for x in rr], [float(x["accuracy"]) for x in rr], marker="o", color=color, alpha=0.55 if mod != "text" else 1.0, label=f"{model} {mod}")
            axes[1].plot([float(x["requested_rate"]) for x in rr], [float(x["macro_f1"]) for x in rr], marker="o", color=color, alpha=0.55 if mod != "text" else 1.0, label=f"{model} {mod}")
    axes[0].set(xlabel="Requested missing rate", ylabel="Accuracy"); axes[1].set(xlabel="Requested missing rate", ylabel="Macro-F1")
    for ax in axes: ax.grid(alpha=.25); ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=7)
    fig.savefig(figure_dir / "missingness_curves.png", dpi=300); fig.savefig(figure_dir / "missingness_curves.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    complete = [r for r in test_rows if r["group"] == "baseline"]
    ab = [r for r in test_rows if r["group"] == "ablation"]
    grouped = {}
    for r in complete + ab: grouped.setdefault((r["model"], r["missing_modalities"] or "complete"), []).append(float(r["accuracy"]))
    labels = list(grouped); values = [np.mean(grouped[x]) for x in labels]
    colors = {"baseline": "#355C7D", "robust_noaug": "#D08C60", "robust": "#C06C84"}
    bars = ax.bar([f"{m}\n{mods.replace('+',' + ')}" for m, mods in labels], values, color=[colors.get(m, "#777777") for m, _ in labels]); ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=2, fontsize=8)
    ax.set_ylim(0, 1); ax.set_ylabel("Accuracy"); ax.grid(axis="y", alpha=.25)
    fig.savefig(figure_dir / "model_ablation.png", dpi=300); fig.savefig(figure_dir / "model_ablation.pdf"); plt.close(fig)
    for payload in histories:
        hist = payload["history"]; fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
        ep = [h["epoch"] for h in hist]
        for ax, metric, label in zip(axes, ("loss", "accuracy", "macro_f1"), ("Loss", "Accuracy", "Macro-F1")):
            ax.plot(ep, [h["train"][metric] for h in hist], label="Training", color="#355C7D")
            ax.plot(ep, [h["valid"][metric] for h in hist], label="Validation", color="#C06C84")
            ax.set(xlabel="Epoch", ylabel=label); ax.grid(alpha=.25); ax.legend(frameon=False)
        fig.savefig(figure_dir / f"training_curves_{payload['model']}_seed_{payload['seed']}.png", dpi=300); fig.savefig(figure_dir / f"training_curves_{payload['model']}_seed_{payload['seed']}.pdf"); plt.close(fig)


def export_attachment3(models, stats, args, device, out):
    rows, summary = [], {}
    for model_name, model in models.items():
        summary[model_name] = {}
        for variant in ("aligned", "unaligned"):
            records = load_missing_records(args.data_root / "attachment_3_missing_modality" / variant, args.text_model, stats)
            class_rows = []
            # Use the existing BERT-AV missing-record protocol, converted to a
            # lightweight dataset by filling unavailable labels with dummies.
            ids = np.stack([r["ids"] for r in records]); masks = np.stack([r["mask"] for r in records]); types = np.stack([r["types"] for r in records])
            audio = np.stack([r["audio"] for r in records]); vision = np.stack([r["vision"] for r in records]); audio_mask = np.stack([r["audio_mask"] for r in records]); vision_mask = np.stack([r["vision_mask"] for r in records])
            split = {"ids": ids, "text_mask": masks, "token_type_ids": types, "audio": audio, "vision": vision, "audio_mask": audio_mask, "vision_mask": vision_mask, "classification": np.zeros(len(records), dtype=np.int64), "regression": np.zeros(len(records), dtype=np.float32), "id": [r["id"] for r in records]}
            pred = predict(model, split, device, args.batch_size * 2)
            for i, rec in enumerate(records):
                row = {"model": model_name, "variant": variant, "id": rec["id"], "predicted_class": int(pred["classification"][i]), "predicted_label": CLASS_NAMES[int(pred["classification"][i])], "regression_prediction": float(pred["regression"][i]), "p_negative": float(pred["probabilities"][i, 0]), "p_neutral": float(pred["probabilities"][i, 1]), "p_positive": float(pred["probabilities"][i, 2]), "valid_ratio_text": float(masks[i].mean()), "valid_ratio_audio": float(audio_mask[i].mean()), "valid_ratio_vision": float(vision_mask[i].mean())}
                rows.append(row)
            summary[model_name][variant] = {"count": len(records), "class_counts": np.bincount(pred["classification"], minlength=3).tolist(), "mean_valid_ratio": {"text": float(masks.mean()), "audio": float(audio_mask.mean()), "vision": float(vision_mask.mean())}}
    write_csv(out / "attachment3_predictions.csv", rows)
    (out / "attachment3_summary.json").write_text(json.dumps({"warning": "Attachment 3 has no labels; predictions are not accuracy.", "models": summary}, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description="BERT-AV controlled missingness experiment")
    p.add_argument("--data-root", type=Path, default=Path("D:/E_math/DATA")); p.add_argument("--text-model", type=Path, default=Path("D:/E_math/models/bert-base-uncased")); p.add_argument("--output-root", type=Path, default=Path("D:/E_math/problem2_experiment2"))
    p.add_argument("--models", default="baseline,robust_noaug,robust", help="baseline, robust_noaug, and/or robust")
    p.add_argument("--seeds", type=parse_seeds, default=[42, 43, 44]); p.add_argument("--epochs", type=int, default=8); p.add_argument("--batch-size", type=int, default=20); p.add_argument("--lr", type=float, default=1.5e-5); p.add_argument("--dropout", type=float, default=.25); p.add_argument("--train-dropout", type=float, default=.35); p.add_argument("--max-blocks", type=int, default=2); p.add_argument("--max-length", type=int, default=10); p.add_argument("--patience", type=int, default=4); p.add_argument("--weight-decay", type=float, default=.01); p.add_argument("--balanced", action="store_true"); p.add_argument("--skip-attachment3", action="store_true"); p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    models_requested = [x.strip() for x in args.models.split(",") if x.strip()]
    if any(x not in ("baseline", "robust_noaug", "robust") for x in models_requested): p.error("--models must contain baseline, robust_noaug, and/or robust")
    out = args.output_root.resolve()
    if out.exists() and any(out.iterdir()) and not args.overwrite: raise SystemExit(f"Output directory is non-empty: {out}; use a new directory or --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw, view = load_aligned(args.data_root / "attachment_2_standard_features" / "aligned_50.pkl")
    # BERT supplies the text representation directly; only audio/vision need
    # train-only feature normalization in this protocol.
    stats = {}
    for modality in ("audio", "vision"):
        values = view["train"][modality][view["train"][f"{modality}_mask"]]
        mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = values.std(axis=0, dtype=np.float64).astype(np.float32)
        std[~np.isfinite(std) | (std < 1e-5)] = 1.0
        mean[~np.isfinite(mean)] = 0.0
        stats[modality] = {"mean": mean, "std": std}
    save_stats(out / "normalization.json", stats)
    normalized = normalize_view(view, stats)
    conditions = build_conditions()
    (out / "experiment_config.json").write_text(json.dumps({"protocol": "aligned_50 only; train-only normalization; attachment-3 labels unused", "device": str(device), "models": models_requested, "seeds": args.seeds, "conditions": [asdict(c) for c in conditions], "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}, ensure_ascii=False, indent=2), encoding="utf-8")
    rows, model_summary, histories, models_for_export = [], [], [], {}
    for model_name in models_requested:
        for seed in args.seeds:
            model, history = train_model(model_name, seed, normalized, args, out / model_name / f"seed_{seed}", device)
            histories.append(history)
            models_for_export.setdefault(model_name, model)
            base_test = metrics_from_prediction(predict(model, normalized["test"], device, args.batch_size * 2))
            model_summary.append({"model": model_name, "seed": seed, "best_epoch": history["best_epoch"], "test_accuracy": base_test["accuracy"], "test_macro_f1": base_test["macro_f1"], "test_mae": base_test["mae"], "test_pearson": base_test["pearson"]})
            for index, condition in enumerate(conditions):
                for split_name in ("valid", "test"):
                    masked, info = apply_condition(normalized[split_name], condition, seed * 10000 + index)
                    result = metrics_from_prediction(predict(model, masked, device, args.batch_size * 2))
                    rows.append({"model": model_name, "seed": seed, "split": split_name, "condition_id": condition.condition_id, "group": condition.group, "missing_modalities": "+".join(condition.missing_modalities), "requested_rate": condition.missing_rate, "position": condition.position, "duration": condition.duration, **info, "accuracy": result["accuracy"], "macro_f1": result["macro_f1"], "mae": result["mae"], "rmse": result["rmse"], "pearson": result["pearson"], "f1_negative": result["f1_by_class"]["Negative"], "f1_neutral": result["f1_by_class"]["Neutral"], "f1_positive": result["f1_by_class"]["Positive"], "valid_sample_count": len(masked["classification"])})
    write_csv(out / "condition_metrics.csv", rows); write_csv(out / "model_summary.csv", model_summary)
    summary_rows = []
    for model_name in models_requested:
        for split_name in ("valid", "test"):
            for condition_id in sorted({r["condition_id"] for r in rows if r["model"] == model_name and r["split"] == split_name}):
                rr = [r for r in rows if r["model"] == model_name and r["split"] == split_name and r["condition_id"] == condition_id]; first = rr[0]
                row = {k: first[k] for k in ("model", "split", "condition_id", "group", "missing_modalities", "requested_rate", "position", "duration")}
                for metric in ("accuracy", "macro_f1", "mae", "rmse", "pearson", "effective_missing_rate"): row[f"{metric}_mean"] = float(np.mean([x[metric] for x in rr])); row[f"{metric}_std"] = float(np.std([x[metric] for x in rr]))
                summary_rows.append(row)
    write_csv(out / "ablation_summary.csv", summary_rows); plot_figures(out, rows, histories)
    if not args.skip_attachment3: export_attachment3(models_for_export, stats, args, device, out)
    (out / "experiment_report.md").write_text("# Problem 2 BERT-AV controlled missingness experiment\n\nThe aligned attachment-2 file is the sole training/validation/test feature version. `baseline` is the original BERT-AV fusion; `robust_noaug` is the same availability-aware architecture without local missingness augmentation; `robust` adds the augmentation. This paired design separates architectural improvement from augmentation. Attachment 3 predictions are unlabeled diagnostics and are not accuracy.\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "device": str(device), "models": models_requested, "seeds": args.seeds, "conditions": len(conditions), "model_summary": model_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__": raise SystemExit(main())
