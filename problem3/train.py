"""Train and export the Problem 3 explainable multimodal model."""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from problem2.data import MODALITIES, pool_to_50
from problem3.model import ExplainableFusionNet


CLASS_NAMES = ("Negative", "Neutral", "Positive")


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def metrics(y_true, y_pred, r_true, r_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = np.zeros((3, 3), dtype=np.int64)
    for y, p in zip(y_true, y_pred): cm[int(y), int(p)] += 1
    f1 = []
    for k in range(3):
        tp = cm[k, k]; precision = tp / max(cm[:, k].sum(), 1); recall = tp / max(cm[k].sum(), 1)
        f1.append(2 * precision * recall / max(precision + recall, 1e-12))
    r_true, r_pred = np.asarray(r_true, dtype=np.float64), np.asarray(r_pred, dtype=np.float64)
    pearson = float(np.corrcoef(r_true, r_pred)[0, 1]) if np.std(r_true) > 1e-12 and np.std(r_pred) > 1e-12 else 0.0
    return {"accuracy": float(np.mean(y_true == y_pred)), "macro_f1": float(np.mean(f1)),
            "f1_by_class": {CLASS_NAMES[i]: float(f1[i]) for i in range(3)}, "confusion_matrix": cm.tolist(),
            "mae": float(np.mean(np.abs(r_true - r_pred))), "rmse": float(np.sqrt(np.mean((r_true - r_pred) ** 2))), "pearson": pearson}


def load_aligned(path):
    with Path(path).open("rb") as handle: raw = pickle.load(handle)
    data = {}
    for name in ("train", "valid", "test"):
        s = raw[name]
        x = {m: np.asarray(s[m], dtype=np.float32) for m in MODALITIES}
        text_bert = np.asarray(s.get("text_bert"))
        text_mask = text_bert[:, 1].astype(bool) if text_bert.ndim == 3 else ~np.all(np.isclose(x["text"], 0.0), axis=-1)
        masks = {"text": text_mask, "audio": ~np.all(np.isclose(x["audio"], 0.0), axis=-1), "vision": ~np.all(np.isclose(x["vision"], 0.0), axis=-1)}
        data[name] = {"x": x, "mask": masks, "classification": np.asarray(s["classification_labels"], dtype=np.int64), "regression": np.asarray(s["regression_labels"], dtype=np.float32), "id": [str(v) for v in s["id"]], "raw_text": [str(v) for v in s.get("raw_text", [""] * len(s["id"]))], "text_bert": text_bert}
    return data


def fit_stats(split):
    stats = {}
    for m in MODALITIES:
        values = split["x"][m][split["mask"][m]]
        mean = values.mean(axis=0, dtype=np.float64).astype(np.float32); std = values.std(axis=0, dtype=np.float64).astype(np.float32)
        mean[~np.isfinite(mean)] = 0.0; std[~np.isfinite(std) | (std < 1e-5)] = 1.0
        stats[m] = {"mean": mean, "std": std}
    return stats


def normalize_split(split, stats):
    out = dict(split); out["x"] = {}
    for m in MODALITIES:
        x = ((split["x"][m] - stats[m]["mean"]) / stats[m]["std"]).astype(np.float32); x[~split["mask"][m]] = 0.0; out["x"][m] = x
    return out


def save_stats(path, stats):
    Path(path).write_text(json.dumps({m: {k: v.tolist() for k, v in d.items()} for m, d in stats.items()}, ensure_ascii=False, indent=2), encoding="utf-8")


class FeatureDataset(Dataset):
    def __init__(self, split): self.split = split
    def __len__(self): return len(self.split["classification"])
    def __getitem__(self, i):
        return {"x": {m: torch.from_numpy(self.split["x"][m][i]) for m in MODALITIES}, "mask": {m: torch.from_numpy(self.split["mask"][m][i].astype(np.bool_)) for m in MODALITIES}, "classification": torch.tensor(self.split["classification"][i], dtype=torch.long), "regression": torch.tensor(self.split["regression"][i], dtype=torch.float32)}


def move_batch(batch, device):
    return {"x": {m: batch["x"][m].to(device) for m in MODALITIES}, "mask": {m: batch["mask"][m].to(device) for m in MODALITIES}, "classification": batch["classification"].to(device), "regression": batch["regression"].to(device)}


def run_epoch(model, loader, device, optimizer=None, cls_loss=None, reg_weight=0.15):
    training = optimizer is not None; model.train(training); ys, ps, rs, prs = [], [], [], []; total = 0.0
    for batch in loader:
        batch = move_batch(batch, device); out = model(batch["x"], batch["mask"])
        loss = cls_loss(out["logits"], batch["classification"]) + reg_weight * nn.functional.smooth_l1_loss(out["regression"], batch["regression"], beta=0.5)
        if training:
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        total += float(loss.detach()) * len(batch["classification"]); ys.append(batch["classification"].detach().cpu().numpy()); ps.append(out["logits"].argmax(-1).detach().cpu().numpy()); rs.append(batch["regression"].detach().cpu().numpy()); prs.append(out["regression"].detach().cpu().numpy())
    result = metrics(np.concatenate(ys), np.concatenate(ps), np.concatenate(rs), np.concatenate(prs)); result["loss"] = total / len(loader.dataset); return result


@torch.inference_mode()
def predict_split(model, split, device, batch_size):
    loader = DataLoader(FeatureDataset(split), batch_size=batch_size, shuffle=False)
    model.eval(); probs, preds, regs, ys, rs, modality, temporal = [], [], [], [], [], [], []
    for batch in loader:
        moved = move_batch(batch, device); out = model(moved["x"], moved["mask"])
        probs.append(torch.softmax(out["logits"], -1).cpu().numpy()); preds.append(out["logits"].argmax(-1).cpu().numpy()); regs.append(out["regression"].cpu().numpy()); ys.append(moved["classification"].cpu().numpy()); rs.append(moved["regression"].cpu().numpy()); modality.append(out["modality_weights"].cpu().numpy()); temporal.append(out["modality_time_weights"].cpu().numpy())
    return {"probabilities": np.concatenate(probs), "classification": np.concatenate(preds), "regression": np.concatenate(regs), "true_classification": np.concatenate(ys), "true_regression": np.concatenate(rs), "modality_weights": np.concatenate(modality), "modality_time_weights": np.concatenate(temporal)}


def video_duration(path):
    try:
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        value = float(probe.stdout.decode().strip())
        if value > 0: return value
    except Exception: pass
    return 1.0


def load_attachment4(root, stats, variant):
    root = Path(root); feature_dir = root / variant / "features"; video_dir = root / variant / "videos"; records = []
    for path in sorted(feature_dir.glob("*.pkl")):
        with path.open("rb") as handle: item = pickle.load(handle)
        text = np.asarray(item["text"], dtype=np.float32); text_bert = np.asarray(item.get("text_bert"))
        text_mask = text_bert[1].astype(bool) if text_bert.ndim == 2 else ~np.all(np.isclose(text, 0.0), axis=-1)
        audio, audio_mask = pool_to_50(item["audio"]); vision, vision_mask = pool_to_50(item["vision"])
        raw = {"text": text, "audio": audio, "vision": vision}; masks = {"text": text_mask, "audio": audio_mask, "vision": vision_mask}; x = {}
        for m in MODALITIES:
            value = ((np.asarray(raw[m], dtype=np.float32) - stats[m]["mean"]) / stats[m]["std"]).astype(np.float32); value[~masks[m]] = 0.0; x[m] = value
        stem = path.stem; video = video_dir / f"{stem}.mp4"
        records.append({"id": str(item.get("id", stem)), "x": x, "mask": masks, "raw_text": str(item.get("raw_text", "")), "text_bert": text_bert, "duration": video_duration(video), "video_path": str(video), "variant": variant})
    return records


def explain_sample(model, record, device, top_k, span, tokenizer=None):
    x = {m: torch.from_numpy(record["x"][m]).unsqueeze(0).to(device) for m in MODALITIES}; masks = {m: torch.from_numpy(record["mask"][m]).unsqueeze(0).to(device) for m in MODALITIES}
    with torch.inference_mode():
        base = model(x, masks); probs = torch.softmax(base["logits"], -1)[0]; predicted = int(probs.argmax()); base_score = float(probs[predicted]); mod_attention = base["modality_weights"][0].cpu().numpy(); mt_attention = base["modality_time_weights"][0].cpu().numpy()
    candidates = []
    for mi, m in enumerate(MODALITIES):
        positions = np.argsort(mt_attention[mi])[::-1][:max(top_k, 1)]
        for pos in positions: candidates.append((mi, m, int(pos)))
    variants = []; meta = []
    for mi, m, pos in candidates:
        xx = {k: v.clone() for k, v in x.items()}; mm = {k: v.clone() for k, v in masks.items()}
        lo, hi = max(0, pos - span // 2), min(xx[m].shape[1], pos - span // 2 + span); xx[m][:, lo:hi] = 0.0; mm[m][:, lo:hi] = False; variants.append((xx, mm)); meta.append((mi, m, pos, lo, hi))
    drops = []
    if variants:
        batch_x = {m: torch.cat([v[0][m] for v in variants], 0) for m in MODALITIES}; batch_m = {m: torch.cat([v[1][m] for v in variants], 0) for m in MODALITIES}
        with torch.inference_mode(): changed = model(batch_x, batch_m); changed_prob = torch.softmax(changed["logits"], -1)[:, predicted].cpu().numpy()
        drops = [base_score - float(v) for v in changed_prob]
    contribution = np.zeros(3, dtype=np.float64)
    evidence = {m: [] for m in MODALITIES}
    for (mi, m, pos, lo, hi), drop in zip(meta, drops):
        contribution[mi] += max(0.0, float(drop)); evidence[m].append({"bin": pos, "start_bin": lo, "end_bin": hi, "occlusion_drop": float(drop), "attention": float(mt_attention[mi, pos])})
    if contribution.sum() <= 1e-10: contribution = mod_attention.astype(np.float64)
    contribution /= max(contribution.sum(), 1e-12)
    for m in MODALITIES: evidence[m] = sorted(evidence[m], key=lambda z: (z["occlusion_drop"], z["attention"]), reverse=True)[:top_k]
    duration = float(record["duration"])
    for e in evidence["audio"] + evidence["vision"]:
        e["start_seconds"] = e["start_bin"] * duration / 50.0; e["end_seconds"] = e["end_bin"] * duration / 50.0
        e["frame_start"] = int(round(e["start_seconds"] * 5)); e["frame_end"] = int(round(e["end_seconds"] * 5))
    tokens = []
    if tokenizer is not None and isinstance(record.get("text_bert"), np.ndarray) and record["text_bert"].ndim == 2:
        ids, valid = record["text_bert"][0], record["text_bert"][1].astype(bool); token_list = tokenizer.convert_ids_to_tokens(ids[valid].astype(int).tolist())
        for e in evidence["text"]:
            ti = min(max(int(round(e["bin"] / 49 * max(len(token_list) - 1, 0))), 0), max(len(token_list) - 1, 0)); e["token_position"] = ti; e["token"] = token_list[ti] if token_list else ""
            tokens.append(e["token"])
    main_index = int(np.argmax(contribution)); main = MODALITIES[main_index]
    return {"id": record["id"], "variant": record["variant"], "video_path": record["video_path"], "raw_text": record["raw_text"], "predicted_class": predicted, "predicted_label": CLASS_NAMES[predicted], "confidence": float(probs[predicted]), "probabilities": probs.cpu().numpy().tolist(), "regression_prediction": float(base["regression"][0]), "modality_attention": {m: float(mod_attention[i]) for i, m in enumerate(MODALITIES)}, "modality_contribution": {m: float(contribution[i]) for i, m in enumerate(MODALITIES)}, "main_modality": main, "evidence": evidence, "text_tokens": tokens, "duration_seconds": duration, "evidence_definition": "Positive occlusion_drop means the predicted-class probability decreased when the local span was masked."}


def write_rows(path, rows):
    if not rows: return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def write_validation_error_analysis(out, split, prediction):
    """Export sample-level validation errors and grouped error diagnostics."""
    y_true = np.asarray(prediction["true_classification"], dtype=np.int64)
    y_pred = np.asarray(prediction["classification"], dtype=np.int64)
    r_true = np.asarray(prediction["true_regression"], dtype=np.float64)
    r_pred = np.asarray(prediction["regression"], dtype=np.float64)
    if "probabilities" in prediction:
        probabilities = np.asarray(prediction["probabilities"], dtype=np.float64)
        confidence = probabilities.max(axis=1)
        order = np.argsort(-probabilities, axis=1)
        margin = probabilities[np.arange(len(probabilities)), order[:, 0]] - probabilities[np.arange(len(probabilities)), order[:, 1]]
    else:
        confidence = np.asarray(prediction.get("confidence", np.zeros(len(y_true))), dtype=np.float64)
        margin = np.full(len(y_true), np.nan, dtype=np.float64)
    abs_error = np.abs(r_true - r_pred)
    correct = y_true == y_pred
    rows = []
    for i, sample_id in enumerate(split["id"]):
        rows.append({
            "id": sample_id,
            "true_label": CLASS_NAMES[int(y_true[i])],
            "predicted_label": CLASS_NAMES[int(y_pred[i])],
            "correct": int(correct[i]),
            "confidence": float(confidence[i]),
            "confidence_margin": None if not np.isfinite(margin[i]) else float(margin[i]),
            "true_intensity": float(r_true[i]),
            "predicted_intensity": float(r_pred[i]),
            "absolute_intensity_error": float(abs_error[i]),
        })
    write_rows(Path(out) / "validation_error_analysis.csv", rows)

    confusion_pairs = {}
    for true_idx in range(3):
        for pred_idx in range(3):
            if true_idx != pred_idx:
                count = int(np.sum((y_true == true_idx) & (y_pred == pred_idx)))
                if count:
                    confusion_pairs[f"{CLASS_NAMES[true_idx]} -> {CLASS_NAMES[pred_idx]}"] = count
    by_class = {}
    for class_idx, label in enumerate(CLASS_NAMES):
        mask = y_true == class_idx
        error_mask = mask & ~correct
        by_class[label] = {
            "support": int(mask.sum()),
            "errors": int(error_mask.sum()),
            "error_rate": float(error_mask.sum() / max(mask.sum(), 1)),
            "mean_confidence": float(confidence[mask].mean()) if mask.any() else 0.0,
            "mean_confidence_on_errors": float(confidence[error_mask].mean()) if error_mask.any() else 0.0,
            "mean_absolute_intensity_error": float(abs_error[mask].mean()) if mask.any() else 0.0,
        }
    high_confidence_error = (~correct) & (confidence >= 0.80)
    summary = {
        "sample_count": int(len(y_true)),
        "correct_count": int(correct.sum()),
        "error_count": int((~correct).sum()),
        "accuracy": float(correct.mean()),
        "error_rate": float((~correct).mean()),
        "high_confidence_error_threshold": 0.80,
        "high_confidence_error_count": int(high_confidence_error.sum()),
        "high_confidence_error_rate_among_errors": float(high_confidence_error.sum() / max((~correct).sum(), 1)),
        "mean_absolute_intensity_error": float(abs_error.mean()),
        "median_absolute_intensity_error": float(np.median(abs_error)),
        "confusion_pairs": confusion_pairs,
        "by_true_class": by_class,
        "top_confident_errors": [rows[i] for i in np.argsort(-confidence * (~correct))[:10] if not correct[i]][:10],
        "top_intensity_errors": [rows[i] for i in np.argsort(-abs_error)[:10]],
        "interpretation": [
            "Error rates are grouped by the true class; a large Neutral error rate indicates ambiguity around the zero-intensity boundary.",
            "High-confidence errors are samples whose predicted probability is at least 0.80 despite an incorrect class.",
            "Absolute intensity error measures the distance between predicted and annotated sentiment strength.",
        ],
    }
    Path(out, "validation_error_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


@torch.inference_mode()
def full_local_occlusion(model, record, device, span=1, batch_size=50):
    """Measure the predicted-class probability drop at every time bin."""
    x = {m: torch.from_numpy(record["x"][m]).unsqueeze(0).to(device) for m in MODALITIES}
    masks = {m: torch.from_numpy(record["mask"][m]).unsqueeze(0).to(device) for m in MODALITIES}
    model.eval()
    base = model(x, masks)
    probabilities = torch.softmax(base["logits"], -1)
    predicted = int(probabilities[0].argmax())
    base_score = float(probabilities[0, predicted])
    drops = np.zeros((len(MODALITIES), 50), dtype=np.float64)
    for mi, modality in enumerate(MODALITIES):
        for start in range(0, 50, batch_size):
            positions = list(range(start, min(start + batch_size, 50)))
            batch_x = {m: x[m].repeat(len(positions), 1, 1) for m in MODALITIES}
            batch_m = {m: masks[m].repeat(len(positions), 1) for m in MODALITIES}
            for row, pos in enumerate(positions):
                lo = max(0, pos - span // 2); hi = min(50, lo + span)
                batch_x[modality][row, lo:hi] = 0.0
                batch_m[modality][row, lo:hi] = False
            changed = model(batch_x, batch_m)
            changed_prob = torch.softmax(changed["logits"], -1)[:, predicted].detach().cpu().numpy()
            drops[mi, positions] = base_score - changed_prob
    return {"predicted_class": predicted, "base_probability": base_score, "drops": drops}


def plot_outputs(out, history, valid_prediction, explanation_cards, error_summary=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Times New Roman", "axes.unicode_minus": False, "font.size": 10})
    fig_dir = Path(out) / "figures"; fig_dir.mkdir(exist_ok=True)
    ep = [h["epoch"] for h in history]; fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
    for ax, key, label in zip(axes, ("loss", "accuracy", "macro_f1"), ("Loss", "Accuracy", "Macro-F1")):
        ax.plot(ep, [h["train"][key] for h in history], label="Training", color="#355C7D"); ax.plot(ep, [h["valid"][key] for h in history], label="Validation", color="#C06C84"); ax.set(xlabel="Epoch", ylabel=label); ax.grid(alpha=.25); ax.legend(frameon=False)
    fig.savefig(fig_dir / "training_curves.png", dpi=300); fig.savefig(fig_dir / "training_curves.pdf"); plt.close(fig)
    cm = np.asarray(metrics(valid_prediction["true_classification"], valid_prediction["classification"], valid_prediction["true_regression"], valid_prediction["regression"])["confusion_matrix"]); fig, ax = plt.subplots(figsize=(4.5, 4), constrained_layout=True); im = ax.imshow(cm, cmap="Blues"); ax.set(xticks=range(3), xticklabels=CLASS_NAMES, yticks=range(3), yticklabels=CLASS_NAMES, xlabel="Predicted", ylabel="True");
    for i in range(3):
        for j in range(3): ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax); fig.savefig(fig_dir / "validation_confusion.png", dpi=300); fig.savefig(fig_dir / "validation_confusion.pdf"); plt.close(fig)
    if explanation_cards:
        mod = np.asarray([[c["modality_contribution"][m] for m in MODALITIES] for c in explanation_cards]); fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True); ax.bar(MODALITIES, mod.mean(0), color=["#355C7D", "#C06C84", "#6C9A8B"]); ax.set_ylabel("Mean contribution"); ax.set_ylim(0, 1); ax.grid(axis="y", alpha=.25); fig.savefig(fig_dir / "modality_contributions.png", dpi=300); fig.savefig(fig_dir / "modality_contributions.pdf"); plt.close(fig)
        heat = np.zeros((3, 50), dtype=np.float64); count = 0
        for c in explanation_cards:
            for mi, m in enumerate(MODALITIES):
                for e in c["evidence"][m]: heat[mi, e["bin"]] += max(0.0, e["attention"])
            count += 1
        if count: heat /= count
        fig, ax = plt.subplots(figsize=(10, 3), constrained_layout=True); im = ax.imshow(heat, aspect="auto", cmap="magma"); ax.set(yticks=range(3), yticklabels=[m.capitalize() for m in MODALITIES], xlabel="Aligned time bin", title="Top-evidence attention distribution"); fig.colorbar(im, ax=ax, label="Attention"); fig.savefig(fig_dir / "temporal_importance.png", dpi=300); fig.savefig(fig_dir / "temporal_importance.pdf"); plt.close(fig)
        primary_heat = np.zeros((3, 50), dtype=np.float64); primary_counts = np.zeros(3, dtype=np.int64)
        for card in explanation_cards:
            primary_idx = MODALITIES.index(card["main_modality"])
            evidence = card["evidence"][card["main_modality"]]
            weights = np.asarray([max(float(e.get("occlusion_drop", 0.0)), 0.0) for e in evidence], dtype=np.float64)
            total = float(weights.sum())
            if total <= 1e-12: continue
            primary_counts[primary_idx] += 1
            for e, weight in zip(evidence, weights): primary_heat[primary_idx, int(e["bin"])] += weight / total
        for mi in range(3):
            if primary_counts[mi] > 0: primary_heat[mi] /= primary_counts[mi]
        write_rows(Path(out) / "primary_modality_local_importance.csv", [{"modality": m, "aligned_bin": b, "mean_normalized_occlusion": float(primary_heat[mi, b]), "sample_count": int(primary_counts[mi])} for mi, m in enumerate(MODALITIES) for b in range(50)])
        fig, axes = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True, constrained_layout=True)
        colors = ["#355C7D", "#C06C84", "#6C9A8B"]
        bins = np.arange(50)
        for mi, (ax, modality) in enumerate(zip(axes, MODALITIES)):
            ax.plot(bins, primary_heat[mi], color=colors[mi], linewidth=2.0)
            ax.fill_between(bins, primary_heat[mi], color=colors[mi], alpha=.18)
            ax.set_ylabel(f"{modality.capitalize()}\n(n={int(primary_counts[mi])})")
            ax.grid(alpha=.22)
        axes[-1].set_xlabel("Aligned time bin")
        fig.savefig(fig_dir / "primary_modality_local_importance.png", dpi=300); fig.savefig(fig_dir / "primary_modality_local_importance.pdf"); plt.close(fig)
    else:
        write_rows(Path(out) / "primary_modality_local_importance.csv", [])

    if error_summary is not None:
        class_labels = list(CLASS_NAMES)
        class_error_rates = [error_summary["by_true_class"][label]["error_rate"] for label in class_labels]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
        axes[0].bar(class_labels, class_error_rates, color=["#355C7D", "#C06C84", "#6C9A8B"])
        axes[0].set_ylabel("Error rate"); axes[0].set_ylim(0, 1); axes[0].grid(axis="y", alpha=.25)
        cm = np.asarray(metrics(valid_prediction["true_classification"], valid_prediction["classification"], valid_prediction["true_regression"], valid_prediction["regression"])["confusion_matrix"])
        offdiag = cm.copy(); np.fill_diagonal(offdiag, 0); im = axes[1].imshow(offdiag, cmap="Oranges")
        axes[1].set(xticks=range(3), xticklabels=CLASS_NAMES, yticks=range(3), yticklabels=CLASS_NAMES, xlabel="Predicted", ylabel="True")
        for i in range(3):
            for j in range(3): axes[1].text(j, i, int(offdiag[i, j]), ha="center", va="center")
        fig.colorbar(im, ax=axes[1], fraction=.046, pad=.04, label="Misclassified samples")
        valid_correct = np.asarray(valid_prediction["true_classification"]) == np.asarray(valid_prediction["classification"])
        abs_regression = np.abs(np.asarray(valid_prediction["true_regression"]) - np.asarray(valid_prediction["regression"]))
        axes[2].boxplot([abs_regression[valid_correct], abs_regression[~valid_correct]], showfliers=False)
        axes[2].set_xticks([1, 2], ["Correct class", "Wrong class"])
        axes[2].set_ylabel("Absolute intensity error"); axes[2].grid(axis="y", alpha=.25)
        fig.savefig(fig_dir / "validation_error_diagnostics.png", dpi=300); fig.savefig(fig_dir / "validation_error_diagnostics.pdf"); plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description="Train the Problem 3 explainable multimodal emotion model")
    p.add_argument("--data-root", type=Path, default=Path("D:/E_math/DATA")); p.add_argument("--text-model", type=Path, default=Path("D:/E_math/models/bert-base-uncased")); p.add_argument("--output-root", type=Path, default=Path("D:/E_math/problem3_outputs")); p.add_argument("--epochs", type=int, default=20); p.add_argument("--batch-size", type=int, default=64); p.add_argument("--hidden", type=int, default=160); p.add_argument("--heads", type=int, default=8); p.add_argument("--layers", type=int, default=2); p.add_argument("--dropout", type=float, default=.15); p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--weight-decay", type=float, default=5e-4); p.add_argument("--reg-weight", type=float, default=.15); p.add_argument("--patience", type=int, default=6); p.add_argument("--seed", type=int, default=42); p.add_argument("--top-k", type=int, default=5); p.add_argument("--occlusion-span", type=int, default=1); p.add_argument("--skip-figures", action="store_true"); p.add_argument("--skip-attachment4", action="store_true"); p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv); seed_everything(args.seed); device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); out = args.output_root.resolve()
    if out.exists() and any(out.iterdir()) and not args.overwrite: raise SystemExit(f"Output directory is non-empty: {out}; use a new directory or --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    raw = load_aligned(args.data_root / "attachment_2_standard_features" / "aligned_50.pkl"); stats = fit_stats(raw["train"]); save_stats(out / "normalization.json", stats); normalized = {k: normalize_split(v, stats) for k, v in raw.items()}
    model = ExplainableFusionNet(hidden=args.hidden, heads=args.heads, layers=args.layers, dropout=args.dropout).to(device); train_loader = DataLoader(FeatureDataset(normalized["train"]), batch_size=args.batch_size, shuffle=True); valid_loader = DataLoader(FeatureDataset(normalized["valid"]), batch_size=args.batch_size * 2, shuffle=False)
    cls_loss = nn.CrossEntropyLoss(); optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * .01); history = []; best_score = -float("inf"); stale = 0; best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        train_result = run_epoch(model, train_loader, device, optimizer, cls_loss, args.reg_weight); valid_result = run_epoch(model, valid_loader, device, None, cls_loss, args.reg_weight); scheduler.step(); history.append({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train": train_result, "valid": valid_result}); print(f"epoch {epoch:03d} train_acc={train_result['accuracy']:.4f} valid_acc={valid_result['accuracy']:.4f} valid_f1={valid_result['macro_f1']:.4f} valid_mae={valid_result['mae']:.4f}", flush=True)
        score = valid_result["macro_f1"] - .05 * valid_result["mae"]
        if score > best_score: best_score, best_epoch, stale = score, epoch, 0; torch.save({"model": model.state_dict(), "config": vars(args), "best_epoch": epoch}, out / "problem3_checkpoint.pt")
        else: stale += 1
        if stale >= args.patience: break
    checkpoint = torch.load(out / "problem3_checkpoint.pt", map_location=device, weights_only=False); model.load_state_dict(checkpoint["model"]); valid_prediction = predict_split(model, normalized["valid"], device, args.batch_size * 2); test_prediction = predict_split(model, normalized["test"], device, args.batch_size * 2); valid_metrics = metrics(valid_prediction["true_classification"], valid_prediction["classification"], valid_prediction["true_regression"], valid_prediction["regression"]); test_metrics = metrics(test_prediction["true_classification"], test_prediction["classification"], test_prediction["true_regression"], test_prediction["regression"])
    (out / "training_history.json").write_text(json.dumps({"best_epoch": best_epoch, "history": history}, ensure_ascii=False, indent=2), encoding="utf-8"); write_rows(out / "training_history.csv", [{"epoch": h["epoch"], "lr": h["lr"], "train_loss": h["train"]["loss"], "valid_loss": h["valid"]["loss"], "train_accuracy": h["train"]["accuracy"], "valid_accuracy": h["valid"]["accuracy"], "train_macro_f1": h["train"]["macro_f1"], "valid_macro_f1": h["valid"]["macro_f1"], "valid_mae": h["valid"]["mae"]} for h in history])
    write_rows(out / "problem3_valid_predictions.csv", [{"id": normalized["valid"]["id"][i], "predicted_class": int(valid_prediction["classification"][i]), "predicted_label": CLASS_NAMES[int(valid_prediction["classification"][i])], "true_class": int(valid_prediction["true_classification"][i]), "true_label": CLASS_NAMES[int(valid_prediction["true_classification"][i])], "regression_prediction": float(valid_prediction["regression"][i]), "true_regression": float(valid_prediction["true_regression"][i]), "confidence": float(valid_prediction["probabilities"][i].max())} for i in range(len(normalized["valid"]["id"]))]); write_rows(out / "problem3_test_predictions.csv", [{"id": normalized["test"]["id"][i], "predicted_class": int(test_prediction["classification"][i]), "predicted_label": CLASS_NAMES[int(test_prediction["classification"][i])], "true_class": int(test_prediction["true_classification"][i]), "true_label": CLASS_NAMES[int(test_prediction["true_classification"][i])], "regression_prediction": float(test_prediction["regression"][i]), "true_regression": float(test_prediction["true_regression"][i]), "confidence": float(test_prediction["probabilities"][i].max())} for i in range(len(normalized["test"]["id"]))])
    cards = []; tokenizer = None
    if not args.skip_attachment4:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.text_model, local_files_only=True)
        except Exception: tokenizer = None
        for variant in ("aligned", "unaligned"):
            records = load_attachment4(args.data_root / "attachment_4_explainability", stats, variant)
            for record in records: cards.append(explain_sample(model, record, device, args.top_k, args.occlusion_span, tokenizer))
        evidence_dir = out / "evidence_cards"; evidence_dir.mkdir(exist_ok=True)
        for card in cards: (evidence_dir / f"{card['variant']}_{card['id']}.json").write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
        rows = []
        for c in cards:
            rows.append({"id": c["id"], "variant": c["variant"], "predicted_class": c["predicted_class"], "predicted_label": c["predicted_label"], "confidence": c["confidence"], "regression_prediction": c["regression_prediction"], "main_modality": c["main_modality"], "text_contribution": c["modality_contribution"]["text"], "audio_contribution": c["modality_contribution"]["audio"], "vision_contribution": c["modality_contribution"]["vision"], "text_evidence": " | ".join(str(e.get("token", e["bin"])) for e in c["evidence"]["text"]), "audio_evidence": " | ".join(f"{e['start_seconds']:.2f}-{e['end_seconds']:.2f}s" for e in c["evidence"]["audio"]), "vision_evidence": " | ".join(f"{e['frame_start']}-{e['frame_end']}" for e in c["evidence"]["vision"])})
        write_rows(out / "attachment4_explanations.csv", rows); (out / "attachment4_explanations.json").write_text(json.dumps(cards, ensure_ascii=False, indent=2), encoding="utf-8")
    error_summary = write_validation_error_analysis(out, normalized["valid"], valid_prediction)
    if not args.skip_figures: plot_outputs(out, history, valid_prediction, cards, error_summary)
    metadata = {"device": str(device), "best_epoch": best_epoch, "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "valid": valid_metrics, "test": test_metrics, "attachment4_count": len(cards), "explanation": "Modality/time attention is accompanied by local occlusion drops for the top evidence bins; Attachment 4 has no labels."}; (out / "problem3_metrics.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps({"output": str(out), "device": str(device), "best_epoch": best_epoch, "valid": valid_metrics, "test": test_metrics, "attachment4_explanations": len(cards)}, ensure_ascii=False, indent=2))


if __name__ == "__main__": raise SystemExit(main())
