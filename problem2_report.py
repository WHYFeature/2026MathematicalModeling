#!/usr/bin/env python3
"""Analyze saved BERT AV predictions and draw publication figures without training.

Only NumPy and Matplotlib are required. Source predictions and model weights
are read-only. Every headline metric is recomputed from the exported CSVs.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
from pathlib import Path

import numpy as np

CLASS_NAMES = ("Negative", "Neutral", "Positive")
COLORS = ("#506DA8", "#CE9542", "#338A7D")
VALID_COLOR, TEST_COLOR = "#93AAC9", "#234F70"
INK, MUTED, GRID = "#203247", "#647487", "#E3E9EF"
FIGURES = (
    ("01_overview", "整体表现与数据组成"),
    ("02_confusion", "验证集与测试集混淆矩阵"),
    ("03_class_errors", "逐类表现与主要误判方向"),
    ("04_regression", "情感强度回归诊断"),
    ("05_error_diagnostics", "情感强度分组与置信度诊断"),
    ("06_attachment3", "附件三两种格式的预测对比"),
)
TRAINING_FIGURE = ("07_training_curves", "训练与验证曲线")


def read_predictions(path, truth):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = {"id", "predicted_class", "regression_prediction", "p_negative", "p_neutral", "p_positive"}
        if truth:
            fields |= {"true_class", "true_regression"}
        if not fields.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing columns in {path}: {fields - set(reader.fieldnames or [])}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty prediction file: {path}")
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids) or any(not x for x in ids):
        raise ValueError(f"Empty or duplicate IDs in {path}")
    p = np.asarray([[float(r[k]) for k in ("p_negative", "p_neutral", "p_positive")] for r in rows])
    predicted = np.asarray([int(r["predicted_class"]) for r in rows])
    regression = np.asarray([float(r["regression_prediction"]) for r in rows])
    if not np.isfinite(p).all() or not np.isfinite(regression).all():
        raise ValueError(f"Non-finite predictions in {path}")
    if (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError(f"Invalid probabilities in {path}")
    if not np.array_equal(predicted, p.argmax(1)):
        raise ValueError(f"Predicted labels do not agree with probability argmax: {path}")
    result = {"ids": ids, "p": p, "pred": predicted, "reg": regression, "rows": rows}
    if truth:
        result["y"] = np.asarray([int(r["true_class"]) for r in rows])
        result["yr"] = np.asarray([float(r["true_regression"]) for r in rows])
        if not np.isin(result["y"], [0, 1, 2]).all() or not np.isfinite(result["yr"]).all():
            raise ValueError(f"Invalid ground truth in {path}")
    return result


def classification_metrics(y, pred):
    cm = np.bincount(y * 3 + pred, minlength=9).reshape(3, 3)
    support, predicted = cm.sum(1), cm.sum(0)
    precision = np.divide(cm.diagonal(), predicted, out=np.zeros(3), where=predicted > 0)
    recall = np.divide(cm.diagonal(), support, out=np.zeros(3), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(3), where=precision + recall > 0)
    return {
        "n": len(y), "correct": int(cm.trace()), "errors": int(len(y) - cm.trace()),
        "accuracy": float(cm.trace() / len(y)), "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall.mean()), "confusion_matrix": cm.tolist(),
        "precision": precision.tolist(), "recall": recall.tolist(), "f1": f1.tolist(),
        "support": support.tolist(), "predicted_counts": predicted.tolist(),
    }


def compute_metrics(data):
    result = classification_metrics(data["y"], data["pred"])
    residual = data["reg"] - data["yr"]
    corr = np.corrcoef(data["reg"], data["yr"])[0, 1] if np.std(data["reg"]) > 1e-12 and np.std(data["yr"]) > 1e-12 else 0.0
    result.update(mae=float(np.abs(residual).mean()), rmse=float(np.sqrt(np.mean(residual ** 2))),
                  pearson=float(corr), mean_residual=float(residual.mean()), median_absolute_error=float(np.median(np.abs(residual))))
    result["regression_by_class"] = [
        {"n": int(np.sum(data["y"] == k)),
         "mean_true": float(data["yr"][data["y"] == k].mean()),
         "mean_predicted": float(data["reg"][data["y"] == k].mean()),
         "mean_residual": float(residual[data["y"] == k].mean()),
         "mae": float(np.abs(residual[data["y"] == k]).mean())}
        if np.any(data["y"] == k) else {"n": 0}
        for k in range(3)
    ]
    confidence, correct = data["p"].max(1), data["pred"] == data["y"]
    bins = np.minimum((confidence * 10).astype(int), 9)
    reliability = []
    for i in range(10):
        group = bins == i
        if group.any():
            reliability.append({"lower": i / 10, "upper": (i + 1) / 10,
                                "n": int(group.sum()), "confidence": float(confidence[group].mean()),
                                "accuracy": float(correct[group].mean())})
    result["reliability"] = reliability
    result["ece_10_bins"] = sum(b["n"] * abs(b["confidence"] - b["accuracy"]) for b in reliability) / len(correct)
    high = confidence >= 0.8
    result["confidence_ge_0_8"] = {"n": int(high.sum()), "errors": int(np.sum(high & ~correct))}
    magnitude = np.abs(data["yr"])
    groups = [("0（中性）", magnitude == 0), ("(0, 0.5]", (magnitude > 0) & (magnitude <= 0.5)),
              ("(0.5, 1]", (magnitude > 0.5) & (magnitude <= 1)),
              ("(1, 2]", (magnitude > 1) & (magnitude <= 2)), ("(2, 3]", (magnitude > 2) & (magnitude <= 3))]
    if not np.all(np.logical_or.reduce([group for _, group in groups])):
        raise ValueError("Intensity groups require ground-truth scores in [-3, 3]")
    result["intensity_groups"] = [{"label": label, "n": int(group.sum()),
                                   "accuracy": float(correct[group].mean()) if group.any() else None}
                                  for label, group in groups]
    return result


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def analyze(root):
    metrics_path = root / "metrics.json"
    metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
    paths = {s: root / f"problem2_{s}_predictions.csv" for s in ("valid", "test", "missing_aligned", "missing_unaligned")}
    data = {s: read_predictions(path, truth=s in ("valid", "test")) for s, path in paths.items()}
    summary = {s: compute_metrics(data[s]) for s in ("valid", "test")}
    overlap = set(data["valid"]["ids"]) & set(data["test"]["ids"])
    if overlap:
        raise ValueError(f"Validation/test IDs overlap: {len(overlap)}")
    for s in ("valid", "test"):
        saved = metadata.get(s)
        if saved is None:
            raise ValueError(f"No {s} metrics found; run --export-only with the latest export script first")
        for metric in ("accuracy", "macro_f1", "mae", "rmse", "pearson"):
            if not np.isclose(saved[metric], summary[s][metric], atol=1e-6):
                raise ValueError(f"CSV and metrics.json disagree: {s}/{metric}")
        if "confusion_matrix" in saved and saved["confusion_matrix"] != summary[s]["confusion_matrix"]:
            raise ValueError(f"Confusion matrix does not match: {s}")
    train_counts = np.asarray(metadata["train_counts"], dtype=int)
    majority = int(train_counts.argmax())
    summary["train_counts"] = train_counts.tolist()
    for s in ("valid", "test"):
        summary[s]["majority_baseline"] = classification_metrics(data[s]["y"], np.full(len(data[s]["y"]), majority))
    aligned, unaligned = data["missing_aligned"], data["missing_unaligned"]
    if set(aligned["ids"]) != set(unaligned["ids"]):
        raise ValueError("Attachment-3 formats must have matching sample IDs for paired display")
    lookup = {sample_id: i for i, sample_id in enumerate(unaligned["ids"])}
    order = [lookup[sample_id] for sample_id in aligned["ids"]]
    paired_reg, paired_cls = unaligned["reg"][order], unaligned["pred"][order]
    delta = paired_reg - aligned["reg"]
    summary["attachment3"] = {
        "n_pairs": len(order), "same_class": int(np.sum(aligned["pred"] == paired_cls)),
        "class_agreement": float(np.mean(aligned["pred"] == paired_cls)),
        "mean_abs_regression_difference": float(np.abs(delta).mean()),
        "pairing": "matched by sample filename ID; these are two views, not independent labeled observations",
        "aligned_counts": np.bincount(aligned["pred"], minlength=3).tolist(),
        "unaligned_counts": np.bincount(paired_cls, minlength=3).tolist(),
        "aligned_range": [float(aligned["reg"].min()), float(aligned["reg"].max())],
        "unaligned_range": [float(paired_reg.min()), float(paired_reg.max())],
    }
    data["paired_reg"], data["paired_cls"] = paired_reg, paired_cls
    summary["best_epoch"] = metadata.get("best_epoch")
    summary["training_history_available"] = bool(metadata.get("history"))
    summary["training_history"] = metadata.get("history") or []
    summary["training_best_epoch"] = metadata.get("best_epoch")
    summary["training_history_provenance"] = None
    summary["sources"] = {p.name: {"path": str(p), "sha256": sha256(p)} for p in [metrics_path, *paths.values()]}
    recovered_path = root / "history_recovery" / "training_history.json"
    if not summary["training_history_available"] and recovered_path.is_file():
        from problem2.training_history import load_history
        recovered = load_history(root)  # Includes reference-file hash checks.
        summary["training_history_available"] = True
        summary["training_history"] = recovered["history"]
        summary["training_best_epoch"] = recovered["best_epoch"]
        summary["training_history_provenance"] = recovered["provenance"]
        summary["sources"]["history_recovery/training_history.json"] = {
            "path": str(recovered_path), "sha256": sha256(recovered_path),
        }
    summary["verification"] = {"metrics_recomputed_from_csv": True, "valid_test_id_overlap": 0,
                               "unique_ids_per_file": True, "probabilities_checked": True}
    return data, summary


def setup_style():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    available = {font.name for font in font_manager.fontManager.ttflist}
    selected = "Times New Roman" if "Times New Roman" in available else None
    if selected is None:
        raise RuntimeError("Times New Roman is required. Install the font and refresh the Matplotlib font cache, then rerun.")
    matplotlib.rcParams.update({
        "font.family": selected, "font.size": 10.5, "axes.titlesize": 12,
        "axes.titleweight": "bold", "axes.titlepad": 16, "axes.labelsize": 10.5,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": "#CCD6E0",
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
        "axes.spines.right": False, "axes.unicode_minus": False, "axes.axisbelow": True,
        "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False,
        "grid.color": GRID, "grid.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })
    return selected


def figure(height=6.2):
    import matplotlib.pyplot as plt
    return plt.figure(figsize=(12.8, height))


def finish(fig, out, name, dpi):
    import matplotlib.pyplot as plt
    # Keep prose in the report, not inside figures; trim unused title margins.
    fig.savefig(out / f"{name}.png", dpi=dpi, bbox_inches="tight", pad_inches=.18)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=.18)
    plt.close(fig)


def overview(data, summary, out, dpi):
    fig = figure(7.0)
    test = summary["test"]
    cards = [("Test accuracy", f"{test['accuracy']:.2%}"), ("Test macro-F1", f"{test['macro_f1']:.4f}"),
             ("Test MAE", f"{test['mae']:.4f}"), ("Test Pearson r", f"{test['pearson']:.4f}")]
    for i, (label, value) in enumerate(cards):
        ax = fig.add_axes([0.065 + i * 0.226, 0.67, 0.20, 0.145])
        ax.set_facecolor("#F1F5F9")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.07, 0.75, label, color=MUTED, fontsize=10.5, transform=ax.transAxes)
        ax.text(0.07, 0.21, value, color=TEST_COLOR, fontsize=25, weight="bold", transform=ax.transAxes)
    ax = fig.add_axes([0.09, 0.16, 0.39, 0.40])
    indices = np.arange(3)
    for offset, split, label, color in ((-0.18, "valid", "Validation", VALID_COLOR), (0.18, "test", "Test", TEST_COLOR)):
        vals = [summary[split][m] for m in ("accuracy", "macro_f1", "balanced_accuracy")]
        bars = ax.bar(indices + offset, vals, width=0.32, color=color, label=label, zorder=3)
        ax.bar_label(bars, labels=[f"{v:.1%}" for v in vals], padding=5, fontsize=9)
    ax.set_xticks(indices, ["Accuracy", "Macro-F1", "Balanced accuracy"])
    ax.set_ylim(0, 1); ax.set_yticks(np.arange(0, 1.01, 0.2), [f"{i}%" for i in range(0, 101, 20)])
    ax.set_title("(a) Classification performance", loc="left"); ax.grid(axis="y"); ax.legend(loc="upper right", ncol=2, fontsize=9)
    ax = fig.add_axes([0.60, 0.16, 0.34, 0.40])
    counts = np.asarray([summary["train_counts"], summary["valid"]["support"], summary["test"]["support"]])
    ratios = counts / counts.sum(1, keepdims=True)
    left = np.zeros(3)
    for k, color in enumerate(COLORS):
        ax.barh(np.arange(3), ratios[:, k], left=left, height=0.48, color=color, label=CLASS_NAMES[k])
        for row in range(3):
            ax.text(left[row] + ratios[row, k] / 2, row, f"{counts[row, k]}\n{ratios[row, k]:.1%}", color="white", ha="center", va="center", fontsize=9)
        left += ratios[:, k]
    ax.set_yticks(range(3), [f"Training  n={counts[0].sum()}", f"Validation  n={counts[1].sum()}", f"Test  n={counts[2].sum()}"])
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
    ax.set_title("(b) Class distribution", loc="left"); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)
    finish(fig, out, FIGURES[0][0], dpi)


def confusion(data, summary, out, dpi):
    from matplotlib.colors import LinearSegmentedColormap
    fig = figure(6.4)
    cmap = LinearSegmentedColormap.from_list("report_blue", ["#F2F6FA", "#A7C4D6", "#346687", "#173A56"])
    for index, (split, label) in enumerate((("valid", "(a) Validation"), ("test", "(b) Test"))):
        ax = fig.add_axes([0.10 + index * 0.425, 0.20, 0.32, 0.58])
        cm = np.asarray(summary[split]["confusion_matrix"])
        norm = np.divide(cm, cm.sum(1, keepdims=True), out=np.zeros_like(cm, dtype=float), where=cm.sum(1, keepdims=True) > 0)
        img = ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
        for row in range(3):
            for col in range(3):
                ax.text(col, row, f"{cm[row,col]}\n({norm[row,col]:.1%})", ha="center", va="center", fontsize=12, color="white" if norm[row,col] > 0.48 else INK)
        ax.set_xticks(range(3), CLASS_NAMES); ax.set_yticks(range(3), CLASS_NAMES)
        ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
        ax.set_title(f"{label}  n={summary[split]['n']}   Accuracy={summary[split]['accuracy']:.2%}", fontsize=12)
        ax.set_xticks(np.arange(-0.5, 3, 1), minor=True); ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=3); ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cax = fig.add_axes([0.91, 0.245, 0.014, 0.44])
    cb = fig.colorbar(img, cax=cax, ticks=[0, 0.25, 0.5, 0.75, 1])
    cb.ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"]); cb.outline.set_visible(False)
    finish(fig, out, FIGURES[1][0], dpi)


def class_errors(data, summary, out, dpi):
    fig = figure(6.2)
    m = summary["test"]
    ax = fig.add_axes([0.08, 0.19, 0.40, 0.59])
    x = np.arange(3)
    for k, (label, color) in enumerate(zip(CLASS_NAMES, COLORS)):
        vals = [m[key][k] for key in ("precision", "recall", "f1")]
        bars = ax.bar(x + (k - 1) * 0.24, vals, width=0.21, color=color, label=label, zorder=3)
        ax.bar_label(bars, labels=[f"{v:.1%}" for v in vals], padding=5, fontsize=8.5)
    ax.set_xticks(x, ["Precision", "Recall", "F1"])
    ax.set_ylim(0, 1); ax.set_yticks(np.arange(0, 1.01, .2), [f"{i}%" for i in range(0, 101, 20)])
    ax.grid(axis="y"); ax.legend(ncol=3, loc="upper left", fontsize=9); ax.set_title("(a) Per-class performance", loc="left")
    ax = fig.add_axes([0.65, 0.19, 0.28, 0.59])
    cm = np.asarray(m["confusion_matrix"])
    errors = sorted([(int(cm[i, j]), i, j) for i in range(3) for j in range(3) if i != j], reverse=True)
    bars = ax.barh(range(6), [v[0] for v in errors], color=[COLORS[v[1]] for v in errors], height=0.60)
    ax.set_yticks(range(6), [f"{CLASS_NAMES[i]} → {CLASS_NAMES[j]}" for _, i, j in errors]); ax.invert_yaxis()
    ax.bar_label(bars, labels=[str(n) for n, _, _ in errors], padding=5, fontsize=10)
    ax.set_xlim(0, max(n for n, _, _ in errors) * 1.3); ax.grid(axis="x"); ax.set_xlabel("Number of errors")
    ax.set_title(f"(b) Error directions (n={m['errors']})", loc="left")
    finish(fig, out, FIGURES[2][0], dpi)


def regression(data, summary, out, dpi):
    fig = figure(9.0)
    lo = min(-3.2, *(float(min(data[s]['yr'].min(), data[s]['reg'].min())) - .15 for s in ('valid', 'test')))
    hi = max(3.2, *(float(max(data[s]['yr'].max(), data[s]['reg'].max())) + .15 for s in ('valid', 'test')))
    for index, (split, label) in enumerate((("valid", "(a) Validation"), ("test", "(b) Test"))):
        ax = fig.add_axes([0.09 + index * 0.46, 0.48, 0.36, 0.34])
        d, m = data[split], summary[split]
        ax.plot([lo, hi], [lo, hi], ls="--", lw=1, color=MUTED, zorder=1)
        for k in range(3):
            keep = d["y"] == k
            ax.scatter(d["yr"][keep], d["reg"][keep], s=13, alpha=.48, linewidths=0, color=COLORS[k], label=CLASS_NAMES[k])
        ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="True sentiment score", ylabel="Predicted sentiment score")
        ax.grid(alpha=.65); ax.set_title(f"{label}   MAE={m['mae']:.3f}   r={m['pearson']:.3f}", loc="left", fontsize=12)
        if index == 0:
            ax.legend(loc="upper left", ncol=3, fontsize=8.5)
    ax = fig.add_axes([0.09, 0.12, 0.36, 0.25])
    for split, label, color in (("valid", "Validation", VALID_COLOR), ("test", "Test", TEST_COLOR)):
        error = np.sort(np.abs(data[split]["reg"] - data[split]["yr"]))
        ax.step(error, np.arange(1, len(error) + 1) / len(error), where="post", label=label, color=color, linewidth=2)
    ax.set_xlabel("Absolute error"); ax.set_ylabel("Cumulative proportion"); ax.set_ylim(0, 1.03)
    ax.grid(); ax.legend(loc="lower right", fontsize=9); ax.set_title("(c) Absolute error ECDF", loc="left")
    ax = fig.add_axes([0.55, 0.12, 0.36, 0.25])
    residual = data["test"]["reg"] - data["test"]["yr"]
    groups = [residual[data["test"]["y"] == k] for k in range(3)]
    bp = ax.boxplot(groups, patch_artist=True, widths=.48, showfliers=True,
                    medianprops={"color": "white", "linewidth": 1.8},
                    flierprops={"marker": ".", "markersize": 3, "alpha": .35})
    for box, color in zip(bp["boxes"], COLORS):
        box.set_facecolor(color); box.set_edgecolor(color)
    ax.axhline(0, color=MUTED, linewidth=1, linestyle="--")
    ax.set_xticks([1, 2, 3], CLASS_NAMES); ax.set_ylabel("Predicted - true score"); ax.grid(axis="y")
    ax.set_title("(d) Test residuals by true class", loc="left")
    finish(fig, out, FIGURES[3][0], dpi)


def error_diagnostics(data, summary, out, dpi):
    fig = figure(6.3)
    m = summary["test"]
    ax = fig.add_axes([0.09, 0.21, 0.38, 0.56])
    groups = m["intensity_groups"]
    vals = [g["accuracy"] if g["accuracy"] is not None else np.nan for g in groups]
    bars = ax.bar(range(len(groups)), vals, width=.62, color=[COLORS[1]] + [TEST_COLOR] * 4, zorder=3)
    for bar, group in zip(bars, groups):
        if group["n"]:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + .025,
                    f"{group['accuracy']:.1%}\nn={group['n']}", ha="center", va="bottom", fontsize=9)
    ax.axhline(m["accuracy"], color=MUTED, ls="--", lw=1, label=f"Overall {m['accuracy']:.1%}")
    ax.set_xticks(range(len(groups)), ["0 (Neutral)", "(0, 0.5]", "(0.5, 1]", "(1, 2]", "(2, 3]"], fontsize=9)
    ax.set_ylim(0, 1.13); ax.set_yticks(np.arange(0, 1.01, .2), [f"{i}%" for i in range(0,101,20)])
    ax.set_xlabel("Absolute true sentiment score"); ax.set_ylabel("Accuracy"); ax.grid(axis="y")
    ax.set_title("(a) Accuracy by sentiment intensity", loc="left"); ax.legend(loc="upper left", fontsize=8.5)
    ax = fig.add_axes([0.61, 0.21, 0.31, 0.56])
    b = m["reliability"]
    x, y = [r["confidence"] for r in b], [r["accuracy"] for r in b]
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1, label="Perfect calibration")
    ax.plot(x, y, color=TEST_COLOR, linewidth=1.8)
    ax.scatter(x, y, color=TEST_COLOR, s=[32 + 150*r["n"]/m["n"] for r in b], edgecolor="white", linewidth=.7, zorder=3)
    ax.set(xlim=(.3, 1.02), ylim=(.3, 1.02), xlabel="Mean confidence", ylabel="Observed accuracy")
    ax.grid(); ax.set_title("(b) Reliability diagram", loc="left")
    ax.text(.05, .92, f"ECE (10 bins) = {m['ece_10_bins']:.3f}", transform=ax.transAxes, fontsize=10)
    ax.legend(loc="lower right", fontsize=9)
    finish(fig, out, FIGURES[4][0], dpi)


def attachment3(data, summary, out, dpi):
    fig = figure(6.5)
    m = summary["attachment3"]
    ax = fig.add_axes([0.09, 0.20, 0.36, 0.58])
    counts = np.asarray([m["aligned_counts"], m["unaligned_counts"]])
    bottom = np.zeros(2)
    for k, color in enumerate(COLORS):
        vals = counts[:, k]
        ax.bar([0, 1], vals, bottom=bottom, color=color, width=.52, label=CLASS_NAMES[k])
        for i in range(2):
            if vals[i]:
                ax.text(i, bottom[i]+vals[i]/2, f"{vals[i]}\n{vals[i]/m['n_pairs']:.1%}", ha="center", va="center", color="white", fontsize=11)
        bottom += vals
    ax.set_xticks([0, 1], ["aligned", "unaligned"]); ax.set_ylim(0, m["n_pairs"] * 1.18)
    ax.set_ylabel("Number of predictions"); ax.grid(axis="y"); ax.set_title("(a) Predicted class distribution", loc="left")
    ax.legend(loc="upper center", ncol=3, fontsize=9)
    ax = fig.add_axes([0.59, 0.20, 0.33, 0.58])
    aligned = data["missing_aligned"]
    same = aligned["pred"] == data["paired_cls"]
    lo = min(aligned["reg"].min(), data["paired_reg"].min()) - .2
    hi = max(aligned["reg"].max(), data["paired_reg"].max()) + .2
    ax.plot([lo, hi], [lo, hi], color=MUTED, ls="--", lw=1)
    ax.scatter(aligned["reg"][same], data["paired_reg"][same], color=TEST_COLOR, s=50, label="Same class", alpha=.8)
    ax.scatter(aligned["reg"][~same], data["paired_reg"][~same], facecolors="none", edgecolors=COLORS[1], s=65, linewidths=1.7, label="Different class")
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Predicted score (aligned)", ylabel="Predicted score (unaligned)")
    ax.grid(); ax.set_title("(b) Paired predictions", loc="left"); ax.legend(loc="lower right", fontsize=9)
    ax.text(.045, .95, f"Class agreement: {m['same_class']}/{m['n_pairs']} ({m['class_agreement']:.1%})", transform=ax.transAxes, fontsize=10, va="top")
    finish(fig, out, FIGURES[5][0], dpi)


def write_report(data, summary, output):
    v, t, a = summary["valid"], summary["test"], summary["attachment3"]
    cm = np.asarray(t["confusion_matrix"])
    errors_involving_neutral = int(cm[1].sum() - cm[1,1] + cm[:,1].sum() - cm[1,1])
    sections = []
    def add(title, paragraphs, table=None, image=None):
        sections.append({"title": title, "paragraphs": paragraphs, "table": table, "image": image})

    is_replay = (summary.get("training_history_provenance") or {}).get("kind") == "verified_replay"
    replay_provenance = summary.get("training_history_provenance") or {}
    is_extended = bool(replay_provenance.get("extended_analysis"))
    if is_replay:
        history_description = (
            f"原始逐轮日志未保存；目前补充了 {len(summary['training_history'])} 轮实测复现记录，"
            f"复现中第 {summary['training_best_epoch']} 轮的全部权重及缓冲区与当前参考模型逐张量完全一致，"
            "归一化统计量也一致。曲线标记为经权重校验的复现过程，不声称找回了当时的原始日志。"
            "参考权重、原始指标与预测文件均保留，文件哈希校验通过。"
        )
        if is_extended:
            history_description += (
                f"前 {replay_provenance['reference_selection_epochs']} 轮对应原实验的复现，"
                f"第 {replay_provenance['extension_start_epoch']} 轮起为延长训练分析。"
                "后续轮次不会替换已确定的参考模型，完整曲线不表示原实验当时训练了这么多轮。"
            )
    elif summary["training_history_available"]:
        history_description = f"本次实验保存了 {len(summary['training_history'])} 轮历史，训练过程见后面的曲线分析。曲线属于当前结果目录对应的实验，不与其他模型的结果混用。"
    else:
        history_description = "原权重未保存最佳轮次、完整超参数及逐轮训练历史，因此 best_epoch 为 null 不表示导出失败。当前没有可核实的训练曲线；只有通过权重完全一致性校验的复现实测记录，才能补充为该模型的复现曲线。"

    add("1. 阶段结论", [
        f"本阶段采用 BERT 文本微调与音视频统计特征融合模型。测试集共 {t['n']} 条，正确分类 {t['correct']} 条，Accuracy 为 {t['accuracy']:.2%}，Macro-F1 为 {t['macro_f1']:.4f}。回归 MAE 为 {t['mae']:.4f}，RMSE 为 {t['rmse']:.4f}，Pearson 相关系数为 {t['pearson']:.4f}。",
        f"模型对正面和负面情感的识别较好，中性类别仍较弱：测试集三类 F1 分别为 {t['f1'][0]:.4f}、{t['f1'][1]:.4f}、{t['f1'][2]:.4f}。该模型作为后续分析的固定基线，当前结果尚不能证明对局部模态缺失具有稳定鲁棒性。",
        history_description,
    ], image=FIGURES[0])
    add("2. 评价协议与基础指标", [
        f"附件二保持原有划分：训练集 {sum(summary['train_counts'])} 条、验证集 {v['n']} 条、测试集 {t['n']} 条。现有训练代码采用验证集 Macro-F1 选择权重；归一化统计量仅由训练集拟合。分类结果使用三类概率的 argmax，本报告没有重新训练或调整决策阈值。",
        "代码中的模型以本地 BERT 为文本编码器，对有效 token 求均值；音频与视觉分别汇总有效帧的均值、标准差和有效比例，投影后与文本拼接，连接三分类头及回归头。代码中的联合损失为交叉熵加 0.15 倍 SmoothL1 回归损失。该结构主要使用音视频统计信息，不能据此声称已实现细粒度跨模态时序注意力。",
        f"测试准确率比验证集高 {(t['accuracy']-v['accuracy'])*100:.2f} 个百分点，但测试 MAE 比验证集高 {t['mae']-v['mae']:.4f}。分类与回归指标衡量不同目标；两个划分的样本组成和难度也不同，不能把它们视为训练前后的提升。",
        f"按训练集多数类固定预测 Positive，测试准确率为 {t['majority_baseline']['accuracy']:.2%}；当前模型高出 {(t['accuracy']-t['majority_baseline']['accuracy'])*100:.2f} 个百分点。Macro-F1 与平均召回率同时报告，以免多数类掩盖中性类表现。历史开发已多次查看测试结果，这不是一个始终未被查看的最终盲测结论。",
    ], table=(["指标", "验证集", "测试集"], [
        ["样本数", str(v["n"]), str(t["n"])],
        *[[label, f"{v[key]:.4f}", f"{t[key]:.4f}"] for key, label in (("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"), ("balanced_accuracy", "平均召回率"), ("mae", "MAE"), ("rmse", "RMSE"), ("pearson", "Pearson r"))],
    ]), image=FIGURES[1])
    add("3. 逐类表现与错误归因", [
        f"测试集中，中性样本 {t['support'][1]} 条，仅 {cm[1,1]} 条识别正确，召回率为 {t['recall'][1]:.2%}；其中 {cm[1,0]} 条被预测为负面、{cm[1,2]} 条被预测为正面。另一方面，模型预测为中性的 {t['predicted_counts'][1]} 条中，有 {cm[0,1] + cm[2,1]} 条实际属于其他类别，中性类别既存在漏判，也存在误报。",
        f"涉及中性类别的错误共 {errors_involving_neutral} 条，占全部 {t['errors']} 条误判的 {errors_involving_neutral/t['errors']:.2%}。直接跨越负面与正面的错误仍有 {cm[0,2]+cm[2,0]} 条，因此问题并非只发生在中性边界。",
        "这些结果支持“中性类别区分不足”的定量结论，但不能单凭混淆矩阵认定标签有错、说话者存在讽刺或某一模态失效。语义级归因仍需按错误样本编号核对文本、音视频原始内容；本报告仅保留可由预测表验证的结论。",
    ], table=(["测试类别", "真实数量", "预测数量", "Precision", "Recall", "F1"], [
        [CLASS_NAMES[k], str(t["support"][k]), str(t["predicted_counts"][k]), *[f"{t[key][k]:.4f}" for key in ("precision", "recall", "f1")]] for k in range(3)
    ]), image=FIGURES[2])
    add("4. 情感强度回归", [
        f"测试集 Pearson r={t['pearson']:.4f} 表明预测分数与真实分数存在正相关，但相关性不等同于数值准确。RMSE={t['rmse']:.4f} 高于 MAE={t['mae']:.4f}，说明较大误差对平方误差指标的影响较明显；绝对误差中位数为 {t['median_absolute_error']:.4f}。",
        f"测试集整体平均残差（预测值减真实值）为 {t['mean_residual']:.4f}。下表进一步按真实类别展示有方向的误差；正残差表示数值高估，负残差表示低估。分组统计用于描述现象，不作为标签或缺失机制的因果解释。",
        f"负面样本的平均预测值由真实的 {t['regression_by_class'][0]['mean_true']:.4f} 向 {t['regression_by_class'][0]['mean_predicted']:.4f} 收缩，正面样本则由 {t['regression_by_class'][2]['mean_true']:.4f} 向 {t['regression_by_class'][2]['mean_predicted']:.4f} 收缩。这表明模型在组平均意义上倾向于低估正负情绪的强烈程度；正负方向的残差相互抵消，不能用接近零的整体平均残差证明模型没有系统偏差。",
    ], table=(["真实类别", "平均真实强度", "平均预测强度", "平均残差", "MAE"], [
        [CLASS_NAMES[k], *[f"{t['regression_by_class'][k][key]:.4f}" for key in ("mean_true", "mean_predicted", "mean_residual", "mae")]] for k in range(3)
    ]), image=FIGURES[3])
    high = t["confidence_ge_0_8"]
    add("5. 错误分布与预测置信度", [
        "按真实情感强度的绝对值分组，0 单独代表中性类，其余区间混合正负两种极性。各组样本量不同，分组差异是描述性结果；没有额外进行显著性检验或独立性假设下的置信区间估计。",
        f"使用 [0,1] 上 10 个等宽概率箱，ECE 为 {t['ece_10_bins']:.4f}。ECE 是各箱平均置信度与实际准确率之差的绝对值，按箱内样本数加权；空箱不参与计算。可靠性曲线位于对角线下方的部分表示过度自信。",
        f"置信度不低于 0.8 的预测有 {high['n']} 条，其中 {high['errors']} 条错误。高预测概率并不保证预测正确。报告同时导出全部分类错误清单，并按错误预测置信度由高到低排序，便于优先人工核查。",
    ], table=(["真实强度绝对值", "样本数", "分类准确率"], [[g["label"], str(g["n"]), f"{g['accuracy']:.2%}" if g["accuracy"] is not None else "无样本"] for g in t["intensity_groups"]]), image=FIGURES[4])
    add("6. 附件三全量预测", [
        f"aligned 与 unaligned 各有 {a['n_pairs']} 条记录，按相同文件编号配对展示，不视为 {2*a['n_pairs']} 条独立带标签样本。aligned 的负面/中性/正面预测数为 {a['aligned_counts']}，unaligned 为 {a['unaligned_counts']}。两个格式均以正面预测居多，但缺少真实标签，无法据此判定偏置方向或预测准确率。",
        f"两种格式的预测类别一致 {a['same_class']}/{a['n_pairs']} 条，一致率为 {a['class_agreement']:.2%}；回归预测的平均绝对差为 {a['mean_abs_regression_difference']:.4f}。一致率仅描述两种输入格式的输出一致程度，不是对真实标签的准确率。",
        "unaligned 的音视频序列在导出前按有效行平均至 50 个区间，文本重新分词后截取到 50 个位置；aligned 使用提供的文本 token 和对应掩码。这些预处理差异也会影响预测，不能把两种格式的输出差异直接解释成某种模态缺失的因果效应。全零行中还可能包含尾部填充，掩码有效比例不能直接当作真实缺失率。",
    ], image=FIGURES[5])
    if summary.get("training_curve_info"):
        info = summary["training_curve_info"]
        add("7. 经权重校验的训练复现过程" if is_replay else "7. 训练与验证过程", [
            ((f"实测过程共 {info['epochs_recorded']} 轮，参考模型在原来的前 {replay_provenance['reference_selection_epochs']} 轮内按验证集 Macro-F1 选择，"
              f"匹配于第 {info['best_epoch']} 轮。后续延长训练的全程最佳验证轮次为 {replay_provenance['replay_selected_epoch']}，单独记录，不替换参考模型。"
              "虚线标记参考模型轮次，浅色背景表示超出原实验轮数的延长训练部分。"
              if is_extended else f"复现实验共 {info['epochs_recorded']} 轮，按验证集 Macro-F1 选中的第 {info['best_epoch']} 轮权重与参考模型完全一致；图中虚线标记该轮。") +
             "测试集不参与此次复现的逐轮选择，也未依据测试分数搜索超参数。原始 best_epoch 字段保留未知，复现轮次单独记录。"
             if is_replay else f"当前目录共记录 {info['epochs_recorded']} 轮，最佳权重对应第 {info['best_epoch']} 轮（如为 None 则未记录最佳轮次）。模型按验证集 Macro-F1 选择；测试集不参与逐轮选择。"),
            ("训练集和验证集曲线均在每轮结束后，以固定权重和 eval 模式评价完整划分，关闭 Dropout，便于比较泛化差距。损失为交叉熵加 0.15 倍 SmoothL1，分类权重沿用该次训练设置。"
             if info["has_train_metrics"] else "旧日志只记录了验证集指标。缺失的训练指标及损失标为未记录，不以零填充或推测。"),
            "曲线使用实测值，不做平滑或插值。训练指标持续改善而验证指标恶化时，支持出现过拟合的判断；不同轮次的最终权重不同，不能把训练过程中的指标直接代替已选权重的评估结果。",
        ], image=TRAINING_FIGURE)
    add(f"{8 if summary.get('training_curve_info') else 7}. 可追溯性与当前边界", [
        "报告从四份预测 CSV 重算指标，逐项与 metrics.json 核对。检查包括文件内 ID 唯一性、验证/测试 ID 不重叠、概率合法性、分类 argmax 一致性，以及混淆矩阵、Accuracy、Macro-F1、MAE、RMSE、Pearson 的一致性。analysis.json 保存计算结果和输入文件 SHA-256，便于对应这一次成果。",
        "现阶段已完成基础预测评估、定量错误诊断及附件三全量预测展示。局部缺失类型/缺失率实验、同一模型下的消融实验和错误样本原素材核查尚未完成。历史不同架构的分数不能替代严格消融；当前 BERT 基线也不能继承旧 Hybrid 模型的缺失增强能力。",
    ])
    md = ["# 问题二阶段性成果分析\n", "本报告对应所选目录中的 BERT 音视频融合模型。图表由保存的预测结果及可用训练记录生成。\n"]
    body = []
    for section in sections:
        md.append(f"## {section['title']}\n")
        body.append(f"<section><h2>{html.escape(section['title'])}</h2>")
        for paragraph in section["paragraphs"]:
            md.append(paragraph + "\n")
            body.append(f"<p>{html.escape(paragraph)}</p>")
        if section["table"]:
            headers, rows = section["table"]
            md += ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
            md += ["| " + " | ".join(row) + " |" for row in rows]
            md.append("")
            body.append("<div class='table-wrap'><table><thead><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in headers) + "</tr></thead><tbody>")
            for row in rows:
                body.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>")
            body.append("</tbody></table></div>")
        if section["image"]:
            name, label = section["image"]
            md.append(f"![{label}](figures/{name}.png)\n")
            image_path = output / "figures" / f"{name}.png"
            encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
            body.append(f"<figure><img alt='{html.escape(label)}' src='data:image/png;base64,{encoded}'><figcaption>{html.escape(label)}</figcaption></figure>")
        body.append("</section>")
    (output / "stage_report.md").write_text("\n".join(md), encoding="utf-8")
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>问题二阶段性成果分析</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f6f8;color:#203247;font:16px/1.9 'Microsoft YaHei',sans-serif}
main{max-width:1100px;margin:36px auto;padding:44px 52px;background:white;border-top:6px solid #234f70}
h1{font-size:30px;line-height:1.5;margin:0 0 12px}h2{font-size:22px;margin:32px 0 14px;border-bottom:1px solid #dce5ee;padding-bottom:8px}
.intro,figcaption{color:#647487}p{margin:12px 0}table{width:100%;border-collapse:collapse;font-size:14px;margin:20px 0}
th{background:#edf3f7}td,th{padding:10px;text-align:right;border-bottom:1px solid #e3e9ef}td:first-child,th:first-child{text-align:left}
.table-wrap{overflow:auto}figure{margin:24px 0}img{width:100%;height:auto;display:block}figcaption{text-align:center;font-size:13px}
@media(max-width:700px){main{margin:0;padding:24px 16px}h1{font-size:25px}h2{font-size:20px}}
@media print{body{background:white}main{margin:0;padding:0;border:0}figure,table{break-inside:avoid}}
</style><main><h1>问题二阶段性成果分析</h1><p class="intro">BERT 音视频融合模型 · 保存预测结果的独立复核与可视化</p>"""
    (output / "stage_report.html").write_text(page + "".join(body) + "</main></html>", encoding="utf-8")


def export_errors(data, output):
    fields = ["id", "true_class", "predicted_class", "true_label", "predicted_label", "confidence", "true_regression", "regression_prediction", "absolute_regression_error"]
    for split in ("valid", "test"):
        d = data[split]
        indices = np.flatnonzero(d["y"] != d["pred"])
        indices = sorted(indices, key=lambda i: -float(d["p"][i].max()))
        with (output / f"errors_{split}.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
            for i in indices:
                writer.writerow({"id": d["ids"][i], "true_class": int(d["y"][i]), "predicted_class": int(d["pred"][i]),
                                 "true_label": CLASS_NAMES[d["y"][i]], "predicted_label": CLASS_NAMES[d["pred"][i]],
                                 "confidence": float(d["p"][i].max()), "true_regression": float(d["yr"][i]),
                                 "regression_prediction": float(d["reg"][i]), "absolute_regression_error": float(abs(d["reg"][i]-d["yr"][i]))})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("D:/E_math/problem2_outputs_bert_av_reproduce"))
    parser.add_argument("--output-root", type=Path, help="Defaults to INPUT_ROOT/stage_report")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)
    if args.dpi < 120:
        parser.error("Use dpi >= 120 for readable figures")
    root = args.input_root.resolve()
    out = args.output_root.resolve() if args.output_root else root / "stage_report"
    data, summary = analyze(root)
    font = setup_style()
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for draw in (overview, confusion, class_errors, regression, error_diagnostics, attachment3):
        draw(data, summary, figures, args.dpi)
        print(f"Created {draw.__name__}", flush=True)
    summary["figure_font"] = font
    summary["figures"] = [{"name": name, "title": title} for name, title in FIGURES]
    if summary["training_history_available"]:
        from problem2.training_history import draw_training_curves
        summary["training_curve_info"] = draw_training_curves(
            {"history": summary["training_history"], "best_epoch": summary["training_best_epoch"],
             "provenance": summary.get("training_history_provenance") or {}},
            figures, name=TRAINING_FIGURE[0], dpi=args.dpi,
        )
        summary["figures"].append({"name": TRAINING_FIGURE[0], "title": TRAINING_FIGURE[1]})
    (out / "analysis.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    export_errors(data, out)
    write_report(data, summary, out)
    print(json.dumps({"report": str(out / "stage_report.html"), "figures": len(summary["figures"]), "formats": ["PNG", "PDF"],
                      "test_accuracy": summary["test"]["accuracy"], "validation_ok": True}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
