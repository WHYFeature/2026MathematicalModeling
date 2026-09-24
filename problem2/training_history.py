"""Persist measured epoch metrics and draw curves without loading a model."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

CURVE_METRICS = ("loss", "accuracy", "macro_f1", "mae")


def write_json_atomic(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def save_training_history(output, history, best_epoch, training_config, *, provenance=None):
    """Commit JSON first; the human-readable CSV can always be rebuilt from it."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1, "history": history, "best_epoch": best_epoch,
        "training_config": training_config,
        "metric_protocol": "Train and valid: full split in eval mode after the same epoch. No test metrics used.",
    }
    if provenance is not None:
        payload["provenance"] = provenance
    write_json_atomic(output / "training_history.json", payload)
    keys = (*CURVE_METRICS, "classification_loss", "regression_loss", "rmse", "pearson")
    fields = ["epoch", "lr", "head_lr", *[f"{split}_{key}" for split in ("train", "valid") for key in keys]]
    path = output / "training_history.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in history:
            row = {key: record.get(key, "") for key in ("epoch", "lr", "head_lr")}
            row.update({f"{split}_{key}": record.get(split, {}).get(key, "")
                        for split in ("train", "valid") for key in keys})
            writer.writerow(row)
    temporary.replace(path)


def prepare_history(payload):
    """Validate recorded values; absent curves stay absent, never synthesized."""
    history = payload.get("history")
    if not history:
        raise ValueError("No recorded epoch history. A final checkpoint cannot recover past training curves; use a new training run with history logging.")
    rows = []
    for record in history:
        epoch = record.get("epoch")
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 1:
            raise ValueError("History epochs must be positive integers")
        # Legacy history contains validation metrics directly in each row.
        row = {"epoch": epoch, "train": record.get("train", {}), "valid": record.get("valid", record)}
        for split in ("train", "valid"):
            for key in CURVE_METRICS:
                value = row[split].get(key)
                if value is not None and (not np.isfinite(value) or value < 0 or (key in ("accuracy", "macro_f1") and value > 1)):
                    raise ValueError(f"Invalid {split}/{key} at epoch {epoch}")
        rows.append(row)
    epochs = [row["epoch"] for row in rows]
    if any(b <= a for a, b in zip(epochs, epochs[1:])):
        raise ValueError("History must have unique, increasing epochs")
    if not any(key in row[split] for row in rows for split in ("train", "valid") for key in CURVE_METRICS):
        raise ValueError("History contains no plottable metrics")
    best_epoch = payload.get("best_epoch")
    if best_epoch is not None and best_epoch not in epochs:
        raise ValueError("Recorded best_epoch is absent from history")
    return rows, best_epoch


def load_history(input_root):
    root = Path(input_root)
    for name in ("training_history.json", "metrics.json"):
        path = root / name
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("history"):
                prepare_history(payload)
                return payload
    recovered = root / "history_recovery" / "training_history.json"
    if recovered.is_file():
        payload = json.loads(recovered.read_text(encoding="utf-8"))
        validate_recovered_history(root, payload)
        return payload
    raise ValueError(f"No recorded epoch history in {root}. Re-exporting old weights cannot recover it; record a new training run in a separate directory.")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_recovered_history(root, payload):
    """Accept a replay only while its exact-match proof still names this model."""
    prepare_history(payload)
    provenance = payload.get("provenance", {})
    if provenance.get("kind") != "verified_replay" or provenance.get("exact_state_match") is not True:
        raise ValueError("Recovered history lacks exact checkpoint verification")
    if provenance.get("normalization_equal") is not True:
        raise ValueError("Recovered history lacks normalization verification")
    if payload.get("best_epoch") not in provenance.get("matched_epochs", []):
        raise ValueError("Recovered history's selected epoch does not match the target model")
    if provenance.get("extended_analysis"):
        horizon = provenance.get("reference_selection_epochs")
        if not isinstance(horizon, int) or not payload["best_epoch"] <= horizon < payload["history"][-1]["epoch"]:
            raise ValueError("Invalid reference selection window in extended history")
        if provenance.get("reference_selected_epoch") != payload["best_epoch"] or provenance.get("previous_verified_prefix_equal") is not True:
            raise ValueError("Extended history does not preserve the verified reference")
    required = {"bert_av_fusion.pt", "metrics.json", "normalization.json"}
    sources = provenance.get("target_files_sha256", {})
    if not required.issubset(sources):
        raise ValueError("Recovered history lacks target provenance")
    for name, expected in sources.items():
        if Path(name).name != name:
            raise ValueError("Invalid target filename in recovered history")
        path = Path(root) / name
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"Target file changed since history recovery: {name}")
    return payload


def draw_training_curves(payload, output, name="training_curves", dpi=300):
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator, PercentFormatter
    from problem2_report import figure, finish, setup_style

    if dpi < 120:
        raise ValueError("Use dpi >= 120 for readable figures")
    rows, best_epoch = prepare_history(payload)
    setup_style()
    epochs = np.asarray([r["epoch"] for r in rows])
    has_train = any(any(key in r["train"] for key in CURVE_METRICS) for r in rows)
    fig = figure(7.8)
    colors = {"train": "#338A7D", "valid": "#506DA8"}
    handles = [Line2D([0], [0], color=colors[s], marker="o" if s == "train" else "s",
                      linestyle="-" if s == "train" else "--", linewidth=2,
                      label=label) for s, label in (("train", "Training"), ("valid", "Validation"))
               if any(any(key in r[s] for key in CURVE_METRICS) for r in rows)]
    if best_epoch is not None:
        marker_label = (f"Reference model (epoch {best_epoch})"
                        if payload.get("provenance", {}).get("kind") == "verified_replay" else
                        f"Best validation macro-F1 (epoch {best_epoch})")
        handles.append(Line2D([0], [0], color="#CE9542", linestyle="--",
                              label=marker_label))
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(.5, .844), ncol=len(handles), fontsize=10)
    labels = (("(a) Joint loss", "Loss"), ("(b) Accuracy", "Accuracy"),
              ("(c) Macro-F1", "Macro-F1"), ("(d) Regression error", "MAE"))
    for index, (key, (panel_title, ylabel)) in enumerate(zip(CURVE_METRICS, labels)):
        ax = fig.add_axes([.085 + (index % 2) * .465, .53 if index < 2 else .15, .385, .255])
        plotted = False
        observed = []
        for split in ("train", "valid"):
            values = np.asarray([r[split].get(key, np.nan) for r in rows], dtype=float)
            if not np.isfinite(values).any():
                continue
            # NaNs break lines where a value was not recorded. Missing epochs
            # also create gaps instead of implying a complete trajectory.
            plot_x, plot_y = [], []
            for i, (epoch, value) in enumerate(zip(epochs, values)):
                if i and epoch - epochs[i - 1] > 1:
                    plot_x.append(np.nan); plot_y.append(np.nan)
                plot_x.append(epoch); plot_y.append(value)
            ax.plot(plot_x, plot_y, color=colors[split], linewidth=2,
                    linestyle="-" if split == "train" else "--",
                    marker="o" if split == "train" else "s",
                    markersize=4 if len(rows) < 40 else 2,
                    markevery=max(1, len(rows) // 15), zorder=3)
            observed.extend(values[np.isfinite(values)].tolist())
            plotted = True
        if not plotted:
            ax.text(.5, .5, "Not recorded", transform=ax.transAxes, ha="center", color="#647487")
        elif key in ("accuracy", "macro_f1"):
            ax.set_ylim(0, 1.05)
            if key == "accuracy":
                ax.yaxis.set_major_formatter(PercentFormatter(1))
        else:
            ax.set_ylim(0, max(max(observed) * 1.15, .05))
        if best_epoch is not None:
            ax.axvline(best_epoch, color="#CE9542", linestyle="--", linewidth=1.3, zorder=2)
        if (payload.get("provenance") or {}).get("extended_analysis"):
            boundary = payload["provenance"]["reference_selection_epochs"] + .5
            ax.axvspan(boundary, epochs[-1] + .4, facecolor="#EAF0F6", alpha=.38, zorder=0)
        ax.set_xlim(epochs[0] - .4, epochs[-1] + .4)
        if len(epochs) <= 8:
            ax.set_xticks(epochs)
        else:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.set_title(panel_title, loc="left"); ax.grid(alpha=.85)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    finish(fig, output, name, dpi)
    return {"epochs_recorded": len(rows), "best_epoch": best_epoch, "has_train_metrics": has_train,
            "png": str(output / f"{name}.png"), "pdf": str(output / f"{name}.pdf")}
