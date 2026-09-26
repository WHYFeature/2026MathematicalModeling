from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from problem3.model import ExplainableFusionNet
from problem3.train import MODALITIES, CLASS_NAMES, full_local_occlusion, load_attachment4, plot_outputs, write_rows, write_validation_error_analysis


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description="Create Problem 3 diagnostics from an existing output directory")
    parser.add_argument("--data-root", type=Path, default=Path("D:/E_math/DATA"))
    parser.add_argument("--output-root", type=Path, default=Path("D:/E_math/problem3_outputs"))
    parser.add_argument("--text-model", type=Path, default=Path("D:/E_math/models/bert-base-uncased"))
    parser.add_argument("--occlusion-span", type=int, default=1)
    args = parser.parse_args()
    out = args.output_root.resolve()
    valid_rows = read_csv(out / "problem3_valid_predictions.csv")
    prediction = {
        "true_classification": np.asarray([int(r["true_class"]) for r in valid_rows], dtype=np.int64),
        "classification": np.asarray([int(r["predicted_class"]) for r in valid_rows], dtype=np.int64),
        "true_regression": np.asarray([float(r["true_regression"]) for r in valid_rows], dtype=np.float64),
        "regression": np.asarray([float(r["regression_prediction"]) for r in valid_rows], dtype=np.float64),
        "confidence": np.asarray([float(r["confidence"]) for r in valid_rows], dtype=np.float64),
    }
    split = {"id": [r["id"] for r in valid_rows]}
    summary = write_validation_error_analysis(out, split, prediction)
    history_payload = json.loads((out / "training_history.json").read_text(encoding="utf-8"))
    cards = json.loads((out / "attachment4_explanations.json").read_text(encoding="utf-8")) if (out / "attachment4_explanations.json").exists() else []
    plot_outputs(out, history_payload.get("history", []), prediction, cards, summary)

    # Recompute all 50 local occlusion drops for each Attachment 4 sample.
    # The original cards keep only the top-k candidates for compactness; this
    # second pass supplies a complete, reproducible distribution for figures.
    stats = json.loads((out / "normalization.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(out / "problem3_checkpoint.pt", map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ExplainableFusionNet(
        hidden=int(config.get("hidden", 160)), heads=int(config.get("heads", 8)),
        layers=int(config.get("layers", 2)), dropout=float(config.get("dropout", 0.15)),
    ).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    card_by_key = {(str(card["variant"]), str(card["id"])): card for card in cards}
    aggregate = np.zeros((len(MODALITIES), 50), dtype=np.float64)
    counts = np.zeros(len(MODALITIES), dtype=np.int64)
    sample_rows = []
    for variant in ("aligned", "unaligned"):
        records = load_attachment4(args.data_root / "attachment_4_explainability", stats, variant)
        for record in records:
            card = card_by_key.get((variant, str(record["id"])))
            if card is None:
                continue
            importance = full_local_occlusion(model, record, device, span=max(int(args.occlusion_span), 1))
            main_modality = card["main_modality"]
            main_index = MODALITIES.index(main_modality)
            raw_drop = importance["drops"][main_index]
            positive = np.maximum(raw_drop, 0.0)
            total = float(positive.sum())
            normalized = positive / total if total > 1e-12 else np.zeros(50, dtype=np.float64)
            counts[main_index] += 1
            aggregate[main_index] += normalized
            for aligned_bin in range(50):
                sample_rows.append({
                    "variant": variant, "id": str(record["id"]), "main_modality": main_modality,
                    "aligned_bin": aligned_bin, "occlusion_drop": float(raw_drop[aligned_bin]),
                    "positive_occlusion_drop": float(positive[aligned_bin]),
                    "normalized_local_importance": float(normalized[aligned_bin]),
                })
    for mi in range(len(MODALITIES)):
        if counts[mi] > 0: aggregate[mi] /= counts[mi]
    write_rows(out / "primary_modality_local_importance_samples.csv", sample_rows)
    write_rows(out / "primary_modality_local_importance.csv", [
        {"modality": modality, "aligned_bin": aligned_bin,
         "mean_normalized_occlusion": float(aggregate[mi, aligned_bin]),
         "sample_count": int(counts[mi])}
        for mi, modality in enumerate(MODALITIES) for aligned_bin in range(50)
    ])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Times New Roman", "axes.unicode_minus": False, "font.size": 10})
    fig_dir = out / "figures"; fig_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True, constrained_layout=True)
    colors = ["#355C7D", "#C06C84", "#6C9A8B"]; bins = np.arange(50)
    for mi, (ax, modality) in enumerate(zip(axes, MODALITIES)):
        ax.plot(bins, aggregate[mi], color=colors[mi], linewidth=2.0)
        ax.fill_between(bins, aggregate[mi], color=colors[mi], alpha=.18)
        ax.set_ylabel(f"{modality.capitalize()}\n(n={int(counts[mi])})"); ax.grid(alpha=.22)
    axes[-1].set_xlabel("Aligned time bin")
    fig.savefig(fig_dir / "primary_modality_local_importance.png", dpi=300); fig.savefig(fig_dir / "primary_modality_local_importance.pdf"); plt.close(fig)
    print(json.dumps({"output": str(out), "validation_samples": len(valid_rows), "attachment4_cards": len(cards), "error_count": summary["error_count"], "full_occlusion_rows": len(sample_rows), "primary_modality_counts": {m: int(counts[i]) for i, m in enumerate(MODALITIES)}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
