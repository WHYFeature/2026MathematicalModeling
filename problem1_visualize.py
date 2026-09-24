#!/usr/bin/env python3
"""Create publication-style diagnostics for Problem 1 aligned features."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


COLORS = {
    "text": "#2563EB",
    "audio": "#D97706",
    "vision": "#059669",
    "negative": "#DC2626",
    "neutral": "#64748B",
    "positive": "#7C3AED",
}


def style():
    plt.rcParams.update({
        "figure.facecolor": "#F8FAFC",
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#CBD5E1",
        "axes.labelcolor": "#1E293B",
        "axes.titlecolor": "#0F172A",
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "text.color": "#0F172A",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.color": "#E2E8F0",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.85,
        "legend.frameon": False,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
    })


def save(fig, path):
    fig.savefig(path, facecolor=fig.get_facecolor())
    # Keep historical root-level copies synchronized with figures/.
    if path.parent.name == "figures":
        fig.savefig(path.parent.parent / path.name, facecolor=fig.get_facecolor())
    plt.close(fig)


def robust_z(x):
    x = np.asarray(x, dtype=np.float64)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = 1.4826 * mad
    if scale < 1e-8:
        scale = np.nanstd(x) or 1.0
    return (x - med) / scale


def boxplot_with_labels(ax, data, labels):
    """Matplotlib 3.8/3.9 compatible boxplot helper."""
    bp = ax.boxplot(data, patch_artist=True, showfliers=False)
    ax.set_xticks(range(1, len(labels) + 1), labels)
    return bp


def load_data(output_root):
    with (output_root / "problem1_aligned_features.pkl").open("rb") as f:
        payload = pickle.load(f)
    samples = payload["samples"]
    return payload, samples


def arrays(samples):
    labels = np.asarray([float(s["label"]) for s in samples])
    durations = np.asarray([float(s["duration_seconds"]) for s in samples])
    coverage = {
        m: np.stack([np.asarray(s["coverage"][m], dtype=np.float64) for s in samples])
        for m in ("text", "audio", "vision")
    }
    valid = {
        m: np.stack([np.asarray(s["valid_masks"][m], dtype=bool) for s in samples])
        for m in ("text", "audio", "vision")
    }
    norms = {
        m: np.stack([
            np.linalg.norm(np.asarray(s[m], dtype=np.float64), axis=1)
            for s in samples
        ])
        for m in ("text", "audio", "vision")
    }
    source_counts = {
        m: np.asarray([
            [len(x) for x in s["source_indices"][m]] for s in samples
        ], dtype=np.float64)
        for m in ("text", "audio", "vision")
    }
    return labels, durations, coverage, valid, norms, source_counts


def make_overview(out, labels, durations, coverage, valid):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2), constrained_layout=True)
    fig.suptitle("Problem 1 · Dataset and Alignment Overview", fontsize=16, fontweight="bold", x=0.03, ha="left")

    ax = axes[0, 0]
    names = ["Negative", "Neutral", "Positive"]
    vals = [int(np.sum(labels < 0)), int(np.sum(labels == 0)), int(np.sum(labels > 0))]
    bars = ax.bar(names, vals, color=[COLORS["negative"], COLORS["neutral"], COLORS["positive"]], width=0.62)
    ax.set_title("Annotation distribution")
    ax.set_ylabel("Number of clips")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.025, str(v), ha="center", fontweight="bold")

    ax = axes[0, 1]
    ax.hist(durations, bins=12, color="#475569", alpha=0.88, edgecolor="#FFFFFF")
    ax.axvline(np.median(durations), color=COLORS["positive"], lw=2, label=f"Median {np.median(durations):.2f}s")
    ax.set_title("Clip duration distribution")
    ax.set_xlabel("Duration (s)")
    ax.set_ylabel("Number of clips")
    ax.legend()

    ax = axes[1, 0]
    x = np.arange(1, coverage["text"].shape[1] + 1)
    for m in ("text", "audio", "vision"):
        mean = coverage[m].mean(axis=0)
        lo = coverage[m].mean(axis=0) - coverage[m].std(axis=0)
        hi = coverage[m].mean(axis=0) + coverage[m].std(axis=0)
        ax.plot(x, mean, lw=2.2, color=COLORS[m], label=m.title())
        ax.fill_between(x, np.clip(lo, 0, 1), np.clip(hi, 0, 1), color=COLORS[m], alpha=0.12)
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("Mean temporal coverage across 50 bins")
    ax.set_xlabel("Aligned time bin")
    ax.set_ylabel("Coverage")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.34))

    ax = axes[1, 1]
    means = [float(valid[m].mean()) for m in ("text", "audio", "vision")]
    bars = ax.bar(["Text", "Audio", "Vision"], means, color=[COLORS["text"], COLORS["audio"], COLORS["vision"]], width=0.62)
    ax.set_ylim(0, 1.08)
    ax.set_title("Valid-bin ratio")
    ax.set_ylabel("Fraction of bins")
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.025, f"{v:.3f}", ha="center", fontweight="bold")
    save(fig, out / "problem1_overview.png")


def make_alignment(out, samples, norms, coverage, valid, source_counts):
    durations = np.asarray([s["duration_seconds"] for s in samples])
    idx = int(np.argsort(durations)[len(durations) // 2])
    sample = samples[idx]
    fig, axes = plt.subplots(3, 1, figsize=(13, 9.0), sharex=True, constrained_layout=True)
    sample_label = str(sample["id"]).replace("$", r"\$")
    fig.suptitle(f"Problem 1 · Temporal Alignment Example · {sample_label}", fontsize=15, fontweight="bold", x=0.03, ha="left")
    x = np.linspace(0, float(sample["duration_seconds"]), 50, endpoint=False)
    width = float(sample["duration_seconds"]) / 50

    ax = axes[0]
    heat = np.vstack([robust_z(np.where(valid[m][idx], norms[m][idx], np.nan)) for m in ("text", "audio", "vision")])
    masked_heat = np.ma.masked_invalid(heat)
    cmap = plt.get_cmap("RdYlBu_r").copy()
    cmap.set_bad("#E2E8F0")
    im = ax.imshow(masked_heat, aspect="auto", cmap=cmap, origin="lower", extent=[0, float(sample["duration_seconds"]), -0.5, 2.5], interpolation="nearest")
    ax.set_yticks([0, 1, 2], ["Text", "Audio", "Vision"])
    ax.set_ylabel("Modality")
    ax.set_title("Robust z-score of per-bin feature norm")
    cbar = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.02)
    cbar.set_label("Robust z-score")

    ax = axes[1]
    for m in ("text", "audio", "vision"):
        ax.plot(x + width / 2, coverage[m][idx], color=COLORS[m], lw=2.2, marker="o", ms=3.5, label=m.title())
    ax.set_ylim(-0.03, 1.05)
    ax.set_ylabel("Coverage")
    ax.set_title("Per-bin temporal coverage")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.4))

    ax = axes[2]
    for m in ("text", "audio", "vision"):
        ax.plot(x + width / 2, source_counts[m][idx], color=COLORS[m], lw=2.0, label=m.title())
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Source intervals")
    ax.set_title("Number of source intervals contributing to each bin")
    ax.set_xlim(0, float(sample["duration_seconds"]))
    save(fig, out / "problem1_alignment_example.png")
    return idx


def make_feature_statistics(out, norms, source_counts, valid):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), constrained_layout=True)
    fig.suptitle("Problem 1 · Feature Quality Diagnostics", fontsize=15, fontweight="bold", x=0.03, ha="left")

    ax = axes[0]
    data = [norms[m][valid[m]] for m in ("text", "audio", "vision")]
    bp = boxplot_with_labels(ax, data, ["Text", "Audio", "Vision"])
    for patch, m in zip(bp["boxes"], ("text", "audio", "vision")):
        patch.set_facecolor(COLORS[m])
        patch.set_alpha(0.25)
        patch.set_edgecolor(COLORS[m])
    for med, m in zip(bp["medians"], ("text", "audio", "vision")):
        med.set_color(COLORS[m])
        med.set_linewidth(2)
    ax.set_title("Per-bin feature norm")
    ax.set_ylabel("L2 norm")
    ax.set_yscale("log")

    ax = axes[1]
    data = [source_counts[m][valid[m]] for m in ("text", "audio", "vision")]
    bp = boxplot_with_labels(ax, data, ["Text", "Audio", "Vision"])
    for patch, m in zip(bp["boxes"], ("text", "audio", "vision")):
        patch.set_facecolor(COLORS[m])
        patch.set_alpha(0.25)
        patch.set_edgecolor(COLORS[m])
    ax.set_title("Source intervals per aligned bin")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    save(fig, out / "problem1_feature_statistics.png")


def make_label_duration(out, labels, durations):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), constrained_layout=True)
    fig.suptitle("Problem 1 · Label and Clip Duration Relationship", fontsize=15, fontweight="bold", x=0.03, ha="left")
    ax = axes[0]
    groups = [(labels < 0, "Negative", COLORS["negative"]), (labels == 0, "Neutral", COLORS["neutral"]), (labels > 0, "Positive", COLORS["positive"])]
    for mask, name, color in groups:
        ax.scatter(durations[mask], labels[mask], s=42, alpha=0.82, color=color, edgecolor="#FFFFFF", linewidth=0.6, label=name)
    ax.axhline(0, color="#94A3B8", lw=1)
    ax.set_xlabel("Duration (s)")
    ax.set_ylabel("Sentiment intensity label")
    ax.set_title("Continuous label versus duration")
    ax.legend()

    ax = axes[1]
    bins = np.linspace(-3, 3, 13)
    ax.hist(labels, bins=bins, color=COLORS["positive"], alpha=0.78, edgecolor="#FFFFFF")
    ax.axvline(0, color="#64748B", lw=1.4)
    ax.set_xlabel("Sentiment intensity label")
    ax.set_ylabel("Number of clips")
    ax.set_title("Regression-label distribution")
    save(fig, out / "problem1_label_duration.png")


def write_report(out, payload, samples, labels, durations, coverage, valid, norms, source_counts, example_idx):
    report = {
        "format_version": payload.get("format_version"),
        "sample_count": len(samples),
        "dimensions": payload.get("dims"),
        "duration_seconds": {"min": float(durations.min()), "median": float(np.median(durations)), "max": float(durations.max()), "mean": float(durations.mean())},
        "label_counts": {"negative": int(np.sum(labels < 0)), "neutral": int(np.sum(labels == 0)), "positive": int(np.sum(labels > 0))},
        "coverage_mean": {m: float(coverage[m].mean()) for m in coverage},
        "valid_bin_ratio": {m: float(valid[m].mean()) for m in valid},
        "feature_norm_median": {m: float(np.median(norms[m])) for m in norms},
        "source_intervals_median": {m: float(np.median(source_counts[m])) for m in source_counts},
        "example_id": samples[example_idx]["id"],
    }
    (out / "problem1_analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 问题一实验结果分析",
        "",
        f"- 样本数：{report['sample_count']}；输出格式：`{report['format_version']}`。",
        f"- 三模态维度：text={payload['dims']['text']}，audio={payload['dims']['audio']}，vision={payload['dims']['vision']}。",
        f"- 视频时长：{report['duration_seconds']['min']:.3f}s–{report['duration_seconds']['max']:.3f}s，中位数 {report['duration_seconds']['median']:.3f}s。",
        f"- 标签分布：Negative {report['label_counts']['negative']}，Neutral {report['label_counts']['neutral']}，Positive {report['label_counts']['positive']}。",
        "",
        "## 时间对齐质量",
        "",
        "文本使用 CTC 强制对齐后再进行 50 段等时长区间的重叠加权池化；音频和视觉特征使用各自源区间直接池化。",
        "",
    ]
    for m in ("text", "audio", "vision"):
        lines.append(f"- {m.title()}：平均覆盖率 {report['coverage_mean'][m]:.4f}，有效 bin 比例 {report['valid_bin_ratio'][m]:.4f}，每 bin 源区间中位数 {report['source_intervals_median'][m]:.1f}。")
    lines += [
        "",
        "## 文件",
        "",
        "- `figures/problem1_overview.png`：数据规模、标签和时间覆盖概览。",
        "- `figures/problem1_alignment_example.png`：中位时长样本的逐 bin 对齐诊断。",
        "- `figures/problem1_feature_statistics.png`：三模态特征范数和源区间数量。",
        "- `figures/problem1_label_duration.png`：情感强度标签与视频时长关系。",
        "- `problem1_analysis.json`：可复用的统计数值。",
    ]
    (out / "problem1_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", type=Path, default=Path("D:/E_math/problem1_outputs"))
    args = p.parse_args()
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    style()
    payload, samples = load_data(out)
    labels, durations, coverage, valid, norms, source_counts = arrays(samples)
    make_overview(figures, labels, durations, coverage, valid)
    example_idx = make_alignment(figures, samples, norms, coverage, valid, source_counts)
    make_feature_statistics(figures, norms, source_counts, valid)
    make_label_duration(figures, labels, durations)
    write_report(out, payload, samples, labels, durations, coverage, valid, norms, source_counts, example_idx)
    print(json.dumps({"output_root": str(out), "sample_count": len(samples), "example_id": samples[example_idx]["id"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
