"""Complete the auditable Problem 1 report from an existing extraction.

The report is deliberately separate from extraction: it validates the PKL,
creates the required 100 x 3 summary table, records provenance, and makes one
representative figure using the real source waveform, frames and transcript.
It does not train a model or re-extract all 100 videos.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data import read_samples, sha256
from .pipeline import executable


MODALITIES = ("text", "audio", "vision")
FEATURE_SCHEMA = {
    "text": {
        "dimension": 768,
        "source": "local BERT last_hidden_state",
        "shape": "(50, 768)",
        "units": "dimensionless embedding",
        "time_axis": "50 equal-duration bins after token/word interval pooling",
    },
    "audio": {
        "dimension": 74,
        "source": "mono FFmpeg waveform, NumPy short-time spectral descriptors",
        "shape": "(50, 74)",
        "units": "mixed: waveform amplitude, FFT-bin descriptors, band magnitudes and first differences",
        "time_axis": "50 equal-duration bins after overlap-weighted pooling of 25 ms frames every 10 ms",
        "features": [
            "rms", "mean_abs", "std", "peak_abs", "sign_change_rate",
            "crest_factor", "spectral_centroid_fft_bin", "spectral_spread_fft_bin",
            "low_85pct_spectral_mass", "spectral_flatness",
            "linear_band_mean_01..32", "delta_linear_band_mean_01..32",
        ],
    },
    "vision": {
        "dimension": 35,
        "source": "FFmpeg RGB frames resized to 64x64, NumPy full-frame descriptors",
        "shape": "(50, 35)",
        "units": "normalized RGB/gray statistics and histogram density",
        "time_axis": "50 equal-duration bins after overlap-weighted pooling of 5 fps frame intervals",
        "features": [
            "rgb_mean_01..03", "rgb_std_01..03", "gray_mean", "gray_std",
            "gray_min", "gray_max", "gray_percentile_05/25/50/75/95",
            "abs_horizontal_diff_mean", "abs_vertical_diff_mean",
            "abs_horizontal_diff_std", "abs_vertical_diff_std", "gray_hist_density_01..16",
        ],
        "limitation": "These are full-frame appearance/texture descriptors, not facial landmarks or a facial emotion model.",
    },
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Validate and report Problem 1 outputs.")
    parser.add_argument("--data-root", type=Path, default=Path("D:/E_math/DATA"))
    parser.add_argument("--output-root", type=Path, default=Path("D:/E_math/problem1_outputs"))
    parser.add_argument("--example-id", default=None, help="Use a specific id such as video$_$clip.")
    parser.add_argument("--no-media", action="store_true", help="Only write tables and metadata; skip waveform/frame figure.")
    return parser.parse_args(argv)


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def load_payload(output: Path):
    with (output / "problem1_aligned_features.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise ValueError("The feature file is not a Problem 1 payload")
    return payload


def read_manifest(path: Path):
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["id"]: row for row in csv.DictReader(handle)}


def validate(payload, label_samples, manifest):
    expected = {s["id"] for s in label_samples}
    actual = {s.get("id") for s in payload["samples"]}
    issues = []
    if len(payload["samples"]) != 100:
        issues.append(f"feature sample count is {len(payload['samples'])}, expected 100")
    if expected != actual:
        issues.append(f"label/feature id mismatch: {sorted(expected ^ actual)[:5]}")
    dims = payload.get("dims", {})
    expected_dims = {"text": 768, "audio": 74, "vision": 35}
    if dims != expected_dims:
        issues.append(f"unexpected dimensions: {dims}")
    shape_counts = {m: 0 for m in MODALITIES}
    finite_counts = {m: 0 for m in MODALITIES}
    mask_counts = {m: 0 for m in MODALITIES}
    interval_counts = {m: 0 for m in MODALITIES}
    for sample in payload["samples"]:
        edges = np.asarray(sample.get("time_edges"), dtype=np.float64)
        if edges.shape != (51,) or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
            issues.append(f"invalid time_edges: {sample.get('id')}")
        for m in MODALITIES:
            x = np.asarray(sample.get(m))
            if x.shape != (50, expected_dims[m]):
                issues.append(f"{sample.get('id')} {m} shape={x.shape}")
            else:
                shape_counts[m] += 1
            if not np.isfinite(x).all():
                issues.append(f"{sample.get('id')} {m} contains non-finite values")
            else:
                finite_counts[m] += 1
            mask = np.asarray(sample.get("valid_masks", {}).get(m), dtype=bool)
            cov = np.asarray(sample.get("coverage", {}).get(m), dtype=np.float64)
            if mask.shape != (50,) or cov.shape != (50,) or np.any(cov < -1e-6) or np.any(cov > 1 + 1e-6):
                issues.append(f"{sample.get('id')} {m} mask/coverage invalid")
            else:
                mask_counts[m] += int(mask.sum())
            interval = sample.get("source_intervals", {}).get(m)
            if interval is not None:
                interval = np.asarray(interval)
                if interval.ndim != 2 or interval.shape[1] != 2:
                    issues.append(f"{sample.get('id')} {m} source intervals invalid")
                else:
                    interval_counts[m] += len(interval)
    return {
        "ok": not issues,
        "issues": issues,
        "sample_count": len(payload["samples"]),
        "id_count_from_labels": len(expected),
        "shape_counts": shape_counts,
        "finite_counts": finite_counts,
        "valid_bin_counts": mask_counts,
        "source_interval_counts": interval_counts,
        "manifest_rows": len(manifest),
        "source_intervals_present_in_payload": any(interval_counts.values()),
    }


def summary_rows(payload, manifest):
    rows = []
    for sample in payload["samples"]:
        duration = float(sample["duration_seconds"])
        for modality in MODALITIES:
            x = np.asarray(sample[modality])
            mask = np.asarray(sample["valid_masks"][modality], dtype=bool)
            cov = np.asarray(sample["coverage"][modality], dtype=np.float64)
            source_indices = sample.get("source_indices", {}).get(modality, [])
            interval = sample.get("source_intervals", {}).get(modality)
            if interval is not None:
                source_rows = int(len(interval))
            else:
                # Legacy PKL files store per-bin lists. Count the union of
                # indices across bins, not the largest single-bin list.
                source_rows = len({int(i) for subset in source_indices for i in subset})
            span_start = ""
            span_end = ""
            if interval is not None and len(interval):
                a = np.asarray(interval, dtype=np.float64)
                span_start, span_end = f"{a[:, 0].min():.6f}", f"{a[:, 1].max():.6f}"
            manifest_row = manifest.get(sample["id"], {})
            rows.append({
                "id": sample["id"],
                "video_id": sample["video_id"],
                "clip_id": sample["clip_id"],
                "modality": modality,
                "annotation": sample["annotation"],
                "label": f"{float(sample['label']):.6g}",
                "duration_seconds": f"{duration:.6f}",
                "raw_feature_rows": source_rows,
                "raw_feature_dim": int(x.shape[1]),
                "aligned_shape": f"{x.shape[0]}x{x.shape[1]}",
                "alignment_granularity": "50 equal-duration bins",
                "valid_bins": int(mask.sum()),
                "mean_coverage": f"{cov.mean():.6f}",
                "source_interval_start_seconds": span_start,
                "source_interval_end_seconds": span_end,
                "text_alignment": sample.get("text_alignment", ""),
                "backend": manifest_row.get(f"{modality}_backend", ""),
                "video_path": manifest_row.get("video_path", ""),
                "video_sha256": manifest_row.get("sha256", ""),
            })
    return rows


def text_truncation_stats(payload, output: Path):
    """Measure loss caused by the fixed 50-position BERT interface.

    Problem 1 keeps 50 positions to match the required aligned-file schema.
    This audit reports how many original BERT token positions would exceed that
    interface; it does not silently change the feature shape.
    """
    run_meta = {}
    metadata_path = output / "run_metadata.json"
    if metadata_path.is_file():
        run_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    limit = int(run_meta.get("parameters", {}).get("max_text_tokens", 50))
    model_name = str(run_meta.get("text_backend", ""))
    model_path = model_name.split(":", 1)[1] if model_name.startswith("transformers:") else ""
    tokenizer = None
    method = "unavailable"
    try:
        from transformers import AutoTokenizer
        if model_path:
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            method = f"transformers:{model_path}"
    except Exception:
        tokenizer = None
    rows = []
    for sample in payload["samples"]:
        text = str(sample.get("raw_text", ""))
        if tokenizer is not None:
            encoded = tokenizer(text, add_special_tokens=True, truncation=False, padding=False)
            full_tokens = int(len(encoded["input_ids"]))
        else:
            full_tokens = None
        bert_meta = np.asarray(sample.get("text_bert"))
        stored_tokens = int(np.asarray(bert_meta[1], dtype=bool).sum()) if bert_meta.shape[0] >= 2 else None
        truncated = bool(full_tokens is not None and full_tokens > limit)
        rows.append({
            "id": sample["id"],
            "raw_word_count": len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)),
            "full_bert_tokens": full_tokens if full_tokens is not None else "",
            "stored_bert_tokens": stored_tokens if stored_tokens is not None else "",
            "token_limit": limit,
            "omitted_bert_tokens": max(0, full_tokens - limit) if full_tokens is not None else "",
            "truncated": truncated if full_tokens is not None else "unknown",
            "raw_text": text,
        })
    known = [r for r in rows if isinstance(r["full_bert_tokens"], int)]
    summary = {
        "method": method,
        "token_limit": limit,
        "sample_count": len(rows),
        "tokenizer_available": tokenizer is not None,
        "truncated_sample_count": sum(bool(r["truncated"]) for r in known),
        "truncated_sample_ratio": (sum(bool(r["truncated"]) for r in known) / len(known)) if known else None,
        "max_full_bert_tokens": max((r["full_bert_tokens"] for r in known), default=None),
        "total_omitted_bert_tokens": sum(int(r["omitted_bert_tokens"]) for r in known),
        "rows": rows,
    }
    write_json(output / "problem1_text_truncation.json", {k: v for k, v in summary.items() if k != "rows"})
    write_csv(output / "problem1_text_truncation.csv", rows)
    return summary


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def decode_audio(path: Path, rate=16000):
    raw = subprocess.run([executable("ffmpeg"), "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(rate), "-f", "f32le", "-"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32), rate


def decode_video(path: Path, duration: float, fps=5.0):
    raw = subprocess.run([executable("ffmpeg"), "-v", "error", "-i", str(path), "-vf", f"fps={fps},scale=96:96", "-frames:v", str(max(1, math.ceil(duration * fps))), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
    size = 96 * 96 * 3
    arr = np.frombuffer(raw[: len(raw) // size * size], dtype=np.uint8)
    return arr.reshape(-1, 96, 96, 3), np.arange(len(arr) // size, dtype=np.float64) / fps


def choose_example(samples, requested=None):
    if requested:
        for i, sample in enumerate(samples):
            if sample["id"] == requested:
                return i
        raise ValueError(f"Example id not found: {requested}")
    candidates = [i for i, s in enumerate(samples) if s.get("text_word_spans")]
    if not candidates:
        candidates = list(range(len(samples)))
    durations = np.asarray([float(samples[i]["duration_seconds"]) for i in candidates])
    return candidates[int(np.argsort(durations)[len(candidates) // 2])]


def make_example(output: Path, sample, source_by_id):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    media = source_by_id[sample["id"]]
    wave, rate = decode_audio(media["path"])
    frames, frame_times = decode_video(media["path"], float(sample["duration_seconds"]))
    duration = float(sample["duration_seconds"])
    spans = sample.get("text_word_spans", [])
    words = []
    for item in spans:
        words.append({"word": str(item.get("word", "")), "start": float(item.get("start", 0)), "end": float(item.get("end", 0)), "confidence": float(item.get("confidence", 0))})
    word_csv = [{"id": sample["id"], "word_index": i, **w} for i, w in enumerate(words)]
    write_csv(output / "typical_sample_words.csv", word_csv)

    edges = np.asarray(sample["time_edges"], dtype=np.float64)
    centers = (edges[:-1] + edges[1:]) / 2
    text_norm = np.linalg.norm(np.asarray(sample["text"], dtype=np.float64), axis=1)
    audio_rms = np.asarray(sample["audio"], dtype=np.float64)[:, 0]
    vision_red = np.asarray(sample["vision"], dtype=np.float64)[:, 0]
    bin_rows = []
    for i, t in enumerate(centers):
        bin_rows.append({
            "id": sample["id"], "bin": i + 1, "start_seconds": f"{edges[i]:.6f}", "end_seconds": f"{edges[i + 1]:.6f}",
            "text_norm": f"{text_norm[i]:.8g}", "audio_rms": f"{audio_rms[i]:.8g}", "vision_rgb_mean_1": f"{vision_red[i]:.8g}",
            "text_coverage": f"{float(sample['coverage']['text'][i]):.6f}", "audio_coverage": f"{float(sample['coverage']['audio'][i]):.6f}", "vision_coverage": f"{float(sample['coverage']['vision'][i]):.6f}",
            "text_source_count": len(sample["source_indices"]["text"][i]), "audio_source_count": len(sample["source_indices"]["audio"][i]), "vision_source_count": len(sample["source_indices"]["vision"][i]),
        })
    write_csv(output / "typical_sample_bins.csv", bin_rows)

    fig = plt.figure(figsize=(15, 11), facecolor="#F8FAFC")
    gs = GridSpec(4, 1, figure=fig, height_ratios=[1.15, 1.25, 1.0, 1.35], hspace=0.42)
    color = {"text": "#2563EB", "audio": "#D97706", "vision": "#059669"}
    ax = fig.add_subplot(gs[0])
    time = np.arange(len(wave), dtype=np.float64) / rate
    # Downsample only for drawing; the CSV remains the full decoded waveform.
    stride = max(1, len(wave) // 30000)
    ax.plot(time[::stride], wave[::stride], color="#475569", lw=0.55)
    ax.set_xlim(0, duration); ax.set_ylabel("Amplitude"); ax.set_title("Actual mono waveform with CTC word intervals", loc="left", fontweight="bold")
    for w in words:
        ax.axvspan(w["start"], w["end"], color="#2563EB", alpha=0.12)
        if w["end"] - w["start"] > duration / 35:
            ax.text((w["start"] + w["end"]) / 2, ax.get_ylim()[1] * 0.82, w["word"], rotation=60, ha="center", va="bottom", fontsize=7, color="#1E3A8A")
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[1])
    for name, values in (("text L2 norm", text_norm), ("audio RMS", audio_rms), ("vision RGB mean 1", vision_red)):
        values = np.asarray(values, dtype=np.float64)
        scale = np.nanpercentile(np.abs(values), 95) or 1.0
        ax.plot(centers, values / scale, lw=1.8, color=color["text" if name.startswith("text") else "audio" if name.startswith("audio") else "vision"], label=f"{name} / p95")
    for w in words:
        ax.axvspan(w["start"], w["end"], color="#94A3B8", alpha=0.06)
    ax.set_xlim(0, duration); ax.set_ylim(-0.05, 1.15); ax.set_ylabel("Normalized value"); ax.set_title("Three stored aligned feature traces (one diagnostic dimension per modality)", loc="left", fontweight="bold"); ax.legend(ncol=3, fontsize=8); ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[2])
    ax.plot(centers, sample["coverage"]["text"], color=color["text"], label="Text coverage")
    ax.plot(centers, sample["coverage"]["audio"], color=color["audio"], label="Audio coverage")
    ax.plot(centers, sample["coverage"]["vision"], color=color["vision"], label="Vision coverage")
    ax.set_xlim(0, duration); ax.set_ylim(-0.03, 1.05); ax.set_ylabel("Coverage"); ax.set_title("Overlap coverage of the 50 equal-duration bins", loc="left", fontweight="bold"); ax.legend(ncol=3, fontsize=8); ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[3])
    ax.axis("off")
    if len(frames):
        show = np.linspace(0, len(frames) - 1, min(6, len(frames)), dtype=int)
        for j, frame_index in enumerate(show):
            inset = ax.inset_axes([j / len(show) + 0.006, 0.07, 0.15, 0.80])
            inset.imshow(frames[frame_index]); inset.axis("off")
            inset.set_title(f"t={frame_times[frame_index]:.2f}s", fontsize=8)
    ax.set_title("Actual RGB frames decoded from the same source video", loc="left", fontweight="bold")
    safe_id = sample["id"].replace("$", "dollar")
    display_id = str(sample["id"]).replace("$", r"\$")
    fig.suptitle(f"Problem 1 typical sample: {display_id}  |  {sample['annotation']}  |  {duration:.2f}s", x=0.02, ha="left", fontsize=15, fontweight="bold")
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "problem1_typical_sample.png", dpi=240, bbox_inches="tight")
    fig.savefig(figures / "problem1_typical_sample.pdf", bbox_inches="tight")
    plt.close(fig)
    return {"id": sample["id"], "duration_seconds": duration, "word_count": len(words), "decoded_audio_samples": int(len(wave)), "decoded_video_frames": int(len(frames)), "figure": "figures/problem1_typical_sample.png", "word_table": "typical_sample_words.csv", "bin_table": "typical_sample_bins.csv"}


def environment_snapshot(output: Path, data_root: Path, payload, report_version="1.0"):
    packages = {}
    for name in ("numpy", "openpyxl", "matplotlib", "torch", "torchaudio", "transformers", "imageio-ffmpeg"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        ffmpeg_text = subprocess.run([executable("ffmpeg"), "-version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True).stdout.decode(errors="replace").splitlines()[0]
    except Exception as exc:
        ffmpeg_text = f"unavailable: {exc}"
    run_meta = {}
    if (output / "run_metadata.json").is_file():
        run_meta = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
    snapshot = {
        "report_version": report_version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages_at_report_time": packages,
        "ffmpeg": ffmpeg_text,
        "cuda_available_at_report_time": None,
        "feature_payload_sha256": sha256(output / "problem1_aligned_features.pkl"),
        "label_file_sha256": sha256(data_root / "attachment_1_raw_samples" / "labels_100.xlsx"),
        "recorded_extraction_metadata": run_meta,
        "provenance_note": "Package versions are measured when this report is generated. If the historical extraction metadata omitted versions, this report does not retroactively claim they were present at extraction time.",
    }
    try:
        import torch
        snapshot["cuda_available_at_report_time"] = bool(torch.cuda.is_available())
        snapshot["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    write_json(output / "problem1_environment.json", snapshot)
    return snapshot


def write_report(output: Path, payload, validation, rows, example_info, environment, truncation):
    labels = np.asarray([float(s["label"]) for s in payload["samples"]])
    durations = np.asarray([float(s["duration_seconds"]) for s in payload["samples"]])
    coverage = {m: np.asarray([np.asarray(s["coverage"][m], dtype=float).mean() for s in payload["samples"]]) for m in MODALITIES}
    report = {
        "format_version": payload.get("format_version"), "sample_count": len(payload["samples"]), "dimensions": payload.get("dims"),
        "duration_seconds": {"min": float(durations.min()), "median": float(np.median(durations)), "mean": float(durations.mean()), "max": float(durations.max())},
        "label_counts": {"Negative": int(np.sum(labels < 0)), "Neutral": int(np.sum(labels == 0)), "Positive": int(np.sum(labels > 0))},
        "mean_per_sample_coverage": {m: float(coverage[m].mean()) for m in MODALITIES},
        "valid_bin_ratio": {m: float(np.asarray([np.asarray(s["valid_masks"][m], dtype=bool).mean() for s in payload["samples"]]).mean()) for m in MODALITIES},
        "validation": validation, "example": example_info, "environment_file": "problem1_environment.json",
        "text_truncation": {k: v for k, v in truncation.items() if k != "rows"},
    }
    write_json(output / "problem1_complete_analysis.json", report)
    lines = [
        "# 问题一补齐材料与核验报告", "",
        "本报告由 `problem1_report.py` 从已生成的 100 条特征文件读取并生成。它不把覆盖率当作时间戳准确率，也不把未覆盖文本 bin 自动解释成静音。",
        "",
        "## 已交付文件", "",
        "- `problem1_feature_summary.csv`：100 条样本 × 3 个模态的 300 行逐模态汇总表。",
        "- `problem1_feature_schema.json`：三类特征的维度、时间轴、计算定义和限制。",
        "- `problem1_environment.json`：报告生成时的 Python/包/FFmpeg 版本、数据哈希和历史运行参数。",
        "- `problem1_validation.json`：样本 ID、形状、有限值、时间边界、掩码和覆盖率检查。",
        "- `problem1_text_truncation.csv/json`：同一 BERT tokenizer 对固定 50-token 接口的截断审计。",
        "- `typical_sample_words.csv`、`typical_sample_bins.csv`：典型样本的词级时间段与 50 个对齐 bin 的可复核明细。",
        "- `figures/problem1_typical_sample.png` 和 `.pdf`：真实波形、CTC 词段、三模态特征曲线、覆盖率和真实视频帧。",
        "",
        "## 数据规模与测量结果", "",
        f"- 样本数：{report['sample_count']}；特征维度：text={report['dimensions']['text']}，audio={report['dimensions']['audio']}，vision={report['dimensions']['vision']}。",
        f"- 视频时长：{report['duration_seconds']['min']:.3f}–{report['duration_seconds']['max']:.3f} s；均值 {report['duration_seconds']['mean']:.3f} s；中位数 {report['duration_seconds']['median']:.3f} s。",
        f"- 标签：Negative {report['label_counts']['Negative']}、Neutral {report['label_counts']['Neutral']}、Positive {report['label_counts']['Positive']}。",
        "- 每个模态都以 50 个等时长 bin 输出；有效 bin 和覆盖率分别保存在 `valid_masks` 与 `coverage`。",
        f"- 文本 50-token 审计：{truncation['truncated_sample_count']}/{truncation['sample_count']} 条样本超过上限，累计省略 {truncation['total_omitted_bert_tokens']} 个 token。",
    ]
    for m in MODALITIES:
        lines.append(f"- {m}：样本平均 bin 覆盖率 {report['mean_per_sample_coverage'][m]:.4f}；有效 bin 比例 {report['valid_bin_ratio'][m]:.4f}。")
    lines += [
        "", "## 数学处理说明", "",
        "令样本时长为 T，等时长边界为 e_k=kT/50。源特征 x_i 在时间区间 [a_i,b_i) 上定义，和第 k 个 bin 的重叠长度为 w_{ik}=max(0,min(b_i,e_{k+1})-max(a_i,e_k))。输出为 y_k=Σ_i w_{ik}x_i/Σ_i w_{ik}；coverage_k 是这些区间并集长度除以 bin 长度，避免重叠音频窗把覆盖率重复计数。Σ_i w_{ik}=0 时输出零向量且 valid_mask_k=false。",
        "- 文本：本地 BERT 的 768 维 hidden state；CTC 强制对齐产生词级区间，再将词元映射到这些区间。CTC 置信度是声学后验诊断量，不是人工标注的时间误差。",
        "- 音频：16 kHz 单声道；25 ms 窗、10 ms hop；10 个标量谱/波形描述、32 个线性谱带均值和 32 个差分，共 74 维。",
        "- 视觉：5 fps、64×64 RGB；RGB/灰度统计、百分位、横纵差分和 16-bin 灰度直方图，共 35 维。视觉特征是全帧外观描述，不等同于面部关键点或面部情绪识别器。",
        "",
        "## 复现命令", "",
        "```powershell",
        "Set-Location D:\\E_math",
        ".\\run_problem1_report.ps1",
        "```",
        "",
        "如需重做完整特征提取，先运行 `.\\run_problem1.ps1 -Overwrite`，再运行上述报告命令。正式提取必须使用本地 BERT 和默认 CTC；hash fallback 与 uniform alignment 只能用于接口检查。",
        "",
        "## 当前结果的边界", "",
        "- 报告中的覆盖率表示源区间在目标 bin 中的时间占用，不等于 CTC 时间戳已经通过人工标注验证。",
        "- 旧 PKL 如果没有 `source_intervals` 字段，汇总表会明确标空源区间的起止时间，并使用已有 `source_indices` 做可追溯计数；修改后的提取器会把文本、音频和视觉源区间直接写入新 PKL。",
        "- 视频帧时间来自 FFmpeg 的 5 fps 重采样时间轴；原始视频若没有逐帧 PTS 导出，报告不会把它写成更精确的原始帧时间。",
        "- 50 个文本位置是题目对齐文件的固定接口，因此不改变输出形状；若有截断样本，明细中列出省略 token 数，不能把截断后的表示写成完整转写表示。",
        "- 当前 PKL 只代表问题一特征结果，不包含原始视频、模型权重或虚拟环境；最终附件是否小于 50 MB 仍需按提交清单整体核算。",
    ]
    (output / "problem1_complete_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def artifact_manifest(output: Path):
    files = []
    total = 0
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "problem1_artifact_manifest.json":
            size = path.stat().st_size
            total += size
            files.append({"path": str(path.relative_to(output)).replace("\\", "/"), "bytes": size})
    obj = {"root": str(output), "total_bytes": total, "total_megabytes": total / 1024**2, "note": "This is the Problem 1 output directory only; raw videos, model weights and virtual environments are excluded from a competition attachment unless explicitly selected.", "files": files}
    write_json(output / "problem1_artifact_manifest.json", obj)


def main(argv=None):
    args = parse_args(argv)
    data_root = args.data_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = load_payload(output)
    label_samples = read_samples(data_root)
    source_by_id = {s["id"]: s for s in label_samples}
    manifest = read_manifest(output / "manifest.csv")
    validation = validate(payload, label_samples, manifest)
    write_json(output / "problem1_validation.json", validation)
    write_json(output / "problem1_feature_schema.json", {"format_version": payload.get("format_version"), "n_bins": payload.get("n_bins"), "modalities": FEATURE_SCHEMA})
    rows = summary_rows(payload, manifest)
    write_csv(output / "problem1_feature_summary.csv", rows)
    truncation = text_truncation_stats(payload, output)
    example_info = {"id": None}
    if not args.no_media:
        example_index = choose_example(payload["samples"], args.example_id)
        example_info = make_example(output, payload["samples"][example_index], source_by_id)
    environment = environment_snapshot(output, data_root, payload)
    report = write_report(output, payload, validation, rows, example_info, environment, truncation)
    artifact_manifest(output)
    print(json.dumps({"output_root": str(output), "sample_count": report["sample_count"], "summary_rows": len(rows), "example_id": example_info.get("id"), "validation_ok": validation["ok"]}, ensure_ascii=False))
    return 0 if validation["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
