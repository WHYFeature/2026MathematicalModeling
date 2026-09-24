"""Problem 1 raw-video feature extraction and 50-bin alignment."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from .core import ctc_viterbi, pool_intervals
from .data import read_samples, sha256, write_json

TEXT_DIM, AUDIO_DIM, VISION_DIM, N_BINS = 768, 74, 35, 50


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Extract Problem 1 features from the 100 raw videos.")
    default_data = Path("D:/E_math/DATA") if os.name == "nt" else Path("/mnt/d/E_math/DATA")
    default_output = Path("D:/E_math/problem1_outputs") if os.name == "nt" else Path("/mnt/d/E_math/problem1_outputs")
    p.add_argument("--data-root", type=Path, default=default_data)
    p.add_argument("--output-root", type=Path, default=default_output)
    p.add_argument("--text-model", default="bert-base-uncased")
    p.add_argument("--text-backend", choices=("auto", "transformers", "hash"), default="auto")
    p.add_argument("--text-align", choices=("uniform", "ctc"), default="ctc")
    p.add_argument("--max-text-tokens", type=int, default=50)
    p.add_argument("--audio-rate", type=int, default=16000)
    p.add_argument("--vision-fps", type=float, default=5.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--allow-fallback", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def executable(name):
    found = shutil.which(name)
    if found:
        return found
    if name == "ffmpeg":
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
    raise RuntimeError(f"{name} not found; install FFmpeg and add it to PATH")


def run(cmd):
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Executable missing: {cmd[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Command failed: " + " ".join(map(str, cmd)) + "\n" + exc.stderr.decode(errors="replace")[-2000:]) from exc
    return p.stdout


def media_metadata(path):
    try:
        ffprobe = executable("ffprobe")
    except RuntimeError:
        # imageio-ffmpeg bundles ffmpeg but commonly omits ffprobe. The
        # input probe still prints enough information to recover duration
        # and frame rate, although an exact frame count is unavailable.
        ffmpeg = executable("ffmpeg")
        p = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        info = p.stderr.decode(errors="replace")
        dm = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", info)
        fm = re.search(r"(?:(\d+(?:\.\d+)?)|\d+/\d+)\s+fps", info)
        if not dm or not fm:
            raise RuntimeError(f"Cannot read video metadata without ffprobe: {path}")
        duration = int(dm.group(1)) * 3600 + int(dm.group(2)) * 60 + float(dm.group(3))
        fps_text = fm.group(0).split()[0]
        if "/" in fps_text:
            num, den = fps_text.split("/", 1)
            fps = float(num) / max(float(den), 1)
        else:
            fps = float(fps_text)
        frames = int(round(duration * fps))
    else:
        out = run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=duration,r_frame_rate,nb_frames:format=duration", "-of", "json", str(path)])
        payload = json.loads(out.decode())
        stream = payload["streams"][0]
        duration = float(stream.get("duration") or payload.get("format", {}).get("duration") or 0)
        num, den = (str(stream.get("r_frame_rate") or "0/1").split("/", 1))
        fps = float(num) / max(float(den), 1)
        try:
            frames = int(stream.get("nb_frames"))
        except (TypeError, ValueError):
            frames = int(round(duration * fps))
    if duration <= 0 or fps <= 0:
        raise ValueError(f"Invalid metadata: {path}")
    return duration, fps, frames


def hash_vec(text, dim=TEXT_DIM):
    out = np.zeros(dim, np.float32)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=64).digest()
    for i, b in enumerate(digest):
        out[(b + i * 131) % dim] += 1 if b & 1 else -1
    norm = np.linalg.norm(out)
    return out / norm if norm else out


class TextEncoder:
    def __init__(self, backend, model_name, max_tokens, allow_fallback):
        self.max_tokens = max_tokens
        self.backend = ""
        self.tokenizer = self.model = self.torch = None
        if backend in ("auto", "transformers"):
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                self.model = AutoModel.from_pretrained(model_name, local_files_only=True)
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model.to(self.device)
                self.model.eval()
                self.torch = torch
                self.backend = f"transformers:{model_name}"
            except Exception as exc:
                if backend == "transformers" and not allow_fallback:
                    raise RuntimeError(f"Local Transformers model unavailable: {model_name}") from exc
        if not self.backend:
            if backend != "hash" and not allow_fallback:
                raise RuntimeError("No local BERT model; use --allow-fallback only for smoke tests")
            self.backend = "hash_fallback"

    def encode(self, text):
        if self.backend == "hash_fallback":
            words = re.findall(r"[A-Za-z0-9']+", text.lower())[: self.max_tokens - 2]
            tokens = ["[CLS]"] + words + ["[SEP]"]
            ids = np.zeros(self.max_tokens, np.int64)
            mask = np.zeros(self.max_tokens, np.int64)
            vectors = np.zeros((self.max_tokens, TEXT_DIM), np.float32)
            n = min(len(tokens), self.max_tokens)
            ids[:n] = [101] + [int.from_bytes(hashlib.sha1(w.encode()).digest()[:4], "little") % 30000 for w in tokens[1:n-1]] + ([102] if n > 1 else [])
            mask[:n] = 1
            for i, token in enumerate(tokens[:n]):
                vectors[i] = hash_vec(token)
            return vectors, np.stack([ids, mask, np.zeros(self.max_tokens, np.int64)]), mask
        enc = self.tokenizer(text, max_length=self.max_tokens, truncation=True, padding="max_length", return_tensors="pt")
        enc = {key: value.to(self.device) for key, value in enc.items()}
        with self.torch.no_grad():
            vectors = self.model(**enc).last_hidden_state[0].cpu().numpy().astype(np.float32)
        ids = enc["input_ids"][0].cpu().numpy().astype(np.int64)
        mask = enc["attention_mask"][0].cpu().numpy().astype(np.int64)
        typ = enc.get("token_type_ids", self.torch.zeros_like(enc["input_ids"]))[0].cpu().numpy().astype(np.int64)
        return vectors, np.stack([ids, mask, typ]), mask

    def token_time_spans(self, text, mask, word_spans, duration):
        """Map BERT positions to forced-aligned word intervals.

        Special tokens inherit the first/last word interval. Subword tokens
        inherit the interval of the word covered by their offset mapping.
        """
        valid = np.flatnonzero(np.asarray(mask, bool))
        if not len(valid):
            return np.zeros((0, 2), np.float64), valid
        if not word_spans:
            n = len(valid)
            return np.c_[duration * np.arange(n) / n, duration * (np.arange(n) + 1) / n], valid
        words = list(re.finditer(r"[A-Za-z]+(?:'[A-Za-z]+)?", text))
        # CTC can omit words containing symbols outside its vocabulary. Keep
        # the mapping total even when the number of aligned words differs
        # from the tokenizer's word count.
        word_count = len(word_spans)
        if not word_count:
            n = len(valid)
            return np.c_[duration * np.arange(n) / n, duration * (np.arange(n) + 1) / n], valid
        if self.backend == "hash_fallback":
            spans = []
            for i in range(len(valid)):
                wi = min(max(i - 1, 0), len(word_spans) - 1)
                spans.append(word_spans[wi])
            return np.asarray(spans, np.float64), valid
        offsets = self.tokenizer(text, max_length=self.max_tokens, truncation=True, padding="max_length", return_offsets_mapping=True)["offset_mapping"]
        offsets = np.asarray(offsets)
        out = []
        for pos in valid:
            a, b = offsets[pos]
            if b <= a:
                out.append(word_spans[0] if pos == valid[0] else word_spans[-1])
                continue
            hit = [i for i, w in enumerate(words) if w.start() < b and w.end() > a]
            if hit:
                left = min(word_count - 1, int(hit[0] * word_count / max(len(words), 1)))
                right = min(word_count - 1, int(hit[-1] * word_count / max(len(words), 1)))
                out.append([word_spans[left][0], word_spans[right][1]])
            else:
                u = min(word_count - 1, int(pos * word_count / max(self.max_tokens, 1)))
                out.append(word_spans[u])
        return np.asarray(out, np.float64), valid


class CTCAligner:
    """Forced-align supplied transcript words with a local torchaudio CTC model."""
    def __init__(self):
        try:
            import torch
            import torchaudio
        except Exception as exc:
            raise RuntimeError("--text-align ctc requires torch and torchaudio") from exc
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.sample_rate = bundle.sample_rate
        self.model = bundle.get_model().to(self.device).eval()
        self.labels = tuple(bundle.get_labels())
        self.to_id = {x: i for i, x in enumerate(self.labels)}
        if "-" in self.to_id:
            self.blank = self.to_id["-"]
        elif "<blank>" in self.to_id:
            self.blank = self.to_id["<blank>"]
        else:
            self.blank = 0

    def align(self, wave, rate, text):
        if rate != self.sample_rate:
            raise ValueError(f"CTC aligner expects {self.sample_rate} Hz, got {rate}")
        words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.upper())
        if not words:
            return []
        targets, owners = [], []
        for wi, word in enumerate(words):
            for ch in word:
                if ch in self.to_id and self.to_id[ch] != self.blank:
                    targets.append(self.to_id[ch]); owners.append(wi)
            if wi + 1 < len(words) and "|" in self.to_id:
                targets.append(self.to_id["|"]); owners.append(-1)
        if not targets:
            return []
        x = self.torch.from_numpy(wave.astype(np.float32))[None, :].to(self.device)
        with self.torch.inference_mode():
            result = self.model(x)
        if isinstance(result, tuple):
            emissions, lengths = result
        else:
            emissions, lengths = result, None
        frame_count = emissions.shape[1] if lengths is None else int(lengths[0].item())
        logp = self.torch.log_softmax(emissions[0, :frame_count], dim=-1).cpu().numpy()
        frame_spans, confidence = ctc_viterbi(logp, np.asarray(targets), self.blank)
        seconds_per_frame = len(wave) / rate / max(len(logp), 1)
        out = []
        for wi, word in enumerate(words):
            ix = np.flatnonzero(np.asarray(owners) == wi)
            if not len(ix):
                continue
            first, last = frame_spans[ix[0], 0], frame_spans[ix[-1], 1]
            out.append({"word": word, "start": float(first * seconds_per_frame), "end": float(last * seconds_per_frame), "confidence": float(confidence[ix].mean())})
        return out


def audio_features(wave, rate):
    frame = max(1, int(rate * 0.025)); hop = max(1, int(rate * 0.010))
    if len(wave) < frame:
        wave = np.pad(wave, (0, frame - len(wave)))
    starts = np.arange(0, max(1, len(wave) - frame + 1), hop, dtype=np.int64)
    all_values = []
    for start in starts:
        x = wave[start:start + frame].astype(np.float64)
        if len(x) < frame: x = np.pad(x, (0, frame - len(x)))
        spec = np.abs(np.fft.rfft(x * np.hanning(frame))) + 1e-8
        freq = np.arange(len(spec), dtype=np.float64)
        centroid = np.sum(freq * spec) / np.sum(spec)
        spread = np.sqrt(np.sum((freq - centroid) ** 2 * spec) / np.sum(spec))
        scalar = [np.sqrt(np.mean(x*x)), np.mean(np.abs(x)), np.std(x), np.max(np.abs(x)), np.mean(np.abs(np.diff(np.signbit(x)))), np.max(np.abs(x))/(np.sqrt(np.mean(x*x))+1e-8), centroid, spread, np.sum(spec[:int(len(spec)*.85)])/np.sum(spec), np.exp(np.mean(np.log(spec)))/np.mean(spec)]
        bands = [float(np.mean(b)) for b in np.array_split(spec, 32)]
        all_values.append(scalar + bands)
    base = np.asarray(all_values, np.float32)
    delta = np.vstack([np.zeros((1, 32), np.float32), np.diff(base[:, -32:], axis=0)])
    feat = np.concatenate([base, delta], axis=1).astype(np.float32)
    return feat, starts.astype(float) / rate, np.minimum((starts + frame) / rate, len(wave) / rate)


def vision_features(frames, starts, duration):
    if frames.ndim != 4 or frames.shape[-1] != 3 or len(frames) == 0:
        raise ValueError("Video decoding produced no RGB frames")
    x = frames.astype(np.float32) / 255
    gray = x.mean(axis=3)
    values = []
    for frame, g in zip(x, gray):
        gx, gy = np.diff(g, axis=1), np.diff(g, axis=0)
        hist, _ = np.histogram(g, bins=16, range=(0, 1), density=True)
        values.append(list(frame.mean((0, 1))) + list(frame.std((0, 1))) + [g.mean(), g.std(), g.min(), g.max()] + list(np.percentile(g, [5,25,50,75,95])) + [np.abs(gx).mean(), np.abs(gy).mean(), np.abs(gx).std(), np.abs(gy).std()] + list(hist))
    values = np.asarray(values, np.float32)
    # FFmpeg's fps filter emits frames at requested timestamps.  A frame owns
    # the interval up to the next emitted timestamp; duration / n_frames can
    # otherwise create artificial gaps or overlaps at the clip boundary.
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.empty_like(starts)
    if len(starts) > 1:
        ends[:-1] = starts[1:]
    ends[-1] = duration
    ends = np.maximum(ends, starts + np.finfo(np.float64).eps)
    return values, starts, np.minimum(ends, duration)


def decode_audio(path, rate):
    raw = run([executable("ffmpeg"), "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(rate), "-f", "f32le", "-"])
    return np.frombuffer(raw, dtype=np.float32)


def decode_video(path, duration, fps):
    width = height = 64
    raw = run([executable("ffmpeg"), "-v", "error", "-i", str(path), "-vf", f"fps={fps},scale={width}:{height}", "-frames:v", str(max(1, math.ceil(duration * fps))), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    size = width * height * 3
    arr = np.frombuffer(raw[: len(raw) // size * size], np.uint8)
    frames = arr.reshape(-1, height, width, 3)
    if len(frames) == 0:
        raise RuntimeError(f"FFmpeg produced no video frames: {path}")
    return frames, np.arange(len(frames), dtype=float) / fps


def main(argv=None):
    args = parse_args(argv)
    data_root = args.data_root.resolve(); output = args.output_root.resolve()
    samples = read_samples(data_root)
    if args.limit is not None: samples = samples[: args.limit]
    if args.dry_run:
        print(json.dumps({"data_root": str(data_root), "samples": len(samples), "labels_verified": True}, ensure_ascii=False, indent=2)); return 0
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is not empty: {output}; use --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    encoder = TextEncoder(args.text_backend, args.text_model, args.max_text_tokens, args.allow_fallback)
    aligner = CTCAligner() if args.text_align == "ctc" else None
    records, manifest = [], []
    for number, sample in enumerate(samples, 1):
        duration, fps, frames_count = media_metadata(sample["path"])
        edges = np.linspace(0, duration, N_BINS + 1)
        text_values, text_bert, text_mask = encoder.encode(sample["text"])
        token_ix = np.flatnonzero(text_mask.astype(bool))
        audio_wave = decode_audio(sample["path"], args.audio_rate)
        word_spans = []
        if aligner is not None:
            word_spans = aligner.align(audio_wave, args.audio_rate, sample["text"])
        text_alignment_used = args.text_align
        if word_spans:
            aligned_spans, token_ix = encoder.token_time_spans(sample["text"], text_mask, [[w["start"], w["end"]] for w in word_spans], duration)
            text, text_valid, text_coverage, text_sources = pool_intervals(text_values[token_ix], aligned_spans, edges)
        elif len(token_ix):
            if args.text_align == "ctc":
                text_alignment_used = "uniform_fallback"
            starts = duration * np.arange(len(token_ix)) / len(token_ix); ends = duration * (np.arange(len(token_ix)) + 1) / len(token_ix)
            aligned_spans = np.c_[starts, ends]
            text, text_valid, text_coverage, text_sources = pool_intervals(text_values[token_ix], aligned_spans, edges)
        else:
            if args.text_align == "ctc":
                text_alignment_used = "uniform_fallback_empty"
            aligned_spans = np.zeros((0, 2), np.float64)
            text = np.zeros((N_BINS, TEXT_DIM), np.float32); text_valid = np.zeros(N_BINS, bool); text_coverage = text_valid.astype(float); text_sources = [[] for _ in range(N_BINS)]
        audio_source, audio_s, audio_e = audio_features(audio_wave, args.audio_rate)
        audio, audio_valid, audio_coverage, audio_sources = pool_intervals(audio_source, np.c_[audio_s, audio_e], edges)
        video, video_s = decode_video(sample["path"], duration, args.vision_fps)
        vision_source, vision_s, vision_e = vision_features(video, video_s, duration)
        vision, vision_valid, vision_coverage, vision_sources = pool_intervals(vision_source, np.c_[vision_s, vision_e], edges)
        records.append({"id": sample["id"], "video_id": sample["video_id"], "clip_id": sample["clip_id"], "raw_text": sample["text"], "text": text, "text_bert": text_bert, "audio": audio, "vision": vision, "valid_masks": {"text": text_valid, "audio": audio_valid, "vision": vision_valid}, "coverage": {"text": text_coverage, "audio": audio_coverage, "vision": vision_coverage}, "source_indices": {"text": text_sources, "audio": audio_sources, "vision": vision_sources}, "source_intervals": {"text": np.asarray(aligned_spans, dtype=np.float32), "audio": np.c_[audio_s, audio_e].astype(np.float32), "vision": np.c_[vision_s, vision_e].astype(np.float32)}, "source_feature_shapes": {"text": [int(len(token_ix)), TEXT_DIM], "audio": list(map(int, audio_source.shape)), "vision": list(map(int, vision_source.shape))}, "time_edges": edges.astype(np.float32), "duration_seconds": duration, "label": sample["label"], "annotation": sample["annotation"], "text_alignment": text_alignment_used, "text_word_spans": word_spans})
        manifest.append({"id": sample["id"], "video_path": str(sample["path"]), "sha256": sha256(sample["path"]), "duration_seconds": duration, "fps": fps, "frame_count": frames_count, "text_valid_bins": int(text_valid.sum()), "audio_valid_bins": int(audio_valid.sum()), "vision_valid_bins": int(vision_valid.sum()), "text_backend": encoder.backend, "audio_backend": "ffmpeg_numpy_74", "vision_backend": "ffmpeg_numpy_35"})
        print(f"[{number}/{len(samples)}] {sample['id']}", flush=True)
    with (output / "problem1_aligned_features.pkl").open("wb") as f:
        pickle.dump({"format_version": "problem1-v2", "n_bins": N_BINS, "dims": {"text": TEXT_DIM, "audio": AUDIO_DIM, "vision": VISION_DIM}, "samples": records}, f, protocol=pickle.HIGHEST_PROTOCOL)
    import csv
    with (output / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0])); writer.writeheader(); writer.writerows(manifest)
    write_json(output / "run_metadata.json", {"format_version": "problem1-v2", "data_root": str(data_root), "sample_count": len(records), "alignment": "50 equal-duration bins; overlap-weighted pooling", "text_alignment": args.text_align, "text_alignment_fallback": any(r["text_alignment"] != "ctc" or not r["text_word_spans"] for r in records), "text_backend": encoder.backend, "audio_backend": "ffmpeg_numpy_74", "vision_backend": "ffmpeg_numpy_35", "parameters": {"max_text_tokens": args.max_text_tokens, "audio_rate": args.audio_rate, "vision_fps": args.vision_fps, "audio_frame_ms": 25, "audio_hop_ms": 10, "vision_size": [64, 64], "n_bins": N_BINS}, "label_file_sha256": sha256(data_root / "attachment_1_raw_samples" / "labels_100.xlsx"), "command": " ".join(map(str, sys.argv))})
    return 0
