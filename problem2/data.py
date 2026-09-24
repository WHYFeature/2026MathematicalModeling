"""Data loading, train-only normalization, and local-missingness utilities."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


MODALITIES = ("text", "audio", "vision")
FEATURE_DIMS = {"text": 768, "audio": 74, "vision": 35}


def _as_float(x):
    return np.asarray(x, dtype=np.float32)


def _standard_mask(split, modality, x):
    if modality == "text":
        # ``text`` is the 50-position BERT sequence.  Its padded hidden
        # states are nonzero (BERT's [PAD] embedding passes through the
        # encoder), so checking for all-zero rows is insufficient.  The
        # tokenizer-side attention row identifies real token positions while
        # preserving the 50-position input interface.
        text_bert = np.asarray(split.get("text_bert"))
        if text_bert.ndim == 3 and text_bert.shape[1:] == (3, x.shape[1]):
            return text_bert[:, 1, :].astype(bool)
        return np.ones(x.shape[:2], dtype=bool)
    return ~np.all(np.isclose(x, 0.0), axis=-1)


def load_standard(path):
    with Path(path).open("rb") as f:
        payload = pickle.load(f)
    result = {}
    for name in ("train", "valid", "test"):
        split = payload[name]
        arrays = {m: _as_float(split[m]) for m in MODALITIES}
        masks = {m: _standard_mask(split, m, arrays[m]) for m in MODALITIES}
        result[name] = {
            "x": arrays,
            "mask": masks,
            "classification": np.asarray(split["classification_labels"], dtype=np.int64),
            "regression": np.asarray(split["regression_labels"], dtype=np.float32),
            "id": [str(x) for x in split["id"]],
        }
    return result


def fit_normalizer(train):
    stats = {}
    for m in MODALITIES:
        x = train["x"][m]
        mask = train["mask"][m]
        flat = x[mask]
        mean = flat.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = flat.std(axis=0, dtype=np.float64).astype(np.float32)
        std[~np.isfinite(std) | (std < 1e-5)] = 1.0
        mean[~np.isfinite(mean)] = 0.0
        stats[m] = {"mean": mean, "std": std}
    return stats


def apply_normalizer(split, stats):
    result = {"x": {}, "mask": split["mask"], "classification": split["classification"], "regression": split["regression"], "id": split["id"]}
    for m in MODALITIES:
        mean, std = stats[m]["mean"], stats[m]["std"]
        x = ((split["x"][m] - mean) / std).astype(np.float32)
        x[~split["mask"][m]] = 0.0
        result["x"][m] = x
    return result


class StandardDataset(Dataset):
    def __init__(self, split, train=False, seed=42):
        self.x = {m: torch.from_numpy(split["x"][m]) for m in MODALITIES}
        self.mask = {m: torch.from_numpy(split["mask"][m].astype(np.bool_)) for m in MODALITIES}
        self.cls = torch.from_numpy(split["classification"].astype(np.int64))
        self.reg = torch.from_numpy(split["regression"].astype(np.float32))
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.cls)

    def __getitem__(self, index):
        return {
            "x": {m: self.x[m][index].clone() for m in MODALITIES},
            "mask": {m: self.mask[m][index].clone() for m in MODALITIES},
            "classification": self.cls[index],
            "regression": self.reg[index],
        }


def collate(batch):
    return {
        "x": {m: torch.stack([b["x"][m] for b in batch]) for m in MODALITIES},
        "mask": {m: torch.stack([b["mask"][m] for b in batch]) for m in MODALITIES},
        "classification": torch.stack([b["classification"] for b in batch]),
        "regression": torch.stack([b["regression"] for b in batch]),
    }


def apply_local_dropout(batch, probability=0.45, max_blocks=2, max_length=10):
    """Drop local contiguous spans independently per modality.

    No complete-modality dropout is used. This matches the attachment-3
    constraint and leaves other time bins available for fusion.
    """
    bsz, steps = next(iter(batch["x"].values())).shape[:2]
    for m in MODALITIES:
        x, mask = batch["x"][m], batch["mask"][m]
        for b in range(bsz):
            if torch.rand(()) > probability:
                continue
            blocks = int(torch.randint(1, max_blocks + 1, ()).item())
            for _ in range(blocks):
                length = int(torch.randint(2, max_length + 1, ()).item())
                start = int(torch.randint(0, max(1, steps - length + 1), ()).item())
                end = min(steps, start + length)
                mask[b, start:end] = False
                x[b, start:end] = 0.0
    return batch


def pool_to_50(x):
    """Average valid source intervals into 50 bins for unaligned attachment 3."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.shape[0] == 50:
        mask = ~np.all(np.isclose(x, 0.0), axis=-1)
        return x, mask
    source_steps = x.shape[0]
    out = np.zeros((50, x.shape[1]), dtype=np.float32)
    mask = np.zeros(50, dtype=bool)
    edges = np.linspace(0, source_steps, 51).round().astype(int)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        block = x[lo:hi]
        good = ~np.all(np.isclose(block, 0.0), axis=-1)
        if good.any():
            out[i] = block[good].mean(axis=0)
            mask[i] = True
    return out, mask


def load_missing_features(directory, text_model, device):
    """Load attachment-3 files and regenerate 768-D text from text_bert IDs."""
    from transformers import AutoModel, AutoTokenizer

    paths = sorted(Path(directory).glob("*.pkl"))
    if not paths:
        raise FileNotFoundError(f"No pickle files found in {directory}")
    raw = []
    text_inputs = []
    text_masks = []
    tokenizer = None
    for p in paths:
        with p.open("rb") as f:
            item = pickle.load(f)
        data = item.get("test", item)
        if "text_bert" in data:
            tb = np.asarray(data["text_bert"])
            if tb.ndim == 3:
                tb = tb[0]
            if tb.shape != (3, 50):
                raise ValueError(f"Unexpected text_bert shape in {p}: {tb.shape}")
            text_mask = (tb[1] > 0) & ~np.all(np.isclose(tb, 0.0), axis=0)
        elif "raw_text" in data:
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(text_model, local_files_only=True)
            text = str(np.asarray(data["raw_text"]).reshape(-1)[0])
            enc = tokenizer(text, max_length=50, truncation=True, padding="max_length")
            tb = np.stack([enc["input_ids"], enc["attention_mask"], enc.get("token_type_ids", [0] * 50)])
            text_mask = np.asarray(enc["attention_mask"], dtype=bool)
        else:
            raise ValueError(f"Missing text_bert/raw_text in {p}")
        text_inputs.append(tb)
        text_masks.append(text_mask)
        raw.append((p.stem, data))
    ids = np.stack(text_inputs).astype(np.int64)
    attention = ids[:, 1, :]
    token_type = ids[:, 2, :]
    bert = AutoModel.from_pretrained(text_model, local_files_only=True).to(device).eval()
    hidden = []
    with torch.inference_mode():
        for start in range(0, len(ids), 16):
            sl = slice(start, start + 16)
            out = bert(
                input_ids=torch.from_numpy(ids[sl, 0]).to(device),
                attention_mask=torch.from_numpy(attention[sl]).to(device),
                token_type_ids=torch.from_numpy(token_type[sl]).to(device),
            ).last_hidden_state.cpu().numpy().astype(np.float32)
            hidden.append(out)
    text = np.concatenate(hidden, axis=0)
    result = []
    for i, (sample_id, data) in enumerate(raw):
        tx = text[i]
        tx_mask = text_masks[i]
        tx[~tx_mask] = 0.0
        audio, audio_mask = pool_to_50(data["audio"])
        vision, vision_mask = pool_to_50(data["vision"])
        result.append({"id": sample_id, "x": {"text": tx, "audio": audio, "vision": vision}, "mask": {"text": tx_mask, "audio": audio_mask, "vision": vision_mask}})
    return result


def save_stats(path, stats):
    serializable = {m: {k: v.tolist() for k, v in d.items()} for m, d in stats.items()}
    Path(path).write_text(json.dumps(serializable), encoding="utf-8")
