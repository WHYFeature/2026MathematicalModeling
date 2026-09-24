"""Pure NumPy alignment operations, with explicit validity and source indices."""
from __future__ import annotations

import numpy as np


def pool_intervals(values, intervals, edges, valid=None):
    """Overlap-weighted pooling; genuine zero observations remain valid.

    Coverage measures the UNION of observed intervals, so overlapping audio
    windows do not artificially inflate coverage above 1.
    """
    values = np.asarray(values, dtype=np.float32)
    spans = np.asarray(intervals, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    if values.ndim != 2 or spans.shape != (len(values), 2):
        raise ValueError("Expected features (T,D) and intervals (T,2)")
    if len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("Time edges must be strictly increasing")
    valid = np.ones(len(values), bool) if valid is None else np.asarray(valid, bool)
    if valid.shape != (len(values),):
        raise ValueError("Validity mask length mismatch")
    if not np.isfinite(values).all() or not np.isfinite(spans).all():
        raise ValueError("Non-finite feature or timestamp")
    if np.any(spans[:, 1] < spans[:, 0]):
        raise ValueError("Negative interval duration")
    out = np.zeros((len(edges) - 1, values.shape[1]), np.float32)
    coverage = np.zeros(len(out), np.float32)
    indices = []
    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        weights = np.maximum(0, np.minimum(spans[:, 1], hi) - np.maximum(spans[:, 0], lo))
        weights *= valid
        ix = np.flatnonzero(weights > 0)
        indices.append(ix.tolist())
        if not len(ix):
            continue
        out[k] = (weights[ix, None] * values[ix]).sum(axis=0) / weights[ix].sum()
        covered = 0.0
        right = lo
        for start, end in sorted((max(lo, spans[i, 0]), min(hi, spans[i, 1])) for i in ix):
            covered += max(0.0, end - max(right, start))
            right = max(right, end)
        coverage[k] = min(1.0, covered / (hi - lo))
    return out, coverage > 0, coverage, indices


def ctc_viterbi(log_probs, tokens, blank_id):
    """Exact transcript CTC path, including leading and trailing blank states.

    Returns half-open frame spans and mean acoustic probabilities for target
    characters. Repeated characters require an intervening blank. Probabilities
    are diagnostics, not calibrated confidence in timestamps.
    """
    log_probs = np.asarray(log_probs, dtype=np.float32)
    tokens = np.asarray(tokens, dtype=np.int64)
    if log_probs.ndim != 2 or not len(tokens) or not len(log_probs):
        raise ValueError("CTC needs nonempty (frames,vocabulary) scores and tokens")
    if np.any(tokens == blank_id) or np.any(tokens < 0) or np.any(tokens >= log_probs.shape[1]):
        raise ValueError("Invalid target token")
    required = len(tokens) + int(np.sum(tokens[1:] == tokens[:-1]))
    if len(log_probs) < required:
        raise ValueError(f"CTC has {len(log_probs)} frames, needs at least {required}")
    states = np.full(2 * len(tokens) + 1, blank_id, dtype=np.int64)
    states[1::2] = tokens
    skip_ok = np.zeros(len(states), bool)
    skip_ok[2:] = (states[2:] != blank_id) & (states[2:] != states[:-2])
    score = np.full(len(states), -np.inf, dtype=np.float32)
    score[:2] = log_probs[0, states[:2]]
    back = np.zeros((len(log_probs), len(states)), dtype=np.uint8)
    for t in range(1, len(log_probs)):
        one = np.full_like(score, -np.inf)
        two = np.full_like(score, -np.inf)
        one[1:] = score[:-1]
        two[2:] = score[:-2]
        two[~skip_ok] = -np.inf
        choices = np.stack([score, one, two])
        back[t] = np.argmax(choices, axis=0)
        score = choices.max(axis=0) + log_probs[t, states]
    state = len(states) - 1 if score[-1] >= score[-2] else len(states) - 2
    if not np.isfinite(score[state]):
        raise ValueError("No feasible CTC path for the supplied transcript")
    path = np.empty(len(log_probs), dtype=np.int32)
    for t in range(len(log_probs) - 1, -1, -1):
        path[t] = state
        if t:
            state -= int(back[t, state])
    spans, confidence = [], []
    for i, token in enumerate(tokens):
        frames = np.flatnonzero(path == 2 * i + 1)
        if not len(frames):
            raise ValueError("Backtracked CTC path omitted a target character")
        spans.append([int(frames[0]), int(frames[-1] + 1)])
        confidence.append(float(np.exp(log_probs[frames, token]).mean()))
    return np.asarray(spans, dtype=np.int32), np.asarray(confidence, dtype=np.float32)
