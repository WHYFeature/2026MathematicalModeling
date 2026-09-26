"""Explainable temporal fusion model for Problem 3."""
from __future__ import annotations

import torch
from torch import nn


class ExplainableFusionNet(nn.Module):
    """Three-branch temporal encoder with explicit evidence tensors.

    ``forward`` returns class logits, regression output, modality weights,
    temporal weights, and modality-by-time weights.  The latter two are used
    as a proposal distribution; final evidence cards additionally apply local
    occlusion so an explanation can be checked by changing the input.
    """

    def __init__(self, dims=(768, 74, 35), hidden=160, heads=8, layers=2, dropout=0.15, max_steps=50):
        super().__init__()
        self.modalities = ("text", "audio", "vision")
        self.max_steps = max_steps
        self.proj = nn.ModuleDict({
            m: nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout))
            for m, d in zip(self.modalities, dims)
        })
        self.position = nn.Parameter(torch.randn(1, max_steps, hidden) * 0.02)
        self.intra = nn.ModuleDict()
        for m in self.modalities:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=heads, dim_feedforward=hidden * 4,
                dropout=dropout, activation="gelu", batch_first=True,
                norm_first=True,
            )
            self.intra[m] = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.cross = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden)
        self.modality_score = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.availability_embedding = nn.Sequential(nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * 4,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(temporal_layer, num_layers=layers, enable_nested_tensor=False)
        self.time_score = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(hidden, 3)
        self.regressor = nn.Linear(hidden, 1)

    @staticmethod
    def _masked_mean(x, mask):
        weights = mask.float().unsqueeze(-1)
        denom = weights.sum(1).clamp_min(1.0)
        return (x * weights).sum(1) / denom

    def forward(self, inputs, masks, return_cross=False):
        encoded, summaries, ratios, present = {}, [], [], []
        for m in self.modalities:
            mask = masks[m].bool()
            h = self.proj[m](inputs[m])
            h = h + self.position[:, :h.shape[1]]
            # PyTorch's attention returns NaN when every key is masked.  A
            # fully unavailable branch is represented by zeros after the
            # encoder, while one harmless sentinel query keeps the operation
            # numerically defined.
            safe_mask = mask.clone()
            empty = ~safe_mask.any(dim=1)
            if empty.any():
                safe_mask[empty, 0] = True
            h = self.intra[m](h, src_key_padding_mask=~safe_mask)
            h = h * mask.unsqueeze(-1).float()
            encoded[m] = h
            summaries.append(self._masked_mean(h, mask))
            ratios.append(mask.float().mean(1, keepdim=True))
            present.append(mask)
        ratio_tensor = torch.cat(ratios, dim=-1)
        summary_tokens = torch.stack(summaries, dim=1)
        cross_out, cross_weights = self.cross(summary_tokens, summary_tokens, summary_tokens, need_weights=True)
        cross_out = self.cross_norm(cross_out + summary_tokens)
        modality_logits = self.modality_score(cross_out).squeeze(-1)
        modality_logits = modality_logits + torch.log(ratio_tensor.clamp_min(1e-5))
        modality_weights = torch.softmax(modality_logits, dim=-1)
        fused = sum(modality_weights[:, i:i + 1, None] * encoded[m] for i, m in enumerate(self.modalities))
        fused = fused + self.availability_embedding(ratio_tensor).unsqueeze(1)
        any_present = torch.stack(present, dim=-1).any(dim=-1)
        temporal = self.temporal(fused, src_key_padding_mask=~any_present)
        time_logits = self.time_score(temporal).squeeze(-1).masked_fill(~any_present, -1e4)
        time_weights = torch.softmax(time_logits, dim=-1)
        pooled = (time_weights.unsqueeze(-1) * temporal).sum(1)
        z = self.head(pooled)
        outputs = {
            "logits": self.classifier(z),
            "regression": self.regressor(z).squeeze(-1),
            "modality_weights": modality_weights,
            "time_weights": time_weights,
            "modality_time_weights": modality_weights.unsqueeze(-1) * time_weights.unsqueeze(1),
            "availability": ratio_tensor,
        }
        if return_cross:
            outputs["cross_attention"] = cross_weights
        return outputs
