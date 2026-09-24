"""Mask-aware temporal multimodal fusion network."""
from __future__ import annotations

import torch
from torch import nn


class RobustFusionNet(nn.Module):
    """Local-missingness-aware three-modal temporal fusion model.

    Each modality is projected independently. At every time bin, learned
    gates normalize only over modalities that are present, then a temporal
    Transformer models the resulting sequence. Missingness indicators are
    retained as inputs so a zero vector is never confused with a real value.
    """

    def __init__(self, dims=(768, 74, 35), hidden=128, heads=4, layers=2, dropout=0.15):
        super().__init__()
        self.modalities = ("text", "audio", "vision")
        self.hidden = hidden
        self.proj = nn.ModuleDict({
            "text": nn.Sequential(nn.LayerNorm(dims[0]), nn.Linear(dims[0], hidden), nn.GELU()),
            "audio": nn.Sequential(nn.LayerNorm(dims[1]), nn.Linear(dims[1], hidden), nn.GELU()),
            "vision": nn.Sequential(nn.LayerNorm(dims[2]), nn.Linear(dims[2], hidden), nn.GELU()),
        })
        self.gates = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(hidden + 1, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
            for m in self.modalities
        })
        self.missing_embedding = nn.Sequential(nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.position = nn.Parameter(torch.zeros(1, 50, hidden))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=layers)
        self.pool_score = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(hidden // 2, 3)
        self.regressor = nn.Linear(hidden // 2, 1)

    def forward(self, inputs, masks):
        # inputs: modality -> [B, 50, D], masks: modality -> [B, 50] bool/float
        encoded = {}
        gate_logits = []
        availability = []
        for m in self.modalities:
            mask = masks[m].float()
            h = self.proj[m](inputs[m])
            h = h * mask.unsqueeze(-1)
            encoded[m] = h
            availability.append(mask)
            gate_logits.append(self.gates[m](torch.cat([h, mask.unsqueeze(-1)], dim=-1)).squeeze(-1))
        avail = torch.stack(availability, dim=-1)  # [B, T, 3]
        logits = torch.stack(gate_logits, dim=-1)
        logits = logits.masked_fill(avail <= 0, -1e4)
        weights = torch.softmax(logits, dim=-1)
        fused = sum(weights[..., i:i + 1] * encoded[m] for i, m in enumerate(self.modalities))
        fused = fused + self.missing_embedding(avail)
        present = avail.any(dim=-1)
        fused = fused + self.position[:, :fused.shape[1]]
        temporal = self.temporal(fused, src_key_padding_mask=~present)
        score = self.pool_score(temporal).squeeze(-1).masked_fill(~present, -1e4)
        pool_weights = torch.softmax(score, dim=-1)
        pooled = (pool_weights.unsqueeze(-1) * temporal).sum(dim=1)
        z = self.head(pooled)
        return self.classifier(z), self.regressor(z).squeeze(-1), weights, present


class TokenFusionNet(nn.Module):
    """Late token fusion that preserves modality-specific information.

    The original early-gated sum is compact but can erase a useful modality
    before temporal attention. This variant sends all 3x50 modality tokens to
    one Transformer, with explicit modality and time embeddings and local
    key-padding masks.
    """

    def __init__(self, dims=(768, 74, 35), hidden=160, heads=8, layers=3, dropout=0.12):
        super().__init__()
        self.modalities = ("text", "audio", "vision")
        self.proj = nn.ModuleDict({
            "text": nn.Sequential(nn.LayerNorm(dims[0]), nn.Linear(dims[0], hidden), nn.GELU()),
            "audio": nn.Sequential(nn.LayerNorm(dims[1]), nn.Linear(dims[1], hidden), nn.GELU()),
            "vision": nn.Sequential(nn.LayerNorm(dims[2]), nn.Linear(dims[2], hidden), nn.GELU()),
        })
        self.modality_embedding = nn.Parameter(torch.randn(1, 1, 3, hidden) * 0.02)
        self.time_embedding = nn.Parameter(torch.randn(1, 50, 1, hidden) * 0.02)
        self.availability_embedding = nn.Sequential(nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=layers)
        self.pool_score = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.head = nn.Sequential(nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(hidden, 3)
        self.regressor = nn.Linear(hidden, 1)

    def forward(self, inputs, masks):
        encoded = []
        mask_list = []
        for i, m in enumerate(self.modalities):
            mask = masks[m].float()
            h = self.proj[m](inputs[m]) * mask.unsqueeze(-1)
            encoded.append(h)
            mask_list.append(mask.bool())
        # [B, T, 3, H] -> [B, 3T, H], keeping all modalities separate.
        tokens = torch.stack(encoded, dim=2)
        avail = torch.stack(mask_list, dim=-1)
        tokens = tokens + self.modality_embedding + self.time_embedding
        tokens = tokens + self.availability_embedding(avail.float()).unsqueeze(2)
        tokens = tokens.reshape(tokens.shape[0], -1, tokens.shape[-1])
        token_mask = avail.reshape(avail.shape[0], -1)
        temporal = self.temporal(tokens, src_key_padding_mask=~token_mask)
        score = self.pool_score(temporal).squeeze(-1).masked_fill(~token_mask, -1e4)
        alpha = torch.softmax(score, dim=-1)
        pooled = (alpha.unsqueeze(-1) * temporal).sum(dim=1)
        # Keep a separate global summary for each modality.
        global_parts = []
        for i, m in enumerate(self.modalities):
            weights = mask_list[i].float()
            global_parts.append((encoded[i] * weights.unsqueeze(-1)).sum(1) / weights.sum(1, keepdim=True).clamp_min(1.0))
        global_summary = torch.stack(global_parts, dim=1).mean(1)
        z = self.head(torch.cat([pooled, global_summary], dim=-1))
        return self.classifier(z), self.regressor(z).squeeze(-1), None, token_mask.any(dim=-1)


class PooledFusionNet(nn.Module):
    """Strong low-variance baseline using masked mean/max modality summaries."""

    def __init__(self, dims=(768, 74, 35), hidden=256, dropout=0.20):
        super().__init__()
        self.modalities = ("text", "audio", "vision")
        self.proj = nn.ModuleDict({
            m: nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU())
            for m, d in zip(self.modalities, dims)
        })
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden * 6 + 3),
            nn.Linear(hidden * 6 + 3, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden, 3)
        self.regressor = nn.Linear(hidden, 1)

    def forward(self, inputs, masks):
        summaries = []
        ratios = []
        for m in self.modalities:
            mask = masks[m].float()
            h = self.proj[m](inputs[m]) * mask.unsqueeze(-1)
            denom = mask.sum(1, keepdim=True).clamp_min(1.0)
            mean = h.sum(1) / denom
            max_value = h.masked_fill(~mask.bool().unsqueeze(-1), -1e4).max(1).values
            max_value = torch.where(mask.any(1, keepdim=True), max_value, torch.zeros_like(max_value))
            summaries.extend([mean, max_value])
            ratios.append(mask.mean(1, keepdim=True))
        z = self.fusion(torch.cat(summaries + ratios, dim=-1))
        return self.classifier(z), self.regressor(z).squeeze(-1), None, torch.ones(z.shape[0], dtype=torch.bool, device=z.device)


class SummaryMLP(nn.Module):
    """Regularized clip-level model using masked mean/std summaries."""

    def __init__(self, dims=(768, 74, 35), hidden=256, dropout=0.35):
        super().__init__()
        self.modalities = ("text", "audio", "vision")
        input_dim = 2 * sum(dims) + 3
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden // 2, 3)
        self.regressor = nn.Linear(hidden // 2, 1)

    def forward(self, inputs, masks):
        pieces, ratios = [], []
        for m in self.modalities:
            x = inputs[m]
            mask = masks[m].float()
            denom = mask.sum(1, keepdim=True).clamp_min(1.0)
            mean = (x * mask.unsqueeze(-1)).sum(1) / denom
            centered = (x - mean.unsqueeze(1)) * mask.unsqueeze(-1)
            std = torch.sqrt((centered.square().sum(1) / denom).clamp_min(1e-8))
            pieces.extend([mean, std])
            ratios.append(mask.mean(1, keepdim=True))
        z = self.net(torch.cat(pieces + ratios, dim=-1))
        return self.classifier(z), self.regressor(z).squeeze(-1), None, torch.ones(z.shape[0], dtype=torch.bool, device=z.device)


class HybridStatsFusionNet(nn.Module):
    """Masked statistics plus cross-modal attention fusion.

    The aligned file contains a fixed 50 positions, but the text branch has
    variable valid length and audio/vision can have local zero spans.  This
    model first forms mean, standard-deviation and max summaries using the
    supplied masks, then lets a small Transformer exchange information among
    the three modality summaries.  Pairwise products retain interactions that
    a plain concatenation can lose while keeping the checkpoint compact.
    """

    def __init__(self, dims=(768, 74, 35), hidden=192, heads=4, layers=2, dropout=0.12):
        super().__init__()
        self.modalities = ("text", "audio", "vision")
        self.proj = nn.ModuleDict({
            m: nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, hidden),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
            )
            for m, d in zip(self.modalities, dims)
        })
        self.summary_proj = nn.ModuleDict({
            m: nn.Sequential(
                nn.LayerNorm(hidden * 3 + 1),
                nn.Linear(hidden * 3 + 1, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for m in self.modalities
        })
        self.modality_embedding = nn.Parameter(torch.randn(1, 3, hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.modality_encoder = nn.TransformerEncoder(layer, num_layers=layers)
        fusion_dim = hidden * 3 + hidden * 3 + 3
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden, 3)
        self.regressor = nn.Linear(hidden, 1)
        # Auxiliary ordinal head.  The three classes have a natural order
        # (negative < neutral < positive); using two cumulative boundaries
        # regularizes the shared representation and discourages the classifier
        # from collapsing the neutral class into a neighbouring class.
        self.ordinal = nn.Linear(hidden, 2)

    @staticmethod
    def _masked_stats(h, mask):
        mask = mask.float()
        keep = mask.unsqueeze(-1)
        denom = mask.sum(1, keepdim=True).clamp_min(1.0)
        mean = (h * keep).sum(1) / denom
        centered = (h - mean.unsqueeze(1)) * keep
        std = torch.sqrt((centered.square().sum(1) / denom).clamp_min(1e-8))
        neg_inf = torch.finfo(h.dtype).min
        max_value = h.masked_fill(~mask.bool().unsqueeze(-1), neg_inf).max(1).values
        max_value = torch.where(mask.any(1, keepdim=True), max_value, torch.zeros_like(max_value))
        return mean, std, max_value, mask.mean(1, keepdim=True)

    def forward(self, inputs, masks):
        summaries, ratios = [], []
        for m in self.modalities:
            h = self.proj[m](inputs[m])
            mean, std, maximum, ratio = self._masked_stats(h, masks[m])
            summaries.append(self.summary_proj[m](torch.cat([mean, std, maximum, ratio], dim=-1)))
            ratios.append(ratio)
        tokens = torch.stack(summaries, dim=1) + self.modality_embedding
        tokens = self.modality_encoder(tokens)
        pooled = tokens.reshape(tokens.shape[0], -1)
        interactions = torch.cat([
            summaries[0] * summaries[1],
            summaries[0] * summaries[2],
            summaries[1] * summaries[2],
        ], dim=-1)
        z = self.head(torch.cat([pooled, interactions, torch.cat(ratios, dim=-1)], dim=-1))
        return self.classifier(z), self.regressor(z).squeeze(-1), self.ordinal(z), torch.ones(z.shape[0], dtype=torch.bool, device=z.device)
