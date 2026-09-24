"""BERT fine-tuning model for the Problem 2 text branch."""
from __future__ import annotations

import torch
from torch import nn


class BertSentiment(nn.Module):
    def __init__(self, model_name, device, dropout=0.25):
        super().__init__()
        from transformers import AutoModel

        self.bert = AutoModel.from_pretrained(model_name, local_files_only=True)
        self.hidden = int(self.bert.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden, 3)
        self.regressor = nn.Linear(self.hidden, 1)
        self.to(device)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
        mask = attention_mask.float().unsqueeze(-1)
        pooled = (out * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        z = self.dropout(pooled)
        return self.classifier(z), self.regressor(z).squeeze(-1)


class BertAVFusion(nn.Module):
    def __init__(self, model_name, device, audio_dim=74, vision_dim=35, hidden=128, dropout=0.25):
        super().__init__()
        from transformers import AutoModel

        self.bert = AutoModel.from_pretrained(model_name, local_files_only=True)
        bert_hidden = int(self.bert.config.hidden_size)
        self.audio = nn.Sequential(nn.LayerNorm(audio_dim * 2 + 1), nn.Linear(audio_dim * 2 + 1, hidden), nn.GELU())
        self.vision = nn.Sequential(nn.LayerNorm(vision_dim * 2 + 1), nn.Linear(vision_dim * 2 + 1, hidden), nn.GELU())
        self.head = nn.Sequential(nn.LayerNorm(bert_hidden + hidden * 2), nn.Linear(bert_hidden + hidden * 2, hidden * 2), nn.GELU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(hidden * 2, 3)
        self.regressor = nn.Linear(hidden * 2, 1)
        self.to(device)

    def forward(self, input_ids, attention_mask, token_type_ids, audio, audio_mask, vision, vision_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
        tm = attention_mask.float().unsqueeze(-1)
        text = (out * tm).sum(1) / tm.sum(1).clamp_min(1.0)
        pieces = [text]
        for x, mask, branch in ((audio, audio_mask, self.audio), (vision, vision_mask, self.vision)):
            m = mask.float(); denom = m.sum(1, keepdim=True).clamp_min(1.0)
            mean = (x * m.unsqueeze(-1)).sum(1) / denom
            centered = (x - mean.unsqueeze(1)) * m.unsqueeze(-1)
            std = torch.sqrt((centered.square().sum(1) / denom).clamp_min(1e-8))
            pieces.append(branch(torch.cat([mean, std, m.mean(1, keepdim=True)], dim=-1)))
        z = self.head(torch.cat(pieces, dim=-1))
        return self.classifier(z), self.regressor(z).squeeze(-1)
