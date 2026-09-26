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


class BertAVRobustFusion(nn.Module):
    """BERT-AV fusion with explicit availability-aware late fusion.

    The baseline above averages the audio and vision branches independently
    and concatenates them with BERT.  This variant keeps those summaries,
    adds the three valid-bin ratios, and learns an audio/vision gate plus a
    missingness embedding.  It therefore receives a signal when a branch is
    absent instead of treating a zero summary as an ordinary observation.
    """

    def __init__(self, model_name, device, audio_dim=74, vision_dim=35, hidden=128, dropout=0.25):
        super().__init__()
        from transformers import AutoModel

        self.bert = AutoModel.from_pretrained(model_name, local_files_only=True)
        bert_hidden = int(self.bert.config.hidden_size)
        self.audio = nn.Sequential(nn.LayerNorm(audio_dim * 2 + 1), nn.Linear(audio_dim * 2 + 1, hidden), nn.GELU(), nn.Dropout(dropout * 0.5))
        self.vision = nn.Sequential(nn.LayerNorm(vision_dim * 2 + 1), nn.Linear(vision_dim * 2 + 1, hidden), nn.GELU(), nn.Dropout(dropout * 0.5))
        self.av_gate = nn.Sequential(nn.Linear(hidden * 2 + 3, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 2))
        self.missing_embedding = nn.Sequential(nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        fusion_dim = bert_hidden + hidden * 4
        self.head = nn.Sequential(nn.LayerNorm(fusion_dim), nn.Linear(fusion_dim, hidden * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(hidden, 3)
        self.regressor = nn.Linear(hidden, 1)
        self.to(device)

    @staticmethod
    def _summary(x, mask):
        m = mask.float()
        denom = m.sum(1, keepdim=True).clamp_min(1.0)
        mean = (x * m.unsqueeze(-1)).sum(1) / denom
        centered = (x - mean.unsqueeze(1)) * m.unsqueeze(-1)
        std = torch.sqrt((centered.square().sum(1) / denom).clamp_min(1e-8))
        return mean, std, m.mean(1, keepdim=True)

    def forward(self, input_ids, attention_mask, token_type_ids, audio, audio_mask, vision, vision_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
        tm = attention_mask.float().unsqueeze(-1)
        text = (out * tm).sum(1) / tm.sum(1).clamp_min(1.0)
        audio_mean, audio_std, audio_ratio = self._summary(audio, audio_mask)
        vision_mean, vision_std, vision_ratio = self._summary(vision, vision_mask)
        audio_z = self.audio(torch.cat([audio_mean, audio_std, audio_ratio], dim=-1))
        vision_z = self.vision(torch.cat([vision_mean, vision_std, vision_ratio], dim=-1))
        availability = torch.cat([attention_mask.float().mean(1, keepdim=True), audio_ratio, vision_ratio], dim=-1)
        gate_input = torch.cat([audio_z, vision_z, availability], dim=-1)
        gates = torch.softmax(self.av_gate(gate_input), dim=-1)
        av_mix = gates[:, :1] * audio_z + gates[:, 1:2] * vision_z
        missing = self.missing_embedding(1.0 - availability)
        z = self.head(torch.cat([text, audio_z, vision_z, av_mix, missing], dim=-1))
        return self.classifier(z), self.regressor(z).squeeze(-1)
