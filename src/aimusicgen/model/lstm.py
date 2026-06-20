"""A small autoregressive LSTM language model over music tokens."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig, PAD


class MusicLSTM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(
            cfg.embed_dim,
            cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.hidden_dim, cfg.vocab_size)

    def forward(self, x: torch.Tensor, hidden=None):
        """x: (batch, seq) long -> logits: (batch, seq, vocab)."""
        emb = self.embed(x)
        out, hidden = self.lstm(emb, hidden)
        logits = self.head(self.dropout(out))
        return logits, hidden
