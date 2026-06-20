"""A small autoregressive LSTM language model over music tokens."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig, PAD


class MusicLSTM(nn.Module):
    """Autoregressive next-token model over the music token vocabulary.

    Pipeline: token-id **embedding** (``PAD`` is a zeroed, no-gradient slot) → a
    stacked **LSTM** → a **linear head** giving one logit per vocabulary token.
    The same network serves training (teacher-forced next-token prediction) and
    generation (one token at a time, threading the hidden state forward).
    """

    def __init__(self, cfg: ModelConfig):
        """Build the layers from ``cfg`` — vocab size, embedding/hidden dims,
        number of LSTM layers, and dropout (see :class:`config.ModelConfig`)."""
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
        """Map token ids to next-token logits.

        Args:
            x: long tensor of token ids, shape ``(batch, seq)``.
            hidden: optional LSTM ``(h, c)`` state to continue from (used when
                generating one token at a time).

        Returns:
            ``(logits, hidden)`` — ``logits`` of shape ``(batch, seq, vocab)`` and
            the updated LSTM state.
        """
        emb = self.embed(x)
        out, hidden = self.lstm(emb, hidden)
        logits = self.head(self.dropout(out))
        return logits, hidden
