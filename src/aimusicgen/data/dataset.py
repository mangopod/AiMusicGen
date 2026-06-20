"""Build a torch Dataset of fixed-length token windows from a MIDI corpus."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from .. import config as C
from .events import midi_to_events


def load_corpus(data_dir=C.DATA_DIR) -> list[list[int]]:
    """Tokenize every .mid/.midi file under ``data_dir`` (a path or list of
    paths — passing several dirs trains one model on the combined corpus)."""
    dirs = data_dir if isinstance(data_dir, (list, tuple)) else [data_dir]
    paths = sorted({p for d in dirs for ext in ("*.mid", "*.midi")
                    for p in Path(d).rglob(ext)})
    songs: list[list[int]] = []
    for p in paths:
        try:
            toks = midi_to_events(p)
        except Exception as exc:  # corrupt / unsupported file -> skip, keep going
            print(f"  ! skipping {p.name}: {exc}")
            continue
        if len(toks) > 1:
            songs.append(toks)
    return songs


class MidiSequenceDataset(Dataset):
    """Sliding windows of length ``seq_len`` (+1 for the shifted target)."""

    def __init__(self, songs: list[list[int]], seq_len: int = 64, stride: int = 1):
        self.seq_len = seq_len
        self.windows: list[list[int]] = []
        for toks in songs:
            if len(toks) < seq_len + 1:
                # pad short songs up to one window
                self.windows.append(toks + [C.PAD] * (seq_len + 1 - len(toks)))
                continue
            for i in range(0, len(toks) - seq_len, stride):
                self.windows.append(toks[i : i + seq_len + 1])

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        w = self.windows[idx]
        x = torch.tensor(w[:-1], dtype=torch.long)
        y = torch.tensor(w[1:], dtype=torch.long)
        return x, y
