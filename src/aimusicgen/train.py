"""Train the MusicLSTM on the MIDI corpus in data/midi/.

Usage:
    python -m aimusicgen.train [--epochs N] [--seq-len N] [--batch-size N]
"""
from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from . import config as C
from .config import TrainConfig, get_device
from .data.dataset import MidiSequenceDataset, load_corpus
from .model.lstm import MusicLSTM


def train(
    cfg: TrainConfig | None = None,
    data_dir=None,
    progress_cb=None,
    save_path=None,
) -> dict:
    """Train and checkpoint. ``data_dir`` is a path or list of paths (combine
    several to train one model on a mixed corpus); ``save_path`` is where the
    best checkpoint is written; ``progress_cb`` (if given) is called once per
    epoch with a dict of metrics — used by the app to drive a live progress bar."""
    cfg = cfg or TrainConfig()
    data_dir = data_dir or C.DATA_DIR
    save_path = save_path or C.CKPT_PATH
    torch.manual_seed(cfg.seed)
    device = get_device()

    print(f"Loading corpus from {data_dir} ...")
    songs = load_corpus(data_dir)
    if not songs:
        raise RuntimeError(
            f"No usable MIDI found in {data_dir}.\n"
            f"Add .mid files, or run:  python scripts/make_demo_data.py"
        )
    total_tokens = sum(len(s) for s in songs)
    print(f"  {len(songs)} songs, {total_tokens} tokens")

    ds = MidiSequenceDataset(songs, seq_len=cfg.seq_len, stride=cfg.stride)
    n_val = max(1, int(len(ds) * cfg.val_split))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.seed)
    )
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size)
    print(f"  {len(ds)} windows -> {n_train} train / {n_val} val | device={device}")

    model = MusicLSTM(cfg.model).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    loss_fn = nn.CrossEntropyLoss(ignore_index=C.PAD)

    best_val = math.inf
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits, _ = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            running += loss.item() * x.size(0)
        sched.step()
        train_loss = running / n_train

        model.eval()
        vrun = 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                vrun += loss.item() * x.size(0)
        val_loss = vrun / n_val

        print(f"epoch {epoch:3d}/{cfg.epochs}  "
              f"train {train_loss:.3f}  val {val_loss:.3f}  "
              f"({time.time() - t0:.1f}s)")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, cfg, epoch, val_loss, save_path)

        if progress_cb is not None:
            progress_cb({
                "epoch": epoch,
                "epochs": cfg.epochs,
                "train_loss": train_loss,
                "val_loss": val_loss,
            })

    print(f"Done. Best val loss {best_val:.3f}. Checkpoint: {save_path}")
    return {"best_val": best_val, "epochs": cfg.epochs, "n_songs": len(songs)}


def save_checkpoint(model: MusicLSTM, cfg: TrainConfig, epoch: int, val: float,
                    path=None) -> None:
    """Write the model weights + ``model_cfg`` + encoding tag to ``path`` (default
    ``CKPT_PATH``). The encoding tag lets loaders reject incompatible checkpoints."""
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_cfg": vars(cfg.model),
            "encoding": C.ENCODING,
            "epoch": epoch,
            "val_loss": val,
        },
        path or C.CKPT_PATH,
    )


def _parse() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train MusicLSTM")
    p.add_argument("--epochs", type=int)
    p.add_argument("--seq-len", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    a = p.parse_args()
    cfg = TrainConfig()
    if a.epochs:
        cfg.epochs = a.epochs
    if a.seq_len:
        cfg.seq_len = a.seq_len
    if a.batch_size:
        cfg.batch_size = a.batch_size
    if a.lr:
        cfg.lr = a.lr
    return cfg


if __name__ == "__main__":
    from . import models

    try:
        out = train(_parse(), save_path=models.model_path("model"))
    except RuntimeError as exc:
        raise SystemExit(str(exc))
    models.register("model", "CLI model", [], out["epochs"], out["best_val"],
                    out["n_songs"])
