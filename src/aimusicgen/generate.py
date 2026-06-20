"""Autoregressively sample a melody from a trained (or fresh) MusicLSTM."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import torch
import torch.nn.functional as F

from . import cadence as cad
from . import config as C
from . import models
from .config import ModelConfig, get_device
from .data.events import events_to_midi
from .model.lstm import MusicLSTM


def checkpoint_compatible(path=None) -> bool:
    """True if the active (or given) checkpoint exists and matches encoding/vocab."""
    if path is None:
        mid = models.get_active()
        path = models.model_path(mid) if mid else None
    if not path or not Path(path).exists():
        return False
    try:
        ckpt = torch.load(path, map_location="cpu")
    except Exception:
        return False
    return (ckpt.get("encoding") == C.ENCODING
            and ckpt.get("model_cfg", {}).get("vocab_size") == ModelConfig().vocab_size)


def load_model(model_id: str | None = None,
               device: torch.device | None = None) -> tuple[MusicLSTM, bool]:
    """Load the active model (or ``model_id``) if compatible. Returns (model, trained?)."""
    device = device or get_device()
    mid = model_id or models.get_active()
    path = models.model_path(mid) if mid else None
    if path and checkpoint_compatible(path):
        ckpt = torch.load(path, map_location=device)
        cfg = ModelConfig(**ckpt["model_cfg"])
        model = MusicLSTM(cfg).to(device)
        model.load_state_dict(ckpt["state_dict"])
        trained = True
    else:
        model = MusicLSTM(ModelConfig()).to(device)
        trained = False
    model.eval()
    return model, trained


@torch.no_grad()
def _interval_bias(prev_pitch: int, rules, strength: float, device) -> torch.Tensor:
    """Additive logit bias over the vocab: NOTE_ON pitches weighted by the
    enabled voice-leading rules relative to ``prev_pitch``; other tokens 0."""
    from . import constraints
    bias = torch.zeros(C.VOCAB_SIZE, device=device)
    pitches = range(C.PITCH_MIN, C.PITCH_MAX + 1)
    vals = [strength * constraints.logbias(p - prev_pitch, rules) for p in pitches]
    base = C.note_on_token(C.PITCH_MIN)
    bias[base:base + C.N_PITCH] = torch.tensor(vals, device=device)
    return bias


def sample(
    model: MusicLSTM,
    length: int = 256,
    temperature: float = 1.0,
    top_k: int = 0,
    seed_tokens: list[int] | None = None,
    device: torch.device | None = None,
    rng_seed: int | None = None,
    constraint_rules: list | None = None,
    constraint_strength: float = 1.0,
    key: str | None = None,
) -> list[int]:
    """Autoregressively sample a token sequence from ``model``.

    Args:
        length: number of tokens to generate.
        temperature: softmax temperature (>1 more random, <1 more conservative).
        top_k: if >0, restrict sampling to the k most likely tokens each step.
        seed_tokens: optional priming tokens (defaults to a single TIME_SHIFT).
        rng_seed: seed for reproducible output.
        constraint_rules: voice-leading rule ids to bias note choice (see
            :mod:`constraints`); ``constraint_strength`` scales their effect.
        key: optional "C major"-style key to hard-restrict pitches to.

    Returns:
        The flat list of generated token ids (including the primer).
    """
    device = device or get_device()
    if rng_seed is not None:
        torch.manual_seed(rng_seed)

    # Optional hard "stay within key" mask: ban out-of-key NOTE_ON pitches.
    key_bias = None
    if key:
        from . import constraints
        pcs = constraints.scale_pcs(key)
        if pcs:
            key_bias = torch.zeros(C.VOCAB_SIZE, device=device)
            for p in range(C.PITCH_MIN, C.PITCH_MAX + 1):
                if (p % 12) not in pcs:
                    key_bias[C.note_on_token(p)] = float("-inf")

    # Prime the model. Default seed = one short time-shift; the model then
    # chooses which notes to start.
    primer = seed_tokens or [C.time_shift_token(1)]
    tokens = list(primer)
    x = torch.tensor([primer], dtype=torch.long, device=device)
    _, hidden = model(x[:, :-1]) if x.size(1) > 1 else (None, None)
    cur = x[:, -1:]
    prev_pitch = next((C.note_on_pitch(t) for t in reversed(primer)
                       if C.is_note_on(t)), None)
    constraint_rules = constraint_rules or []

    temperature = max(1e-4, float(temperature))
    for _ in range(length):
        logits, hidden = model(cur, hidden)
        logits = logits[:, -1, :].squeeze(0) / temperature
        logits[C.PAD] = -float("inf")  # never emit padding
        if key_bias is not None:
            logits = logits + key_bias
        if constraint_rules and prev_pitch is not None:
            logits = logits + _interval_bias(prev_pitch, constraint_rules,
                                             constraint_strength, device)
        if top_k > 0:
            kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
            logits[logits < kth] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        nxt = int(torch.multinomial(probs, 1).item())
        tokens.append(nxt)
        if C.is_note_on(nxt):
            prev_pitch = C.note_on_pitch(nxt)
        cur = torch.tensor([[nxt]], dtype=torch.long, device=device)
    return tokens


def generate_midi(
    length: int = 512,
    temperature: float = 1.0,
    top_k: int = 0,
    tempo: int = C.DEFAULT_TEMPO,
    rng_seed: int | None = None,
    out_path: str | Path | None = None,
    model_id: str | None = None,
    constraint_rules: list | None = None,
    constraint_strength: float = 1.0,
    cadence: str | None = None,
    key: str | None = None,
) -> dict:
    """Generate one melody and write it to a .mid file. Returns metadata."""
    device = get_device()
    model, trained = load_model(model_id=model_id, device=device)
    tokens = sample(
        model,
        length=length,
        temperature=temperature,
        top_k=top_k,
        device=device,
        rng_seed=rng_seed,
        constraint_rules=constraint_rules,
        constraint_strength=constraint_strength,
        key=key,
    )
    if out_path is None:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = C.OUTPUT_DIR / f"gen-{stamp}.mid"
    pm = events_to_midi(tokens, tempo=tempo)
    kind = cad.resolve_kind(cadence)
    if kind:
        cad.append_cadence(pm, kind, key=key)   # match the chosen key if set
    pm.write(str(out_path))
    return {
        "path": str(out_path),
        "trained": trained,
        "n_tokens": len(tokens),
        "n_notes": sum(len(i.notes) for i in pm.instruments),
        "cadence": kind,
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Generate a melody")
    p.add_argument("--length", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--tempo", type=int, default=C.DEFAULT_TEMPO)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--rules", default="",
                   help="comma-separated voice-leading rules: stepwise,direction,rare_leaps")
    p.add_argument("--constraint-strength", type=float, default=1.0)
    p.add_argument("--cadence", choices=["half", "plagal", "either"], default=None,
                   help="append a final cadence")
    p.add_argument("--key", default=None, help='restrict to a key, e.g. "C major"')
    a = p.parse_args()
    rules = [r.strip() for r in a.rules.split(",") if r.strip()]
    info = generate_midi(
        length=a.length, temperature=a.temperature, top_k=a.top_k,
        tempo=a.tempo, rng_seed=a.seed,
        constraint_rules=rules, constraint_strength=a.constraint_strength,
        cadence=a.cadence, key=a.key,
    )
    tag = "trained model" if info["trained"] else "UNTRAINED model (run training first)"
    print(f"Wrote {info['path']}  [{info['n_notes']} notes, {tag}]")
