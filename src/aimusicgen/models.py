"""Saved-model library: per-composer and mixed-composer checkpoints.

Each model is ``checkpoints/<id>.pt``; ``checkpoints/models.json`` holds the
friendly name, the composers each model was trained on, metrics, and which
model is *active* (the one generation loads). The id is derived from the
composer set, so retraining the same set updates the same model.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from . import config as C

MANIFEST = C.CKPT_DIR / "models.json"


def model_path(model_id: str) -> Path:
    """Filesystem path of a model's checkpoint (``checkpoints/<id>.pt``)."""
    return C.CKPT_DIR / f"{model_id}.pt"


def make_id(composers) -> str:
    """Stable model id from a composer set, e.g. ['bach','beethoven'] →
    'bach+beethoven' (so retraining the same set updates the same model)."""
    return "+".join(sorted(composers)) if composers else "model"


def default_name(composers) -> str:
    """Human-readable default name for a model, e.g. 'Bach + Beethoven'."""
    if not composers:
        return "model"
    pretty = {"curated": "Keepers"}
    return " + ".join(pretty.get(c, c[:1].upper() + c[1:]) for c in sorted(composers))


def _load() -> dict:
    try:
        d = json.loads(MANIFEST.read_text())
    except Exception:
        d = {}
    d.setdefault("active", None)
    d.setdefault("models", {})
    return d


def _save(d: dict) -> None:
    try:
        MANIFEST.write_text(json.dumps(d, indent=2))
    except Exception:
        pass


def register(model_id: str, name: str, composers, epochs, val_loss, n_songs) -> None:
    """Record a freshly trained model and make it active."""
    d = _load()
    d["models"][model_id] = {
        "name": name,
        "composers": list(composers),
        "epochs": epochs,
        "val_loss": round(float(val_loss), 4),
        "n_songs": n_songs,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    d["active"] = model_id
    _save(d)


def list_models() -> dict:
    """Every checkpoint on disk merged with manifest metadata, newest first."""
    d = _load()
    meta = d["models"]
    out = []
    for p in C.CKPT_DIR.glob("*.pt"):
        mid = p.stem
        m = meta.get(mid, {})
        out.append({
            "id": mid,
            "name": m.get("name") or mid,
            "composers": m.get("composers", []),
            "epochs": m.get("epochs"),
            "val_loss": m.get("val_loss"),
            "n_songs": m.get("n_songs"),
            "created": m.get("created") or datetime.fromtimestamp(
                p.stat().st_mtime).isoformat(timespec="seconds"),
        })
    out.sort(key=lambda e: e["created"], reverse=True)
    active = d.get("active")
    if active and not model_path(active).exists():
        active = None
    return {"active": active, "models": out}


def get_active() -> str | None:
    """The active model's id (the one generation loads), or None if it's gone."""
    d = _load()
    a = d.get("active")
    return a if a and model_path(a).exists() else None


def set_active(model_id: str) -> bool:
    """Make ``model_id`` active. Returns False if its checkpoint is missing."""
    if not model_path(model_id).exists():
        return False
    d = _load()
    d["active"] = model_id
    _save(d)
    return True


def rename(model_id: str, name: str) -> bool:
    """Set a model's friendly name in the manifest."""
    if not model_path(model_id).exists():
        return False
    d = _load()
    entry = d["models"].get(model_id, {"composers": []})
    entry["name"] = name.strip() or model_id
    entry.setdefault("created", datetime.fromtimestamp(
        model_path(model_id).stat().st_mtime).isoformat(timespec="seconds"))
    d["models"][model_id] = entry
    _save(d)
    return True


def delete(model_id: str) -> bool:
    """Delete a model's checkpoint + manifest entry; if it was active, fall back
    to the most recently modified remaining checkpoint."""
    p = model_path(model_id)
    if not p.exists():
        return False
    try:
        p.unlink()
    except Exception:
        return False
    d = _load()
    d["models"].pop(model_id, None)
    if d.get("active") == model_id:
        remaining = sorted(C.CKPT_DIR.glob("*.pt"),
                           key=lambda x: x.stat().st_mtime, reverse=True)
        d["active"] = remaining[0].stem if remaining else None
    _save(d)
    return True
