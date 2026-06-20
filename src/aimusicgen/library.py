"""Persistent library of generated pieces.

Files in output/ are the source of truth; a small JSON manifest holds the extra
metadata (friendly name, generation params, note count) keyed by filename.
The manifest tolerates files added or removed outside the app.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from . import config as C

MANIFEST = C.OUTPUT_DIR / "library.json"

# Generations the user "keeps" are copied here and become training data
# (the human-curated feedback loop). It's a sibling of the composer folders,
# so training can include it like any other source.
CURATED_DIR = C.DATA_DIR / "curated"


def _load() -> dict:
    try:
        return json.loads(MANIFEST.read_text())
    except Exception:
        return {}


def _save(meta: dict) -> None:
    try:
        MANIFEST.write_text(json.dumps(meta, indent=2))
    except Exception:
        pass


def safe_path(file: str) -> Path:
    """Resolve a filename to output/, stripping any path traversal."""
    return C.OUTPUT_DIR / Path(file).name


def add_entry(path: str | Path, n_notes: int, params: dict) -> None:
    """Record a freshly written generation in the manifest (name, time, params)."""
    path = Path(path)
    meta = _load()
    meta[path.name] = {
        "name": path.stem,
        "created": datetime.now().isoformat(timespec="seconds"),
        "n_notes": n_notes,
        "params": params,
    }
    _save(meta)


def list_entries() -> list[dict]:
    """Every .mid in output/, newest first, merged with manifest metadata."""
    meta = _load()
    entries: list[dict] = []
    for p in C.OUTPUT_DIR.glob("*.mid"):
        m = meta.get(p.name, {})
        created = m.get("created") or datetime.fromtimestamp(
            p.stat().st_mtime).isoformat(timespec="seconds")
        entries.append({
            "file": p.name,
            "name": m.get("name") or p.stem,
            "created": created,
            "n_notes": m.get("n_notes"),
            "params": m.get("params", {}),
            "kept": (CURATED_DIR / p.name).exists(),
        })
    entries.sort(key=lambda e: e["created"], reverse=True)
    return entries


def keep(file: str) -> bool:
    """Copy a saved generation into the curated training pool."""
    src = safe_path(file)
    if not src.exists():
        return False
    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(src, CURATED_DIR / src.name)
        return True
    except Exception:
        return False


def unkeep(file: str) -> bool:
    """Remove ``file`` from the curated pool (no-op if absent)."""
    p = CURATED_DIR / Path(file).name
    if p.exists():
        try:
            p.unlink()
        except Exception:
            return False
    return True


def keepers_count() -> int:
    """Number of .mid files in the curated pool."""
    return sum(1 for _ in CURATED_DIR.glob("*.mid")) if CURATED_DIR.exists() else 0


def list_curated() -> list[dict]:
    """Files in the drop/training pool (data/midi/curated/), newest first."""
    if not CURATED_DIR.exists():
        return []
    out = []
    for p in sorted(CURATED_DIR.glob("*.mid")):
        n_notes = None
        try:
            import pretty_midi
            pm = pretty_midi.PrettyMIDI(str(p))
            n_notes = sum(len(i.notes) for i in pm.instruments)
        except Exception:
            pass
        out.append({
            "file": p.name,
            "name": p.stem,
            "n_notes": n_notes,
            "size_kb": round(p.stat().st_size / 1024, 1),
            "created": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
        })
    out.sort(key=lambda e: e["created"], reverse=True)
    return out


def curated_path(file: str):
    """Resolve a filename safely inside the curated pool dir."""
    return CURATED_DIR / Path(file).name


def rename(file: str, name: str) -> bool:
    """Set the friendly name of a saved generation in the manifest."""
    p = safe_path(file)
    if not p.exists():
        return False
    meta = _load()
    entry = meta.get(p.name, {})
    entry["name"] = name.strip() or p.stem
    entry.setdefault("created", datetime.fromtimestamp(
        p.stat().st_mtime).isoformat(timespec="seconds"))
    meta[p.name] = entry
    _save(meta)
    return True


def delete(file: str) -> bool:
    """Delete a generation file from output/ and drop its manifest entry."""
    p = safe_path(file)
    if not p.exists():
        return False
    try:
        p.unlink()
    except Exception:
        return False
    meta = _load()
    if p.name in meta:
        del meta[p.name]
        _save(meta)
    return True
