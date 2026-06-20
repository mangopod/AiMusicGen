"""Query the bundled music21 corpus (works offline).

Used by the app's corpus-browser widget: pick a composer/collection, list its
works, render a readable dump of a selected work, and play / train on it.
"""
from __future__ import annotations

import base64
import functools
import os
import tempfile
from pathlib import Path

import mido
import pretty_midi
from music21 import common, converter, corpus, stream

from .. import config as C

DEFAULT_COMPOSER = "bach"
_MAX_RAW_CHARS = 120_000
SCORE_EXTS = ("*.mxl", "*.xml", "*.musicxml", "*.krn", "*.abc", "*.mid",
              "*.midi", "*.rntxt", "*.capx")


def corpus_base() -> Path:
    return Path(common.getCorpusFilePath())


def composer_dir(composer: str) -> Path:
    """Where imported MIDI for a composer lands (a per-composer subfolder)."""
    return C.DATA_DIR / composer


@functools.lru_cache(maxsize=1)
def list_composers() -> list[dict]:
    """Every top-level composer/collection folder in the corpus, with a work
    count. Alphabetical so it reads well in a dropdown."""
    out: list[dict] = []
    for d in corpus_base().iterdir():
        if not d.is_dir() or d.name.startswith((".", "_")):
            continue
        n = sum(1 for ext in SCORE_EXTS for _ in d.rglob(ext))
        if n:
            out.append({"name": d.name, "count": n})
    out.sort(key=lambda c: c["name"].lower())
    return out


@functools.lru_cache(maxsize=32)
def list_works(composer: str = DEFAULT_COMPOSER) -> list[dict]:
    """All works for a composer. Fast — just file listing, no parsing.
    ``name`` is the path relative to the composer folder (handles nesting)."""
    root = corpus_base() / composer
    works: list[dict] = []
    seen: set[str] = set()
    for ext in SCORE_EXTS:                     # SCORE_EXTS order = format preference
        for p in root.rglob(ext):
            name = str(p.relative_to(root).with_suffix(""))
            if name in seen:                   # same piece in another format -> skip
                continue
            seen.add(name)
            works.append({"path": str(p), "name": name, "format": p.suffix.lstrip(".")})
    works.sort(key=lambda w: w["name"].lower())
    return works


def _as_score(parsed):
    """A multi-tune file (e.g. .abc) parses to an Opus; take its first score."""
    if isinstance(parsed, stream.Opus):
        scores = parsed.scores
        return scores[0] if scores else parsed
    return parsed


def _score_tempo(score, default: int = C.DEFAULT_TEMPO) -> int:
    for mm in score.recurse().getElementsByClass("MetronomeMark"):
        if mm.number:
            return int(mm.number)
    return default


def score_to_pretty_midi(score, tempo: int = C.DEFAULT_TEMPO) -> pretty_midi.PrettyMIDI:
    """Build a PrettyMIDI straight from the parsed notes (one instrument per
    part). Bypasses music21's MIDI writer, which chokes on some corpus files
    (unexpandable repeats, duplicate conductor elements, etc.)."""
    spq = 60.0 / tempo                          # seconds per quarter note
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(tempo))
    for part in (list(score.parts) or [score]):
        inst = pretty_midi.Instrument(program=0)
        for el in part.flatten().notesAndRests:
            if el.isRest:
                continue
            start = float(el.offset) * spq
            dur = float(el.duration.quarterLength) * spq
            if dur <= 0:
                continue
            pitches = el.pitches if el.isChord else [el.pitch]
            for p in pitches:
                if 0 <= p.midi <= 127:
                    inst.notes.append(pretty_midi.Note(
                        velocity=80, pitch=p.midi, start=start, end=start + dur))
        if inst.notes:
            pm.instruments.append(inst)
    return pm


def _soprano_part(score):
    """Pick the soprano voice; fall back to the first/top part."""
    for p in score.parts:
        if "soprano" in (p.partName or "").lower():
            return p
    return score.parts[0] if score.parts else score


def import_works(composer: str = DEFAULT_COMPOSER, n: int | None = None,
                 voice: str = "all", progress_cb=None) -> dict:
    """Export the first ``n`` works of ``composer`` (or all) to MIDI in
    ``data/midi/<composer>/`` so the MIDI->token training pipeline can use them.

    ``voice="all"`` keeps every voice (full polyphony); ``voice="soprano"`` keeps
    only the top line. Returns a summary.
    """
    dest = composer_dir(composer)
    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.glob("*.mid"):       # rebuild from scratch so n_works is exact
        old.unlink()
    works = list_works(composer)
    if n:
        works = works[:n]

    imported, skipped = 0, 0
    total = len(works)
    for i, w in enumerate(works, 1):
        try:
            score = _as_score(converter.parse(w["path"]))
            src = score if voice == "all" else _soprano_part(score)
            # normalize tempo to 120 BPM so the token grid is consistent
            pm = score_to_pretty_midi(src, tempo=C.DEFAULT_TEMPO)
            if not pm.instruments:
                raise ValueError("no notes")
            stem = w["name"].replace("/", "_")   # flat filename, no subdirs
            pm.write(str(dest / (stem + ".mid")))
            imported += 1
        except Exception:
            skipped += 1
        if progress_cb is not None:
            progress_cb(i, total, imported)
    return {"imported": imported, "skipped": skipped, "total": total,
            "dest": str(dest)}


def _describe(el) -> str:
    if el.isRest:
        return "rest"
    if el.isChord:
        return "[" + " ".join(pt.nameWithOctave for pt in el.pitches) + "]"
    return el.nameWithOctave


def _score_to_data_uri(score) -> str | None:
    """Render a parsed score to MIDI bytes as a base64 data URI (for playback),
    keeping the score's own tempo so it sounds right."""
    tmp = None
    try:
        pm = score_to_pretty_midi(score, tempo=_score_tempo(score))
        if not pm.instruments:
            return None
        fd, tmp = tempfile.mkstemp(suffix=".mid")
        os.close(fd)
        pm.write(tmp)
        data = Path(tmp).read_bytes()
        return "data:audio/midi;base64," + base64.b64encode(data).decode()
    except Exception:
        return None
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)


def notes_via_mido(path: str) -> list[int]:
    """Render the work to MIDI and read its note onsets back with Mido."""
    score = _as_score(converter.parse(path))
    pm = score_to_pretty_midi(score, tempo=_score_tempo(score))
    fd, tmp = tempfile.mkstemp(suffix=".mid")
    os.close(fd)
    try:
        pm.write(tmp)
        mf = mido.MidiFile(tmp)
        return [m.note for m in mf if m.type == "note_on" and m.velocity > 0]
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def raw_text(path: str) -> dict:
    """Parse one work and return a human-readable dump of its raw data."""
    score = _as_score(converter.parse(path))
    md = score.metadata
    title = (md.title if md and md.title else None) or Path(path).stem

    try:
        key = str(score.analyze("key"))
    except Exception:
        key = "?"
    ts = score.recurse().getElementsByClass("TimeSignature")
    time_sig = ts[0].ratioString if ts else "?"

    parts = list(score.parts) or [score]
    total_notes = sum(len(p.flatten().notes) for p in parts)

    lines = [
        f"# {title}",
        f"file   : {Path(path).name}",
        f"key    : {key}",
        f"meter  : {time_sig}",
        f"parts  : {len(parts)}",
        f"notes  : {total_notes}",
        f"format : offset(quarters)  pitch  q=duration",
    ]
    for i, part in enumerate(parts):
        name = part.partName or getattr(part, "id", None) or f"part {i + 1}"
        lines.append(f"\n## Part {i + 1}: {name}")
        for el in part.flatten().notesAndRests:
            off = round(float(el.offset), 3)
            dur = round(float(el.duration.quarterLength), 3)
            lines.append(f"  {off:>8}  {_describe(el):<14} q={dur}")

    text = "\n".join(lines)
    truncated = len(text) > _MAX_RAW_CHARS
    if truncated:
        text = text[:_MAX_RAW_CHARS] + "\n\n… (truncated)"

    return {
        "name": title,
        "key": key,
        "time_signature": time_sig,
        "n_parts": len(parts),
        "n_notes": total_notes,
        "truncated": truncated,
        "text": text,
        "midi_data_uri": _score_to_data_uri(score),
    }
