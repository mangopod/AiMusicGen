"""Append a harmonic cadence to a generated piece.

The model has no notion of key/chords, so a cadence can't be reliably coaxed by
sampling — instead we detect the key (Krumhansl-Schmuckler) from the generated
notes and append the cadence chords in that key:
  - half cadence:   ... -> V        (ends on the dominant, sounds suspended)
  - plagal cadence: ... -> IV -> I   ("Amen")
"""
from __future__ import annotations

import random

import numpy as np
import pretty_midi

from . import config as C

# Krumhansl-Kessler key profiles.
_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def detect_key(notes) -> tuple[int, str]:
    """Return (tonic_pitch_class, mode) for a list of pretty_midi Notes."""
    hist = np.zeros(12)
    for n in notes:
        hist[n.pitch % 12] += max(0.0, n.end - n.start)
    if hist.sum() == 0 or np.std(hist) == 0:
        return 0, "major"
    best = (-2.0, 0, "major")
    for mode, prof in (("major", _MAJ), ("minor", _MIN)):
        for t in range(12):
            r = float(np.nan_to_num(np.corrcoef(hist, np.roll(prof, t))[0, 1]))
            if r > best[0]:
                best = (r, t, mode)
    return best[1], best[2]


def _triad(root_pc: int, quality: str, center: int) -> list[int]:
    """A close-position triad + an octave-lower bass, sitting near ``center``."""
    third = 4 if quality == "maj" else 3
    root = root_pc + 12 * round((center - root_pc) / 12)
    pitches = [root - 12, root, root + third, root + 7]
    return [p for p in pitches if C.PITCH_MIN <= p <= C.PITCH_MAX]


def _parse_key(key_str):
    """'C# minor' -> (1, 'minor'); None/invalid -> None."""
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    try:
        name, mode = key_str.rsplit(" ", 1)
        return names.index(name), ("minor" if mode == "minor" else "major")
    except (ValueError, AttributeError):
        return None


def append_cadence(pm: pretty_midi.PrettyMIDI, kind: str,
                   key: str | None = None) -> pretty_midi.PrettyMIDI:
    """Append a half ('half') or plagal ('plagal') cadence. Uses ``key`` if given,
    otherwise detects the key from the notes."""
    notes = [n for inst in pm.instruments for n in inst.notes]
    if not notes or not pm.instruments:
        return pm
    parsed = _parse_key(key)
    tonic, mode = parsed if parsed else detect_key(notes)
    center = int(np.median([n.pitch for n in notes]))
    step = C.SECONDS_PER_STEP
    beat = step * 4                                   # a quarter note
    start = (round(max(n.end for n in notes) / step) + 2) * step   # short gap

    if kind == "half":                                # ... -> V (major dominant)
        chords = [(_triad((tonic + 7) % 12, "maj", center), beat * 3)]
    else:                                             # plagal: IV -> I (diatonic)
        q = "maj" if mode == "major" else "min"
        chords = [(_triad((tonic + 5) % 12, q, center), beat * 2),
                  (_triad(tonic % 12, q, center), beat * 4)]

    inst = pm.instruments[0]
    cur = start
    for pitches, dur in chords:
        for p in pitches:
            inst.notes.append(pretty_midi.Note(velocity=85, pitch=p,
                                               start=cur, end=cur + dur))
        cur += dur
    return pm


def resolve_kind(cadence: str | None) -> str | None:
    if cadence in ("half", "plagal"):
        return cadence
    if cadence == "either":
        return random.choice(["half", "plagal"])
    return None
