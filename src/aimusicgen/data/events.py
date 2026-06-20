"""Event-based polyphonic encoding: MIDI <-> a MIDI-style event token stream.

All voices/instruments are merged into one chronological event stream on the
16th-note grid (see config). NOTE_ON / NOTE_OFF / TIME_SHIFT tokens capture
arbitrary polyphony — e.g. the four simultaneous voices of a Bach chorale.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pretty_midi

from .. import config as C


def midi_to_events(path: str | Path) -> list[int]:
    """Load a MIDI file and return a flat list of event tokens."""
    pm = pretty_midi.PrettyMIDI(str(path))
    notes = [
        n
        for inst in pm.instruments
        if not inst.is_drum
        for n in inst.notes
        if C.PITCH_MIN <= n.pitch <= C.PITCH_MAX
    ]
    if not notes:
        return []

    sps = C.SECONDS_PER_STEP
    ons: dict[int, list[int]] = defaultdict(list)
    offs: dict[int, list[int]] = defaultdict(list)
    for n in notes:
        s0 = int(round(n.start / sps))
        s1 = max(s0 + 1, int(round(n.end / sps)))
        ons[s0].append(n.pitch)
        offs[s1].append(n.pitch)

    steps = sorted(set(ons) | set(offs))
    tokens: list[int] = []
    cur = steps[0]  # start the clock at the first event (drop leading silence)
    for s in steps:
        delta = s - cur
        while delta > 0:
            k = min(delta, C.MAX_SHIFT)
            tokens.append(C.time_shift_token(k))
            delta -= k
        for p in sorted(offs.get(s, [])):   # close before (re)opening at same instant
            tokens.append(C.note_off_token(p))
        for p in sorted(ons.get(s, [])):
            tokens.append(C.note_on_token(p))
        cur = s
    return tokens


def _close(inst, pitch, s0, s1, sps):
    if s1 > s0:
        inst.notes.append(
            pretty_midi.Note(velocity=90, pitch=pitch, start=s0 * sps, end=s1 * sps)
        )


def events_to_midi(tokens: list[int], tempo: int = C.DEFAULT_TEMPO,
                   program: int = 0) -> pretty_midi.PrettyMIDI:
    """Decode an event token stream back into a (possibly polyphonic) score."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(tempo))
    inst = pretty_midi.Instrument(program=program)
    sps = C.SECONDS_PER_STEP

    t = 0                      # current time, in grid steps
    active: dict[int, int] = {}  # pitch -> onset step (insertion order = age)
    for tok in tokens:
        if tok == C.PAD:
            continue
        if C.is_time_shift(tok):
            t += C.shift_amount(tok)
            # auto-close notes sustained past the limit (dangling-note guard)
            for p in [p for p, s0 in active.items() if t - s0 >= C.MAX_NOTE_STEPS]:
                s0 = active.pop(p)
                _close(inst, p, s0, s0 + C.MAX_NOTE_STEPS, sps)
        elif C.is_note_on(tok):
            p = C.note_on_pitch(tok)
            if p in active:        # re-onset without an off: close the old segment
                _close(inst, p, active[p], t, sps)
            elif len(active) >= C.MAX_VOICES:  # keep the texture playable
                old = next(iter(active))
                _close(inst, old, active.pop(old), t, sps)
            active[p] = t
        elif C.is_note_off(tok):
            p = C.note_off_pitch(tok)
            if p in active:
                _close(inst, p, active.pop(p), t, sps)

    for p, s0 in active.items():   # close anything still sounding at the end
        _close(inst, p, s0, min(t + 1, s0 + C.MAX_NOTE_STEPS), sps)

    pm.instruments.append(inst)
    return pm


def save_events_as_midi(tokens: list[int], path: str | Path,
                        tempo: int = C.DEFAULT_TEMPO) -> Path:
    """Decode ``tokens`` to MIDI and write it to ``path`` (returns the path)."""
    path = Path(path)
    events_to_midi(tokens, tempo=tempo).write(str(path))
    return path
