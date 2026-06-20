"""Generate simple, musical demo MIDI files so the pipeline is runnable today.

Writes scales, arpeggios and small motifs across several keys into data/midi/.
Replace these with a real corpus when you have one — same folder, same format.

    python scripts/make_demo_data.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pretty_midi  # noqa: E402

from aimusicgen import config as C  # noqa: E402

MAJOR = [0, 2, 4, 5, 7, 9, 11, 12]
TRIAD = [0, 4, 7, 12]
KEYS = [60, 62, 64, 65, 67, 69]  # C D E F G A (octave 4)
STEP = C.SECONDS_PER_STEP


def _write(notes: list[tuple[int, float, float]], path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(C.DEFAULT_TEMPO))
    inst = pretty_midi.Instrument(program=0)
    for pitch, start, end in notes:
        inst.notes.append(pretty_midi.Note(velocity=90, pitch=pitch,
                                            start=start, end=end))
    pm.instruments.append(inst)
    pm.write(str(path))


def scale_song(root: int, descending: bool = False) -> list[tuple[int, float, float]]:
    degs = MAJOR[::-1] if descending else MAJOR
    notes, t = [], 0.0
    for _ in range(2):
        for d in degs:
            notes.append((root + d, t, t + STEP * 2))
            t += STEP * 2
    return notes


def arpeggio_song(root: int) -> list[tuple[int, float, float]]:
    notes, t = [], 0.0
    for _ in range(4):
        for d in TRIAD:
            notes.append((root + d, t, t + STEP))
            t += STEP
    return notes


def motif_song(root: int, rng: random.Random) -> list[tuple[int, float, float]]:
    notes, t = [], 0.0
    for _ in range(24):
        d = rng.choice(MAJOR)
        dur = rng.choice([STEP, STEP * 2, STEP * 2])
        notes.append((root + d, t, t + dur))
        t += dur
        if rng.random() < 0.15:        # occasional rest
            t += STEP
    return notes


def main() -> None:
    rng = random.Random(0)
    out = C.DATA_DIR
    n = 0
    for root in KEYS:
        _write(scale_song(root), out / f"scale_up_{root}.mid"); n += 1
        _write(scale_song(root, descending=True), out / f"scale_down_{root}.mid"); n += 1
        _write(arpeggio_song(root), out / f"arp_{root}.mid"); n += 1
        for k in range(4):
            _write(motif_song(root, rng), out / f"motif_{root}_{k}.mid"); n += 1
    print(f"Wrote {n} demo MIDI files to {out}")


if __name__ == "__main__":
    main()
