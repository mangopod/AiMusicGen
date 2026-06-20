"""Render a MIDI file as MusicXML so the webview can show real staff notation.

music21's MusicXML export is pure-Python (no MuseScore/Lilypond needed); the
in-app staff is drawn by OpenSheetMusicDisplay (OSMD) in the webview.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from music21 import converter


def midi_to_musicxml(midi_path: str | Path) -> str:
    """Parse a MIDI file and return its MusicXML as a string."""
    score = converter.parse(str(midi_path))
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(suffix=".musicxml")
        os.close(fd)
        score.write("musicxml", fp=tmp)
        return Path(tmp).read_text(encoding="utf-8")
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)
