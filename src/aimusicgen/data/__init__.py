"""Data layer.

Modules:
- ``events``   — MIDI ⇄ polyphonic event-token stream (the training encoding).
- ``dataset``  — corpus loading + sliding-window torch ``Dataset``.
- ``corpus``   — the bundled music21 corpus: list composers/works, raw dumps,
  MIDI playback, and importing works to ``data/midi/`` for training.
- ``notation`` — render a generated MIDI to MusicXML for in-app staff display.
"""
