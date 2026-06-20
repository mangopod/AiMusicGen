"""AiMusicGen — local-ML music generation in a native macOS app.

A PyTorch LSTM is trained on MIDI corpora (the bundled music21 library or your own
files) and generates polyphonic music, shown in a pywebview window with in-app
playback, staff notation, saved-model/generation libraries, voice-leading & key
constraints, and a rule-based species-counterpoint / fugue generator.

Module map:
- ``config``      — paths, the event-token scheme, hyperparameters, device.
- ``data``        — MIDI⇄token encoding, datasets, the music21 corpus, notation.
- ``model``       — the ``MusicLSTM`` network.
- ``train`` / ``generate`` — training loop and autoregressive sampling.
- ``models`` / ``library`` — saved-checkpoint and saved-generation libraries.
- ``constraints`` / ``cadence`` / ``counterpoint`` — music-theory tools.
- ``mistral``     — note analysis via a local Ollama model.
- ``app``         — the native window and the JS↔Python bridge (``app.api.Api``).

See README.md and Training.md for usage and the data pipeline.
"""
__version__ = "0.1.0"
