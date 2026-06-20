# AiMusicGen

Local-ML music generation in a **native mac app**. A PyTorch **LSTM** is trained
on a corpus of MIDI files (or the bundled **music21 Bach corpus**), then generates
new **polyphonic** music. The UI is a native window (via `pywebview`) wrapping an
HTML/JS front end, with in-page playback and a piano-roll visualizer (Web Audio —
no external synth needed), plus a Bach corpus browser and one-click training.

## Architecture

```
run.py                       launches the native window
src/aimusicgen/
  config.py                  paths, event tokenization, hyperparams, device (MPS/CUDA/CPU)
  data/
    events.py                MIDI <-> polyphonic event-token stream
    corpus.py                music21 Bach corpus: list / raw view / import to MIDI
    dataset.py               corpus loading + sliding-window torch Dataset
  model/lstm.py              MusicLSTM (embedding -> LSTM -> linear head)
  train.py                   training loop + checkpointing  (python -m aimusicgen.train)
  generate.py                autoregressive sampling -> .mid (python -m aimusicgen.generate)
  app/
    __init__.py              pywebview window
    api.py                   JS<->Python bridge (generate / status / corpus / train)
    ui/                      index.html, style.css, app.js
scripts/make_demo_data.py    writes simple training MIDIs so it runs today
data/midi/                   training corpus (put .mid files here; bach/ = imported chorales)
checkpoints/                 saved model weights (tagged with encoding version)
output/                      generated .mid files
```

### How music is represented
A single **event-token stream** on a 16th-note grid (MIDI-style), so arbitrary
**polyphony** is supported (e.g. all four voices of a chorale):

- `TIME_SHIFT_k` — advance the clock by *k* grid steps
- `NOTE_ON_p` / `NOTE_OFF_p` — start / end a note (MIDI pitch 21–108)

Vocabulary size: **193**. On decode, two safeguards keep generated output musical
regardless of model maturity: a max sustain length (`MAX_NOTE_STEPS`) and a max
simultaneous-voice cap (`MAX_VOICES`). These affect generation only, not training.

> Known v1 limitation: a single stream can't represent two voices playing the
> *same pitch* at the same instant — rare unison doublings collapse to one note.

For the full data pipeline (MIDI → tokens → windows → batches) and the training
loop, see **[Training.md](Training.md)**.

## Setup

A venv already exists (`.venv`, reusing the system MPS-enabled PyTorch). To
recreate elsewhere:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Quick start (runnable now)

```bash
# 1. make a small demo corpus (scales / arpeggios / motifs)
.venv/bin/python scripts/make_demo_data.py

# 2. train (uses Apple-Silicon GPU if available; ~a minute on demo data)
.venv/bin/python -m aimusicgen.train

# 3. launch the app, then click Generate
.venv/bin/python run.py
```

CLI generation without the app:

```bash
.venv/bin/python -m aimusicgen.generate --length 512 --temperature 0.9 --top-k 12
```

## Train on Bach (in-app, one click)
The app's **Bach corpus** panel lists all 433 works (music21, bundled/offline),
lets you view any work's raw data, and has a **Train on Bach (4-part)** button:
it imports the soprano+alto+tenor+bass of *N* chorales to `data/midi/bach/`, then
trains on the full polyphony with a live progress bar. Training **overwrites** the
shared checkpoint; `generate()` is disabled while it runs.

## Using your own data
Drop `.mid` / `.midi` files into `data/midi/` (subfolders are fine) and retrain.
All voices/instruments are merged into one polyphonic event stream — no flattening.

## Roadmap
- Transformer backbone option
- Conditioning (key / tempo / style), temperature schedules
- Per-voice / multi-track encoding (preserve unison doublings & voice identity)
- `.app` packaging (briefcase / py2app)
