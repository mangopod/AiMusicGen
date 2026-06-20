# Training — data pipeline & model

How AiMusicGen turns MIDI files into the tensors the LSTM trains on, and how the
training loop works. Code references point at the modules under `src/aimusicgen/`.

```
.mid files ──► tokenize ──► list of token sequences ──► sliding windows ──► batches ──► LSTM
(data/midi/)  (events.py)   (dataset.load_corpus)      (MidiSequenceDataset)  (DataLoader)  (model/lstm.py)
```

## 1. Data sources

Training reads every `.mid` / `.midi` file under a folder (e.g. `data/midi/bach/`).
Files arrive two ways:

- **Corpus import** — `corpus.import_works(composer, n)` renders music21 works to
  MIDI, **normalized to 120 BPM**, into `data/midi/<composer>/`. (Normalizing
  matters: the tokenizer's time grid assumes 120 BPM — see §2.)
- **Your own MIDI** — drop files into `data/midi/curated/` ("My MIDI" pool) or any
  subfolder.

You can train on one folder or several at once (a mixed-composer model).

## 2. MIDI → event tokens  (`data/events.py` → `midi_to_events`)

Each file becomes a flat list of integers (an event stream, MIDI-performance style):

1. **Parse** with `pretty_midi`; collect every note from non-drum instruments whose
   pitch is in the 88-key range (MIDI **21–108**). All voices are merged into one
   chronological stream — this is why the encoding is polyphonic.
2. **Quantize to a 16th-note grid**: `step = round(time / SECONDS_PER_STEP)` with
   `SECONDS_PER_STEP = 0.125 s` (a 16th note at 120 BPM).
3. **Emit events** walking the grid: at each active step, a `TIME_SHIFT` for the gap
   since the previous event, then the `NOTE_OFF`s, then the `NOTE_ON`s. Leading
   silence is dropped.

### Vocabulary — 193 tokens (`config.py`, `ENCODING = "events-v1"`)

| token | id range | meaning |
|---|---|---|
| `PAD` | 0 | padding (never generated) |
| `TIME_SHIFT_k` | 1–16 | advance the clock by *k* 16th-steps (`MAX_SHIFT = 16`; longer gaps chain) |
| `NOTE_ON_p` | 17–104 | start pitch *p* (21–108) |
| `NOTE_OFF_p` | 105–192 | end pitch *p* |

### Worked example

A C-major triad (C4 E4 G4) held a quarter note (4 steps), then released:

```
step 0:  NOTE_ON 60, NOTE_ON 64, NOTE_ON 67
step 4:  TIME_SHIFT 4, NOTE_OFF 60, NOTE_OFF 64, NOTE_OFF 67
→ [56, 60, 63,  4,  144, 148, 151]
    └ on C,E,G ┘ └sh┘ └  off C,E,G ┘
```

(`note_on(60) = 17 + (60−21) = 56`, `time_shift(4) = 4`, `note_off(60) = 105 + 39 = 144`.)

Decoding is the inverse (`events_to_midi`): walk the tokens, advance time on
`TIME_SHIFT`, open/close notes on `NOTE_ON`/`NOTE_OFF`. The decoder also applies two
**generation-only** safeguards (`MAX_NOTE_STEPS`, `MAX_VOICES`) so under-trained
output stays playable; these never affect the training encoder.

## 3. Folder → list of songs  (`data/dataset.py` → `load_corpus`)

`rglob`s the folder(s), runs `midi_to_events` on each file, and returns a **list of
token sequences** (one per file). Corrupt/unsupported files are skipped. Passing a
list of folders concatenates their songs (mixed-corpus training).

## 4. Songs → training windows  (`MidiSequenceDataset`)

The model is trained on **next-token prediction**, so every example is a window and
its one-step-shifted copy:

- Slide a window of length `seq_len + 1` across each song with a `stride`, producing
  many overlapping windows.
- For a window `w`: `x = w[:-1]`, `y = w[1:]` — at every position the target is the
  *next* token.
- Songs shorter than a window are padded with `PAD` up to one window.

```
window:  [t0 t1 t2 … t127 t128]
x     :  [t0 t1 t2 … t127]        input
y     :  [t1 t2 t3 … t128]        target (shifted by one)
```

## 5. Model  (`model/lstm.py` → `MusicLSTM`)

```
token id ─► Embedding(193 → 128) ─► LSTM(2 layers, hidden 256, dropout 0.3) ─► Linear(256 → 193) ─► logits
```

`forward(x)` returns logits of shape `(batch, seq, 193)` — a distribution over the
next token at every position — plus the LSTM hidden state (threaded through during
autoregressive generation).

## 6. Training loop  (`train.py` → `train`)

1. `load_corpus(data_dir)` → songs → `MidiSequenceDataset(seq_len, stride)`.
2. `random_split` holds out `val_split` for validation.
3. `DataLoader` shuffles and batches the training windows.
4. Per epoch: forward → **cross-entropy** loss of logits vs `y` with
   `ignore_index=PAD` (padding doesn't count) → backprop → gradient clip → Adam step.
   Then the validation loss is computed in `eval()` mode (no grad).
5. The **best-validation** checkpoint is saved to `checkpoints/<model>.pt`, tagged
   with the encoding version and `model_cfg` (so incompatible old checkpoints are
   detected and ignored on load).
6. A cosine learning-rate schedule decays the LR across the epochs.

### Defaults (`config.py`)

| | value |
|---|---|
| `seq_len` | 128 |
| `stride` | 4 |
| `batch_size` | 128 |
| `epochs` | 40 (UI passes 30) |
| `lr` | 2e-3 (Adam) |
| `grad_clip` | 1.0 |
| `val_split` | 0.1 |
| `seed` | 1337 |
| embedding / hidden / layers / dropout | 128 / 256 / 2 / 0.3 |
| device | MPS (Apple GPU) → CUDA → CPU |

### `train` / `val` losses
`train` = cross-entropy on the data the weights update on; `val` = the same on the
held-out 10%. Both should fall; if `train` keeps dropping while `val` plateaus or
rises, the model is overfitting — only the best-`val` epoch is kept.

## 7. Running training

From the project root:

```bash
# CLI — trains on everything under data/midi/ (saves a "CLI" model)
.venv/bin/python -m aimusicgen.train --epochs 30

# In the app — Model training view: pick composer(s), Works/Epochs, Train.
.venv/bin/python run.py
```

Each training run starts from **random weights** (no warm-start/resume) and trains
from scratch; in the app, training a composer/mix saves it as its own checkpoint in
the model library.

## 8. Generation (the reverse)

`generate.py` loads the active checkpoint, primes the model, samples a token, feeds
it back (carrying the LSTM state), and decodes the event stream to MIDI via
`events_to_midi`. Optional voice-leading constraints, a key restriction, and a final
cadence are applied during/after sampling.
