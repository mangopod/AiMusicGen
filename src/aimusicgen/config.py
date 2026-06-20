"""Central configuration: paths, tokenization scheme, model hyperparameters, device."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

# --- Paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "midi"
CKPT_DIR = ROOT / "checkpoints"
OUTPUT_DIR = ROOT / "output"
CKPT_PATH = CKPT_DIR / "model.pt"

for _d in (DATA_DIR, CKPT_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Tokenization (event-based, polyphonic) ------------------------------
# A single token stream of MIDI-style events on a fixed 16th-note grid:
#   PAD                padding (never generated)
#   TIME_SHIFT_<k>     advance the clock by k grid steps (1..MAX_SHIFT)
#   NOTE_ON_<pitch>    start a note   (pitch in [PITCH_MIN, PITCH_MAX])
#   NOTE_OFF_<pitch>   end a note
# This represents arbitrary polyphony — any number of simultaneous voices,
# e.g. all four voices of a Bach chorale. Bump ENCODING when the scheme
# changes so old checkpoints are detected as incompatible.
ENCODING = "events-v1"

PAD = 0
PITCH_MIN, PITCH_MAX = 21, 108           # 88-key piano range
N_PITCH = PITCH_MAX - PITCH_MIN + 1       # 88
MAX_SHIFT = 16                            # steps per TIME_SHIFT token (one 4/4 bar)

STEPS_PER_BEAT = 4                        # 16th-note grid
SECONDS_PER_STEP = 0.125                  # 16th note @ 120 BPM
DEFAULT_TEMPO = 120

# Decoder-only safeguards (shape generated output; do not affect the encoder /
# training data). Keep an under-trained model from stacking dangling notes.
MAX_NOTE_STEPS = 32                       # auto-close a note after 2 bars
MAX_VOICES = 6                            # cap simultaneous notes when decoding

_SHIFT_BASE = 1                              # tokens 1 .. MAX_SHIFT
_NOTE_ON_BASE = _SHIFT_BASE + MAX_SHIFT      # 17
_NOTE_OFF_BASE = _NOTE_ON_BASE + N_PITCH     # 105
VOCAB_SIZE = _NOTE_OFF_BASE + N_PITCH        # 1 + 16 + 88 + 88 = 193


def time_shift_token(k: int) -> int:
    """Token for advancing by k grid steps (1 <= k <= MAX_SHIFT)."""
    return _SHIFT_BASE + (k - 1)


def note_on_token(pitch: int) -> int:
    """Token id for starting note ``pitch`` (PITCH_MIN..PITCH_MAX)."""
    return _NOTE_ON_BASE + (pitch - PITCH_MIN)


def note_off_token(pitch: int) -> int:
    """Token id for ending note ``pitch``."""
    return _NOTE_OFF_BASE + (pitch - PITCH_MIN)


def is_time_shift(tok: int) -> bool:
    """True if ``tok`` is a TIME_SHIFT token."""
    return _SHIFT_BASE <= tok < _SHIFT_BASE + MAX_SHIFT


def is_note_on(tok: int) -> bool:
    """True if ``tok`` is a NOTE_ON token."""
    return _NOTE_ON_BASE <= tok < _NOTE_ON_BASE + N_PITCH


def is_note_off(tok: int) -> bool:
    """True if ``tok`` is a NOTE_OFF token."""
    return _NOTE_OFF_BASE <= tok < _NOTE_OFF_BASE + N_PITCH


def shift_amount(tok: int) -> int:
    """Number of grid steps a TIME_SHIFT token advances (inverse of time_shift_token)."""
    return tok - _SHIFT_BASE + 1


def note_on_pitch(tok: int) -> int:
    """MIDI pitch of a NOTE_ON token (inverse of note_on_token)."""
    return tok - _NOTE_ON_BASE + PITCH_MIN


def note_off_pitch(tok: int) -> int:
    """MIDI pitch of a NOTE_OFF token (inverse of note_off_token)."""
    return tok - _NOTE_OFF_BASE + PITCH_MIN


# --- Model / training ----------------------------------------------------
@dataclass
class ModelConfig:
    """Architecture hyperparameters for :class:`model.lstm.MusicLSTM`
    (vocabulary size, embedding & hidden widths, LSTM depth, dropout)."""
    vocab_size: int = VOCAB_SIZE
    embed_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.3


@dataclass
class TrainConfig:
    """Training hyperparameters (window size/hop, batch, epochs, LR, etc.) plus
    the nested :class:`ModelConfig`. See Training.md for what each controls."""
    seq_len: int = 128            # events are denser than frames -> longer context
    stride: int = 4              # window hop (keeps polyphonic corpora tractable)
    batch_size: int = 128
    epochs: int = 40
    lr: float = 2e-3
    grad_clip: float = 1.0
    val_split: float = 0.1
    seed: int = 1337
    model: ModelConfig = field(default_factory=ModelConfig)


def get_device() -> torch.device:
    """Prefer Apple-Silicon GPU (MPS), then CUDA, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
