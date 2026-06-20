"""Send extracted notes to a locally-running Mistral (via Ollama) for a
key / interval analysis.

No API key needed — talks to the local Ollama server (default
http://localhost:11434, model `mistral`). Override with OLLAMA_HOST / OLLAMA_MODEL.
Uses stdlib urllib (no extra dependency).
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


def _host() -> str:
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def _model() -> str:
    return os.environ.get("OLLAMA_MODEL") or os.environ.get("MISTRAL_MODEL") or "mistral"


def analyze_notes(name: str, note_names: list[str], intervals: list[int],
                  question: str = "", timeout: int = 120) -> str:
    """Ask the local model for a key/interval/voice-leading analysis of a note
    list. Builds a prompt from ``note_names`` (and the optional user ``question``),
    POSTs it to Ollama, and returns the reply text. Raises RuntimeError if Ollama
    is unreachable or the model is missing."""
    model = _model()
    seq = " ".join(note_names[:400])
    more = "" if len(note_names) <= 400 else f"  (+{len(note_names) - 400} more)"
    prompt = (
        "You are a music-theory analyst. The note onsets below were extracted "
        f'from a MIDI rendering of "{name}" using the Mido library.\n\n'
        f"Notes ({len(note_names)}): {seq}{more}\n\n"
        "Give a concise analysis:\n"
        "1. Most likely key, with the evidence (scale, accidentals, final notes).\n"
        "2. Interval makeup: approximate %% of repeated notes, steps (m2/M2) and "
        "leaps (3rd+), and which leaps occur.\n"
        "3. Notable voice-leading / melodic traits.\n"
        "Keep it under ~200 words."
    )
    if question and question.strip():
        prompt += ("\n\nThen also answer this question from the user about the "
                   f"piece:\n{question.strip()}")
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.2},
    }
    req = urllib.request.Request(
        _host() + "/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="ignore")[:300]
        raise RuntimeError(
            f"Ollama error {e.code}: {detail}\n(Is the model pulled? `ollama pull {model}`)")
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Couldn't reach Ollama at {_host()} ({e.reason}). "
            f"Start it with `ollama serve` and `ollama pull {model}`.")
    except Exception as e:
        raise RuntimeError(f"Ollama request failed: {e}")
    return (data.get("message", {}).get("content") or "(empty response)").strip()
