"""Melodic voice-leading constraints, as independently toggleable rules.

Each rule contributes a per-interval log-weight; the enabled rules' weights are
summed and (× a global strength) added to NOTE_ON logits during sampling, so the
model's note choices are biased toward the chosen statistics.
"""
from __future__ import annotations

import math

# Metadata shown in the UI (id, name, description).
RULES = [
    {
        "id": "stepwise",
        "name": "Stepwise motion dominates",
        "desc": "~50% of moves are m2/M2 seconds and ~19% repeated notes; only "
                "~31% are leaps of a third or larger.",
    },
    {
        "id": "direction",
        "name": "Balanced direction",
        "desc": "~39% up, ~41% down, ~19% repeated — smooth, conjunct lines with "
                "a slight downward tendency.",
    },
    {
        "id": "rare_leaps",
        "name": "Large / dissonant leaps rare",
        "desc": "Sixths and sevenths essentially absent; the largest common leap "
                "is the octave (~2.4%).",
    },
]


def _stepwise_w(i: int) -> float:
    # magnitude preference: favour repeats & seconds over leaps
    return {0: 0.19, 1: 0.18, 2: 0.32, 3: 0.11, 4: 0.08,
            5: 0.06, 6: 0.05, 7: 0.04}.get(abs(i), 0.03)


def _direction_w(i: int) -> float:
    # up / down / repeat balance (slight downward bias)
    return 0.19 if i == 0 else (0.41 if i < 0 else 0.39)


def _rare_w(i: int) -> float:
    a = abs(i)
    if a in (8, 9, 10, 11):
        return 0.02      # 6ths & 7ths — essentially absent
    if a == 6:
        return 0.2       # tritone — rare
    if a == 12:
        return 0.3       # octave — allowed, but the rarest "common" leap
    if a > 12:
        return 0.01      # wider than an octave — almost never
    return 1.0           # seconds..fifths — untouched by this rule


_RULE_FN = {"stepwise": _stepwise_w, "direction": _direction_w, "rare_leaps": _rare_w}


def list_rules() -> list[dict]:
    """The selectable voice-leading rules (id, name, description) for the UI."""
    return [dict(r) for r in RULES]


# --- "stay within key" constraint -------------------------------------------
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
_MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10)   # natural minor


def list_keys() -> list[str]:
    """All 24 key names ('C major' … 'B minor')."""
    return [f"{n} {m}" for m in ("major", "minor") for n in NOTE_NAMES]


def scale_pcs(key_str: str):
    """Pitch classes (set of 0-11) allowed in ``key_str`` e.g. 'C# minor'."""
    try:
        name, mode = key_str.rsplit(" ", 1)
        pc = NOTE_NAMES.index(name)
    except (ValueError, AttributeError):
        return None
    scale = _MAJOR_SCALE if mode == "major" else _MINOR_SCALE
    return {(pc + s) % 12 for s in scale}


def logbias(interval: int, enabled) -> float:
    """Summed log-weight for a signed interval across the enabled rule ids."""
    total = 0.0
    for rid in enabled:
        fn = _RULE_FN.get(rid)
        if fn:
            total += math.log(max(fn(interval), 1e-9))
    return total
