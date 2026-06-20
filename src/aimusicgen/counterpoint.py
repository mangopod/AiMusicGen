"""First-species (note-against-note) counterpoint, after Fux's Gradus ad Parnassum.

Generates a cantus firmus, then searches (backtracking) for a counterpoint line
that obeys the species-1 rules: only consonances; perfect consonances approached
only by contrary/oblique motion (no parallel/hidden 5ths/8ves); stepwise-leaning
melody with leaps recovered; begins and ends on a perfect consonance with a
stepwise leading-tone cadence.
"""
from __future__ import annotations

import math
import random

import pretty_midi

from . import config as C

MAJOR = (0, 2, 4, 5, 7, 9, 11)
NAT_MINOR = (0, 2, 3, 5, 7, 8, 10)
_LEGAL_MELODIC = {1, 2, 3, 4, 5, 7, 8, 12}   # m2..P5, m6, P8 (no tritone/7th/M6)


def _perfect(d: int) -> bool:
    return d % 12 in (0, 7)


def _consonant(d: int) -> bool:
    return d % 12 in (0, 3, 4, 7, 8, 9)       # excludes P4(5), tritone(6), 2nds/7ths


def _ladder(tonic_midi: int, mode: str, span: int = 12) -> list[int]:
    base = MAJOR if mode == "major" else NAT_MINOR
    out, o = [], 0
    while True:
        for s in base:
            p = tonic_midi + 12 * o + s
            if p - tonic_midi > span:
                return out
            out.append(p)
        o += 1


def make_cantus_firmus(tonic: int, mode: str, length: int = 10,
                       rng: random.Random | None = None) -> list[int]:
    """A diatonic cantus firmus: starts/ends on the tonic, ends 2->1, mostly
    stepwise, one climax, within an octave."""
    rng = rng or random.Random()
    lad = _ladder(48 + tonic, mode, span=12)   # one octave of diatonic pitches
    top = min(len(lad) - 1, 7)
    for _ in range(800):
        idx = [0]
        for pos in range(1, length):
            if pos == length - 1:
                idx.append(0)
                break
            if pos == length - 2:
                idx.append(1)            # supertonic -> tonic cadence
                continue
            prev = idx[-1]
            last = idx[-1] - idx[-2] if len(idx) >= 2 else 0
            moves = [1, -1, 2, -2, 3, -3]
            rng.shuffle(moves)
            chosen = None
            for m in moves:
                ni = prev + m
                if ni < 0 or ni > top:
                    continue
                if abs(last) >= 2 and (m * last > 0 or abs(m) >= 2):
                    continue            # recover a leap with an opposite-direction step
                if ni == prev:
                    continue
                chosen = ni
                break
            if chosen is None:
                break
            idx.append(chosen)
        if len(idx) != length or idx[-2] != 1 or idx[-1] != 0:
            continue
        if any(idx[i] == idx[i - 1] for i in range(1, length)):
            continue
        pitches = [lad[i] for i in idx]
        if pitches.count(max(pitches)) != 1:     # single climax
            continue
        return pitches
    patt = ([0, 1, 2, 3, 4, 3, 2, 1] + [1] * length)[:length]
    patt[-1], patt[-2] = 0, 1
    return [lad[min(i, top)] for i in patt]


def first_species(cf: list[int], tonic: int, mode: str, above: bool = True,
                  rng: random.Random | None = None) -> list[int] | None:
    """Backtracking search for a species-1 counterpoint to ``cf``."""
    rng = rng or random.Random()
    n = len(cf)
    pcs = {(tonic + s) % 12 for s in (MAJOR if mode == "major" else NAT_MINOR)}
    if mode == "minor":
        pcs.add((tonic + 11) % 12)               # raised leading tone for the cadence
    diat = lambda p: p % 12 in pcs
    final_cp = cf[-1] + 12 if above else cf[-1] - 12   # end on the octave

    def candidates(cfp: int) -> list[int]:
        lo, hi = (cfp + 1, cfp + 16) if above else (cfp - 16, cfp - 1)
        return [p for p in range(lo, hi + 1)
                if diat(p) and _consonant(abs(p - cfp))]

    def melodic_ok(a: int, b: int) -> bool:
        return abs(a - b) in _LEGAL_MELODIC

    def motion_ok(p0: int, c0: int, p1: int, c1: int) -> bool:
        if _perfect(abs(p1 - c1)):
            cp_dir, cf_dir = p1 - p0, c1 - c0
            if cp_dir and cf_dir and (cp_dir > 0) == (cf_dir > 0):
                return False                      # similar motion into a perfect = forbidden
        return True

    def order(pos: int, cands: list[int], cp: list[int]) -> list[int]:
        def score(p):
            d = abs(p - cf[pos]); s = rng.random()
            if d % 12 in (3, 4, 8, 9):
                s += 2                            # prefer imperfect consonances
            if cp:
                if abs(p - cp[-1]) <= 2:
                    s += 2                        # prefer stepwise
                if cf[pos] - cf[pos - 1] and (p - cp[-1] > 0) != (cf[pos] - cf[pos - 1] > 0):
                    s += 1                        # prefer contrary motion
            return s
        return sorted(cands, key=score, reverse=True)

    def dfs(pos: int, cp: list[int]) -> bool:
        if pos == n:
            return True
        if pos == n - 1:
            p = final_cp
            if not diat(p) or not _perfect(abs(p - cf[pos])):
                return False
            if not (melodic_ok(cp[-1], p) and motion_ok(cp[-1], cf[pos - 1], p, cf[pos])):
                return False
            cp.append(p)
            return True
        for p in order(pos, candidates(cf[pos]), cp):
            if pos == 0:
                if not _perfect(abs(p - cf[0])):
                    continue
            else:
                if not (melodic_ok(cp[-1], p) and motion_ok(cp[-1], cf[pos - 1], p, cf[pos])):
                    continue
            cp.append(p)
            if dfs(pos + 1, cp):
                return True
            cp.pop()
        return False

    cp: list[int] = []
    return cp if dfs(0, cp) else None


# --- shared helpers for species 2-5 ---------------------------------------
def _diatonic_set(tonic, mode):
    pcs = {(tonic + s) % 12 for s in (MAJOR if mode == "major" else NAT_MINOR)}
    if mode == "minor":
        pcs.add((tonic + 11) % 12)
    return pcs


def _candidates(cfp, above, pcs):
    lo, hi = (cfp + 1, cfp + 16) if above else (cfp - 16, cfp - 1)
    return [p for p in range(lo, hi + 1) if p % 12 in pcs]


def _melodic_ok(a, b):
    return abs(a - b) in _LEGAL_MELODIC


def _motion_ok(p0, c0, p1, c1):
    if _perfect(abs(p1 - c1)):
        cpd, cfd = p1 - p0, c1 - c0
        if cpd and cfd and (cpd > 0) == (cfd > 0):
            return False        # similar motion into a perfect → parallel/hidden
    return True


def _is_step(a, b):
    return abs(a - b) in (1, 2)


def _pending_ok(pending, diss, p):
    """The resolution ``p`` of a weak dissonance ``diss``: continue stepwise in
    the approach direction (passing tone) or return to ``before`` (neighbour)."""
    before, direction, allow_neighbor = pending
    if p != diss and _is_step(diss, p) and ((p > diss) == (direction > 0)):
        return True
    return bool(allow_neighbor and p == before)


def _florid(cf, tonic, mode, above, subs, rng):
    """Species 2/3/5: ``subs[m]`` cells per measure. Downbeats consonant; weak
    cells may be dissonant only as passing/neighbour tones."""
    pcs = _diatonic_set(tonic, mode)
    final_cp = cf[-1] + 12 if above else cf[-1] - 12
    cells = [(m, pos, s) for m, s in enumerate(subs) for pos in range(s)]
    cfc = [cf[m] for (m, _p, _s) in cells]
    N = len(cells)

    def dfs(j, cp, pending):
        if j == N:
            return True
        _m, pos, s = cells[j]
        cfp = cfc[j]
        if j == N - 1:                       # forced final tonic (perfect)
            p = final_cp
            if p % 12 not in pcs or not _perfect(abs(p - cfp)):
                return False
            if not (_melodic_ok(cp[-1], p) and _motion_ok(cp[-1], cfc[j - 1], p, cfp)):
                return False
            if pending and not _pending_ok(pending, cp[-1], p):
                return False
            cp.append(p)
            return True
        cands = _candidates(cfp, above, pcs)

        def score(p):
            d = abs(p - cfp); sc = rng.random()
            if d % 12 in (3, 4, 8, 9):
                sc += 2
            if cp and abs(p - cp[-1]) <= 2:
                sc += 2
            return sc
        cands.sort(key=score, reverse=True)
        for p in cands:
            if j > 0 and pending and not _pending_ok(pending, cp[-1], p):
                continue
            cons = _consonant(abs(p - cfp))
            if pos == 0 and not cons:
                continue                     # downbeat must be consonant
            if j == 0:
                if not _perfect(abs(p - cfp)):
                    continue                 # begin on a perfect consonance
            else:
                if not (_melodic_ok(cp[-1], p) and _motion_ok(cp[-1], cfc[j - 1], p, cfp)):
                    continue
            new_pending = None
            if not cons:                     # weak dissonance → must be a passing/nbr
                if j == 0 or not _is_step(cp[-1], p):
                    continue
                new_pending = (cp[-1], 1 if p > cp[-1] else -1, s >= 4)
            cp.append(p)
            if dfs(j + 1, cp, new_pending):
                return True
            cp.pop()
        return False

    cp = []
    return cp if dfs(0, cp, None) else None


def _fourth_species(cf, tonic, mode, above, rng):
    """Suspensions: each note is prepared (consonant), held over the barline,
    and if dissonant there it resolves DOWN by step."""
    n = len(cf)
    pcs = _diatonic_set(tonic, mode)
    final_cp = cf[-1] + 12 if above else cf[-1] - 12

    def dfs(i, q):
        if i == n - 1:
            if not _perfect(abs(final_cp - cf[-1])):
                return False
            if q and not _consonant(abs(q[-1] - cf[-1])):   # last held note is a suspension
                if (q[-1] - final_cp) not in (1, 2):        # must resolve down by step
                    return False
            q.append(final_cp)
            return True
        cfp = cf[i]
        cands = [p for p in _candidates(cfp, above, pcs) if _consonant(abs(p - cfp))]
        cands.sort(key=lambda p: (rng.random() + (2 if (q and abs(p - q[-1]) <= 2) else 0)),
                   reverse=True)
        for p in cands:
            if i == 0:
                if not _perfect(abs(p - cfp)):
                    continue
            else:
                held = q[-1]                 # sounds on the downbeat of measure i
                if not _consonant(abs(held - cf[i])):       # it's a suspension
                    if (held - p) not in (1, 2):            # must resolve down by step
                        continue
                elif not _melodic_ok(held, p):
                    continue
            q.append(p)
            if dfs(i + 1, q):
                return True
            q.pop()
        return False

    q = []
    return q if dfs(0, q) else None


def _subs_for(species, n, rng):
    if species == 2:
        return [2] * (n - 1) + [1]
    if species == 3:
        return [4] * (n - 1) + [1]
    if species == 5:                          # florid: mixed note values per measure
        return [rng.choice([1, 2, 2, 4]) for _ in range(n - 1)] + [1]
    return [1] * n


def generate(tonic: int = 0, mode: str = "major", length: int = 10,
             above: bool = True, species: int = 1, W: float = 2.0,
             rng_seed: int | None = None) -> pretty_midi.PrettyMIDI:
    """Build a 2-voice (cantus firmus + counterpoint) PrettyMIDI for any species."""
    rng = random.Random(rng_seed)
    cf = cp_cells = None
    for _ in range(60):                        # retry fresh CFs until one solves
        cf = make_cantus_firmus(tonic, mode, length, rng)
        if species == 1:
            cp = first_species(cf, tonic, mode, above, rng)
            cp_cells = [(p, m * W, W) for m, p in enumerate(cp)] if cp else None
        elif species == 4:
            q = _fourth_species(cf, tonic, mode, above, rng)
            if q:                              # offset (syncopated) half-tied notes
                cp_cells = []
                for i, p in enumerate(q[:-1]):
                    cp_cells.append((p, i * W + W / 2, W))
                cp_cells.append((q[-1], (len(q) - 1) * W, W))
        else:
            subs = _subs_for(species, len(cf), rng)
            cp = _florid(cf, tonic, mode, above, subs, rng)
            if cp:
                cp_cells, j = [], 0
                for m, s in enumerate(subs):
                    for pos in range(s):
                        cp_cells.append((cp[j], m * W + pos * (W / s), W / s))
                        j += 1
        if cp_cells:
            break
    if not cp_cells:
        raise RuntimeError("Could not find a valid counterpoint — try again.")

    pm = pretty_midi.PrettyMIDI(initial_tempo=float(C.DEFAULT_TEMPO))
    cf_inst = pretty_midi.Instrument(program=0, name="Cantus firmus")
    cp_inst = pretty_midi.Instrument(program=0, name="Counterpoint")
    for m, a in enumerate(cf):
        cf_inst.notes.append(pretty_midi.Note(velocity=78, pitch=a,
                                              start=m * W, end=(m + 1) * W))
    for p, t0, dur in cp_cells:
        cp_inst.notes.append(pretty_midi.Note(velocity=84, pitch=p,
                                              start=t0, end=t0 + dur))
    pm.instruments.extend([cf_inst, cp_inst])
    return pm


# --- fugue: subject + tonal answer ----------------------------------------
def make_subject(tonic: int, mode: str, length: int = 8,
                 rng: random.Random | None = None) -> list[tuple[int, float]]:
    """A short diatonic fugue subject as [(midi, duration_in_beats)]. Starts on
    the tonic or dominant (often with a tonic-dominant head) so the tonal answer
    is meaningful."""
    rng = rng or random.Random()
    lad = _ladder(48 + tonic, mode, span=16)
    top = min(len(lad) - 1, 9)
    start = rng.choice([0, 4])                       # tonic or dominant
    idx = [start]
    if rng.random() < 0.5:                           # strong tonic<->dominant head
        idx.append(4 if start == 0 else 0)
    while len(idx) < length:
        prev = idx[-1]
        last = idx[-1] - idx[-2] if len(idx) >= 2 else 0
        nxt = None
        for m in [1, -1, 2, -2, 1, -1, 3, -3]:
            if rng.random() < 0.5:
                continue
            ni = prev + m
            if ni < 0 or ni > top:
                continue
            if abs(last) >= 3 and abs(m) >= 2 and m * last > 0:
                continue                            # don't stack big leaps same way
            nxt = ni
            break
        idx.append(nxt if nxt is not None else max(0, prev - 1))
    pitches = [lad[i] for i in idx[:length]]
    return list(zip(pitches, _subject_rhythm(length, rng)))


def _subject_rhythm(n: int, rng: random.Random) -> list[float]:
    """A varied, characterful rhythm (beats): a longer head, eighth/sixteenth
    runs, dotted figures, and a long final note."""
    body = [0.25, 0.5, 0.5, 0.75, 1.0, 1.0, 1.5]     # weighted palette
    durs = [rng.choice([1.0, 1.5, 2.0])]             # head: a longer note
    while len(durs) < n - 1:
        room = (n - 1) - len(durs)
        if room >= 2 and rng.random() < 0.45:        # an eighth/sixteenth run
            run = rng.choice([0.25, 0.5])
            durs += [run] * min(rng.randint(2, 3), room)
        else:
            durs.append(rng.choice(body))
    durs.append(rng.choice([1.5, 2.0]))              # long final note
    return durs[:n]


def tonal_answer(subject: list[tuple[int, float]], tonic: int, mode: str
                 ) -> list[tuple[int, float]]:
    """Transpose the subject to the dominant; at the head, a dominant (degree 5)
    is answered up a fourth (to the tonic) instead of up a fifth — the tonal
    adjustment. Everything else is a real transposition up a fifth."""
    lad = _ladder(48 + tonic - 24, mode, span=72)    # wide ladder spanning both registers
    idx_of = {p: i for i, p in enumerate(lad)}

    def to_idx(p):
        return idx_of.get(p) or min(range(len(lad)), key=lambda k: abs(lad[k] - p))

    sidx = [to_idx(p) for p, _ in subject]
    degrees = [i % 7 for i in sidx]                  # 0 = tonic, 4 = dominant
    head_len = 0
    for deg in degrees:
        if deg in (0, 4):
            head_len += 1
        else:
            break

    ans = []
    for k, (p, d) in enumerate(subject):
        i = sidx[k]
        up = 3 if (k < head_len and i % 7 == 4) else 4   # 4th for head dominant, else 5th
        ans.append((lad[min(i + up, len(lad) - 1)], d))
    return ans


def _counter_line(given, tonic, mode, above, rng):
    """Best-effort note-against-note consonant counter-line (no strict cadence)."""
    pcs = _diatonic_set(tonic, mode)
    n = len(given)

    def dfs(i, line):
        if i == n:
            return True
        g = given[i]
        cands = [p for p in _candidates(g, above, pcs) if _consonant(abs(p - g))]
        cands.sort(key=lambda p: rng.random()
                   + (2 if abs(p - g) % 12 in (3, 4, 8, 9) else 0)
                   + (2 if line and abs(p - line[-1]) <= 2 else 0), reverse=True)
        for p in cands:
            if i > 0:
                if not _melodic_ok(line[-1], p) or not _motion_ok(line[-1], given[i - 1], p, g):
                    continue
            line.append(p)
            if dfs(i + 1, line):
                return True
            line.pop()
        return False

    line = []
    return line if dfs(0, line) else None


def _counter_line_multi(givens, tonic, mode, lo, hi, rng):
    """A counter-line within [lo,hi] that is consonant with EVERY line in
    ``givens`` at each time index (note-against-note), stepwise-leaning, and
    free of parallel perfects against the first given (the entering voice)."""
    pcs = _diatonic_set(tonic, mode)
    n = len(givens[0])

    def dfs(i, line):
        if i == n:
            return True
        cands = [p for p in range(lo, hi + 1) if p % 12 in pcs
                 and all(_consonant(abs(p - g[i])) for g in givens)]
        cands.sort(key=lambda p: rng.random()
                   + (2 if line and abs(p - line[-1]) <= 2 else 0)
                   + (1 if all(abs(p - g[i]) % 12 in (3, 4, 8, 9) for g in givens) else 0),
                   reverse=True)
        for p in cands:
            if i > 0:
                if not _melodic_ok(line[-1], p):
                    continue
                if not _motion_ok(line[-1], givens[0][i - 1], p, givens[0][i]):
                    continue
            line.append(p)
            if dfs(i + 1, line):
                return True
            line.pop()
        return False

    line = []
    return line if dfs(0, line) else None


def _fit_octave(pseq, center):
    """Octave-transpose a pitch sequence so its mean sits near ``center``."""
    shift = round((center - sum(pseq) / len(pseq)) / 12) * 12
    return [p + shift for p in pseq]


def _notes_from(pitches, durs):
    """[(pitch, start_beat, end_beat)] from parallel pitch/duration lists."""
    out, t = [], 0.0
    for p, d in zip(pitches, durs):
        out.append((p, t, t + d))
        t += d
    return out


def _pitch_at(notes, t):
    for (p, s, e) in notes:
        if s - 1e-9 <= t < e - 1e-9:
            return p
    return None


def _beat_points(a, b):
    """The onset plus each whole beat inside [a, b) — the metrically strong points
    where a held counter-note must be consonant (off-beat notes are passing)."""
    pts = [a]
    k = math.floor(a) + 1
    while k < b - 1e-9:
        pts.append(float(k))
        k += 1
    return pts


def _independent_counter(placed, tonic, mode, lo, hi, total, rng):
    """A rhythmically INDEPENDENT counter-voice (calmer half/dotted/quarter notes)
    spanning ``total`` beats. A held note must be consonant with the other voices
    only on the beats it spans, so it can sustain across the entry's fast runs —
    keeping the texture consonant but far less dense."""
    pcs = _diatonic_set(tonic, mode)
    durs, rem = [], total
    while rem > 1e-6:                                  # own, slower rhythm (prefer halves)
        d = 2.0 if rem >= 2 and rng.random() < 0.6 else (1.0 if rem >= 1 else rem)
        durs.append(min(d, rem))
        rem -= durs[-1]
    wins, t = [], 0.0
    for d in durs:
        wins.append((t, t + d))
        t += d

    def dfs(i, line):
        if i == len(wins):
            return True
        a, b = wins[i]
        must = set()
        for pt in _beat_points(a, b):
            for pl in placed:
                p = _pitch_at(pl, pt + 1e-6)
                if p is not None:
                    must.add(p)
        cands = [p for p in range(lo, hi + 1) if p % 12 in pcs
                 and all(_consonant(abs(p - m)) for m in must)]
        cands.sort(key=lambda p: rng.random()
                   + (2 if line and abs(p - line[-1]) <= 2 else 0), reverse=True)
        for p in cands:
            if i > 0 and not _melodic_ok(line[-1], p):
                continue
            line.append(p)
            if dfs(i + 1, line):
                return True
            line.pop()
        return False

    line = []
    if not dfs(0, line):
        return None
    return [(line[i], wins[i][0], wins[i][1]) for i in range(len(wins))]


def fugue_exposition(tonic: int = 0, mode: str = "major", length: int = 8,
                     voices: int = 4, beat: float = 0.5,
                     rng_seed: int | None = None) -> pretty_midi.PrettyMIDI:
    """Full N-voice fugal exposition (N=2..4). Voices enter one per statement,
    alternating Subject (tonic) / Answer (dominant) in descending registers;
    each already-entered voice continues in consonant counterpoint."""
    rng = random.Random(rng_seed)
    voices = max(2, min(voices, 4))
    subject = make_subject(tonic, mode, length, rng)
    answer = tonal_answer(subject, tonic, mode)
    durs = [d for _, d in subject]
    s_pitch = [p for p, _ in subject]
    a_pitch = [p for p, _ in answer]

    T = 48 + tonic
    spread = {2: [12, 0], 3: [12, 0, -12], 4: [24, 12, 0, -12]}[voices]
    centers = [T + o for o in spread]                 # registers, high → low
    entries = [_fit_octave(s_pitch if v % 2 == 0 else a_pitch, centers[v])
               for v in range(voices)]                # tonic/dominant alternation

    total_beats = sum(durs)
    stmt_len = total_beats * beat
    voice_notes = [[] for _ in range(voices)]         # (pitch, start_s, dur_s) per voice

    for s in range(voices):                           # stage s: voice s enters
        entry_notes = _notes_from(entries[s], durs)   # subject/answer rhythm (varied)
        placed = [entry_notes]
        stage = {s: entry_notes}
        for v in range(s):                            # voices already in → free counterpoint
            lo, hi = centers[v] - 10, centers[v] + 10
            cn = _independent_counter(placed, tonic, mode, lo, hi, total_beats, rng)
            stage[v] = cn
            if cn:
                placed.append(cn)
        t_off = s * stmt_len
        for v, notes in stage.items():
            if not notes:
                continue
            for p, sb, eb in notes:
                voice_notes[v].append((p, t_off + sb * beat, (eb - sb) * beat))

    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    names = ["Voice 1 (S)", "Voice 2 (A)", "Voice 3 (S)", "Voice 4 (A)"]
    for v in range(voices):
        inst = pretty_midi.Instrument(program=0, name=names[v])
        for p, t0, dur in voice_notes[v]:
            inst.notes.append(pretty_midi.Note(82, p, t0, t0 + dur))
        pm.instruments.append(inst)
    return pm
