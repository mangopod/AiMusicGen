"""JS<->Python bridge exposed to the webview.

Methods on this class are callable from the UI as ``window.pywebview.api.<name>``.
Generation returns the MIDI bytes as a base64 data URI so the in-page
<midi-player> (Web Audio) can play it without any native synth.
"""
from __future__ import annotations

import base64
import subprocess
import threading
from pathlib import Path

from .. import config as C
from .. import constraints
from .. import library
from .. import models
from ..config import TrainConfig
from ..data import corpus
from ..data.notation import midi_to_musicxml
from ..generate import checkpoint_compatible, generate_midi
from ..train import train as run_training


CURATED = "curated"  # pseudo-composer id for the kept-generations pool


def _idle_train_state() -> dict:
    return {
        "running": False,
        "phase": "idle",        # idle | importing | training | done | error
        "epoch": 0,
        "epochs": 0,
        "train_loss": None,
        "val_loss": None,
        "n_songs": 0,
        "message": "",
        "error": None,
    }


class Api:
    """The object exposed to the webview as ``window.pywebview.api``.

    Every public method here is callable from the front-end JavaScript and
    returns JSON-serializable data. Methods are grouped below: generation, the
    saved-generations library, the "My MIDI" pool, the music21 corpus browser,
    constraints, training, the saved-model library, counterpoint/fugue, and the
    Mistral analysis. ``_``-prefixed members are internal helpers.
    """

    def __init__(self):
        self._lock = threading.Lock()       # serialises generation (one model/device)
        self._train = _idle_train_state()   # background-training progress, polled by the UI

    @staticmethod
    def _model_name(model_id):
        """Friendly name for a model id (falls back to the id)."""
        if not model_id:
            return None
        m = next((x for x in models.list_models()["models"]
                  if x["id"] == model_id), None)
        return m["name"] if m else model_id

    def status(self) -> dict:
        """App status for the UI: whether a usable model is active, its name, the
        corpus file count, and the output directory."""
        n_midi = sum(1 for _ in C.DATA_DIR.rglob("*.mid")) + \
                 sum(1 for _ in C.DATA_DIR.rglob("*.midi"))
        active = models.get_active()
        active_name = self._model_name(active)
        return {
            "trained": checkpoint_compatible(),
            "active_model": active,
            "active_model_name": active_name,
            "corpus_files": n_midi,
            "output_dir": str(C.OUTPUT_DIR),
        }

    @staticmethod
    def _midi_payload(path) -> dict:
        """Base64 MIDI data URI + staff MusicXML for a .mid on disk."""
        path = Path(path)
        data = path.read_bytes()
        payload = {
            "data_uri": "data:audio/midi;base64," + base64.b64encode(data).decode(),
            "file": path.name,
        }
        try:                                   # staff notation (best-effort)
            payload["musicxml"] = midi_to_musicxml(path)
        except Exception:
            payload["musicxml"] = None
        return payload

    def generate(self, params: dict | None = None) -> dict:
        """Generate a piece with the active model and return it for playback.

        ``params`` (all optional): ``length``, ``temperature``, ``top_k``,
        ``tempo``, ``seed``, ``constraints`` (rule ids), ``constraint_strength``,
        ``cadence`` (half/plagal/either), ``key`` (e.g. "C major"). The result is
        saved to the library and returned with a base64 MIDI ``data_uri`` and
        staff ``musicxml``.
        """
        params = params or {}
        if self._train["running"]:
            raise RuntimeError("Training in progress — try again when it finishes.")
        active = models.get_active()
        rules = params.get("constraints") or []
        used = {
            "length": int(params.get("length", 256)),
            "temperature": float(params.get("temperature", 1.0)),
            "top_k": int(params.get("top_k", 0)),
            "tempo": int(params.get("tempo", C.DEFAULT_TEMPO)),
            "seed": params.get("seed"),
            "constraints": list(rules),
            "constraint_strength": float(params.get("constraint_strength", 1.0)),
            "cadence": params.get("cadence") or None,
            "key": params.get("key") or None,
            "model": active,
            "model_name": self._model_name(active),
        }
        # Serialize generation; one model, one device.
        with self._lock:
            info = generate_midi(
                rng_seed=used["seed"],
                constraint_rules=used["constraints"],
                constraint_strength=used["constraint_strength"],
                cadence=used["cadence"],
                key=used["key"],
                **{k: used[k] for k in ("length", "temperature", "top_k", "tempo")},
            )
        library.add_entry(info["path"], info["n_notes"], used)
        info.update(self._midi_payload(info["path"]))
        info["name"] = Path(info["path"]).name
        return info

    def list_constraints(self) -> dict:
        """The selectable voice-leading rules (id, name, description)."""
        return {"rules": constraints.list_rules()}

    # --- species counterpoint generator (Fux) ------------------------
    def generate_counterpoint(self, params: dict | None = None) -> dict:
        """Generate a cantus firmus + rule-valid first-species counterpoint."""
        params = params or {}
        if self._train["running"]:
            raise RuntimeError("Training in progress — try again when it finishes.")
        from datetime import datetime
        from .. import counterpoint
        key = params.get("key") or "C major"
        try:
            name, mode = key.rsplit(" ", 1)
            tonic = constraints.NOTE_NAMES.index(name)
            mode = mode if mode in ("major", "minor") else "major"
        except (ValueError, AttributeError):
            tonic, mode = 0, "major"
        length = max(6, min(int(params.get("length", 10)), 16))
        above = bool(params.get("above", True))
        species = max(1, min(int(params.get("species", 1)), 5))
        try:
            pm = counterpoint.generate(tonic, mode, length, above, species=species,
                                       rng_seed=params.get("seed"))
        except Exception as exc:
            return {"error": str(exc), "text": str(exc)}
        out = C.OUTPUT_DIR / f"cp-{datetime.now().strftime('%Y%m%d-%H%M%S')}.mid"
        pm.write(str(out))
        n_notes = sum(len(i.notes) for i in pm.instruments)
        library.add_entry(out, n_notes,
                          {"type": "counterpoint", "key": key, "length": length,
                           "species": species, "voice": "above" if above else "below"})
        info = {"path": str(out), "name": out.name, "n_notes": n_notes, "trained": True}
        info.update(self._midi_payload(out))
        return info

    def generate_fugue(self, params: dict | None = None) -> dict:
        """Generate a 2-voice fugal exposition: subject + tonal answer (+CS)."""
        params = params or {}
        if self._train["running"]:
            raise RuntimeError("Training in progress — try again when it finishes.")
        from datetime import datetime
        from .. import counterpoint
        key = params.get("key") or "C major"
        try:
            name, mode = key.rsplit(" ", 1)
            tonic = constraints.NOTE_NAMES.index(name)
            mode = mode if mode in ("major", "minor") else "major"
        except (ValueError, AttributeError):
            tonic, mode = 0, "major"
        length = max(6, min(int(params.get("length", 8)), 12))
        voices = max(2, min(int(params.get("voices", 4)), 4))
        try:
            pm = counterpoint.fugue_exposition(tonic, mode, length, voices=voices,
                                               rng_seed=params.get("seed"))
        except Exception as exc:
            return {"error": str(exc), "text": str(exc)}
        out = C.OUTPUT_DIR / f"fugue-exp-{datetime.now().strftime('%Y%m%d-%H%M%S')}.mid"
        pm.write(str(out))
        n_notes = sum(len(i.notes) for i in pm.instruments)
        library.add_entry(out, n_notes,
                          {"type": "fugue-exposition", "key": key, "length": length,
                           "voices": voices})
        info = {"path": str(out), "name": out.name, "n_notes": n_notes, "trained": True}
        info.update(self._midi_payload(out))
        return info

    # --- saved-generations library -----------------------------------
    def list_generations(self) -> dict:
        """All saved generations in output/, newest first (metadata only)."""
        entries = library.list_entries()
        return {"count": len(entries), "generations": entries}

    def load_generation(self, file: str) -> dict:
        """Reload a saved generation for replay (audio + staff)."""
        path = library.safe_path(file)
        if not path.exists():
            return {"error": "not found"}
        meta = next((e for e in library.list_entries() if e["file"] == path.name), {})
        out = {"name": meta.get("name", path.stem), "n_notes": meta.get("n_notes")}
        out.update(self._midi_payload(path))
        return out

    def rename_generation(self, file: str, name: str) -> dict:
        """Set the friendly name of a saved generation."""
        return {"ok": library.rename(file, name)}

    def delete_generation(self, file: str) -> dict:
        """Delete a saved generation (file + manifest entry)."""
        return {"ok": library.delete(file)}

    # --- curated training pool (human feedback loop) ------------------
    def keep_generation(self, file: str) -> dict:
        """Copy a generation into the curated training pool ("keep" it)."""
        return {"ok": library.keep(file), "count": library.keepers_count()}

    def unkeep_generation(self, file: str) -> dict:
        """Remove a generation from the curated pool."""
        return {"ok": library.unkeep(file), "count": library.keepers_count()}

    def keepers_count(self) -> dict:
        """Number of files in the curated pool (drives the '★ My MIDI' chip)."""
        return {"count": library.keepers_count()}

    # --- "My MIDI" drop folder (data/midi/curated) --------------------
    def list_my_midi(self) -> dict:
        """List the drop-folder files (kept generations + user-dropped MIDI)."""
        files = library.list_curated()
        return {"count": len(files), "dir": str(library.CURATED_DIR), "files": files}

    def load_my_midi(self, file: str) -> dict:
        """Load a My-MIDI file for playback (audio + staff)."""
        path = library.curated_path(file)
        if not path.exists():
            return {"error": "not found"}
        out = {"name": path.stem}
        out.update(self._midi_payload(path))
        return out

    def delete_my_midi(self, file: str) -> dict:
        """Remove a file from the My-MIDI drop folder."""
        return {"ok": library.unkeep(file), "count": library.keepers_count()}

    def reveal_my_midi(self) -> bool:
        """Open the My-MIDI drop folder in Finder."""
        try:
            library.CURATED_DIR.mkdir(parents=True, exist_ok=True)
            subprocess.run(["open", str(library.CURATED_DIR)], check=False)
            return True
        except Exception:
            return False

    # --- music21 corpus browser --------------------------------------
    def list_composers(self) -> dict:
        """Every composer/collection in the bundled corpus, with work counts."""
        comps = corpus.list_composers()
        return {"composers": comps, "default": corpus.DEFAULT_COMPOSER}

    def list_corpus(self, composer: str = corpus.DEFAULT_COMPOSER) -> dict:
        """All works for ``composer`` in the bundled music21 corpus."""
        works = corpus.list_works(composer)
        return {"composer": composer, "count": len(works), "works": works}

    def corpus_raw(self, path: str) -> dict:
        """Parsed raw data of one corpus work, as readable text."""
        try:
            return corpus.raw_text(path)
        except Exception as exc:
            return {"error": str(exc), "text": f"Failed to load:\n{exc}"}

    def analyze_corpus_work(self, path: str, question: str = "") -> dict:
        """Extract notes with Mido and send them (plus an optional user question)
        to the local Mistral/Ollama for a key/interval analysis."""
        try:
            import pretty_midi
            from .. import mistral
            pitches = corpus.notes_via_mido(path)
            if not pitches:
                return {"text": "No notes found in this work."}
            names = [pretty_midi.note_number_to_name(p) for p in pitches]
            intervals = [pitches[i] - pitches[i - 1] for i in range(1, len(pitches))]
            text = mistral.analyze_notes(Path(path).stem, names, intervals,
                                         question=question or "")
            return {"text": text, "n_notes": len(pitches)}
        except Exception as exc:
            return {"error": str(exc), "text": str(exc)}

    # --- training: per-composer or mixed model -----------------------
    def training_status(self) -> dict:
        """Snapshot of background-training progress (phase, epoch, losses,
        message) — the UI polls this to drive the progress bar."""
        return dict(self._train)

    def train_corpus(self, params: dict | None = None) -> dict:
        """Import N works of one or more composers, then train a model on the
        combined corpus (background thread). Poll training_status() for progress."""
        params = params or {}
        if self._train["running"]:
            return {"started": False, "reason": "training already running"}
        composers = params.get("composers")
        if not composers:
            composers = [params.get("composer") or corpus.DEFAULT_COMPOSER]
        composers = [c for c in composers if c]
        if not composers:
            return {"started": False, "reason": "no composers selected"}
        n = params.get("n_works")
        n = int(n) if n else None  # None / 0 -> all works
        epochs = max(1, int(params.get("epochs", 30)))
        name = (params.get("name") or "").strip() or None
        self._train = _idle_train_state()
        self._train.update(running=True, phase="importing", epochs=epochs,
                           message="Importing works…")
        threading.Thread(target=self._train_worker,
                         args=(composers, n, epochs, name), daemon=True).start()
        return {"started": True}

    def _train_worker(self, composers: list, n, epochs: int, name) -> None:
        try:
            # the curated pool is already on disk (kept generations) — don't
            # re-import it; import only real music21 composers.
            real = [c for c in composers if c != CURATED]
            total_imported = 0
            for i, c in enumerate(real, 1):
                def import_cb(done, total, imported, c=c, i=i):
                    self._train["message"] = (
                        f"Importing {c} ({i}/{len(real)})… {done}/{total}")
                res = corpus.import_works(c, n=n, progress_cb=import_cb)
                total_imported += res["imported"]
            if real and total_imported == 0:
                raise RuntimeError("No works could be imported.")

            label = " + ".join(composers)
            self._train.update(phase="training",
                               message=f"Imported {total_imported} works — training {label}…")

            def epoch_cb(p):
                self._train.update(
                    epoch=p["epoch"],
                    train_loss=round(p["train_loss"], 3),
                    val_loss=round(p["val_loss"], 3),
                    message=(f"Training {label}… epoch {p['epoch']}/{p['epochs']}  "
                             f"val {p['val_loss']:.3f}"),
                )

            cfg = TrainConfig()
            cfg.epochs = epochs
            model_id = models.make_id(composers)
            out = run_training(
                cfg,
                data_dir=[corpus.composer_dir(c) for c in composers],
                progress_cb=epoch_cb,
                save_path=models.model_path(model_id),
            )
            models.register(model_id, name or models.default_name(composers),
                            composers, epochs, out["best_val"], out["n_songs"])
            self._train.update(
                running=False, phase="done",
                n_songs=out["n_songs"],
                message=(f"Saved model “{name or models.default_name(composers)}” — "
                         f"{total_imported} works, best val {out['best_val']:.3f}."),
            )
        except Exception as exc:
            self._train.update(running=False, phase="error",
                               error=str(exc), message=f"Failed: {exc}")

    # --- saved-model library -----------------------------------------
    def list_models(self) -> dict:
        """All saved checkpoints + which one is active."""
        return models.list_models()

    def set_active_model(self, model_id: str) -> dict:
        """Make ``model_id`` the model generation loads from ('reload')."""
        return {"ok": models.set_active(model_id)}

    def rename_model(self, model_id: str, name: str) -> dict:
        """Set a model's friendly name."""
        return {"ok": models.rename(model_id, name)}

    def delete_model(self, model_id: str) -> dict:
        """Delete a saved checkpoint (re-points 'active' if it was active)."""
        return {"ok": models.delete(model_id)}

    def reveal_output(self) -> bool:
        """Open the output folder in Finder."""
        try:
            subprocess.run(["open", str(C.OUTPUT_DIR)], check=False)
            return True
        except Exception:
            return False
