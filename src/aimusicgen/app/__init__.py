"""Native app shell: a pywebview window hosting the HTML/JS UI in ``ui/``, with an
:class:`api.Api` instance bridged in as ``window.pywebview.api``."""
from __future__ import annotations

from pathlib import Path

import webview

from .api import Api

UI_DIR = Path(__file__).resolve().parent / "ui"


def run() -> None:
    """Create the native window, attach the JS↔Python API, and start the event
    loop (blocks until the window is closed)."""
    api = Api()
    webview.create_window(
        "AiMusicGen",
        str(UI_DIR / "index.html"),
        js_api=api,
        width=900,
        height=680,
        min_size=(640, 520),
    )
    webview.start()


if __name__ == "__main__":
    run()
