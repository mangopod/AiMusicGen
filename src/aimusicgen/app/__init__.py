"""Launch the native window."""
from __future__ import annotations

from pathlib import Path

import webview

from .api import Api

UI_DIR = Path(__file__).resolve().parent / "ui"


def run() -> None:
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
