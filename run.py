"""Launch the AiMusicGen native window.

    python run.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from aimusicgen.app import run  # noqa: E402

if __name__ == "__main__":
    run()
