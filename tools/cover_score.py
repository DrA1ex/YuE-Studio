"""Validate the native YuE/SheetSage ABC dialect and free the cover's harmony."""
import sys
from pathlib import Path

HELPERS = Path(__file__).resolve().parents[1] / "skills" / "yue2-music" / "scripts"
if str(HELPERS) not in sys.path:
    sys.path.insert(0, str(HELPERS))
from abc_tools import strip_chords


def prepare_cover_score(abc):
    if not isinstance(abc, str) or not abc.strip():
        raise ValueError("A cover requires a melody score. Transcribe audio or import an ABC file first.")
    # The parser checks pitches, rhythms and meter before/after, and leaves header quotes intact.
    return strip_chords(abc.strip() + "\n")
