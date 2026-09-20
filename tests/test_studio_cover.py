"""Native app request regression tests; no model load or generated stand-in output."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from cover_score import prepare_cover_score
from abc_tools import parse, compare
from test_skill_abc_tools import score

spec = importlib.util.spec_from_file_location("studio_worker", ROOT / "tools/yue2_worker.py")
worker = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = worker
spec.loader.exec_module(worker)

class CoverTests(unittest.TestCase):
    def test_cover_preserves_notes_rhythm_and_headers(self):
        original = score('"C"C8D8"Am"E8F8|', 'G32|')
        melody = prepare_cover_score(original)
        self.assertTrue(compare(parse(original), parse(melody))["match"])
        self.assertEqual(parse(melody).voices["Vocal"].chords, [])
        self.assertIn('name="Vocal Melody"', melody)

    def test_rejects_empty_and_invalid_abc(self):
        for invalid in (None, "", "random text", score("C12|")):
            with self.subTest(abc=invalid), self.assertRaises(ValueError):
                prepare_cover_score(invalid)

    def test_cover_request_uses_melody_and_persists_identity(self):
        submitted = []
        events = []
        with tempfile.TemporaryDirectory() as directory, patch.object(worker, "OUTPUT_DIR", Path(directory)), patch.object(worker.SCHED, "submit", submitted.extend), patch.object(worker, "emit", lambda **event: events.append(event)):
            worker.submit_generate(dict(style="Jazz", lyrics="Words", abc=score('"C"C32|'), kind="COVER", source_path="/original.wav", title="My Cover", request_id="req1"))
        song = submitted[0]
        self.assertEqual(song.request.cot, "melody")
        self.assertEqual(parse(song.request.abc).voices["Vocal"].chords, [])
        started = next(e for e in events if e["event"] == "started")
        self.assertEqual(started["request_id"], "req1")
        self.assertEqual(started["songs"][0]["lyrics"], "Words")
        self.assertEqual(song.source_path, "/original.wav")

    def test_cover_allows_empty_style_and_matches_source_duration(self):
        submitted = []
        with tempfile.TemporaryDirectory() as directory, patch.object(worker, "OUTPUT_DIR", Path(directory)), patch.object(worker.SCHED, "submit", submitted.extend), patch.object(worker, "emit"):
            worker.submit_generate(dict(
                style="", lyrics="Words", abc=score("C32|"), kind="COVER",
                source_path="/original.wav", target_seconds=42.0,
                prompt_fidelity=0.8, style_fidelity=0.7, source_fidelity=1.0,
            ))
        song = submitted[0]
        self.assertEqual(song.request.style, "")
        self.assertEqual(song.limit, 1050)
        self.assertEqual(song.min_tokens, 1048)
        self.assertLess(song.temperature, 1.0)
        self.assertLess(song.top_p, 0.95)

    def test_imported_instrumental_score_silences_vocals(self):
        submitted = []
        with tempfile.TemporaryDirectory() as directory, patch.object(worker, "OUTPUT_DIR", Path(directory)), patch.object(worker.SCHED, "submit", submitted.extend), patch.object(worker, "emit"):
            worker.submit_generate(dict(style="Piano", lyrics="", instrumental=True, abc=score("C32|", "E32|")))
        result = parse(submitted[0].request.abc)
        self.assertEqual(result.voices["Vocal"].notes, [])
        self.assertEqual(result.voices["Ins"].notes, [[0, 64, 4]])

    def test_invalid_request_error_is_correlated(self):
        events = []
        with patch.object(worker, "emit", lambda **event: events.append(event)), patch.object(worker, "log"), patch.object(worker.traceback, "print_exc"):
            worker.submit(dict(cmd="generate", style="", lyrics="", request_id="bad"))
        self.assertEqual(events[-1]["event"], "error")
        self.assertEqual(events[-1]["request_id"], "bad")

if __name__ == "__main__":
    unittest.main()
