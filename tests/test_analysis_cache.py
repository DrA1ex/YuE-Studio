"""Synthetic cache tests; no downloads, UI or music inference."""
import json
import sys
import tempfile
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from analysis_cache import StageCache


class StageCacheTests(unittest.TestCase):
    def test_stage_cache_reuses_matching_payload_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "audio.bin"
            audio.write_bytes(b"synthetic audio")
            first = StageCache(root / "cache", audio)
            signature = {"revision": "test", "mode": "segment"}
            first.save("lyrics", signature, {"text": "repeat"}, elapsed_seconds=.1)

            second = StageCache(root / "cache", audio)
            self.assertEqual(second.load("lyrics", signature), {"text": "repeat"})
            self.assertEqual(second.records["lyrics"]["status"], "reused")
            self.assertIsNone(second.load("lyrics", {"revision": "changed"}))

    def test_failed_stage_is_inspectable_without_becoming_reusable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "audio.bin"
            audio.write_bytes(b"synthetic audio")
            cache = StageCache(root / "cache", audio)
            signature = {"revision": "test"}
            cache.fail("style", signature, "synthetic failure")
            failure = next((root / "cache" / cache.source_sha256).glob("style.failure.json"))
            self.assertEqual(json.loads(failure.read_text())["error"], "synthetic failure")
            self.assertIsNone(cache.load("style", signature))


if __name__ == "__main__":
    unittest.main()
