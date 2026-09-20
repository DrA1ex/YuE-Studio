"""Synthetic songs only: no external lyrics, model downloads or UI tests."""
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from lyrics_structure import format_lyrics
from transcribe_cover import (_merge_transcription_results, filter_silent_words, resolve_whisper_model, segment_diagnostics,
                             WHISPER_MODEL, WHISPER_REVISION, LEGACY_WHISPER_MODEL)

INTRO = "Across the ocean I hear your call. Before the sunrise we leave it all."
CHORUS = "We sail together under silver skies. We chase the dawn with open eyes. The wind will carry every song. This is the place where we belong."
VERSE = "The captain writes another letter. Her sister waits beside the gate. No winter storm can ever stop us. We lift the anchor never late."
VERSE_TWO = "The empty streets are far behind us. A distant bell begins to ring. We watch the fading shore in silence. A brighter season starts to sing."
PRE = "Now take my hand and face the water. The stars are bright above the bay."
BRIDGE = "A quiet island waits beyond us. A hidden shore we have not seen."


def text_only(result):
    return " ".join(line for line in result.splitlines() if line and not line.startswith("["))


class LyricStructureTests(unittest.TestCase):
    def test_complete_refrains_not_two_line_fragments(self):
        text = " ".join([INTRO, CHORUS, VERSE, PRE, CHORUS, VERSE_TWO, PRE, CHORUS, BRIDGE, CHORUS, "Goodnight my friends."])
        result = format_lyrics({"text": text})
        self.assertEqual(text_only(result), text)
        self.assertEqual(result.count("[chorus]"), 4)
        self.assertEqual(result.count("[pre-chorus]"), 2)
        self.assertEqual(result.count("[bridge]"), 1)
        self.assertTrue(result.startswith("[intro]"))
        self.assertIn("[outro]\nGoodnight my friends.", result)
        for block in result.split("\n\n"):
            if block.startswith("[chorus]"):
                self.assertEqual(text_only(block), CHORUS)

    def test_tolerates_substitution_without_rewriting_words(self):
        variant = CHORUS.replace("silver", "golden").replace("every song", "our song")
        text = " ".join([INTRO, CHORUS, VERSE, variant])
        result = format_lyrics({"text": text})
        self.assertEqual(text_only(result), text)
        self.assertEqual(result.count("[chorus]"), 2)
        self.assertIn("golden", result)

    def test_different_segment_boundaries_keep_same_sections(self):
        tokens = " ".join([INTRO, CHORUS, VERSE, CHORUS]).split()
        single = format_lyrics({"text": " ".join(tokens)})
        chunked = format_lyrics({"chunks": [{"text": " ".join(tokens[i:i + 7])} for i in range(0, len(tokens), 7)]})
        self.assertEqual([line for line in single.splitlines() if line.startswith("[")], [line for line in chunked.splitlines() if line.startswith("[")])
        self.assertEqual(text_only(chunked), " ".join(tokens))

    def test_word_timestamps_split_at_pause(self):
        words = [{"text": text, "timestamp": (start, end)} for text, start, end in
                 [("Hold", 0, .3), ("my", .3, .5), ("hand", .5, .9), ("Follow", 2, 2.3), ("me", 2.3, 2.6)]]
        result = format_lyrics({"chunks": words})
        self.assertEqual(result, "[verse]\nHold my hand\nFollow me")

    def test_no_arbitrary_four_line_verse_splits(self):
        text = " ".join([INTRO, VERSE, VERSE_TWO])
        result = format_lyrics({"text": text})
        self.assertEqual(result.count("[verse]"), 1)
        self.assertEqual(text_only(result), text)

    def test_only_proven_silent_words_are_removed(self):
        audio = np.zeros(16000, dtype=np.float32)
        audio[0:8000] = .1
        chunks = [{"text": "Sung", "timestamp": (0, .4)}, {"text": "ghost", "timestamp": (.7, .9)},
                  {"text": "unknown", "timestamp": (None, None)}, {"text": "brief", "timestamp": (.9, .92)}]
        result, count = filter_silent_words({"chunks": chunks, "text": "Sung ghost unknown brief"}, audio, 16000)
        self.assertEqual(count, 1)
        self.assertEqual([chunk["text"] for chunk in result["chunks"]], ["Sung", "unknown", "brief"])
        self.assertEqual(chunks[1]["text"], "ghost")

    def test_all_silence_does_not_resurrect_raw_text(self):
        result, _ = filter_silent_words({"chunks": [{"text": "ghost", "timestamp": (0, 1)}], "text": "ghost"}, np.zeros(16000), 16000)
        self.assertEqual(format_lyrics(result), "")

    def test_segment_diagnostics_preserve_invalid_timing_as_evidence(self):
        diagnostics = segment_diagnostics({"chunks": [
            {"text": "valid phrase", "timestamp": (1.0, 2.0)},
            {"text": "uncertain phrase", "timestamp": (None, None)},
            {"text": "reversed", "timestamp": (4.0, 3.0)},
        ]})
        self.assertEqual([item["timestamp_valid"] for item in diagnostics], [True, False, False])
        self.assertEqual(diagnostics[0]["token_count_estimate"], 2)

    def test_window_merge_offsets_and_deduplicates_overlap(self):
        merged = _merge_transcription_results([
            ({"segments": [{"text": "same refrain", "start": 0.0, "end": 2.0}]}, 0.0, 3.0),
            ({"segments": [{"text": "same refrain", "start": 0.0, "end": 2.0}, {"text": "new line", "start": 2.0, "end": 3.0}]}, 1.5, 4.5),
        ])
        self.assertEqual([chunk["text"] for chunk in merged["chunks"]], ["same refrain", "new line"])
        self.assertEqual(merged["chunks"][1]["timestamp"], (3.5, 4.5))

    def test_incomplete_upgrade_falls_back_to_existing_complete_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            complete, partial = root / "complete", root / "partial"
            complete.mkdir(); partial.mkdir()
            for filename in ("config.json", "generation_config.json", "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors"):
                (complete / filename).write_text("fixture")
            with patch("huggingface_hub.snapshot_download", side_effect=[str(partial), str(complete)]):
                path, model, _ = resolve_whisper_model()
            self.assertEqual(path, str(complete))
            self.assertEqual(model, LEGACY_WHISPER_MODEL)
            with patch("huggingface_hub.snapshot_download", return_value=str(complete)) as download:
                _, model, revision = resolve_whisper_model()
            self.assertEqual((model, revision), (WHISPER_MODEL, WHISPER_REVISION))
            self.assertEqual(download.call_count, 1)
            self.assertTrue(download.call_args.kwargs["local_files_only"])
            with patch("huggingface_hub.snapshot_download", return_value=str(partial)), self.assertRaises(FileNotFoundError):
                resolve_whisper_model()


if __name__ == "__main__":
    unittest.main()
