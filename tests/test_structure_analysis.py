"""Synthetic acoustic-structure checks with generated signals only."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from structure_analysis import analyze_acoustic_structure, estimate_tempo


class StructureAnalysisTests(unittest.TestCase):
    def test_empty_audio_has_a_stable_contract(self):
        result = analyze_acoustic_structure(np.array([], dtype=np.float32), 16000)
        self.assertEqual(result["repeated_regions"], [])
        self.assertEqual(result["tempo_candidates"], [])

    def test_repeated_tone_produces_acoustic_evidence_without_semantic_labels(self):
        rate = 8000
        tone = np.sin(2 * np.pi * 220 * np.arange(rate * 8) / rate).astype(np.float32)
        result = analyze_acoustic_structure(np.concatenate([tone, np.zeros(rate * 2), tone]), rate)
        self.assertEqual(result["source"], "spectral-repetition-fallback")
        self.assertTrue(all("similarity" in item for item in result["repeated_regions"]))

    def test_tempo_returns_only_bounded_candidates(self):
        rate = 8000
        pulse = np.zeros(rate * 8, dtype=np.float32)
        for start in range(0, len(pulse), int(rate * .5)):
            pulse[start:start + 80] = 1.0
        candidates = estimate_tempo(pulse, rate)
        self.assertTrue(all(45 <= value <= 240 for value in candidates))


if __name__ == "__main__":
    unittest.main()
