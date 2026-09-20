"""Synthetic regression checks: no UI automation, downloads or music inference."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from audio_analysis import excerpts, format_lyrics, select_descriptors, summarize_genre


class AnalysisTests(unittest.TestCase):
    def test_empty_lyrics_do_not_invent_sections(self):
        self.assertEqual(format_lyrics({"text": ""}), "")

    def test_refrain_preserves_all_words_and_repetitions(self):
        lines = ["Walking down this road alone", "Waiting for the morning light", "We will always sing together", "You will always be my home", "Another night another day", "Finding all the words to say", "We will always sing together", "You will always be my home"]
        result = format_lyrics({"chunks": [{"text": x} for x in lines]})
        self.assertEqual(result.count("[chorus]"), 2)
        self.assertEqual([x for x in result.splitlines() if x and not x.startswith("[")], lines)

    def test_adlibs_are_not_a_chorus(self):
        result = format_lyrics({"chunks": [{"text": x} for x in ["Oh", "Yeah", "Oh", "Yeah"]]})
        self.assertNotIn("[chorus]", result)

    def test_fallback_and_unicode(self):
        self.assertEqual(format_lyrics({"text": "È già sera. Tornerò qui!"}), "[verse]\nÈ già sera.\nTornerò qui!")

    def test_excerpts_cover_beginning_middle_end_with_bounded_size(self):
        clips = excerpts(np.arange(1000), 10, seconds=10)
        self.assertEqual([clip[0] for clip in clips], [0, 450, 900])
        self.assertTrue(all(len(clip) == 100 for clip in clips))
        self.assertEqual(len(excerpts(np.arange(12), 10)), 1)
        self.assertEqual(excerpts(np.array([]), 10), [])

    def test_uncertain_descriptors_are_omitted(self):
        self.assertEqual(select_descriptors([.3, .29], ["piano", "guitar"]), [])
        self.assertEqual(select_descriptors([.1, .03], ["piano", "guitar"]), [])
        self.assertEqual(select_descriptors([.4, .2], ["piano", "guitar"])[0]["label"], "piano")

    def test_close_genre_scores_are_reported_as_hybrid_evidence(self):
        self.assertEqual(
            summarize_genre([{"label": "rock", "score": .42}, {"label": "hip_hop", "score": .31}]),
            "Rock / Hip-Hop",
        )

    def test_weak_genre_scores_abstain(self):
        self.assertEqual(summarize_genre([{"label": "rock", "score": .21}]), "Mixed / uncertain")

    def test_full_analysis_contract_without_model_weights(self):
        import json
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch
        from scipy.io import wavfile
        import transcribe_cover
        recognizer = MagicMock()
        recognizer.alignment_warnings = []
        recognizer.timestamp_mode = "segment"
        recognizer.return_value = {"text": "Our song is here.", "chunks": [{"text": "Our song is here.", "timestamp": (0, 1)}]}
        classifier = MagicMock()
        classifier.feature_extractor.sampling_rate = 16000
        classifier.model.config.num_labels = 2
        classifier.return_value = [{"label": "hip_hop", "score": .8}, {"label": "pop", "score": .2}]
        melody = MagicMock()
        melody.eval.return_value.to.return_value.transcribe.return_value = {"abc": "score", "warnings": []}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "input.wav"
            wavfile.write(audio, 16000, np.sin(np.arange(16000) * .1).astype(np.float32))
            with patch.object(sys, "argv", ["transcribe_cover", str(audio), "--output", str(root / "out")]), \
                 patch("transformers.AutoModel.from_pretrained", return_value=melody), \
                 patch("transformers.pipeline", side_effect=[recognizer, classifier]), \
                 patch("huggingface_hub.snapshot_download", return_value="/local/model"), \
                 patch.object(transcribe_cover, "resolve_whisper_model", return_value=("/local/model", transcribe_cover.WHISPER_MODEL, transcribe_cover.WHISPER_REVISION)), \
                 patch("cover_score.prepare_cover_score", return_value="valid score"), \
                 patch.object(transcribe_cover, "describe_style", return_value=[{"label": "deep bass", "similarity": .4}]), \
                 patch.object(transcribe_cover, "event"):
                self.assertEqual(transcribe_cover.main(), 0)
            result = json.loads((root / "out/cover_analysis.json").read_text())
            self.assertEqual(result["style"], "Hip-Hop, deep bass")
            self.assertEqual(result["lyrics"], "[verse]\nOur song is here.")
            self.assertEqual(result["lyrics_raw"], "Our song is here.")
            self.assertEqual(result["lyrics_decoding"]["timestamp_mode"], "segment")
            self.assertEqual(result["lyrics_segment_diagnostics"][0]["timestamp_valid"], True)
            self.assertTrue(recognizer.call_args.kwargs["return_timestamps"])
            self.assertEqual(recognizer.call_args.kwargs["generate_kwargs"]["num_beams"], 1)
            self.assertNotIn("chunk_length_s", recognizer.call_args.kwargs)
            self.assertIsNone(recognizer.generation_config.forced_decoder_ids)
            self.assertEqual(classifier.call_args.kwargs["top_k"], 2)

    def test_synthetic_whisper_beam_search_and_word_alignment(self):
        import torch
        from transformers import WhisperConfig, WhisperForConditionalGeneration
        config = WhisperConfig(
            vocab_size=32, num_mel_bins=8, d_model=16, encoder_layers=1, decoder_layers=1,
            encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32,
            decoder_ffn_dim=32, max_source_positions=20, max_target_positions=32,
            pad_token_id=2, bos_token_id=1, eos_token_id=2, decoder_start_token_id=1,
            suppress_tokens=[], begin_suppress_tokens=[], forced_decoder_ids=None,
        )
        config._attn_implementation = "eager"
        with torch.random.fork_rng(), torch.inference_mode():
            torch.manual_seed(7)
            model = WhisperForConditionalGeneration(config).eval()
            model.generation_config.no_timestamps_token_id = 10
            model.generation_config.alignment_heads = [[0, 0]]
            model.generation_config.is_multilingual = False
            model.generation_config.prev_sot_token_id = 3
            result = model.generate(
                input_features=torch.zeros(1, 8, 40), attention_mask=torch.ones(1, 40, dtype=torch.long),
                return_timestamps=True, return_token_timestamps=True, num_beams=3, max_new_tokens=8,
                condition_on_prev_tokens=False, temperature=(0.0, .2, .4),
                compression_ratio_threshold=2.4, logprob_threshold=-1.0, no_speech_threshold=.6,
                return_legacy_cache=False,
            )
        # Whisper can strip closing timestamp/EOS tokens from sequences while
        # retaining their alignment entries; every retained token must be covered.
        self.assertEqual(result["sequences"].shape[0], result["token_timestamps"].shape[0])
        self.assertGreaterEqual(result["token_timestamps"].shape[1], result["sequences"].shape[1])
        self.assertTrue(torch.isfinite(result["token_timestamps"]).all())

    def test_whisper_preprocessing_passes_real_mask_short_and_long(self):
        # Use the installed Transformers implementation, real feature extraction,
        # and a synthetic config; no weights or tokenizers are needed.
        import torch
        from transformers import WhisperFeatureExtractor
        from transformers.pipelines.automatic_speech_recognition import AutomaticSpeechRecognitionPipeline
        recognizer = object.__new__(AutomaticSpeechRecognitionPipeline)
        recognizer.type = "seq2seq_whisper"
        recognizer.feature_extractor = WhisperFeatureExtractor()
        from types import SimpleNamespace
        recognizer.model = SimpleNamespace(dtype=torch.float32)
        for seconds in (1, 31):
            processed = next(recognizer.preprocess(np.zeros(seconds * 16000, dtype=np.float32)))
            self.assertIn("attention_mask", processed)
            self.assertEqual(int(processed["attention_mask"].sum()), seconds * 100)
            self.assertEqual(processed["attention_mask"].shape[-1], processed["input_features"].shape[-1])


if __name__ == "__main__":
    unittest.main()
