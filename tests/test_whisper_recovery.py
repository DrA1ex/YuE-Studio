"""Real tokenizer regression with synthetic vocabulary; no model downloads or UI."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

import torch
from transformers import AutomaticSpeechRecognitionPipeline, WhisperTokenizer, EncoderDecoderCache
from transformers.models.whisper.tokenization_whisper import bytes_to_unicode

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from whisper_transcriber import RobustWhisperPipeline, normalize_decoder_cache


class WhisperRecoveryTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        root = Path(folder.name)
        vocab = {value: i for i, value in enumerate(bytes_to_unicode().values())}
        specials = ["<|endoftext|>", "<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|startofprev|>", "<|notimestamps|>"]
        vocab.update({token: 256 + i for i, token in enumerate(specials)})
        (root / "vocab.json").write_text(json.dumps(vocab))
        (root / "merges.txt").write_text("#version: 0.2\n")
        tokenizer = WhisperTokenizer(str(root / "vocab.json"), str(root / "merges.txt"), additional_special_tokens=specials)
        self.pipe = object.__new__(RobustWhisperPipeline)
        self.pipe.framework = "pt"
        self.pipe.type = "seq2seq_whisper"
        self.pipe.tokenizer = tokenizer
        self.pipe.model = SimpleNamespace(config=SimpleNamespace(max_source_positions=1500))
        self.pipe.feature_extractor = SimpleNamespace(chunk_length=30, sampling_rate=16000)
        self.pipe.alignment_warnings = []
        self.pipe.timestamp_mode = "word"
        self.pipe.generation_config = SimpleNamespace()
        timestamp = vocab["<|notimestamps|>"] + 1
        ids = [vocab["<|startoftranscript|>"], vocab["<|en|>"], vocab["<|transcribe|>"], timestamp]
        ids += tokenizer.encode(" Hello world", add_special_tokens=False) + [timestamp + 50, vocab["<|endoftext|>"]]
        self.tokens = torch.tensor([ids])

    def test_old_decoder_crashes_new_decoder_preserves_transcript(self):
        outputs = [{"tokens": self.tokens, "token_timestamps": torch.zeros(1, 2)}]
        with self.assertRaisesRegex(IndexError, "list index out of range"):
            AutomaticSpeechRecognitionPipeline.postprocess(self.pipe, outputs, return_timestamps="word")
        with contextlib.redirect_stderr(io.StringIO()):
            result = self.pipe.postprocess(outputs, return_timestamps="word")
        self.assertEqual(result["text"].strip(), "Hello world")
        self.assertEqual(result["chunks"][0]["timestamp"], (0., 1.))
        self.assertEqual(self.pipe.timestamp_mode, "segment")
        self.assertTrue(self.pipe.alignment_warnings)

    def test_valid_word_alignment_is_retained(self):
        times = torch.arange(self.tokens.shape[1], dtype=torch.float32).unsqueeze(0) * .02
        result = self.pipe.postprocess([{"tokens": self.tokens, "token_timestamps": times}], return_timestamps="word")
        self.assertEqual(result["text"].strip(), "Hello world")
        self.assertEqual(len(result["chunks"]), 2)
        self.assertEqual(self.pipe.timestamp_mode, "word")
        self.assertEqual(self.pipe.alignment_warnings, [])

    def test_generate_failure_retries_once_with_same_features_and_mask(self):
        self.pipe.model.generate = MagicMock(side_effect=[IndexError("alignment"), self.tokens])
        features, mask = torch.zeros(1, 8, 40), torch.ones(1, 40)
        with contextlib.redirect_stderr(io.StringIO()):
            output = self.pipe._forward({"input_features": features, "attention_mask": mask, "is_last": True}, return_timestamps="word", num_beams=3)
        first, second = self.pipe.model.generate.call_args_list
        for call in (first, second):
            self.assertIs(call.kwargs["input_features"], features)
            self.assertIs(call.kwargs["attention_mask"], mask)
            self.assertNotIn("inputs", call.kwargs)
        self.assertFalse(second.kwargs["return_token_timestamps"])
        self.assertFalse(second.kwargs["use_cache"])
        self.assertNotIn("return_legacy_cache", second.kwargs)
        self.assertEqual(second.kwargs["num_beams"], 1)
        self.assertEqual(self.pipe.postprocess([output], return_timestamps="word")["text"].strip(), "Hello world")

    def test_recovery_failure_is_not_hidden_or_retried_forever(self):
        self.pipe.model.generate = MagicMock(side_effect=IndexError("bad generation"))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(IndexError):
            self.pipe._forward({"input_features": torch.zeros(1, 8, 40), "attention_mask": torch.ones(1, 40), "is_last": True}, return_timestamps="word")
        self.assertEqual(self.pipe.model.generate.call_count, 2)

    def test_empty_longform_segments_return_empty_text(self):
        self.pipe.model.generate = MagicMock(return_value={"sequences": torch.empty(1, 0, dtype=torch.long), "segments": [[]]})
        output = self.pipe._forward({"input_features": torch.zeros(1, 8, 80), "attention_mask": torch.ones(1, 80), "is_last": True}, return_timestamps="word")
        self.assertEqual(self.pipe.postprocess([output], return_timestamps="word"), {"text": "", "chunks": []})

    def test_cache_conversion_preserves_values_and_respects_disabled_cache(self):
        module = SimpleNamespace(config=SimpleNamespace(use_cache=True))
        tensor = torch.ones(1, 2, 3, 4)
        _, kwargs = normalize_decoder_cache(module, (), {"past_key_values": ((tensor, tensor, tensor, tensor),)})
        self.assertIsInstance(kwargs["past_key_values"], EncoderDecoderCache)
        self.assertEqual(kwargs["past_key_values"].get_seq_length(), 3)
        _, empty = normalize_decoder_cache(module, (), {})
        self.assertIsInstance(empty["past_key_values"], EncoderDecoderCache)
        self.assertEqual(empty["past_key_values"].get_seq_length(), 0)
        _, disabled = normalize_decoder_cache(module, (), {"use_cache": False})
        self.assertNotIn("past_key_values", disabled)


if __name__ == "__main__":
    unittest.main()
