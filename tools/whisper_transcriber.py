"""Single-file Whisper adapter for the pinned Transformers cover runtime.

Word alignment is optional: invalid alignment must not discard a transcript.
The normal path retains beam search, real attention masks and decoder caching.
"""
import copy
import sys
import traceback

import torch
from transformers import AutomaticSpeechRecognitionPipeline, EncoderDecoderCache


class AlignmentError(ValueError):
    pass


def normalize_decoder_cache(module, args, kwargs):
    """Convert legacy caches supplied by Whisper's internal no-speech pass."""
    cache = kwargs.get("past_key_values")
    use_cache = kwargs.get("use_cache")
    if use_cache is None:
        use_cache = module.config.use_cache
    if isinstance(cache, tuple) or (cache is None and use_cache):
        kwargs["past_key_values"] = EncoderDecoderCache.from_legacy_cache(cache)
    return args, kwargs


class RobustWhisperPipeline(AutomaticSpeechRecognitionPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alignment_warnings = []
        self.timestamp_mode = "word"
        self._cache_hook = self.model.register_forward_pre_hook(normalize_decoder_cache, with_kwargs=True)

    def _notice(self, message, *, exception=False):
        self.alignment_warnings.append(message)
        print(f"Whisper: {message}", file=sys.stderr, flush=True)
        if exception:
            traceback.print_exc()

    def _forward(self, model_inputs, return_timestamps=False, **generate_kwargs):
        if model_inputs.get("stride") is not None:
            raise ValueError("Use sequential Whisper decoding; chunked input is not supported by this adapter")
        features = model_inputs["input_features"]
        mask = model_inputs["attention_mask"]
        word_mode = return_timestamps == "word"
        self.timestamp_mode = "word" if word_mode else "segment"

        def generate(words):
            options = dict(generate_kwargs)
            options["generation_config"] = copy.deepcopy(options.get("generation_config", self.generation_config))
            options.update(return_timestamps=True, return_token_timestamps=words, return_segments=words,
                           return_legacy_cache=False)
            if words:
                options["num_frames"] = model_inputs.get("num_frames")
            else:
                # Bounded recovery: bypass alignment and beam/cache interactions.
                # `return_legacy_cache` has no meaning when caching is off and
                # triggers a broken warning call in Transformers 4.45.2.
                options.pop("return_legacy_cache", None)
                options.update(num_beams=1, use_cache=False, output_attentions=False,
                               return_dict_in_generate=False)
            return self.model.generate(input_features=features, attention_mask=mask, **options)

        try:
            generated = generate(word_mode)
        except IndexError:
            if not word_mode:
                raise
            self._notice("Word timestamp pass failed; retrying with segment timestamps.", exception=True)
            self.timestamp_mode = "segment"
            generated = generate(False)

        if isinstance(generated, torch.Tensor):
            output = {"tokens": generated}
        else:
            output = {"tokens": generated["sequences"]}
            if "segments" in generated:
                timestamps = []
                for segments in generated["segments"]:
                    pieces = [segment["token_timestamps"] for segment in segments if "token_timestamps" in segment]
                    timestamps.append(torch.cat(pieces) if pieces else torch.empty(0, device=features.device))
                output["token_timestamps"] = timestamps
            elif "token_timestamps" in generated:
                output["token_timestamps"] = generated["token_timestamps"]
        return {"is_last": model_inputs["is_last"], **output}

    def postprocess(self, model_outputs, return_timestamps=None, **kwargs):
        if all(output["tokens"].numel() == 0 for output in model_outputs):
            return {"text": "", "chunks": []}
        words = return_timestamps == "word" and self.timestamp_mode == "word"
        if words:
            try:
                for output in model_outputs:
                    timestamps = output.get("token_timestamps")
                    if timestamps is None or len(timestamps) != output["tokens"].shape[0]:
                        raise AlignmentError("Missing token alignment")
                    for tokens, times in zip(output["tokens"], timestamps):
                        if len(times) < len(tokens) or not torch.isfinite(times).all() or (times[1:] < times[:-1]).any():
                            raise AlignmentError("Incomplete or invalid token alignment")
                # Some language/token combinations fail during subword-to-word
                # grouping even with valid lengths. Preserve tokens for recovery.
                return super().postprocess(copy.deepcopy(model_outputs), return_timestamps="word", **kwargs)
            except (IndexError, AlignmentError):
                self._notice("Word alignment unavailable; retained transcript with segment timestamps.", exception=True)
                self.timestamp_mode = "segment"
        return super().postprocess(model_outputs, return_timestamps=True, **kwargs)
