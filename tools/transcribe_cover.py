#!/usr/bin/env python3
"""Transcribe an audio file to a melody-only ABC score for YuE2 covers.

Runs in the separate Python 3.11 cover environment. AVFoundation supplies a mono WAV;
array input avoids an external FFmpeg dependency. Model revisions are pinned.
"""
import argparse
import json
import sys
import gc
import traceback
import warnings
from audio_analysis import STYLE_MODEL, STYLE_REVISION, describe_style, excerpts, format_lyrics
from pathlib import Path

SHEETSAGE_REVISION = "80af707174fc7ee521c25925d5f014729f0e61ae"
MERT_REVISION = "d8ba1c745e733b3908ce6ad16ebeb17ac7600a42"
WHISPER_MODEL = "openai/whisper-large-v3-turbo"
WHISPER_REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
LEGACY_WHISPER_MODEL = "openai/whisper-small"
LEGACY_WHISPER_REVISION = "973afd24965f72e36ca33b3055d56a652f456b4d"
GENRE_MODEL = "dima806/music_genres_classification"
GENRE_REVISION = "5f71fb1e2c6bedcddb2bfb1e929fc70655780902"

def event(stage, message):
    print(json.dumps({"event": "transcription", "stage": stage, "message": message}), flush=True)


def _resample(waveform, source_rate, target_rate):
    if source_rate == target_rate:
        return waveform
    from scipy.signal import resample_poly
    import math
    divisor = math.gcd(int(source_rate), int(target_rate))
    return resample_poly(waveform, int(target_rate) // divisor, int(source_rate) // divisor).astype("float32", copy=False)


def _clean_genre(label):
    text = str(label or "").replace("_", " ").replace("-", " ").strip()
    return " ".join(part.capitalize() for part in text.split())

def resolve_whisper_model():
    """Prefer the complete upgraded snapshot; keep existing installs usable offline."""
    from huggingface_hub import snapshot_download
    for model, revision in ((WHISPER_MODEL, WHISPER_REVISION), (LEGACY_WHISPER_MODEL, LEGACY_WHISPER_REVISION)):
        try:
            path = Path(snapshot_download(model, revision=revision, local_files_only=True))
        except OSError:
            continue
        required = ("config.json", "generation_config.json", "preprocessor_config.json", "tokenizer_config.json")
        weights = any((path / filename).is_file() for filename in ("model.safetensors", "pytorch_model.bin"))
        tokenizer = (path / "tokenizer.json").is_file() or all((path / filename).is_file() for filename in ("vocab.json", "merges.txt"))
        if weights and tokenizer and all((path / filename).is_file() for filename in required):
            return str(path), model, revision
    raise FileNotFoundError("No complete Whisper model is installed. Update / Repair Engine in Settings.")


def filter_silent_words(result, audio, rate):
    """Discard only timestamped words wholly inside near-digital silence.

    This is not vocal activity detection: music alone is not proof of silence.
    Uncertain, missing and degenerate timestamps always preserve the text.
    """
    import numpy as np
    chunks = result.get("chunks") or []
    kept = []
    for chunk in chunks:
        start, end = chunk.get("timestamp") or (None, None)
        if start is not None and end is not None and 0 <= start < end <= len(audio) / rate:
            clip = audio[int(start * rate):int(end * rate)]
            if len(clip) >= int(.08 * rate) and float(np.max(np.abs(clip))) < 1e-5:
                continue
        kept.append(chunk)
    if len(kept) == len(chunks):
        return result, 0
    return {**result, "chunks": kept, "text": " ".join(str(chunk.get("text", "")).strip() for chunk in kept)}, len(chunks) - len(kept)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path, nargs="?")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    parser.add_argument("--dtype", default="fp32", choices=("fp32",))
    args = parser.parse_args()
    if args.install:
        from huggingface_hub import snapshot_download
        import torch, torchaudio, scipy, transformers
        for model, revision in (("m-a-p/SheetSage2", SHEETSAGE_REVISION), ("m-a-p/MERT-v2-FullSong", MERT_REVISION)):
            event("download", f"Downloading {model}")
            snapshot_download(model, revision=revision, allow_patterns=["*.py", "*.json", "*.safetensors", "*.txt", "*.model"])
        for model, revision in ((WHISPER_MODEL, WHISPER_REVISION), (GENRE_MODEL, GENRE_REVISION), (STYLE_MODEL, STYLE_REVISION)):
            event("download", f"Downloading {model}")
            snapshot_download(model, revision=revision, allow_patterns=["*.json", "*.txt", "*.safetensors"] + (["pytorch_model.bin"] if model == STYLE_MODEL else []))
        event("installed", "Pinned transcription models cached locally")
        return 0
    if args.audio is None or args.output is None:
        parser.error("audio and --output are required for transcription")
    if not args.audio.is_file():
        raise FileNotFoundError(args.audio)
    args.output.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, pipeline
    from scipy.io import wavfile
    from whisper_transcriber import RobustWhisperPipeline
    import numpy as np
    import torch
    from cover_score import prepare_cover_score

    # AVFoundation writes optional metadata chunks SciPy does not interpret.
    # Ignore only that benign notice; malformed/truncated WAV warnings remain.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"Chunk \(non-data\) not understood, skipping it\.", category=wavfile.WavFileWarning)
        rate, waveform = wavfile.read(args.audio)
    if waveform.ndim != 1 or not np.issubdtype(waveform.dtype, np.floating):
        raise ValueError("Expected mono float32 WAV prepared by YuE Studio")
    if not np.isfinite(waveform).all() or waveform.size < 1025:
        raise ValueError("Audio is empty, too short, or contains invalid samples")
    event("loading", "Loading SheetSage2 on CPU in FP32")

    model = AutoModel.from_pretrained("m-a-p/SheetSage2", revision=SHEETSAGE_REVISION,
                                      code_revision=SHEETSAGE_REVISION, trust_remote_code=True,
                                      local_files_only=True).eval().to(args.device)
    event("transcribing", "Transcribing source melody")
    def progress(update):
        stage = update.get("stage", "transcribing")
        detail = ", ".join(f"{key}={value}" for key, value in update.items() if key != "stage")
        event(stage, f"{stage.capitalize()}: {detail}" if detail else stage.capitalize())
    result = model.transcribe(
        waveform, sampling_rate=rate, output_dir=str(args.output), melody_only=True,
        dtype=args.dtype, preset="default", progress=progress,
    )
    score = result.get("abc")
    if not score or result.get("abc_error"):
        raise RuntimeError(result.get("abc_error") or "SheetSage2 produced no ABC melody")
    score = prepare_cover_score(score)
    score_path = args.output / "score.abc"
    score_path.write_text(score, encoding="utf-8")

    del model
    gc.collect()

    whisper_path, whisper_model, whisper_revision = resolve_whisper_model()
    event("lyrics", f"Transcribing sung text with {whisper_model}")
    speech = _resample(waveform.astype(np.float32, copy=False), rate, 16000)
    recognizer = pipeline(
        "automatic-speech-recognition", model=whisper_path, pipeline_class=RobustWhisperPipeline,
        device=-1, torch_dtype=torch.float32, model_kwargs={"local_files_only": True, "attn_implementation": "eager"},
    )
    # Sequential long-form decoding passes the real padding mask in Transformers 4.45.
    # The old chunk iterator omitted it. Do not silence the warning or invent a mask.
    recognizer.model.generation_config.forced_decoder_ids = None
    recognizer.model.config.forced_decoder_ids = None
    recognizer.generation_config.forced_decoder_ids = None
    with torch.inference_mode():
        speech_result = recognizer(speech, return_timestamps="word", generate_kwargs={
            "task": "transcribe", "condition_on_prev_tokens": False, "num_beams": 3,
            "compression_ratio_threshold": 2.4, "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6, "temperature": (0.0, 0.2, 0.4), "return_legacy_cache": False,
        })
    alignment_warnings = recognizer.alignment_warnings
    timestamp_mode = recognizer.timestamp_mode
    filtered_speech, silent_words = filter_silent_words(speech_result, speech, 16000)
    lyrics = format_lyrics(filtered_speech)
    del recognizer
    gc.collect()

    event("genre", "Suggesting musical genre")
    classifier = pipeline("audio-classification", model=snapshot_download(GENRE_MODEL, revision=GENRE_REVISION, local_files_only=True), device=-1, model_kwargs={"local_files_only": True})
    genre_rate = int(getattr(classifier.feature_extractor, "sampling_rate", rate))
    genre_audio = _resample(waveform.astype(np.float32, copy=False), rate, genre_rate)
    totals = {}
    clips = excerpts(genre_audio, genre_rate, seconds=10)
    with torch.inference_mode():
        for clip in clips:
            for item in classifier(clip, top_k=classifier.model.config.num_labels):
                label = _clean_genre(item["label"])
                totals[label] = totals.get(label, 0.0) + float(item["score"]) / len(clips)
    genre_candidates = [{"label": label, "score": score} for label, score in
                        sorted(totals.items(), key=lambda item: item[1], reverse=True)[:3]]
    genre = genre_candidates[0]["label"] if genre_candidates else ""
    del classifier
    gc.collect()
    analysis_warnings = list(result.get("warnings") or []) + alignment_warnings
    analysis_warnings.append("Lyric sections are inferred from repeated passages and position; review before generating.")
    if whisper_model == LEGACY_WHISPER_MODEL:
        analysis_warnings.append("Using Whisper small. Update / Repair Engine in Settings to install Whisper large-v3-turbo.")
    if silent_words:
        analysis_warnings.append(f"Omitted {silent_words} timestamped words in digital silence; raw transcription is retained.")
    descriptors = []
    try:
        event("style", "Analyzing instrumentation, vocals, mood and production")
        descriptors = describe_style(_resample(waveform, rate, 48000), 48000)
    except OSError:
        analysis_warnings.append("Detailed style model unavailable locally. Update the cover engine in Settings.")
    style = ", ".join([genre] + [item["label"] for item in descriptors])

    for warning in analysis_warnings:
        event("warning", str(warning))

    analysis = {
        "status": "complete", "model": "m-a-p/SheetSage2", "revision": SHEETSAGE_REVISION,
        "melody_only": True,
        "source": str(args.audio), "warnings": analysis_warnings,
        "style": style, "style_descriptors": descriptors, "style_model": STYLE_MODEL,
        "style_revision": STYLE_REVISION, "lyrics_raw": speech_result.get("text", ""),
        "lyrics_segments": speech_result.get("chunks", []),
        "lyrics": lyrics, "genre": genre, "genre_candidates": genre_candidates,
        "lyrics_model": whisper_model, "lyrics_revision": whisper_revision,
        "lyrics_decoding": {"requested_num_beams": 3, "word_timestamps": timestamp_mode == "word", "timestamp_mode": timestamp_mode, "condition_on_prev_tokens": False},
        "genre_model": GENRE_MODEL, "genre_revision": GENRE_REVISION,
    }
    (args.output / "cover_transcription.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    (args.output / "cover_analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    event("complete", str(score_path))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        traceback.print_exc()
        print(f"transcribe_cover: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
