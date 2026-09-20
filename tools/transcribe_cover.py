#!/usr/bin/env python3
"""Transcribe an audio file to a melody-only ABC score for YuE2 covers.

Runs in the separate Python 3.11 cover environment. AVFoundation supplies a mono WAV;
array input avoids an external FFmpeg dependency. Model revisions are pinned.
"""
import argparse
import json
import sys
import gc
import os
import subprocess
import time
import traceback
import warnings
from analysis_cache import StageCache
from audio_analysis import STYLE_MODEL, STYLE_REVISION, clean_genre_label, describe_style, excerpts, format_lyrics, summarize_genre
from pathlib import Path
from structure_analysis import STRUCTURE_REVISION, analyze_acoustic_structure
from voice_activity import (SEPARATION_MODEL_NAME, SEPARATION_REVISION,
                            activity_windows, download_separation_model, separation_model_available,
                            separate_vocal_activity)

SHEETSAGE_REVISION = "80af707174fc7ee521c25925d5f014729f0e61ae"
MERT_REVISION = "d8ba1c745e733b3908ce6ad16ebeb17ac7600a42"
WHISPER_MODEL = "openai/whisper-large-v3-turbo"
WHISPER_REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
LEGACY_WHISPER_MODEL = "openai/whisper-small"
LEGACY_WHISPER_REVISION = "973afd24965f72e36ca33b3055d56a652f456b4d"
GENRE_MODEL = "dima806/music_genres_classification"
GENRE_REVISION = "5f71fb1e2c6bedcddb2bfb1e929fc70655780902"
MLX_MODEL = "mlx-community/whisper-large-v3-turbo"
MLX_REVISION = "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb"

COMPONENTS = ("melody", "lyrics", "genre", "style", "mlxWhisper", "vocalActivity")

def event(stage, message):
    print(json.dumps({"event": "transcription", "stage": stage, "message": message}), flush=True)


def _resample(waveform, source_rate, target_rate):
    if source_rate == target_rate:
        return waveform
    from scipy.signal import resample_poly
    import math
    divisor = math.gcd(int(source_rate), int(target_rate))
    return resample_poly(waveform, int(target_rate) // divisor, int(source_rate) // divisor).astype("float32", copy=False)


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


def resolve_mlx_model():
    """Resolve the converted model without allowing analysis to download it."""
    from huggingface_hub import snapshot_download
    try:
        return str(snapshot_download(MLX_MODEL, revision=MLX_REVISION, local_files_only=True,
                                     allow_patterns=["config.json", "weights.safetensors", "*.json"]))
    except OSError:
        return None


def _shift_segments(result, offset: float, fallback_end: float):
    chunks = []
    for segment in result.get("segments") or result.get("chunks") or []:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        start = segment.get("start")
        end = segment.get("end")
        timestamp = segment.get("timestamp")
        if start is None and isinstance(timestamp, (list, tuple)) and len(timestamp) == 2:
            start, end = timestamp
        if not isinstance(start, (int, float)):
            start = 0.0
        if not isinstance(end, (int, float)) or end < start:
            end = fallback_end
        chunks.append({"text": text, "timestamp": (float(start) + offset, float(end) + offset)})
    return chunks


def _merge_transcription_results(results):
    """Merge bounded windows while removing exact overlap duplicates."""
    from difflib import SequenceMatcher
    chunks = []
    for result, offset, fallback_end in results:
        for chunk in _shift_segments(result, offset, fallback_end):
            normalized = " ".join(chunk["text"].casefold().split())
            if chunks:
                previous = chunks[-1]
                previous_text = " ".join(previous["text"].casefold().split())
                overlap = min(previous["timestamp"][1], chunk["timestamp"][1]) - max(previous["timestamp"][0], chunk["timestamp"][0])
                if normalized == previous_text or (overlap > 0 and SequenceMatcher(None, previous_text, normalized).ratio() >= .88):
                    if len(normalized) > len(previous_text):
                        chunks[-1] = chunk
                    continue
            chunks.append(chunk)
    text = " ".join(chunk["text"] for chunk in chunks).strip()
    return {"text": text, "chunks": chunks}


def _run_mlx_transcription(mlx_python: Path, worker: Path, model_path: str, audio, output: Path, windows):
    import numpy as np
    audio_path = output / "mlx-audio.npy"
    windows_path = output / "mlx-windows.json"
    np.save(audio_path, np.asarray(audio, dtype=np.float32))
    arguments = [str(mlx_python), str(worker), str(audio_path), "--model", model_path]
    if windows:
        windows_path.write_text(json.dumps(windows), encoding="utf-8")
        arguments.extend(["--windows", str(windows_path)])
    completed = subprocess.run(arguments, capture_output=True, text=True, check=False,
                               env=os.environ.copy())
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or "MLX Whisper worker failed")
    for line in reversed(completed.stdout.splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError("MLX Whisper worker returned no JSON result")


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


def segment_diagnostics(result):
    """Keep alignment evidence without making alignment a hard dependency."""
    diagnostics = []
    for index, chunk in enumerate(result.get("chunks") or []):
        timestamp = chunk.get("timestamp")
        start, end = timestamp if isinstance(timestamp, (list, tuple)) and len(timestamp) == 2 else (None, None)
        valid = (
            isinstance(start, (int, float)) and isinstance(end, (int, float))
            and start >= 0 and end >= start
        )
        diagnostics.append({
            "index": index,
            "start": start,
            "end": end,
            "text_characters": len(str(chunk.get("text", ""))),
            "token_count_estimate": len(str(chunk.get("text", "")).split()),
            "timestamp_valid": valid,
            "timestamp_count": 2 if isinstance(timestamp, (list, tuple)) else 0,
        })
    return diagnostics


def install_component(component: str) -> None:
    """Download exactly one analysis component into the shared local cache."""
    from huggingface_hub import snapshot_download

    if component == "melody":
        for model, revision in (("m-a-p/SheetSage2", SHEETSAGE_REVISION), ("m-a-p/MERT-v2-FullSong", MERT_REVISION)):
            event("download", f"Downloading {model}")
            snapshot_download(model, revision=revision, allow_patterns=["*.py", "*.json", "*.safetensors", "*.txt", "*.model"])
    elif component == "lyrics":
        event("download", f"Downloading {WHISPER_MODEL}")
        snapshot_download(WHISPER_MODEL, revision=WHISPER_REVISION,
                          allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model"])
    elif component == "genre":
        event("download", f"Downloading {GENRE_MODEL}")
        snapshot_download(GENRE_MODEL, revision=GENRE_REVISION,
                          allow_patterns=["*.json", "*.txt", "*.safetensors", "*.bin"])
    elif component == "style":
        event("download", f"Downloading {STYLE_MODEL}")
        snapshot_download(STYLE_MODEL, revision=STYLE_REVISION,
                          allow_patterns=["*.json", "*.txt", "*.safetensors", "pytorch_model.bin"])
    elif component == "mlxWhisper":
        event("download", f"Downloading {MLX_MODEL}")
        snapshot_download(MLX_MODEL, revision=MLX_REVISION,
                          allow_patterns=["config.json", "weights.safetensors", "*.json"])
    elif component == "vocalActivity":
        event("download", "Downloading Hybrid Demucs vocal-boundary model")
        path = download_separation_model()
        event("download", f"Hybrid Demucs asset ready at {path.name}")
    else:
        raise ValueError(f"Unknown analysis component: {component}")
    event("installed", f"{component} component ready")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path, nargs="?")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--install-component", choices=COMPONENTS)
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    parser.add_argument("--dtype", default="fp32", choices=("fp32",))
    parser.add_argument("--cache-dir", type=Path, help="Persistent directory for resumable stage evidence")
    parser.add_argument("--mlx-python", type=Path, help="Optional isolated MLX Whisper Python executable")
    parser.add_argument("--disable-separation", action="store_true", help="Use the original mix without vocal-boundary analysis")
    args = parser.parse_args()
    if args.install_component:
        install_component(args.install_component)
        return 0
    if args.install:
        from huggingface_hub import snapshot_download
        import torch, torchaudio, scipy, transformers
        for model, revision in (("m-a-p/SheetSage2", SHEETSAGE_REVISION), ("m-a-p/MERT-v2-FullSong", MERT_REVISION)):
            event("download", f"Downloading {model}")
            snapshot_download(model, revision=revision, allow_patterns=["*.py", "*.json", "*.safetensors", "*.txt", "*.model"])
        for model, revision in ((WHISPER_MODEL, WHISPER_REVISION), (GENRE_MODEL, GENRE_REVISION), (STYLE_MODEL, STYLE_REVISION)):
            event("download", f"Downloading {model}")
            snapshot_download(model, revision=revision, allow_patterns=["*.json", "*.txt", "*.safetensors"] + (["pytorch_model.bin"] if model == STYLE_MODEL else []))
        try:
            event("download", f"Downloading {MLX_MODEL}")
            snapshot_download(MLX_MODEL, revision=MLX_REVISION, allow_patterns=["config.json", "weights.safetensors", "*.json"])
        except OSError as exc:
            event("warning", f"MLX Whisper model unavailable; Transformers fallback remains active: {exc}")
        try:
            import torch
            import torchaudio
            event("download", "Downloading Hybrid Demucs vocal-boundary model")
            torchaudio.pipelines.HDEMUCS_HIGH_MUSDB.get_model()
            del torch, torchaudio
        except (OSError, RuntimeError) as exc:
            event("warning", f"Vocal-boundary model unavailable; full-mix transcription remains active: {exc}")
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
    if waveform.ndim not in (1, 2) or not np.issubdtype(waveform.dtype, np.floating):
        raise ValueError("Expected float32 WAV prepared by YuE Studio")
    if waveform.ndim == 2 and waveform.shape[1] > 2:
        waveform = waveform[:, :2]
    stereo_waveform = waveform.astype(np.float32, copy=False)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1, dtype=np.float32)
    else:
        stereo_waveform = waveform[:, None]
    if not np.isfinite(waveform).all() or waveform.size < 1025:
        raise ValueError("Audio is empty, too short, or contains invalid samples")
    cache = StageCache(args.cache_dir, args.audio) if args.cache_dir else StageCache(None, args.audio)
    cache_signature = lambda **values: values
    event("loading", "Loading SheetSage2 on CPU in FP32")
    melody_signature = cache_signature(stage="melody", revision=SHEETSAGE_REVISION, dtype=args.dtype, preset="default")
    result = cache.load("melody", melody_signature)
    if result is not None:
        event("cache", "Reusing cached melody analysis")
        score = result["score"]
    else:
        melody_started = time.monotonic()
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
            cache.fail("melody", melody_signature, str(result.get("abc_error") or "SheetSage2 produced no ABC melody"))
            raise RuntimeError(result.get("abc_error") or "SheetSage2 produced no ABC melody")
        score = prepare_cover_score(score)
        cache.save("melody", melody_signature, {
            "score": score,
            "warnings": list(result.get("warnings") or []),
            "diagnostics": result.get("diagnostics"),
            "model": "m-a-p/SheetSage2",
            "revision": SHEETSAGE_REVISION,
        }, elapsed_seconds=time.monotonic() - melody_started)
    score_path = args.output / "score.abc"
    score_path.write_text(score, encoding="utf-8")
    melody_warnings = list(result.get("warnings") or [])
    melody_diagnostics = result.get("diagnostics")
    if "model" in locals():
        del model
    gc.collect()

    duration = len(waveform) / float(rate)
    structure_signature = cache_signature(stage="structure", revision=STRUCTURE_REVISION, sample_rate=rate)
    structure_stage = cache.load("structure", structure_signature)
    if structure_stage is not None:
        event("cache", "Reusing cached acoustic structure evidence")
        acoustic_structure = structure_stage
    else:
        structure_started = time.monotonic()
        event("structure", "Analyzing acoustic repetitions and tempo")
        acoustic_structure = analyze_acoustic_structure(waveform, rate)
        cache.save("structure", structure_signature, acoustic_structure, elapsed_seconds=time.monotonic() - structure_started)

    separation_signature = cache_signature(stage="voice_activity", revision=SEPARATION_REVISION, model=SEPARATION_MODEL_NAME, sample_rate=rate)
    separation_stage = cache.load("voice_activity", separation_signature)
    vocal_windows = None
    separation_status = "unavailable"
    if args.disable_separation:
        separation_status = "disabled"
        event("structure", "Vocal-boundary analysis disabled; using the original mix")
    elif separation_stage is not None:
        event("cache", "Reusing cached vocal-boundary windows")
        vocal_windows = [tuple(window) for window in separation_stage.get("windows", [])]
        separation_status = separation_stage.get("status", "cached")
    elif separation_model_available():
        separation_started = time.monotonic()
        try:
            event("separation", "Finding singing windows with Hybrid Demucs")
            activity = separate_vocal_activity(stereo_waveform, rate)
            vocal_windows = activity_windows(activity["scores"], activity["frame_seconds"], duration)
            separation_status = "complete"
            cache.save("voice_activity", separation_signature, {
                "windows": vocal_windows,
                "status": separation_status,
                "model": SEPARATION_MODEL_NAME,
                "revision": SEPARATION_REVISION,
                "frame_seconds": activity["frame_seconds"],
                "score_count": len(activity["scores"]),
            }, elapsed_seconds=time.monotonic() - separation_started)
        except (OSError, RuntimeError, ValueError) as exc:
            cache.fail("voice_activity", separation_signature, str(exc))
            event("warning", f"Vocal-boundary analysis unavailable; using the original mix: {exc}")
    else:
        event("warning", "Hybrid Demucs is not installed; using the original mix for transcription")

    speech = _resample(waveform.astype(np.float32, copy=False), rate, 16000)
    mlx_model_path = None
    if args.mlx_python and args.mlx_python.is_file():
        mlx_model_path = resolve_mlx_model()
    try:
        whisper_path, transformer_model, transformer_revision = resolve_whisper_model()
    except FileNotFoundError:
        whisper_path, transformer_model, transformer_revision = None, WHISPER_MODEL, WHISPER_REVISION
    backend = "mlx" if mlx_model_path else "transformers"
    lyrics_model = MLX_MODEL if backend == "mlx" else transformer_model
    lyrics_revision = MLX_REVISION if backend == "mlx" else transformer_revision
    lyrics_signature = cache_signature(stage="lyrics", backend=backend, revision=lyrics_revision,
                                       timestamp_mode="segment", num_beams=1,
                                       condition_on_prev_tokens=False, windowing=separation_status)
    lyrics_stage = cache.load("lyrics", lyrics_signature)
    if lyrics_stage is not None:
        event("cache", "Reusing cached lyric transcription")
        speech_result = lyrics_stage["speech_result"]
        lyrics = lyrics_stage["lyrics"]
        alignment_warnings = lyrics_stage.get("alignment_warnings", [])
        timestamp_mode = lyrics_stage.get("timestamp_mode", "segment")
        silent_words = int(lyrics_stage.get("silent_words", 0))
        lyrics_diagnostics = lyrics_stage.get("segment_diagnostics", segment_diagnostics(speech_result))
        backend = lyrics_stage.get("backend", backend)
        lyrics_model = lyrics_stage.get("model", lyrics_model)
        lyrics_revision = lyrics_stage.get("revision", lyrics_revision)
    else:
        lyrics_started = time.monotonic()
        alignment_warnings = []
        speech_result = None
        timestamp_mode = "segment"
        if backend == "mlx":
            try:
                event("lyrics", "Transcribing sung text with MLX Whisper on Apple Silicon")
                worker = Path(__file__).with_name("mlx_whisper_worker.py")
                speech_result = _run_mlx_transcription(args.mlx_python, worker, mlx_model_path, speech, args.output, vocal_windows)
            except (OSError, RuntimeError, ValueError) as exc:
                event("warning", f"MLX Whisper unavailable; using Transformers fallback: {exc}")
                alignment_warnings.append(f"MLX Whisper unavailable; used the Transformers fallback: {exc}")
                backend = "transformers"
                lyrics_model = transformer_model
                lyrics_revision = transformer_revision
                lyrics_signature = cache_signature(stage="lyrics", backend=backend, revision=lyrics_revision,
                                                   timestamp_mode="segment", num_beams=1,
                                                   condition_on_prev_tokens=False, windowing=separation_status)
        if speech_result is None:
            if whisper_path is None:
                event("warning", "Whisper is not installed; continuing without lyric transcription")
                cache.fail("lyrics", lyrics_signature, "No complete Whisper model is installed")
                speech_result = {"text": "", "chunks": []}
                backend = "unavailable"
                lyrics_model = "unavailable"
                lyrics_revision = ""
            else:
                event("lyrics", f"Transcribing sung text with {transformer_model}")
                recognizer = pipeline(
                    "automatic-speech-recognition", model=whisper_path, pipeline_class=RobustWhisperPipeline,
                    device=-1, torch_dtype=torch.float32, model_kwargs={"local_files_only": True, "attn_implementation": "eager"},
                )
                recognizer.model.generation_config.forced_decoder_ids = None
                recognizer.model.config.forced_decoder_ids = None
                recognizer.generation_config.forced_decoder_ids = None
                windows = vocal_windows or [(0.0, duration)]
                results = []
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=r"The input name `inputs` is deprecated.*", category=FutureWarning)
                    warnings.filterwarnings("ignore", message=r"Passing a tuple of `past_key_values` is deprecated.*", category=FutureWarning)
                    with torch.inference_mode():
                        for start, end in windows:
                            left = max(0, int(start * 16000))
                            right = min(len(speech), max(left + 16000, int(end * 16000)))
                            clip = speech[left:right]
                            result = recognizer(clip, return_timestamps=True, generate_kwargs={
                                "task": "transcribe", "condition_on_prev_tokens": False, "num_beams": 1,
                                "compression_ratio_threshold": 2.4, "logprob_threshold": -1.0,
                                "no_speech_threshold": 0.6, "temperature": (0.0, 0.2, 0.4), "return_legacy_cache": False,
                            })
                            results.append((result, left / 16000.0, right / 16000.0))
                            alignment_warnings.extend(getattr(recognizer, "alignment_warnings", []))
                speech_result = _merge_transcription_results(results)
                timestamp_mode = recognizer.timestamp_mode
                del recognizer
        filtered_speech, silent_words = filter_silent_words(speech_result, speech, 16000)
        speech_result = filtered_speech
        lyrics = format_lyrics(speech_result, acoustic_structure)
        lyrics_diagnostics = segment_diagnostics(speech_result)
        cache.save("lyrics", lyrics_signature, {
            "speech_result": speech_result,
            "lyrics": lyrics,
            "alignment_warnings": alignment_warnings,
            "timestamp_mode": timestamp_mode,
            "silent_words": silent_words,
            "segment_diagnostics": lyrics_diagnostics,
            "model": lyrics_model,
            "revision": lyrics_revision,
            "backend": backend,
            "window_count": len(vocal_windows or [(0.0, duration)]),
        }, elapsed_seconds=time.monotonic() - lyrics_started)
    gc.collect()

    genre_warnings = []
    genre_signature = cache_signature(stage="genre", revision=GENRE_REVISION, excerpt_seconds=10, excerpt_count=3)
    genre_stage = cache.load("genre", genre_signature)
    if genre_stage is not None:
        event("cache", "Reusing cached genre evidence")
        genre_candidates = genre_stage["genre_candidates"]
        genre_excerpt_candidates = genre_stage.get("genre_excerpt_candidates", [])
        genre = genre_stage["genre"]
    else:
        genre_started = time.monotonic()
        event("genre", "Suggesting musical genre")
        try:
            classifier = pipeline("audio-classification", model=snapshot_download(GENRE_MODEL, revision=GENRE_REVISION, local_files_only=True), device=-1, model_kwargs={"local_files_only": True})
            genre_rate = int(getattr(classifier.feature_extractor, "sampling_rate", rate))
            genre_audio = _resample(waveform.astype(np.float32, copy=False), rate, genre_rate)
            totals = {}
            genre_excerpt_candidates = []
            clips = excerpts(genre_audio, genre_rate, seconds=10)
            with torch.inference_mode():
                for clip in clips:
                    excerpt = []
                    for item in classifier(clip, top_k=classifier.model.config.num_labels):
                        label = clean_genre_label(item["label"])
                        score_value = float(item["score"])
                        totals[label] = totals.get(label, 0.0) + score_value / max(1, len(clips))
                        excerpt.append({"label": label, "score": score_value})
                    genre_excerpt_candidates.append(sorted(excerpt, key=lambda item: item["score"], reverse=True)[:3])
            genre_candidates = [{"label": label, "score": score_value} for label, score_value in
                                sorted(totals.items(), key=lambda item: item[1], reverse=True)[:5]]
            genre = summarize_genre(genre_candidates, genre_excerpt_candidates)
            cache.save("genre", genre_signature, {
                "genre": genre,
                "genre_candidates": genre_candidates,
                "genre_excerpt_candidates": genre_excerpt_candidates,
                "model": GENRE_MODEL,
                "revision": GENRE_REVISION,
            }, elapsed_seconds=time.monotonic() - genre_started)
            del classifier
        except (OSError, RuntimeError, ValueError) as exc:
            cache.fail("genre", genre_signature, str(exc))
            warning = f"Genre classifier unavailable; continuing without it: {exc}"
            event("warning", warning)
            genre_warnings.append(warning)
            genre_candidates = []
            genre_excerpt_candidates = []
            genre = ""
    gc.collect()
    analysis_warnings = melody_warnings + alignment_warnings + genre_warnings
    if lyrics.strip():
        analysis_warnings.append("Lyric sections are inferred from repeated passages and position; review before generating.")
    if lyrics_model == "unavailable":
        analysis_warnings.append("Lyric transcription is unavailable locally. Install the Lyrics component in Settings.")
    if lyrics_model == LEGACY_WHISPER_MODEL:
        analysis_warnings.append("Using Whisper small. Update / Repair Engine in Settings to install Whisper large-v3-turbo.")
    if silent_words:
        analysis_warnings.append(f"Omitted {silent_words} timestamped words in digital silence; raw transcription is retained.")
    if separation_status == "unavailable":
        analysis_warnings.append("Vocal activity detection is unavailable; Whisper used the original mix.")
    style_signature = cache_signature(stage="style", revision=STYLE_REVISION, excerpt_seconds=10, excerpt_count=3)
    style_stage = cache.load("style", style_signature)
    if style_stage is not None:
        event("cache", "Reusing cached style evidence")
        descriptors = style_stage.get("descriptors", [])
    else:
        style_started = time.monotonic()
        descriptors = []
        try:
            event("style", "Analyzing instrumentation, vocals, mood and production")
            descriptors = describe_style(_resample(waveform, rate, 48000), 48000)
            cache.save("style", style_signature, {"descriptors": descriptors, "model": STYLE_MODEL, "revision": STYLE_REVISION}, elapsed_seconds=time.monotonic() - style_started)
        except (OSError, RuntimeError) as exc:
            cache.fail("style", style_signature, str(exc))
            analysis_warnings.append("Detailed style model unavailable locally. Update the cover engine in Settings.")
    style_parts = [part for part in [genre] + [item["label"] for item in descriptors if item.get("label")] if part]
    style = ", ".join(style_parts)

    for warning in analysis_warnings:
        event("warning", str(warning))

    analysis = {
        "status": "complete", "model": "m-a-p/SheetSage2", "revision": SHEETSAGE_REVISION,
        "melody_only": True,
        "source": str(args.audio), "warnings": analysis_warnings,
        "style": style, "style_descriptors": descriptors, "style_model": STYLE_MODEL,
        "style_revision": STYLE_REVISION, "lyrics_raw": speech_result.get("text", ""),
        "lyrics_segments": speech_result.get("chunks", []),
        "lyrics_segment_diagnostics": lyrics_diagnostics,
        "lyrics": lyrics, "genre": genre, "genre_candidates": genre_candidates,
        "genre_excerpt_candidates": genre_excerpt_candidates,
        "lyrics_model": lyrics_model, "lyrics_revision": lyrics_revision,
        "lyrics_decoding": {"backend": backend, "requested_num_beams": 1, "word_timestamps": False, "timestamp_mode": timestamp_mode, "condition_on_prev_tokens": False, "window_count": len(vocal_windows or [(0.0, duration)])},
        "genre_model": GENRE_MODEL, "genre_revision": GENRE_REVISION,
        "acoustic_structure": acoustic_structure,
        "voice_activity": {"status": separation_status, "windows": vocal_windows or [], "model": SEPARATION_MODEL_NAME, "revision": SEPARATION_REVISION},
        "cache_key": cache.source_sha256, "stages": cache.records,
    }
    (args.output / "cover_transcription.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    (args.output / "cover_analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    cache.save("complete", {"stage": "complete", "revision": 1}, analysis)
    event("complete", str(score_path))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        traceback.print_exc()
        print(f"transcribe_cover: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
