"""Optional Hybrid Demucs vocal activity extraction.

The separator is used to find singing windows, not as an unconditional
replacement for the original mix. Whisper still receives the original mix so
separation artefacts cannot directly rewrite the recognized lyrics.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

SEPARATION_REVISION = "torchaudio-2.8.0-hdemucs-high-musdb"
SEPARATION_MODEL_NAME = "HDEMUCS_HIGH_MUSDB"
SEPARATION_ASSET = "models/hdemucs_high_musdbhq_only.pt"
SEPARATION_URL = "https://download.pytorch.org/torchaudio/" + SEPARATION_ASSET
SEPARATION_MIN_BYTES = 100 * 1024 * 1024


def separation_asset_path() -> Path:
    import torch
    return Path(torch.hub.get_dir()) / "torchaudio" / SEPARATION_ASSET


def separation_model_available() -> bool:
    try:
        path = separation_asset_path()
        return path.is_file() and path.stat().st_size >= SEPARATION_MIN_BYTES
    except Exception:
        return False


def download_separation_model() -> Path:
    """Download the asset without constructing the 319 MB model in memory."""
    import torch

    path = separation_asset_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size < SEPARATION_MIN_BYTES:
        path.unlink()
    # Keep the download path independent from TorchAudio's deprecated asset
    # helper. PyTorch's downloader writes a partial file and moves it only
    # after completion, so an interrupted transfer cannot look ready.
    torch.hub.download_url_to_file(SEPARATION_URL, str(path), progress=True)
    downloaded = path
    if not downloaded.is_file() or downloaded.stat().st_size < SEPARATION_MIN_BYTES:
        raise OSError(f"Hybrid Demucs download is incomplete: {downloaded}")
    return downloaded


def _split_windows(start: float, end: float, maximum: float = 28.0, overlap: float = 2.0) -> list[tuple[float, float]]:
    if end <= start:
        return []
    if end - start <= maximum:
        return [(round(start, 3), round(end, 3))]
    result = []
    cursor = start
    while cursor < end:
        boundary = min(end, cursor + maximum)
        result.append((round(cursor, 3), round(boundary, 3)))
        if boundary >= end:
            break
        cursor = max(cursor + 0.5, boundary - overlap)
    return result


def activity_windows(scores, frame_seconds: float, duration: float, maximum: float = 28.0) -> list[tuple[float, float]]:
    """Convert vocal energy into padded, bounded transcription windows."""
    import numpy as np

    values = np.asarray(scores, dtype=np.float32)
    if values.size == 0 or duration <= 0 or not np.isfinite(values).any():
        return _split_windows(0.0, duration, maximum)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    positive = values[values > 1e-7]
    if positive.size == 0:
        return _split_windows(0.0, duration, maximum)
    # The relative threshold survives different recording gains and leaves a
    # full-song fallback when the separator sees continuous vocals.
    threshold = max(float(np.percentile(positive, 35)) * 0.75, float(values.max()) * 0.16)
    active = values >= threshold
    gap_frames = max(1, int(round(1.25 / frame_seconds)))
    active_list = active.tolist()
    index = 0
    while index < len(active_list):
        if active_list[index]:
            index += 1
            continue
        end = index
        while end < len(active_list) and not active_list[end]:
            end += 1
        if index > 0 and end < len(active_list) and end - index <= gap_frames:
            active_list[index:end] = [True] * (end - index)
        index = end
    spans = []
    index = 0
    while index < len(active_list):
        if not active_list[index]:
            index += 1
            continue
        end = index + 1
        while end < len(active_list) and active_list[end]:
            end += 1
        start_time = max(0.0, index * frame_seconds - 0.5)
        end_time = min(duration, end * frame_seconds + 0.5)
        if end_time - start_time >= 0.8:
            spans.append((start_time, end_time))
        index = end
    if not spans or sum(end - start for start, end in spans) >= duration * 0.92:
        return _split_windows(0.0, duration, maximum)
    merged = []
    for start, end in spans:
        if merged and start <= merged[-1][1] + 0.4:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return [window for start, end in merged for window in _split_windows(start, end, maximum)]


def separate_vocal_activity(audio, rate: int, *, device: str = "cpu", chunk_seconds: float = 12.0,
                            overlap_seconds: float = 2.0, frame_seconds: float = 0.5) -> dict[str, Any]:
    """Run one Demucs chunk at a time and return only a compact energy track."""
    import gc
    import numpy as np
    import torch
    import torchaudio

    if not separation_model_available():
        raise FileNotFoundError("Hybrid Demucs model is not installed")
    values = np.asarray(audio, dtype=np.float32)
    if values.ndim == 1:
        values = np.repeat(values[:, None], 2, axis=1)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("Expected audio with shape (frames, channels)")
    channels = min(2, values.shape[1])
    stereo = torch.from_numpy(values[:, :channels].T.copy())
    if channels == 1:
        stereo = stereo.repeat(2, 1)
    bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB
    model = bundle.get_model().to(device).eval()
    model_rate = int(bundle.sample_rate)
    if int(rate) != model_rate:
        stereo = torchaudio.functional.resample(stereo, int(rate), model_rate)
    total_frames = stereo.shape[-1]
    chunk_frames = max(int(chunk_seconds * model_rate), model_rate)
    overlap_frames = int(overlap_seconds * model_rate)
    step_frames = max(model_rate, chunk_frames - overlap_frames)
    output_frames = max(1, int(math.ceil(values.shape[0] / rate / frame_seconds)))
    totals = np.zeros(output_frames, dtype=np.float64)
    weights = np.zeros(output_frames, dtype=np.float64)
    # The loaded model exposes the source order; do not call get_model() again
    # because that would reload the separator and double its memory use.
    vocal_index = list(model.sources).index("vocals") if hasattr(model, "sources") else 3
    with torch.inference_mode():
        for start in range(0, total_frames, step_frames):
            stop = min(total_frames, start + chunk_frames)
            chunk = stereo[:, start:stop]
            actual = chunk.shape[-1]
            if actual < chunk_frames:
                chunk = torch.nn.functional.pad(chunk, (0, chunk_frames - actual))
            stems = model(chunk.unsqueeze(0))
            vocal = stems[0, vocal_index, :, :actual]
            rms = torch.sqrt(torch.mean(vocal * vocal, dim=0) + 1e-8).detach().cpu().numpy()
            global_start = start / model_rate * rate
            global_end = (start + actual) / model_rate * rate
            first_frame = max(0, int(global_start / rate / frame_seconds))
            last_frame = min(output_frames, int(math.ceil(global_end / rate / frame_seconds)))
            for frame in range(first_frame, last_frame):
                sample_start = max(0, int(frame * frame_seconds * model_rate - start * 1.0))
                sample_end = min(len(rms), max(sample_start + 1, int((frame + 1) * frame_seconds * model_rate - start * 1.0)))
                if sample_start < sample_end:
                    totals[frame] += float(np.mean(rms[sample_start:sample_end]))
                    weights[frame] += 1.0
            del stems, vocal, rms
    del model
    gc.collect()
    scores = (totals / np.maximum(weights, 1.0)).astype(np.float32)
    return {
        "scores": scores.tolist(),
        "frame_seconds": frame_seconds,
        "sample_rate": int(rate),
        "model": SEPARATION_MODEL_NAME,
        "revision": SEPARATION_REVISION,
    }
