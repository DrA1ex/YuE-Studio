"""Lightweight acoustic repetition and tempo evidence.

This is a deterministic fallback for section analysis. It never invents a
semantic label by itself; lyric repetition remains required before a section
is called a chorus.
"""
from __future__ import annotations

import math


STRUCTURE_REVISION = "spectral-repetition-v1"


def _feature(audio, rate: int, start: int, size: int):
    import numpy as np

    clip = np.asarray(audio[start:start + size], dtype=np.float32)
    if len(clip) < size:
        clip = np.pad(clip, (0, size - len(clip)))
    window = np.hanning(len(clip)).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(clip * window, n=4096))
    frequencies = np.fft.rfftfreq(4096, 1.0 / rate)
    edges = np.geomspace(60.0, min(12000.0, rate / 2.0), 25)
    bands = []
    for low, high in zip(edges[:-1], edges[1:]):
        selected = spectrum[(frequencies >= low) & (frequencies < high)]
        bands.append(float(np.log1p(np.mean(selected) if len(selected) else 0.0)))
    rms = float(np.sqrt(np.mean(clip * clip) + 1e-9))
    centroid = float(np.sum(frequencies * spectrum) / max(np.sum(spectrum), 1e-8)) / max(rate, 1)
    vector = np.asarray(bands + [math.log1p(rms), centroid], dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def _cosine(left, right):
    import numpy as np
    return float(np.dot(left, right) / max(np.linalg.norm(left) * np.linalg.norm(right), 1e-8))


def estimate_tempo(audio, rate: int):
    """Return cautious BPM candidates from an onset-like envelope."""
    import numpy as np
    from scipy.signal import find_peaks

    values = np.asarray(audio, dtype=np.float32)
    if len(values) < rate * 2:
        return []
    hop = max(1, int(rate * 0.02))
    envelope = np.asarray([np.sqrt(np.mean(values[i:i + hop] ** 2) + 1e-9) for i in range(0, len(values) - hop, hop)])
    onset = np.maximum(0.0, np.diff(envelope, prepend=envelope[:1]))
    centered = onset - np.median(onset)
    autocorrelation = np.correlate(centered, centered, mode="full")[len(centered) - 1:]
    minimum = max(1, int(60.0 / 240.0 / .02))
    maximum = min(len(autocorrelation) - 1, int(60.0 / 45.0 / .02))
    if maximum <= minimum:
        return []
    peaks, _ = find_peaks(autocorrelation[minimum:maximum], distance=max(1, int(.12 / .02)))
    if len(peaks) == 0:
        return []
    ranked = sorted(((float(autocorrelation[minimum + peak]), 60.0 / ((minimum + peak) * .02)) for peak in peaks), reverse=True)
    best = ranked[0][1]
    candidates = sorted({round(best, 1), round(best / 2.0, 1), round(best * 2.0, 1)})
    return [bpm for bpm in candidates if 45.0 <= bpm <= 240.0]


def analyze_acoustic_structure(audio, rate: int) -> dict:
    import numpy as np

    values = np.asarray(audio, dtype=np.float32)
    frame_seconds, hop_seconds = 8.0, 2.0
    size, hop = max(1, int(frame_seconds * rate)), max(1, int(hop_seconds * rate))
    if len(values) == 0:
        return {"revision": STRUCTURE_REVISION, "frames": [], "repeated_regions": [], "tempo_candidates": []}
    frames = [_feature(values, rate, start, size) for start in range(0, max(1, len(values) - size + 1), hop)]
    repeats = []
    for left in range(len(frames)):
        for right in range(left + 3, len(frames)):
            similarity = _cosine(frames[left], frames[right])
            if similarity >= .93:
                repeats.append({
                    "left_start": round(left * hop_seconds, 3),
                    "right_start": round(right * hop_seconds, 3),
                    "duration": frame_seconds,
                    "similarity": round(similarity, 4),
                })
    return {
        "revision": STRUCTURE_REVISION,
        "frame_seconds": frame_seconds,
        "hop_seconds": hop_seconds,
        "frame_count": len(frames),
        "repeated_regions": repeats[:128],
        "tempo_candidates": estimate_tempo(values, rate),
        "source": "spectral-repetition-fallback",
    }
