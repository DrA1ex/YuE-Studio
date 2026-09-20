#!/usr/bin/env python3
"""Minimal MLX Whisper worker used from the isolated MLX environment."""
import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--windows", type=Path)
    args = parser.parse_args()
    import mlx_whisper
    import numpy as np

    clip_timestamps = "0"
    if args.windows and args.windows.is_file():
        windows = json.loads(args.windows.read_text(encoding="utf-8"))
        clip_timestamps = [value for window in windows for value in window]
    result = mlx_whisper.transcribe(
        np.load(args.audio),
        path_or_hf_repo=args.model,
        verbose=None,
        word_timestamps=False,
        condition_on_previous_text=False,
        clip_timestamps=clip_timestamps,
        temperature=(0.0, 0.2, 0.4),
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
        fp16=True,
    )
    print(json.dumps(result, ensure_ascii=False, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
