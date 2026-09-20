"""Bounded local music analysis; section labels and descriptors are suggestions."""
from lyrics_structure import format_lyrics

STYLE_MODEL = "laion/clap-htsat-unfused"
STYLE_REVISION = "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a"
STYLE_GROUPS = {
    "instruments": ["acoustic guitar", "distorted electric guitar", "piano", "orchestral strings", "synthesizers", "electronic drums", "acoustic drums", "deep bass", "brass instruments"],
    "vocals": ["rap vocals", "melodic singing", "a choir", "spoken vocals", "instrumental music without vocals"],
    "mood": ["a warm relaxed mood", "a dark atmospheric mood", "an uplifting joyful mood", "a melancholic mood", "an aggressive energetic mood"],
    "production": ["acoustic production", "electronic production", "lo-fi production", "orchestral production"],
    "pace": ["a slow laid-back pulse", "a moderate steady groove", "a fast driving rhythm"],
}


def excerpts(audio, rate, seconds=10, count=3):
    import numpy as np
    length = min(len(audio), int(rate * seconds))
    if not length:
        return []
    starts = sorted(set(int(x) for x in np.linspace(0, len(audio) - length, count)))
    return [audio[start:start + length] for start in starts]


def select_descriptors(scores, labels, minimum=.20, margin=.025):
    """Cosine similarity is evidence, not a calibrated probability."""
    ranked = sorted(zip(labels, map(float, scores)), key=lambda item: item[1], reverse=True)
    if not ranked or ranked[0][1] < minimum or (len(ranked) > 1 and ranked[0][1] - ranked[1][1] < margin):
        return []
    return [{"label": ranked[0][0], "similarity": ranked[0][1]}]


def describe_style(audio, rate):
    import torch
    from transformers import ClapModel, ClapProcessor
    processor = ClapProcessor.from_pretrained(STYLE_MODEL, revision=STYLE_REVISION, local_files_only=True)
    model = ClapModel.from_pretrained(STYLE_MODEL, revision=STYLE_REVISION, local_files_only=True).eval()
    labels = [label for group in STYLE_GROUPS.values() for label in group]
    with torch.inference_mode():
        text = processor(text=[f"Music featuring {label}." for label in labels], return_tensors="pt", padding=True)
        text_features = model.get_text_features(**text)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)
        similarities = []
        for clip in excerpts(audio, rate):
            inputs = processor(audios=clip, sampling_rate=rate, return_tensors="pt")
            features = torch.nn.functional.normalize(model.get_audio_features(**inputs), dim=-1)
            similarities.append((features @ text_features.T)[0])
        scores = torch.stack(similarities).mean(0).tolist()
    selected, offset = [], 0
    for group in STYLE_GROUPS.values():
        selected.extend(select_descriptors(scores[offset:offset+len(group)], group))
        offset += len(group)
    return selected
