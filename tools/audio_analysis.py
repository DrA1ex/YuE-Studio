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


def clean_genre_label(label):
    text = str(label or "").replace("_", " ").replace("-", " ").strip()
    words = " ".join(part.capitalize() for part in text.split())
    return words.replace("Hiphop", "Hip-Hop").replace("Hip Hop", "Hip-Hop")


def summarize_genre(candidates, excerpt_candidates=None):
    """Return a cautious label and preserve the uncalibrated evidence.

    GTZAN-style classifier scores are useful for ranking, but they are not
    probabilities. A close second label is therefore exposed as a hybrid
    suggestion instead of silently discarded.
    """
    ranked = sorted(
        ({"label": clean_genre_label(item.get("label")), "score": float(item.get("score", 0.0))}
         for item in candidates if item.get("label")),
        key=lambda item: item["score"], reverse=True,
    )
    if not ranked:
        return "Mixed / uncertain"
    top = ranked[0]
    if top["score"] < 0.28:
        return "Mixed / uncertain"
    if len(ranked) > 1:
        second = ranked[1]
        if second["score"] >= 0.18 and top["score"] - second["score"] < 0.12:
            return f"{top['label']} / {second['label']}"
    if excerpt_candidates:
        excerpt_winners = [clean_genre_label(items[0]["label"]) for items in excerpt_candidates if items]
        if excerpt_winners and len(set(excerpt_winners)) > 1:
            return f"{top['label']} (mixed across sections)"
    return top["label"]


def excerpts(audio, rate, seconds=10, count=3):
    import numpy as np
    length = min(len(audio), int(rate * seconds))
    if not length:
        return []
    starts = sorted(set(int(x) for x in np.linspace(0, len(audio) - length, count)))
    return [audio[start:start + length] for start in starts]


def select_descriptors(scores, labels, minimum=.28, margin=.025, support=None, max_items=2):
    """Cosine similarity is evidence, not a calibrated probability."""
    ranked = sorted(zip(labels, map(float, scores)), key=lambda item: item[1], reverse=True)
    if not ranked or ranked[0][1] < minimum:
        return []
    if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < margin:
        return []
    selected = []
    for label, score in ranked:
        index = list(labels).index(label)
        item_support = float(support[index]) if support is not None else 1.0
        if score < minimum or item_support < .5:
            continue
        if selected and score < ranked[0][1] - .12:
            continue
        selected.append({"label": label, "similarity": score, "support": item_support})
        if len(selected) >= max_items:
            break
    return selected


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
        scores = torch.stack(similarities).mean(0).squeeze(0).tolist()
    selected, offset = [], 0
    per_clip = torch.stack(similarities).squeeze(1).tolist()
    for group in STYLE_GROUPS.values():
        group_scores = scores[offset:offset+len(group)]
        support = [sum(row[offset + index] >= .28 for row in per_clip) / len(per_clip) for index in range(len(group))]
        selected.extend(select_descriptors(group_scores, group, support=support))
        offset += len(group)
    return selected
