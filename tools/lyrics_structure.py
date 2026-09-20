"""Conservative, word-preserving lyric layout from timestamped ASR output.

Labels are structural suggestions, never a correction of recognized words.
"""
import math
import re
from collections import defaultdict
from difflib import SequenceMatcher

# Ignore optional interjections for matching only. They remain in the output.
_INTERJECTIONS = {"uh", "oh", "hey", "ayy", "ay", "yeah"}


def _key(word):
    return re.sub(r"[^\w]", "", word.casefold())


def _words(result):
    words = []
    chunks = result.get("chunks") or [{"text": result.get("text", "")}]
    for chunk in chunks:
        tokens = str(chunk.get("text") or "").split()
        times = chunk.get("timestamp") or (None, None)
        start, end = times
        for index, token in enumerate(tokens):
            # Only word-level timestamps are used for pauses. Segment-level
            # chunks provide boundaries, not fabricated word alignment.
            words.append({"text": token, "start": start if index == 0 else None,
                          "end": end if index == len(tokens) - 1 else None,
                          "boundary": index == len(tokens) - 1 and len(tokens) > 1})
    return words


def _passages(keys):
    """Find recurring passages, tolerating small substitutions and ad-libs.

    Four-token anchors bound the search. Two-line hooks need at least eight
    meaningful words; up to 100 words per occurrence are aligned.
    """
    anchors = defaultdict(list)
    for i in range(len(keys) - 3):
        anchors[tuple(keys[i:i + 4])].append(i)
    groups = []
    for starts in anchors.values():
        if len(starts) < 2 or len(starts) > 32:
            continue
        matches = []
        for a_index, a in enumerate(starts[:-1]):
            for b in starts[a_index + 1:]:
                if b - a < 8:
                    continue
                length = min(100, b - a)
                left, right = keys[a:a + length], keys[b:b + length]
                blocks = SequenceMatcher(None, left, right, autojunk=False).get_matching_blocks()
                end_a = end_b = matched = 0
                for block in blocks:
                    if block.size == 0:
                        break
                    gap_a, gap_b = block.a - end_a, block.b - end_b
                    if gap_a > 5 or gap_b > 5 or abs(gap_a - gap_b) > 3:
                        break
                    end_a, end_b = block.a + block.size, block.b + block.size
                    matched += block.size
                if min(end_a, end_b) >= 8 and matched / max(end_a, end_b) >= .78:
                    matches.append((a, a + end_a, b, b + end_b))
        # Compare recurring cores separately from longer passages that only
        # happen to repeat twice (e.g. chorus + verse + pre-chorus).
        for length in sorted({min(a_end - a, b_end - b) for a, a_end, b, b_end in matches}):
            spans = {}
            for a, a_end, b, b_end in matches:
                if min(a_end - a, b_end - b) < length:
                    continue
                spans[a] = min(spans.get(a, a_end), a_end)
                spans[b] = min(spans.get(b, b_end), b_end)
            nonoverlap = []
            for start, end in sorted(spans.items()):
                if not nonoverlap or start >= nonoverlap[-1][1]:
                    nonoverlap.append((start, end))
            if len(nonoverlap) >= 2:
                coverage = sum(end - start for start, end in nonoverlap)
                groups.append((coverage * math.log2(1 + len(nonoverlap)), nonoverlap))
    return sorted(groups, key=lambda item: item[0], reverse=True)


def _lines(words):
    lines, current = [], []
    previous_end = None
    for word in words:
        pause = (word["start"] - previous_end if word["start"] is not None and previous_end is not None else 0)
        if current and (pause >= .65 or len(current) >= 12):
            lines.append(" ".join(current)); current = []
        current.append(word["text"])
        punctuation = word["text"].endswith((".", "!", "?", ";", ":"))
        clause = word["text"].endswith(",") and len(current) >= 5
        if punctuation or clause or word["boundary"]:
            lines.append(" ".join(current)); current = []
        previous_end = word["end"]
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


def format_lyrics(result):
    words = _words(result)
    if not words:
        return ""
    indexed = [(i, _key(word["text"])) for i, word in enumerate(words)]
    indexed = [(i, key) for i, key in indexed if key and key not in _INTERJECTIONS]
    keys = [key for _, key in indexed]
    # Keep layout bounded for unusually long files, without truncating text.
    groups = _passages(keys) if 8 <= len(keys) <= 3000 else []
    if not groups:
        return "[verse]\n" + _lines(words)
    spans = groups[0][1]
    chorus = [(indexed[start][0], indexed[end - 1][0] + 1) for start, end in spans]
    # Keep punctuation/ad-libs belonging to a refrain inside that refrain.
    for i, (start, end) in enumerate(chorus):
        floor = chorus[i - 1][1] if i else 0
        while start > floor and _key(words[start - 1]["text"]) in _INTERJECTIONS:
            start -= 1
        ceiling = chorus[i + 1][0] if i + 1 < len(chorus) else len(words)
        while end < ceiling and _key(words[end]["text"]) in _INTERJECTIONS:
            end += 1
        chorus[i] = (start, end)

    # Repeated passages immediately preceding two or more choruses are
    # possible pre-choruses. Do not label arbitrary unique lines as such.
    pre = {}
    for _, candidates in groups[1:]:
        matched = []
        for start, end in candidates:
            a, b = indexed[start][0], indexed[end - 1][0] + 1
            for c, _ in chorus:
                if a < c and b >= c - 4 and 8 <= c - a <= 70 and not any(a < tail and head < c for head, tail in chorus):
                    matched.append((c, a))
        if len({c for c, _ in matched}) >= 2:
            for c, a in matched:
                # Never extend backward through an earlier chorus.
                previous = max((end for _, end in chorus if end <= c), default=0)
                if a >= previous:
                    pre[c] = min(pre.get(c, a), a)

    sections, cursor = [], 0
    for index, (start, end) in enumerate(chorus):
        before = pre.get(start, start)
        if before > cursor:
            label = "verse"
            if index == 0 and before <= 30:
                label = "intro"
            elif len(chorus) >= 3 and index == len(chorus) - 1 and start not in pre and before - cursor <= 70:
                label = "bridge"
            sections.append((label, cursor, before))
        if before < start:
            sections.append(("pre-chorus", before, start))
        sections.append(("chorus", start, end))
        cursor = end
    if cursor < len(words):
        sections.append(("outro" if len(words) - cursor <= 25 else "verse", cursor, len(words)))
    return "\n\n".join(f"[{label}]\n" + _lines(words[start:end]) for label, start, end in sections)
