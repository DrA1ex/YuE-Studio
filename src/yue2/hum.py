"""Turn a transcribed hum (SheetSage2, melody-vocal) into an opening the planner continues.

The score keeps YuE2's two-voice format (``V: Vocal`` / ``V: Ins`` bodies under the header).
``promote_hum_to_vocal`` makes sure the hummed line is the Vocal voice; ``open_score`` drops
the end of the score (trailing rests, the final bar line) so the last thing the planner sees is
the last sung note, and it carries on from there.
"""
from __future__ import annotations
import re

VOICE = re.compile(r"^V:\s*(\w+)\s*$")
REST_BAR = re.compile(r"^(?:\"[^\"]*\")?(?:z\d*|Z\d*)+$")     # a bar of rests, optionally with a chord symbol


def _bars(line):
    return [b.strip() for b in line.strip().rstrip("|]").split("|")]


def _rest_only(line):
    bars = [b for b in _bars(line) if b]
    return bool(bars) and all(REST_BAR.fullmatch(b) for b in bars)


def promote_hum_to_vocal(abc):
    """If every Vocal body is rests and Ins carries the notes, swap the body labels."""
    lines = abc.splitlines()
    vocal_notes = ins_notes = False
    current = None
    for line in lines:
        m = VOICE.match(line)
        if m:
            current = m.group(1); continue
        if line.startswith("%") or not line.strip() or current is None:
            continue
        if not _rest_only(line):
            if current == "Vocal":
                vocal_notes = True
            elif current == "Ins":
                ins_notes = True
    if vocal_notes or not ins_notes:
        return abc
    out = []
    for line in lines:
        m = VOICE.match(line)
        if m and m.group(1) in ("Vocal", "Ins"):
            out.append("V: " + ("Ins" if m.group(1) == "Vocal" else "Vocal"))
        else:
            out.append(line)
    return "\n".join(out) + ("\n" if abc.endswith("\n") else "")


def open_score(abc):
    """Leave the score open after its last sung note: strip trailing whitespace, a final bar
    line, and rest-only bars (and rest-only Ins/Vocal lines) at the end."""
    lines = abc.rstrip().splitlines()
    while lines:
        line = lines[-1].rstrip()
        if not line or line.startswith("%"):
            lines.pop(); continue
        if VOICE.match(line):
            lines.pop(); continue
        bars = _bars(line)
        while bars and (not bars[-1] or REST_BAR.fullmatch(bars[-1])):
            bars.pop()
        if not bars:
            lines.pop(); continue
        # Trim trailing rests inside the last bar (z8 at the end of "e4G8z16z4").
        last = re.sub(r"(?:z\d*)+$", "", bars[-1])
        bars[-1] = last if last else bars[-1]
        lines[-1] = "|".join(bars) + "|"
        break
    text = "\n".join(lines)
    return text if text.endswith("|") else text + "|"


def hum_opening(abc):
    return open_score(promote_hum_to_vocal(abc))
