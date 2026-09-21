from yue2.hum import open_score, promote_hum_to_vocal, hum_opening
from yue2.protocol import SongRequest, token_prefixes, ABC_START, ABC_END, MUSIC_START, EOD
from yue2.pipeline import complete_score_ids


class Tokenizer:
    def encode(self, text):
        return [ord(c) % 1000 for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


HEADER = "X:1\nT:\nM:4/4\nL:1/32\nQ:1/4=86\nV: Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\"\nV: Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\"\nK:G\n"


def test_open_score_strips_trailing_rests_and_bar_line():
    abc = HEADER + "V: Vocal\nz8B2d6d4B2d6B2d2-|d6z24z2|z32|\nV: Ins\nZ3|\n"
    out = open_score(abc)
    assert out.endswith("V: Vocal\nz8B2d6d4B2d6B2d2-|d6|")
    assert "V: Ins" not in out.splitlines()[-2:]


def test_open_score_keeps_a_score_that_ends_on_a_note():
    abc = HEADER + "V: Vocal\nB4c4d4e4|]"
    assert open_score(abc).endswith("V: Vocal\nB4c4d4e4|")


def test_promote_swaps_when_the_hum_landed_in_ins():
    abc = HEADER + "V: Vocal\nz32|z32|\nV: Ins\nB4c4d4e4|f8g8|\n"
    out = promote_hum_to_vocal(abc)
    body = out.split("K:G\n", 1)[1]
    assert body.startswith("V: Ins\nz32|z32|\nV: Vocal\nB4c4d4e4|")
    assert "V: Vocal clef=treble" in out          # header definitions untouched
    assert promote_hum_to_vocal(HEADER + "V: Vocal\nB4|\n") == HEADER + "V: Vocal\nB4|\n"


def test_open_request_prompts_and_completes_score_ids():
    t = Tokenizer()
    opening = hum_opening(HEADER + "V: Vocal\nB4c4|z32|\n")
    r = SongRequest("style", "lyrics", cot="melody", abc=opening, abc_open=True)
    planner = token_prefixes(r, t)
    assert planner[:1] == [EOD] and planner[-len(t.encode(opening)) - 1] == ABC_START and ABC_END not in planner
    full = complete_score_ids(r, t, [7, 8, 9])
    assert full == t.encode(opening) + [7, 8, 9]
    assert token_prefixes(r, t, full)[-3:] == [9, ABC_END, MUSIC_START]
    closed = SongRequest("style", "lyrics", cot="melody", abc=opening)
    assert token_prefixes(closed, t)[-2:] == [ABC_END, MUSIC_START]


def test_abc_open_requires_abc():
    import pytest
    with pytest.raises(ValueError):
        SongRequest("s", "l", cot="melody", abc_open=True)
