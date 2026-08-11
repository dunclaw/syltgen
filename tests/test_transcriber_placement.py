"""Unit tests for lyric-line placement helpers in :mod:`syltgen.transcriber`."""

from __future__ import annotations

import pytest

from syltgen.transcriber import (
    _align_lines_by_global_token_match,
    _alignment_failure_signals,
    _apply_intro_onset_floor,
    _choose_better_placement,
    _looks_like_failed_alignment,
    _placement_badness,
    _split_long_segments,
)


def _words(spec: list[tuple[str, float]], *, duration: float = 0.4) -> list[dict]:
    return [
        {"token": token, "start": start, "end": start + duration} for token, start in spec
    ]


TRANSCRIPT = _words(
    [
        ("hello", 10.0),
        ("darkness", 10.5),
        ("my", 11.0),
        ("old", 11.5),
        ("friend", 12.0),
        ("i've", 14.0),
        ("come", 14.5),
        ("to", 15.0),
        ("talk", 15.5),
        ("with", 16.0),
        ("you", 16.5),
        ("again", 17.0),
        ("because", 20.0),
        ("a", 20.5),
        ("vision", 21.0),
        ("softly", 21.5),
        ("creeping", 22.0),
    ]
)

LINES = [
    "Hello darkness, my old friend",
    "I've come to talk with you again",
    "Because a vision softly creeping",
]


def test_global_match_places_every_line_on_its_own_words():
    result = _align_lines_by_global_token_match(LINES, TRANSCRIPT)
    assert [round(seg["start"], 2) for seg in result] == [10.0, 14.0, 20.0]
    assert [seg["text"] for seg in result] == LINES


def test_global_match_skips_audio_the_lyric_sheet_does_not_cover():
    """An unlisted ad-lib between verses must not drag later lines earlier."""
    transcript = TRANSCRIPT[:5] + _words(
        [("yeah", 30.0), ("yeah", 30.5), ("oh", 31.0), ("baby", 31.5)]
    ) + [
        {"token": w["token"], "start": w["start"] + 30.0, "end": w["end"] + 30.0}
        for w in TRANSCRIPT[5:]
    ]
    result = _align_lines_by_global_token_match(LINES, transcript)
    assert result[0]["start"] == pytest.approx(10.0)
    assert result[1]["start"] == pytest.approx(44.0)
    assert result[2]["start"] == pytest.approx(50.0)


def test_global_match_interpolates_a_line_the_transcriber_missed():
    lines = [LINES[0], "A verse whisper never picked up", LINES[2]]
    result = _align_lines_by_global_token_match(lines, TRANSCRIPT)
    assert result[0]["start"] == pytest.approx(10.0)
    assert result[2]["start"] == pytest.approx(20.0)
    assert 10.0 < result[1]["start"] < 20.0


def test_global_match_uses_audio_onset_for_leading_unmatched_lines():
    lines = ["An intro line nobody transcribed"] + LINES
    result = _align_lines_by_global_token_match(lines, TRANSCRIPT, audio_onset=6.0)
    assert 6.0 <= result[0]["start"] < 10.0


def test_global_match_output_is_monotonic():
    lines = LINES + ["Another line not in the transcript", LINES[0]]
    result = _align_lines_by_global_token_match(lines, TRANSCRIPT)
    starts = [seg["start"] for seg in result]
    assert starts == sorted(starts)


def test_global_match_returns_empty_without_input():
    assert _align_lines_by_global_token_match([], TRANSCRIPT) == []
    assert _align_lines_by_global_token_match(LINES, []) == []


def test_global_match_spreads_lines_when_nothing_matches():
    result = _align_lines_by_global_token_match(
        ["zzz qqq", "wwww vvvv", "xxxx yyyy"], TRANSCRIPT
    )
    starts = [seg["start"] for seg in result]
    assert starts == sorted(starts)
    assert starts[0] >= TRANSCRIPT[0]["start"]
    assert starts[-1] <= TRANSCRIPT[-1]["end"]


def test_intro_onset_floor_clamps_early_first_line():
    segments = [
        {"text": "a", "start": 2.0, "end": 3.0},
        {"text": "b", "start": 8.0, "end": 9.0},
    ]
    result = _apply_intro_onset_floor(segments, 5.0)
    assert result[0]["start"] == pytest.approx(5.0)
    assert result[1]["start"] == pytest.approx(8.0)


class _FakeWord:
    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end


class _FakeSegment:
    def __init__(self, words: list[_FakeWord]) -> None:
        self.words = words


class _FakeResult:
    def __init__(self, segments: list[_FakeSegment]) -> None:
        self.segments = segments


def _healthy_result(count: int = 20) -> _FakeResult:
    return _FakeResult(
        [_FakeSegment([_FakeWord(i * 3.0, i * 3.0 + 1.5)]) for i in range(count)]
    )


def _healthy_segments(count: int = 20) -> list[dict]:
    return [
        {"text": f"line {i}", "start": i * 3.0, "end": i * 3.0 + 2.5}
        for i in range(count)
    ]


def _silence(seconds: float):
    import numpy as np

    return np.zeros(int(seconds * 16000), dtype="float32")


def test_healthy_alignment_is_not_flagged():
    signals = _alignment_failure_signals(
        _healthy_result(), _healthy_segments(), _silence(65.0)
    )
    assert not _looks_like_failed_alignment(signals)


def test_unplaced_words_flag_failure():
    result = _FakeResult(
        [_FakeSegment([_FakeWord(1.0, 1.0)]) for _ in range(5)]
        + [_FakeSegment([_FakeWord(i * 3.0, i * 3.0 + 1.0)]) for i in range(15)]
    )
    signals = _alignment_failure_signals(result, _healthy_segments(), _silence(65.0))
    assert signals["unplaced_word_ratio"] == pytest.approx(0.25)
    assert _looks_like_failed_alignment(signals)


def test_collapsed_lines_flag_failure():
    segments = _healthy_segments(10) + [
        {"text": "crammed", "start": 30.0, "end": 30.4} for _ in range(10)
    ]
    signals = _alignment_failure_signals(_healthy_result(), segments, _silence(65.0))
    assert signals["collapsed_line_ratio"] > 0.4
    assert _looks_like_failed_alignment(signals)


def test_long_instrumental_outro_alone_is_not_a_failure():
    """A big tail with clean word timing is just an outro, not a broken align."""
    signals = _alignment_failure_signals(
        _healthy_result(), _healthy_segments(), _silence(200.0)
    )
    assert signals["tail_fraction"] > 0.5
    assert not _looks_like_failed_alignment(signals)


def test_tail_plus_mild_collapse_flags_failure():
    segments = _healthy_segments(19) + [{"text": "x", "start": 54.02, "end": 54.5}]
    signals = _alignment_failure_signals(_healthy_result(), segments, _silence(200.0))
    assert 0.0 < signals["collapsed_line_ratio"] < 0.10
    assert signals["tail_fraction"] > 0.25
    assert _looks_like_failed_alignment(signals)


# --- placement badness / candidate selection ---------------------------------


def _segs(starts: list[float], *, dur: float = 2.0) -> list[dict]:
    return [
        {"text": f"line {i}", "start": s, "end": s + dur} for i, s in enumerate(starts)
    ]


def test_badness_prefers_spread_placement_over_collapsed_one():
    good = _placement_badness(_segs([10.0, 20.0, 30.0, 40.0]), 50.0)
    collapsed = _placement_badness(_segs([10.0, 10.0, 10.01, 10.02]), 50.0)
    assert good < collapsed


def test_badness_penalises_backwards_lines():
    forward = _placement_badness(_segs([10.0, 20.0, 30.0]), 40.0)
    backwards = _placement_badness(_segs([10.0, 30.0, 20.0]), 40.0)
    assert backwards > forward


def test_badness_penalises_timing_that_stops_far_before_the_end():
    covering = _placement_badness(_segs([10.0, 40.0, 80.0]), 100.0)
    early = _placement_badness(_segs([1.0, 2.0, 3.0]), 100.0)
    assert early > covering


def test_badness_of_empty_placement_is_infinite():
    assert _placement_badness([], 100.0) == float("inf")


def test_empty_fallback_keeps_alignment():
    aligned = _segs([10.0, 20.0])
    assert (
        _choose_better_placement(aligned, {}, [], {}, audio_duration=60.0) is aligned
    )


def test_poorly_anchored_fallback_is_rejected():
    aligned = _segs([1.0, 1.01, 1.02])  # obviously collapsed
    fallback = _segs([10.0, 30.0, 50.0])
    kept = _choose_better_placement(
        aligned,
        {"collapsed_line_ratio": 1.0},
        fallback,
        {"anchored_ratio": 0.2, "approximate": 0.0},
        audio_duration=60.0,
    )
    assert kept is aligned


def test_approximate_fallback_is_rejected_even_when_fully_anchored():
    aligned = _segs([1.0, 1.01, 1.02])
    fallback = _segs([10.0, 30.0, 50.0])
    kept = _choose_better_placement(
        aligned,
        {"collapsed_line_ratio": 1.0},
        fallback,
        {"anchored_ratio": 1.0, "approximate": 1.0},
        audio_duration=60.0,
    )
    assert kept is aligned


def test_healthy_fallback_replaces_collapsed_alignment():
    aligned = _segs([1.0, 1.01, 1.02])
    fallback = _segs([10.0, 30.0, 50.0])
    kept = _choose_better_placement(
        aligned,
        {"collapsed_line_ratio": 1.0},
        fallback,
        {"anchored_ratio": 0.9, "approximate": 0.0},
        audio_duration=60.0,
    )
    assert kept is fallback


def test_global_match_reports_anchored_ratio():
    stats: dict = {}
    _align_lines_by_global_token_match(
        ["hello darkness my old friend", "completely unrelated words here"],
        TRANSCRIPT,
        stats=stats,
    )
    assert stats["anchored_ratio"] == pytest.approx(0.5)


# --- line splitting -------------------------------------------------------


def _segment(text: str, *, word_dur: float = 0.32, gap: float = 0.05,
             pauses: dict[int, float] | None = None) -> dict:
    """Build a dense segment with uniform word timings plus explicit pauses.

    ``pauses`` maps a word index to the silence that follows it, which is how
    the real transcriber sees a breath or a held note mid-phrase.
    """
    tokens = text.split()
    pauses = pauses or {}
    words = []
    t = 0.0
    for idx, token in enumerate(tokens):
        words.append({"word": token, "start": t, "end": t + word_dur})
        t += word_dur + pauses.get(idx, gap)
    return {
        "text": text,
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "words": words,
    }


def _pause_before(text: str, word: str, seconds: float = 0.5) -> dict[int, float]:
    """Silence immediately before ``word`` -- i.e. after the token preceding it."""
    tokens = [t.strip(".,;:!?").lower() for t in text.split()]
    return {tokens.index(word.lower()) - 1: seconds}


def _split_texts(text: str, **kwargs) -> list[str]:
    return [s["text"] for s in _split_long_segments([_segment(text, **kwargs)])]


def _break_pairs(lines: list[str]) -> set[tuple[str, str]]:
    """(last word of a line, first word of the next) for every break."""
    pairs = set()
    for prev, nxt in zip(lines, lines[1:]):
        prev_words = prev.split()
        next_words = nxt.split()
        if prev_words and next_words:
            pairs.add((prev_words[-1].strip(".,;:!?").lower(), next_words[0].strip(".,;:!?").lower()))
    return pairs


def test_does_not_break_after_a_copula():
    text = (
        "Not a cloud in the sky but the feeling is lightning "
        "Didn't know heaven was a place just like this"
    )
    lines = _split_texts(text, pauses=_pause_before(text, "lightning"))
    assert ("is", "lightning") not in _break_pairs(lines)


def test_does_not_break_after_a_possessive_determiner():
    text = (
        "It's all right here in front of my eyes "
        "I think I found my happy place and I am never leaving"
    )
    lines = _split_texts(text, pauses=_pause_before(text, "eyes"))
    assert ("my", "eyes") not in _break_pairs(lines)


def test_does_not_break_before_the_pronoun_i():
    text = (
        "And oh, there was a time when you and I were standing "
        "on the edge of a mountain, looking down at everything below"
    )
    lines = _split_texts(text, pauses=_pause_before(text, "I", 0.45))
    assert ("and", "i") not in _break_pairs(lines)


def test_unpunctuated_dense_text_is_not_chunked_at_fixed_width():
    text = (
        "Not a cloud in the sky but the feeling is lightning "
        "Didn't know heaven was a place just like this "
        "Every colour brighter than the one before it "
        "Holding on to something that I never want to miss"
    )
    counts = [len(line.split()) for line in _split_texts(text)]
    assert len(set(counts)) > 1, f"degenerate fixed-width split: {counts}"


def test_split_preserves_every_word_in_order():
    text = (
        "Not a cloud in the sky but the feeling is lightning "
        "Didn't know heaven was a place just like this"
    )
    joined = " ".join(_split_texts(text))
    assert joined.split() == text.split()


def test_short_segments_are_left_alone():
    seg = _segment("Just a short line here")
    assert [s["text"] for s in _split_long_segments([seg])] == ["Just a short line here"]




def _after(previous: dict, text: str, *, gap: float = 0.4, **kwargs) -> dict:
    """A segment starting ``gap`` seconds after ``previous`` ends."""
    seg = _segment(text, **kwargs)
    shift = float(previous["end"]) + gap - float(seg["start"])
    for word in seg["words"]:
        word["start"] += shift
        word["end"] += shift
    seg["start"] += shift
    seg["end"] += shift
    return seg


def test_repairs_a_dangling_break_between_whisper_segments():
    """Whisper's own boundaries are the main source of unnatural breaks."""
    first = _segment("Not a cloud in the sky but the feeling is")
    second = _after(first, "lightning didn't know heaven was a place")

    lines = [s["text"] for s in _split_long_segments([first, second])]
    assert ("is", "lightning") not in _break_pairs(lines)
    assert " ".join(lines).split() == (first["text"] + " " + second["text"]).split()


def test_does_not_merge_across_a_long_instrumental_gap():
    first = _segment("Every night I dream about the")
    second = _after(first, "morning light that never comes to me", gap=30.0)

    lines = _split_long_segments([first, second])
    assert len(lines) == 2


def test_repair_leaves_well_formed_segments_alone():
    first = _segment("Hello darkness my old friend")
    second = _after(first, "I've come to talk with you again")

    lines = [s["text"] for s in _split_long_segments([first, second])]
    assert lines == [first["text"], second["text"]]
