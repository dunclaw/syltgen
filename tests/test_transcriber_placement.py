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
