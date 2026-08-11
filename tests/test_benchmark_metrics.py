"""Unit tests for benchmark scoring (no audio, models, or GPU required)."""

from __future__ import annotations

import math

import pytest

from tests.benchmark.metrics import (
    normalize_tokens,
    score_file,
    score_lines,
    score_tokens,
)


def _seg(text: str, start: float, end: float | None = None) -> dict:
    return {"text": text, "start": start, "end": start + 2.0 if end is None else end}


REFERENCE = [
    _seg("Hello darkness my old friend", 10.0),
    _seg("I've come to talk with you again", 14.0),
    _seg("Because a vision softly creeping", 18.0),
]


def test_normalize_tokens_strips_punctuation_and_case():
    assert normalize_tokens("Hello, Darkness — my OLD friend!") == [
        "hello",
        "darkness",
        "my",
        "old",
        "friend",
    ]


def test_normalize_tokens_keeps_curly_apostrophes():
    assert normalize_tokens("I\u2019ve come") == ["i've", "come"]


def test_perfect_alignment_scores_zero():
    score = score_lines(REFERENCE, REFERENCE)
    assert score.median_abs_error == 0.0
    assert score.mean_abs_error == 0.0
    assert score.first_line_error == 0.0
    assert score.gross_error_rate == 0.0
    assert score.within["0.25s"] == 1.0


def test_constant_offset_is_reported_as_offset_not_jitter():
    shifted = [_seg(s["text"], s["start"] + 1.5) for s in REFERENCE]
    score = score_lines(REFERENCE, shifted)
    assert score.median_signed_error == pytest.approx(1.5)
    assert score.jitter == pytest.approx(0.0)
    assert score.median_abs_error == pytest.approx(1.5)


def test_scatter_is_reported_as_jitter():
    scattered = [
        _seg(REFERENCE[0]["text"], 8.0),
        _seg(REFERENCE[1]["text"], 14.0),
        _seg(REFERENCE[2]["text"], 20.0),
    ]
    score = score_lines(REFERENCE, scattered)
    assert score.median_signed_error == pytest.approx(0.0)
    assert score.jitter == pytest.approx(2.0)


def test_gross_error_rate_counts_badly_placed_lines():
    broken = [
        _seg(REFERENCE[0]["text"], 10.0),
        _seg(REFERENCE[1]["text"], 14.2),
        _seg(REFERENCE[2]["text"], 90.0),
    ]
    score = score_lines(REFERENCE, broken)
    assert score.gross_error_rate == pytest.approx(1 / 3)
    assert score.max_abs_error == pytest.approx(72.0)


def test_monotonic_violations_are_counted():
    out_of_order = [
        _seg(REFERENCE[0]["text"], 10.0),
        _seg(REFERENCE[1]["text"], 9.0),
        _seg(REFERENCE[2]["text"], 18.0),
    ]
    assert score_lines(REFERENCE, out_of_order).monotonic_violations == 1


def test_line_count_mismatch_raises():
    with pytest.raises(ValueError):
        score_lines(REFERENCE, REFERENCE[:2])


def test_token_score_survives_resegmentation():
    """Merging two reference lines must still score as accurate."""
    merged = [
        _seg("Hello darkness my old friend I've come to talk with you again", 10.0, 18.0),
        _seg("Because a vision softly creeping", 18.0, 22.0),
    ]
    score = score_tokens(REFERENCE, merged)
    assert score.coverage == pytest.approx(1.0)
    # Interpolation across the merged line keeps every token within a second.
    assert score.median_abs_error < 1.0
    assert score.within["2s"] == 1.0


def test_token_score_flags_wholesale_displacement():
    displaced = [_seg(s["text"], s["start"] + 30.0) for s in REFERENCE]
    score = score_tokens(REFERENCE, displaced)
    # Reference spans are inferred from the next onset while hypothesis spans use
    # their own end times, so intra-line interpolation differs by up to ~1 s.
    assert score.median_abs_error == pytest.approx(30.0, abs=1.5)
    assert score.gross_error_rate == 1.0


def test_token_score_reports_partial_coverage():
    partial = [_seg(REFERENCE[0]["text"], 10.0)]
    score = score_tokens(REFERENCE, partial)
    assert 0.0 < score.coverage < 0.5


def test_token_score_handles_empty_hypothesis():
    score = score_tokens(REFERENCE, [])
    assert score.n_matched_tokens == 0
    assert score.coverage == 0.0
    assert math.isnan(score.median_abs_error)


def test_score_file_returns_both_scorers():
    result = score_file(REFERENCE, REFERENCE)
    assert result["lines"]["median_abs_error"] == 0.0
    assert result["tokens"]["coverage"] == pytest.approx(1.0)


def test_score_file_omits_line_score_on_mismatch():
    result = score_file(REFERENCE, REFERENCE[:2])
    assert result["lines"] is None
    assert result["tokens"]["n_matched_tokens"] > 0
