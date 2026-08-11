"""Scoring functions for lyric placement accuracy.

Two complementary scorers:

``score_lines``
    Line-level 1-to-1 comparison.  Valid when the hypothesis has exactly one
    line per reference line, which is what the USLT forced-alignment path
    produces for strict ground-truth files.  This is the metric that most
    directly reflects "is the lyric shown at the right moment".

``score_tokens``
    Structure-independent comparison via word-stream alignment.  Works when the
    hypothesis re-segments the lyrics (the full-transcription path), and also
    detects when a nominally 1-to-1 result has drifted onto the wrong words.

Both are pure functions over plain dicts so they can be unit tested without
audio, models, or GPU.
"""

from __future__ import annotations

import difflib
import re
import statistics
from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence

_TOKEN_RE = re.compile(r"[a-z0-9']+")

#: Tolerance buckets (seconds) reported as "fraction of lines within X".
TOLERANCES: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 5.0)

#: A line off by more than this is not "slightly late", it is in the wrong place.
GROSS_ERROR_SECONDS = 3.0

#: Assumed duration of the final reference line, which has no successor to
#: bound it.
_TRAILING_LINE_SECONDS = 4.0


def normalize_tokens(text: str) -> list[str]:
    """Lower-case word tokens used for text matching."""
    return _TOKEN_RE.findall(text.lower().replace("\u2019", "'"))


@dataclass
class LineScore:
    """Line-level 1-to-1 accuracy result."""

    n_lines: int = 0
    median_abs_error: float = float("nan")
    mean_abs_error: float = float("nan")
    p90_abs_error: float = float("nan")
    max_abs_error: float = float("nan")
    median_signed_error: float = float("nan")
    jitter: float = float("nan")
    first_line_error: float = float("nan")
    gross_error_rate: float = float("nan")
    monotonic_violations: int = 0
    within: dict[str, float] = field(default_factory=dict)


@dataclass
class TokenScore:
    """Structure-independent accuracy result based on word-stream matching."""

    n_ref_tokens: int = 0
    n_matched_tokens: int = 0
    coverage: float = 0.0
    median_abs_error: float = float("nan")
    mean_abs_error: float = float("nan")
    p90_abs_error: float = float("nan")
    median_signed_error: float = float("nan")
    gross_error_rate: float = float("nan")
    within: dict[str, float] = field(default_factory=dict)


def _percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile; ``values`` need not be sorted."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100.0 * len(ordered) + 0.5) - 1))
    return ordered[index]


def _within_buckets(abs_errors: Sequence[float]) -> dict[str, float]:
    if not abs_errors:
        return {f"{t:g}s": float("nan") for t in TOLERANCES}
    total = len(abs_errors)
    return {
        f"{t:g}s": sum(1 for e in abs_errors if e <= t) / total for t in TOLERANCES
    }


def score_lines(reference: Sequence[dict], hypothesis: Sequence[dict]) -> LineScore:
    """Score a hypothesis that has one line per reference line.

    Raises ``ValueError`` when the line counts differ — callers should fall
    back to :func:`score_tokens` in that case.
    """
    if len(reference) != len(hypothesis):
        raise ValueError(
            f"line count mismatch: {len(reference)} reference vs {len(hypothesis)} hypothesis"
        )
    if not reference:
        return LineScore()

    signed = [
        float(hyp["start"]) - float(ref["start"])
        for ref, hyp in zip(reference, hypothesis)
    ]
    abs_errors = [abs(e) for e in signed]
    median_signed = statistics.median(signed)

    starts = [float(h["start"]) for h in hypothesis]
    violations = sum(1 for a, b in zip(starts, starts[1:]) if b < a - 1e-6)

    return LineScore(
        n_lines=len(reference),
        median_abs_error=statistics.median(abs_errors),
        mean_abs_error=statistics.fmean(abs_errors),
        p90_abs_error=_percentile(abs_errors, 90),
        max_abs_error=max(abs_errors),
        median_signed_error=median_signed,
        jitter=statistics.median([abs(e - median_signed) for e in signed]),
        first_line_error=signed[0],
        gross_error_rate=sum(1 for e in abs_errors if e > GROSS_ERROR_SECONDS)
        / len(abs_errors),
        monotonic_violations=violations,
        within=_within_buckets(abs_errors),
    )


def _timed_token_stream(
    segments: Sequence[dict], *, use_end: bool
) -> tuple[list[str], list[float]]:
    """Flatten segments into a token stream with an interpolated time each.

    ``use_end`` selects how each line's span is determined:

    * ``True``  – the segment's own ``end`` field (hypothesis segments carry
      real end times from the aligner).
    * ``False`` – the next segment's ``start`` (reference SYLT frames only
      store onsets).
    """
    tokens: list[str] = []
    times: list[float] = []

    spans: list[tuple[float, float]] = []
    for index, seg in enumerate(segments):
        start = float(seg["start"])
        if use_end and seg.get("end") is not None:
            end = float(seg["end"])
        elif index + 1 < len(segments):
            end = float(segments[index + 1]["start"])
        else:
            end = start + _TRAILING_LINE_SECONDS
        spans.append((start, max(start, end)))

    for seg, (start, end) in zip(segments, spans):
        line_tokens = normalize_tokens(str(seg.get("text", "")))
        if not line_tokens:
            continue
        span = max(0.0, end - start)
        step = span / len(line_tokens)
        for offset, token in enumerate(line_tokens):
            tokens.append(token)
            times.append(start + offset * step)

    return tokens, times


def score_tokens(reference: Sequence[dict], hypothesis: Sequence[dict]) -> TokenScore:
    """Score without assuming any correspondence between line structures."""
    ref_tokens, ref_times = _timed_token_stream(reference, use_end=False)
    hyp_tokens, hyp_times = _timed_token_stream(hypothesis, use_end=True)

    if not ref_tokens:
        return TokenScore()
    if not hyp_tokens:
        return TokenScore(n_ref_tokens=len(ref_tokens))

    matcher = difflib.SequenceMatcher(None, ref_tokens, hyp_tokens, autojunk=False)
    signed: list[float] = []
    for ref_i, hyp_i, size in matcher.get_matching_blocks():
        for offset in range(size):
            signed.append(hyp_times[hyp_i + offset] - ref_times[ref_i + offset])

    if not signed:
        return TokenScore(n_ref_tokens=len(ref_tokens))

    abs_errors = [abs(e) for e in signed]
    return TokenScore(
        n_ref_tokens=len(ref_tokens),
        n_matched_tokens=len(signed),
        coverage=len(signed) / len(ref_tokens),
        median_abs_error=statistics.median(abs_errors),
        mean_abs_error=statistics.fmean(abs_errors),
        p90_abs_error=_percentile(abs_errors, 90),
        median_signed_error=statistics.median(signed),
        gross_error_rate=sum(1 for e in abs_errors if e > GROSS_ERROR_SECONDS)
        / len(abs_errors),
        within=_within_buckets(abs_errors),
    )


def score_file(
    reference: Sequence[dict], hypothesis: Sequence[dict]
) -> dict[str, Optional[dict]]:
    """Score one file with both scorers, tolerating structure mismatch."""
    line_score: Optional[LineScore]
    try:
        line_score = score_lines(reference, hypothesis)
    except ValueError:
        line_score = None

    return {
        "lines": asdict(line_score) if line_score is not None else None,
        "tokens": asdict(score_tokens(reference, hypothesis)),
    }
