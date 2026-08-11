"""Line-break quality scoring for the no-USLT transcription path.

The USLT path inherits its line structure from the lyric sheet, so line breaks
are only ever chosen by ``syltgen`` when there is no sheet.  Two complementary
scorers are used here because neither is sufficient alone:

* :func:`score_break_positions` compares the chosen breaks against the
  human-authored breaks in a library file's own lyrics.  This is the closest
  thing to ground truth, but it only exists for files that already have tags,
  and the transcript never matches the sheet word-for-word.
* :func:`score_dangling_breaks` needs no reference at all: it counts breaks
  that leave a line ending on a word which cannot end a phrase ("you and" /
  "in front of my").  This is exactly the defect users notice, and it can be
  measured on any output file.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

#: Words that cannot end a natural lyric line: they syntactically require a
#: continuation, so a break straight after one always reads as a mistake.
DANGLING_END_WORDS = {
    # determiners / possessives
    "a", "an", "the", "my", "your", "our", "their", "his", "her", "its",
    "this", "that", "these", "those", "some", "any", "no", "every", "each",
    # coordinating and subordinating conjunctions
    "and", "or", "but", "nor", "so", "yet", "if", "than", "as", "because",
    "cause", "coz", "though", "although", "unless", "until", "till", "while",
    "when", "where", "whether", "since",
    # prepositions
    "to", "of", "in", "on", "at", "for", "with", "from", "by", "into", "onto",
    "over", "under", "through", "about", "above", "below", "between", "across",
    "against", "around", "before", "after", "behind", "beside", "beyond",
    "during", "inside", "outside", "toward", "towards", "upon", "within",
    "without", "like",
    # copulas and auxiliaries
    "am", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    "gonna", "wanna", "gotta",
    # relative pronouns / complementisers
    "who", "whom", "whose", "which", "what",
    # very common contractions that must attach to what follows
    "i'm", "i've", "i'll", "i'd", "you're", "you've", "you'll", "you'd",
    "we're", "we've", "we'll", "we'd", "they're", "they've", "they'll",
    "he's", "she's", "it's", "there's", "that's", "don't", "doesn't", "didn't",
    "can't", "won't", "ain't", "isn't", "aren't", "wasn't", "weren't",
}

_WORD_RE = re.compile(r"[A-Za-z']+")


def normalize_word(token: str) -> str:
    """Lowercase a token and strip punctuation, keeping internal apostrophes."""
    token = token.replace("\u2019", "'")
    return re.sub(r"^[^A-Za-z']+|[^A-Za-z']+$", "", token).lower()


def line_tokens(line: str) -> list[str]:
    """Normalized word tokens for one lyric line."""
    return [w.lower() for w in _WORD_RE.findall(line.replace("\u2019", "'"))]


@dataclass
class BreakScore:
    """Agreement between chosen line breaks and reference line breaks."""

    n_reference_breaks: int
    n_hypothesis_breaks: int
    n_matched: int
    precision: float
    recall: float
    f1: float
    #: Fraction of reference tokens that could be aligned to the transcript.
    coverage: float


@dataclass
class DanglingScore:
    """Reference-free count of breaks that leave a line syntactically open."""

    n_breaks: int
    n_dangling: int
    dangling_rate: float
    examples: list[str]


def _token_stream(lines: Sequence[str]) -> tuple[list[str], list[int]]:
    """Flatten lines to tokens plus the token index each break falls before."""
    tokens: list[str] = []
    breaks: list[int] = []
    for index, line in enumerate(lines):
        if index > 0:
            breaks.append(len(tokens))
        tokens.extend(line_tokens(line))
    return tokens, [b for b in breaks if b > 0]


def _reference_to_hypothesis_map(
    ref_tokens: Sequence[str], hyp_tokens: Sequence[str]
) -> dict[int, int]:
    """Map reference token indices onto hypothesis token indices."""
    matcher = difflib.SequenceMatcher(None, ref_tokens, hyp_tokens, autojunk=False)
    mapping: dict[int, int] = {}
    for ref_i, hyp_i, size in matcher.get_matching_blocks():
        for offset in range(size):
            mapping[ref_i + offset] = hyp_i + offset
    return mapping


def _nearest_unused_break(
    position: int, tolerance: int, all_breaks: set[int], unused: set[int]
) -> Optional[int]:
    """Closest still-unmatched hypothesis break within *tolerance* tokens."""
    for delta in range(tolerance + 1):
        for candidate in (position + delta, position - delta):
            if candidate in all_breaks and candidate in unused:
                return candidate
    return None


def score_break_positions(    reference_lines: Sequence[str],
    hypothesis_lines: Sequence[str],
    *,
    tolerance: int = 0,
) -> BreakScore:
    """Score chosen breaks against human-authored breaks in aligned token space.

    The transcript is never word-identical to the lyric sheet, so reference and
    hypothesis token streams are aligned first and only breaks sitting in
    aligned regions can match.  ``tolerance`` allows a break to count as correct
    when it lands within that many tokens of the reference break.
    """
    ref_tokens, ref_breaks = _token_stream(reference_lines)
    hyp_tokens, hyp_breaks = _token_stream(hypothesis_lines)

    if not ref_tokens or not hyp_tokens:
        return BreakScore(len(ref_breaks), len(hyp_breaks), 0, 0.0, 0.0, 0.0, 0.0)

    mapping = _reference_to_hypothesis_map(ref_tokens, hyp_tokens)
    coverage = len(mapping) / len(ref_tokens)

    # A break sits *before* a token, so project it through the first aligned
    # token at or after that position.
    expected: list[int] = []
    for position in ref_breaks:
        projected = next(
            (mapping[i] for i in range(position, len(ref_tokens)) if i in mapping),
            None,
        )
        if projected is not None:
            expected.append(projected)

    hyp_break_set = set(hyp_breaks)
    unused = set(hyp_breaks)
    matched = 0
    for position in expected:
        hit = _nearest_unused_break(position, tolerance, hyp_break_set, unused)
        if hit is not None:
            unused.discard(hit)
            matched += 1

    precision = matched / len(hyp_breaks) if hyp_breaks else 0.0
    recall = matched / len(expected) if expected else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return BreakScore(
        n_reference_breaks=len(expected),
        n_hypothesis_breaks=len(hyp_breaks),
        n_matched=matched,
        precision=precision,
        recall=recall,
        f1=f1,
        coverage=coverage,
    )


def score_dangling_breaks(
    lines: Sequence[str], *, max_examples: int = 5
) -> DanglingScore:
    """Count line breaks that leave the line ending on a continuation word.

    Needs no reference, so it can be run over any produced ``.lrc`` or SYLT.
    """
    texts = [line for line in lines if line_tokens(line)]
    dangling: list[str] = []
    breaks = 0

    for index, line in enumerate(texts[:-1]):
        tokens = line_tokens(line)
        if not tokens:
            continue
        breaks += 1
        if normalize_word(tokens[-1]) in DANGLING_END_WORDS:
            following = line_tokens(texts[index + 1])
            dangling.append(
                f"...{' '.join(tokens[-3:])} | {' '.join(following[:3])}..."
            )

    return DanglingScore(
        n_breaks=breaks,
        n_dangling=len(dangling),
        dangling_rate=len(dangling) / breaks if breaks else 0.0,
        examples=dangling[:max_examples],
    )


def read_lrc_lines(path) -> list[str]:
    """Read the lyric text out of an ``.lrc`` file, dropping the timestamps."""
    from pathlib import Path

    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = []
    for raw in text.splitlines():
        stripped = re.sub(r"^(\[[^\]]*\])+", "", raw).strip()
        if stripped:
            lines.append(stripped)
    return lines


def segments_to_lines(segments: Iterable[dict]) -> list[str]:
    """Line texts from a list of ``{"text", "start", "end"}`` segments."""
    return [str(seg.get("text", "")).strip() for seg in segments]


def uslt_lines(text: Optional[str]) -> list[str]:
    """Non-empty lines of a USLT lyric sheet."""
    if not text:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]
