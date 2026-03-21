"""Transcription and forced alignment using WhisperX."""

import logging
import os
import re
import difflib
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Disable pyannote anonymous telemetry calls (otel.pyannote.ai) so verbose runs
# are quieter and do not emit repeated HTTPS connection debug lines.
os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")

_WEAK_BOUNDARY_END = {
    "a", "an", "the", "and", "or", "but", "to", "of", "in", "on", "at", "for", "with", "from", "by",
    "as", "if", "that", "which", "who", "when", "while", "because", "cause", "my", "your", "our", "their",
    "his", "her", "its", "i",
}
_WEAK_BOUNDARY_START = {
    "and", "or", "but", "to", "of", "in", "on", "at", "for", "with", "from", "by", "as", "if", "that",
    "which", "who", "when", "while", "because", "cause",
}

_LIKELY_FILLER_WORDS = {
    "uh", "um", "oh", "ah", "ooh", "aah", "la", "na", "da", "so", "yo", "hey", "yeah",
}

_LOW_CONTENT_WORDS = _WEAK_BOUNDARY_END | _WEAK_BOUNDARY_START | {
    "am", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had",
    "he", "she", "we", "they", "me", "him", "them", "you",
    "it", "this", "that", "these", "those",
    "not", "no", "yes", "all",
}

# Whisper model to use.  "large-v2" is most accurate; for faster results use
# "tiny", "base", "small", or "medium" at the cost of accuracy.
DEFAULT_WHISPER_MODEL = "large-v2"
DEFAULT_COMPUTE_TYPE = "float16"


def _default_device() -> str:
    """Return 'cuda' if a CUDA-capable GPU is available, otherwise 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


DEFAULT_DEVICE = _default_device()


def transcribe_and_align(
    audio_path: str | os.PathLike,
    unsynced_lyrics: Optional[str] = None,
    model_name: str = DEFAULT_WHISPER_MODEL,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
    language: str = "en",
) -> list[dict]:
    """
    Transcribe and/or align *audio_path* using WhisperX.

    If ``device`` is ``'cuda'`` but CUDA is not available the function
    automatically falls back to ``'cpu'``.

    Parameters
    ----------
    audio_path:
        Path to the audio file (typically the vocals-only WAV).
    unsynced_lyrics:
        Plain-text lyrics to use for *forced alignment*.  When provided the
        model only determines timing; no transcription is performed.
    model_name:
        Whisper checkpoint name (``"base"``, ``"small"``, ``"medium"``,
        ``"large-v2"``).
    device:
        ``"cuda"`` for NVIDIA GPU (recommended) or ``"cpu"``.
    compute_type:
        ``"int8"`` (fast, low VRAM) or ``"float16"`` (higher accuracy on GPU).
    language:
        ISO 639-1 language code for transcription (e.g. ``"en"``).

    Returns
    -------
    list[dict]
        Segments with ``"start"`` (seconds), ``"end"`` (seconds), and
        ``"text"`` keys.
    """
    try:
        import whisperx
    except ImportError as exc:
        raise ImportError(
            "whisperx is not installed. See README for installation instructions."
        ) from exc

    # Gracefully fall back to CPU when CUDA was requested but isn't available.
    try:
        import torch as _torch
        cuda_available = _torch.cuda.is_available()
        if device == "auto":
            device = "cuda" if cuda_available else "cpu"
        if device == "cuda" and not cuda_available:
            logger.warning("CUDA requested but not available – falling back to CPU.")
            device = "cpu"
            if compute_type == "float16":
                compute_type = "int8"
    except Exception:
        pass

    audio_path = Path(audio_path)
    logger.info("Loading audio from '%s'…", audio_path.name)
    audio = whisperx.load_audio(str(audio_path))

    if unsynced_lyrics:
        logger.info("Forced alignment mode – using provided lyrics text.")
        segments = _forced_align(
            whisperx,
            audio,
            unsynced_lyrics,
            device,
            language,
            model_name=model_name,
            compute_type=compute_type,
        )
    else:
        logger.info("Full transcription mode (no lyrics provided).")
        segments = _transcribe(whisperx, audio, model_name, device, compute_type, language)

    return segments


def _transcribe(whisperx, audio, model_name, device, compute_type, language):
    """Run Whisper transcription then word-level alignment."""
    model = whisperx.load_model(model_name, device, compute_type=compute_type, language=language)
    result = model.transcribe(audio, batch_size=16)
    segments = result["segments"]

    if _looks_probably_instrumental(segments):
        logger.info("No credible lyrics detected; treating track as instrumental.")
        return []

    align_model, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    aligned = whisperx.align(segments, align_model, metadata, audio, device, return_char_alignments=False)

    # Post-process: produce consistent, phrase-like line lengths.
    refined = _split_long_segments(aligned["segments"])
    return refined


def _looks_probably_instrumental(segments: list[dict]) -> bool:
    """Return ``True`` when Whisper output looks like instrumental hallucination.

    This is intentionally conservative and only triggers on very sparse,
    low-confidence text such as isolated words, note symbols, or tiny scraps.
    """
    if not segments:
        return True

    alpha_word_count = 0
    alpha_char_count = 0
    textful_segments = 0
    short_or_symbolic_segments = 0
    logprobs: list[float] = []
    unique_words: set[str] = set()
    non_filler_word_count = 0
    contentful_word_count = 0
    timed_duration = 0.0

    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        textful_segments += 1
        seg_start = seg.get("start")
        seg_end = seg.get("end")
        if isinstance(seg_start, (int, float)) and isinstance(seg_end, (int, float)) and seg_end > seg_start:
            timed_duration += float(seg_end) - float(seg_start)

        words = re.findall(r"[A-Za-z']+", text)
        meaningful_words = [w for w in words if re.search(r"[A-Za-z]", w)]
        alpha_word_count += len(meaningful_words)
        alpha_char_count += sum(len(w) for w in meaningful_words)
        normalized_words = [w.lower() for w in meaningful_words]
        unique_words.update(normalized_words)
        non_filler_word_count += sum(1 for w in normalized_words if w not in _LIKELY_FILLER_WORDS)
        contentful_word_count += sum(
            1
            for w in normalized_words
            if w not in _LIKELY_FILLER_WORDS and w not in _LOW_CONTENT_WORDS and len(w) >= 4
        )

        stripped = re.sub(r"[A-Za-z']", "", text)
        is_symbolic = not meaningful_words or all(ch in "♪♫♬♩.,!?-–—()[]{}:;\"' " for ch in stripped)
        if len(meaningful_words) <= 1 or is_symbolic:
            short_or_symbolic_segments += 1

        avg_logprob = seg.get("avg_logprob")
        if isinstance(avg_logprob, (int, float)):
            logprobs.append(float(avg_logprob))

    if textful_segments == 0:
        return True

    mean_logprob = sum(logprobs) / len(logprobs) if logprobs else 0.0

    # Strong signal: effectively no readable lyric content.
    if alpha_word_count == 0:
        return True

    # Sparse scraps with low confidence are typically instrumental hallucinations.
    if alpha_word_count <= 3 and alpha_char_count <= 16 and mean_logprob <= -0.9:
        return True

    if textful_segments <= 2 and short_or_symbolic_segments == textful_segments and alpha_word_count <= 4:
        return True

    lexical_density = alpha_word_count / max(1.0, timed_duration)
    if (
        timed_duration >= 20.0
        and lexical_density < 0.20
        and mean_logprob <= -1.0
        and (non_filler_word_count == 0 or len(unique_words) <= 3)
    ):
        return True

    if (
        timed_duration >= 20.0
        and textful_segments <= 6
        and alpha_word_count <= 10
        and mean_logprob <= -1.2
        and contentful_word_count == 0
    ):
        return True

    symbolic_ratio = short_or_symbolic_segments / max(1, textful_segments)
    if (
        timed_duration >= 120.0
        and textful_segments <= 12
        and alpha_word_count <= 28
        and lexical_density < 0.18
        and mean_logprob <= -0.85
        and (symbolic_ratio >= 0.34 or contentful_word_count <= 6)
    ):
        return True

    return False


def _split_long_segments(
    segments: list[dict],
    *,
    target_words: int = 10,
    min_words: int = 4,
    max_words: int = 13,
    pause_threshold: float = 0.35,
    max_duration: float = 5.8,
) -> list[dict]:
    """Split aligned segments into consistent lyric lines.

    Uses word-level timestamps when available to prefer boundaries at natural
    pauses/punctuation while keeping line lengths reasonably uniform.
    """
    result: list[dict] = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue

        words_in_text = len(text.split())
        if words_in_text <= max_words:
            result.append({"text": text, "start": seg["start"], "end": seg["end"]})
            continue

        split_segments = _split_segment_consistently(
            seg,
            target_words=target_words,
            min_words=min_words,
            max_words=max_words,
            pause_threshold=pause_threshold,
            max_duration=max_duration,
        )
        result.extend(split_segments)

    return _merge_tiny_neighbor_lines(result, min_words=min_words, max_words=max_words)


def _split_segment_consistently(
    seg: dict,
    *,
    target_words: int,
    min_words: int,
    max_words: int,
    pause_threshold: float,
    max_duration: float,
) -> list[dict]:
    """Split one aligned segment into consistent phrase-level chunks."""
    timed_words = _extract_timed_words(seg)
    if len(timed_words) < 2:
        return _fallback_split_without_word_times(seg, target_words=target_words, max_words=max_words)

    n = len(timed_words)
    boundary_bonus = [0.0] * n
    for idx, word in enumerate(timed_words):
        next_word = timed_words[idx + 1] if idx + 1 < n else None
        gap_to_next = 0.0
        if next_word is not None:
            gap_to_next = max(0.0, next_word["start"] - word["end"])

        bonus = 0.0
        token = word["token"]
        next_token = next_word["token"] if next_word is not None else ""
        end_word = _clean_boundary_word(token)
        start_word = _clean_boundary_word(next_token)

        is_sentence_punct = bool(re.search(r"[.!?]$", token))
        is_soft_punct = bool(re.search(r"[,;:]$", token))
        next_is_upper = bool(next_word is not None and re.match(r"[A-Z]", next_token))
        next_is_lower = bool(next_word is not None and re.match(r"[a-z]", next_token))

        if is_sentence_punct:
            bonus += 2.2
        elif is_soft_punct:
            bonus += 1.2

        if gap_to_next >= pause_threshold:
            bonus += min(3.6, gap_to_next * 6.5)
        elif gap_to_next < 0.08:
            bonus -= 0.7

        if next_is_upper:
            bonus += 1.4

        # Very important: avoid splitting inside a flowing phrase when the next
        # token starts lowercase and there is no real pause/punctuation boundary.
        if next_is_lower and not is_sentence_punct and not is_soft_punct and gap_to_next < (pause_threshold * 0.7):
            bonus -= 2.8

        if end_word in _WEAK_BOUNDARY_END:
            bonus -= 3.0
        if start_word in _WEAK_BOUNDARY_START:
            bonus -= 1.6

        boundary_bonus[idx] = bonus

    max_span = max_words + 4
    inf = 1e18
    dp = [inf] * (n + 1)
    nxt = [-1] * (n + 1)
    dp[n] = 0.0

    for i in range(n - 1, -1, -1):
        best_cost = inf
        best_j = -1
        limit = min(n, i + max_span)
        for j in range(i, limit):
            count = j - i + 1
            start = timed_words[i]["start"]
            end = timed_words[j]["end"]
            duration = max(0.01, end - start)

            cost = 0.0
            # Keep lengths somewhat consistent, but do not dominate pause/grammar cues.
            cost += 0.22 * ((count - target_words) ** 2)
            if count < min_words:
                cost += (min_words - count) * 4.0
            if count > max_words:
                cost += (count - max_words) * 4.5

            if duration > max_duration:
                cost += (duration - max_duration) * 3.0
            elif duration < 1.1 and count >= min_words:
                cost += (1.1 - duration) * 1.2

            if j < n - 1:
                cost -= 2.3 * boundary_bonus[j]

            total = cost + dp[j + 1]
            if total < best_cost:
                best_cost = total
                best_j = j

        dp[i] = best_cost
        nxt[i] = best_j

    if nxt[0] < 0:
        return [{"text": seg.get("text", "").strip(), "start": seg["start"], "end": seg["end"]}]

    chunks: list[dict] = []
    i = 0
    while i < n and nxt[i] >= i:
        j = nxt[i]
        text = _join_tokens([w["token"] for w in timed_words[i : j + 1]])
        if text:
            chunks.append(
                {
                    "text": text,
                    "start": timed_words[i]["start"],
                    "end": timed_words[j]["end"],
                }
            )
        i = j + 1

    return chunks if chunks else [{"text": seg.get("text", "").strip(), "start": seg["start"], "end": seg["end"]}]


def _clean_boundary_word(token: str) -> str:
    """Normalize a token for boundary-language heuristics."""
    return re.sub(r"^[^A-Za-z']+|[^A-Za-z']+$", "", token).lower()


def _normalize_alignment_text(text: str) -> str:
    """Normalize lyric/transcript text for fuzzy matching."""
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _seed_segments_from_coarse_alignment(lines: list[str], coarse_segments: list[dict]) -> list[dict]:
    """Map lyric lines onto coarse transcript spans in order.

    This uses fuzzy text matching against short contiguous groups of coarse
    transcript segments so the initial timing windows follow the song structure
    instead of being spread uniformly over the full vocal span.
    """
    normalized_lines = [_normalize_alignment_text(line) for line in lines]
    coarse = []
    for seg in coarse_segments:
        text = str(seg.get("text", "")).strip()
        start = seg.get("start")
        end = seg.get("end")
        if not text or start is None or end is None:
            continue
        coarse.append(
            {
                "text": text,
                "norm": _normalize_alignment_text(text),
                "start": float(start),
                "end": float(end),
            }
        )

    if not coarse:
        return []

    seeded: list[dict] = []
    cursor = 0
    max_skip = 3
    max_group = 4

    for idx, (line, norm_line) in enumerate(zip(lines, normalized_lines)):
        remaining_lines = len(lines) - idx - 1
        best: tuple[float, int, int] | None = None

        search_end = min(len(coarse), cursor + max_skip + max_group)
        for start_idx in range(cursor, search_end):
            max_end = min(len(coarse), start_idx + max_group)
            for end_idx in range(start_idx, max_end):
                remaining_coarse = len(coarse) - (end_idx + 1)
                if remaining_coarse < max(0, remaining_lines - max_skip):
                    continue

                candidate_text = " ".join(seg["norm"] for seg in coarse[start_idx : end_idx + 1]).strip()
                if not candidate_text:
                    continue

                similarity = difflib.SequenceMatcher(None, norm_line, candidate_text).ratio()
                skip_penalty = 0.05 * (start_idx - cursor)
                span_penalty = 0.03 * (end_idx - start_idx)
                score = similarity - skip_penalty - span_penalty

                if best is None or score > best[0]:
                    best = (score, start_idx, end_idx)

        if best is None:
            break

        _, start_idx, end_idx = best
        seeded.append(
            {
                "text": line,
                "start": coarse[start_idx]["start"],
                "end": coarse[end_idx]["end"],
            }
        )
        cursor = end_idx + 1

    return seeded


def _seed_segments_by_coarse_durations(lines: list[str], coarse_segments: list[dict]) -> list[dict]:
    """Fallback seeding that maps many lyric lines into few coarse chunks.

    Distributes lines proportionally to each coarse segment's *available span*.
    For short inter-segment gaps (< 2× average segment duration) the available
    span extends to the next segment's start, covering lines that sing in the
    gap.  For large gaps (instrumental bridges) the span is capped at the
    segment's own end, preventing bridge silence from inflating a block's
    allocation and pushing adjacent lines too late.
    """
    if not lines:
        return []

    coarse = []
    for seg in coarse_segments:
        start = seg.get("start")
        end = seg.get("end")
        if start is None or end is None:
            continue
        s = float(start)
        e = float(end)
        if e <= s:
            continue
        coarse.append({"start": s, "end": e, "duration": e - s})

    if not coarse:
        return []

    n_lines = len(lines)
    n_coarse = len(coarse)

    # Threshold for detecting instrumental bridges between coarse segments.
    avg_dur = sum(s["duration"] for s in coarse) / n_coarse
    bridge_threshold = 2.0 * avg_dur

    # For each coarse block, include the gap to the next block only when it is
    # short (likely just a breath or mild silence with lyrics); skip it when it
    # is a long instrumental bridge.
    available_ends = []
    for i in range(n_coarse - 1):
        gap = coarse[i + 1]["start"] - coarse[i]["end"]
        if gap > bridge_threshold:
            available_ends.append(coarse[i]["end"])    # bridge — stop at own end
        else:
            available_ends.append(coarse[i + 1]["start"])  # short gap — include it
    available_ends.append(coarse[-1]["end"])

    available_spans = [available_ends[i] - coarse[i]["start"] for i in range(n_coarse)]
    total_available = sum(max(0.01, s) for s in available_spans)

    if n_lines <= n_coarse:
        allocation = [1 if i < n_lines else 0 for i in range(n_coarse)]
    else:
        allocation = [1] * n_coarse
        remaining = n_lines - n_coarse
        raw_extra = [remaining * (max(0.01, s) / total_available) for s in available_spans]
        extra_int = [int(x) for x in raw_extra]
        for i, n in enumerate(extra_int):
            allocation[i] += n
        left = remaining - sum(extra_int)
        if left > 0:
            remainders = sorted(
                ((raw_extra[i] - extra_int[i], i) for i in range(n_coarse)),
                reverse=True,
            )
            for _, idx in remainders[:left]:
                allocation[idx] += 1

    seeded: list[dict] = []
    line_idx = 0
    for coarse_idx, (seg, k) in enumerate(zip(coarse, allocation)):  # noqa: B007
        if k <= 0:
            continue
        block_lines = lines[line_idx : line_idx + k]
        if not block_lines:
            break

        block_start = seg["start"]
        block_end = available_ends[coarse_idx]
        block_span = max(0.12, block_end - block_start)

        weights = [max(1, len(line)) for line in block_lines]
        total_w = sum(weights)
        cursor = block_start

        for i, (line, w) in enumerate(zip(block_lines, weights)):
            frac = w / total_w if total_w > 0 else (1.0 / len(block_lines))
            dur = block_span * frac
            end = block_end if i == len(block_lines) - 1 else cursor + dur
            start = cursor
            if end <= start:
                end = start + 0.12
            seeded.append({"text": line, "start": start, "end": end})
            cursor = end

        line_idx += len(block_lines)

    while line_idx < n_lines:
        prev_end = seeded[-1]["end"] if seeded else coarse[0]["start"]
        seeded.append({"text": lines[line_idx], "start": prev_end, "end": prev_end + 1.0})
        line_idx += 1

    return seeded


def _extract_timed_words(seg: dict) -> list[dict]:
    """Extract words with timestamps from a WhisperX aligned segment."""
    words = seg.get("words") or []
    timed: list[dict] = []
    for w in words:
        token = str(w.get("word", "")).strip()
        start = w.get("start")
        end = w.get("end")
        if not token:
            continue
        if start is None or end is None:
            continue
        timed.append({"token": token, "start": float(start), "end": float(end)})
    return timed


def _extract_all_timed_words(segments: list[dict]) -> list[dict]:
    """Flatten word-level timestamps from all whisperx aligned segments."""
    words: list[dict] = []
    for seg in segments:
        for w in (seg.get("words") or []):
            token = str(w.get("word", "")).strip()
            start = w.get("start")
            end = w.get("end")
            if not token or start is None or end is None:
                continue
            words.append({"token": token, "start": float(start), "end": float(end)})
    return words


def _normalize_token(token: str) -> str:
    """Normalize a single word token for lyric matching."""
    return re.sub(r"[^a-z']", "", token.lower().replace("\u2019", "'"))


def _align_uslt_to_transcribed_words(
    uslt_lines: list[str],
    timed_words: list[dict],
) -> list[dict]:
    """Match USLT lyric lines to transcribed word timestamps.

    For each USLT line the best-matching consecutive sequence of transcribed
    words is found by greedy forward search with fuzzy scoring.  The line is
    assigned the *start* timestamp of its first matched word, so timing accuracy
    depends entirely on WhisperX word alignment rather than coarse-segment
    seeding — eliminating the progressive drift that accumulates when seed
    windows stretch across musical gaps.
    """
    if not timed_words:
        return []

    n_words = len(timed_words)
    norm_trans = [_normalize_token(w["token"]) for w in timed_words]

    result: list[dict] = []
    word_cursor = 0

    for line_idx, line in enumerate(uslt_lines):
        if word_cursor >= n_words:
            last_start = float(timed_words[-1]["start"])
            last_end = float(timed_words[-1]["end"])
            result.append({"text": line, "start": last_start, "end": last_end})
            continue

        line_tokens = [
            _normalize_token(t)
            for t in re.findall(r"[A-Za-z']+", line)
            if t.strip()
        ]

        if not line_tokens:
            prev_start = result[-1]["start"] if result else timed_words[0]["start"]
            prev_end = result[-1]["end"] if result else timed_words[0]["end"]
            result.append({"text": line, "start": prev_start, "end": prev_end})
            continue

        n_line = len(line_tokens)
        remaining_uslt = max(1, len(uslt_lines) - line_idx - 1)
        remaining_words = max(1, n_words - word_cursor)

        # How far ahead to search for this line's first word.  Leave enough
        # words downstream for remaining USLT lines so we don't over-consume.
        words_per_remaining = max(1, remaining_words // max(1, remaining_uslt))
        skip_budget = min(
            remaining_words - max(0, remaining_uslt - 1),
            max(n_line * 2, words_per_remaining * 3),
        )

        start_floor = min(max(0, word_cursor), n_words - 1)
        best_score = -1.0
        best_start_idx = start_floor
        best_end_idx = min(start_floor + n_line - 1, n_words - 1)

        for skip in range(max(0, skip_budget) + 1):
            start_idx = word_cursor + skip
            if start_idx >= n_words:
                break

            # Try window sizes close to n_line to handle minor transcription
            # length differences (contractions, dropped words, etc.).
            max_delta = min(4, n_words - start_idx - n_line + 1)
            for delta in range(-min(3, n_line - 1), max_delta + 1):
                window = n_line + delta
                if window <= 0:
                    continue
                end_idx = start_idx + window - 1
                if end_idx >= n_words:
                    break

                score = difflib.SequenceMatcher(
                    None, line_tokens, norm_trans[start_idx : end_idx + 1]
                ).ratio()
                # Small penalty for skipping transcribed words so we prefer
                # the earliest good match when scores are otherwise equal.
                score -= skip * 0.015

                if score > best_score:
                    best_score = score
                    best_start_idx = start_idx
                    best_end_idx = end_idx

        result.append({
            "text": line,
            "start": timed_words[best_start_idx]["start"],
            "end": timed_words[best_end_idx]["end"],
        })
        word_cursor = min(best_end_idx + 1, n_words)

    return result


def _fallback_split_without_word_times(seg: dict, *, target_words: int, max_words: int) -> list[dict]:
    """Fallback splitting when WhisperX does not provide timed words."""
    text = seg.get("text", "").strip()
    words = text.split()
    if not text or len(words) <= max_words:
        return [{"text": text, "start": seg["start"], "end": seg["end"]}] if text else []

    clauses = [part.strip() for part in re.split(r"(?<=[,.;:!?])\s+", text) if part.strip()]
    if len(clauses) == 1:
        clauses = _chunk_words(words, target_words)

    total_duration = max(0.01, float(seg["end"]) - float(seg["start"]))
    starts_ends: list[tuple[float, float]] = []
    cursor = float(seg["start"])
    total_chars = sum(max(1, len(c)) for c in clauses)
    for i, clause in enumerate(clauses):
        weight = max(1, len(clause)) / total_chars
        dur = total_duration * weight
        end = float(seg["end"]) if i == len(clauses) - 1 else cursor + dur
        starts_ends.append((cursor, end))
        cursor = end

    return [{"text": c, "start": s, "end": e} for c, (s, e) in zip(clauses, starts_ends)]


def _chunk_words(words: list[str], target_words: int) -> list[str]:
    """Chunk raw word list into near-target-size chunks."""
    if not words:
        return []
    chunks: list[str] = []
    for i in range(0, len(words), target_words):
        chunks.append(" ".join(words[i : i + target_words]))
    return chunks


def _join_tokens(tokens: list[str]) -> str:
    """Join tokens into readable lyric text."""
    text = " ".join(t.strip() for t in tokens if t.strip())
    return re.sub(r"\s+([,.;:!?])", r"\1", text).strip()


def _merge_tiny_neighbor_lines(segments: list[dict], *, min_words: int, max_words: int) -> list[dict]:
    """Merge very short lines with neighbors to avoid choppy output."""
    if not segments:
        return []
    cleaned = [
        {"text": s.get("text", "").strip(), "start": s["start"], "end": s["end"]}
        for s in segments
        if s.get("text", "").strip()
    ]

    i = 0
    while i < len(cleaned):
        current = cleaned[i]
        current_words = len(current["text"].split())
        ends_emphatic = bool(re.search(r"[!?]$", current["text"]))
        if ends_emphatic and current_words <= 3:
            i += 1
            continue
        if current_words >= min_words:
            i += 1
            continue

        # Prefer merging forward so sentence starters are not stranded,
        # but avoid crossing strong punctuation into overlong lines.
        if i + 1 < len(cleaned):
            nxt = cleaned[i + 1]
            combined_words = current_words + len(nxt["text"].split())
            current_ends_hard_stop = bool(re.search(r"[.!?]$", current["text"]))
            if not current_ends_hard_stop and combined_words <= max_words + 2:
                cleaned[i + 1] = {
                    "text": _join_tokens([current["text"], nxt["text"]]),
                    "start": current["start"],
                    "end": nxt["end"],
                }
                del cleaned[i]
                continue

        # If this is the final tiny line, merge backward.
        if i > 0:
            prev = cleaned[i - 1]
            prev_words = len(prev["text"].split())
            prev_ends_hard_stop = bool(re.search(r"[.!?]$", prev["text"]))
            if not prev_ends_hard_stop and (prev_words + current_words) <= max_words + 2:
                cleaned[i - 1] = {
                    "text": _join_tokens([prev["text"], current["text"]]),
                    "start": prev["start"],
                    "end": current["end"],
                }
                del cleaned[i]
                i -= 1
                continue

        i += 1

    return cleaned


def _forced_align(
    whisperx,
    audio,
    unsynced_lyrics: str,
    device: str,
    language: str,
    *,
    model_name: str,
    compute_type: str,
):
    """Derive synchronized line timings from USLT lyrics + full transcription.

    Strategy:
    1. Transcribe the audio with Whisper to detect the true vocal content.
    2. Run whisperx word-level alignment to get a per-word timestamp list.
    3. Greedily match each USLT line to the closest consecutive word sequence
       using fuzzy text scoring.
    4. Assign each line the *start* timestamp of its first matched word.

    This completely avoids the progressive timing drift caused by coarse-segment
    seeding, where accumulated stretch errors between musical gaps push later
    lines progressively late.
    """
    lines = [line.strip() for line in unsynced_lyrics.splitlines() if line.strip()]
    if not lines:
        logger.warning("No lyrics text provided for alignment.")
        return []

    audio_duration = len(audio) / 16000.0
    logger.debug("Audio duration: %.1f s, %d USLT lines", audio_duration, len(lines))

    # Full Whisper transcription pass.
    model = whisperx.load_model(model_name, device, compute_type=compute_type, language=language)
    tx_result = model.transcribe(audio, batch_size=16)
    raw_segments = tx_result.get("segments", [])

    if _looks_probably_instrumental(raw_segments):
        logger.info("Track appears instrumental; skipping USLT alignment.")
        return []

    # Word-level alignment to get per-word timestamps.
    align_language = tx_result.get("language") or language
    logger.debug("Running word-level alignment (lang=%s).", align_language)
    align_model, metadata = whisperx.load_align_model(language_code=align_language, device=device)
    aligned = whisperx.align(raw_segments, align_model, metadata, audio, device, return_char_alignments=False)

    timed_words = _extract_all_timed_words(aligned["segments"])
    logger.debug("Extracted %d timed words.", len(timed_words))

    if len(timed_words) >= 5:
        result = _align_uslt_to_transcribed_words(lines, timed_words)
        logger.debug("Matched %d USLT lines to word timestamps.", len(result))
        return result

    # Sparse fallback: not enough word timestamps (very quiet / sparse vocal).
    # Fall back to segment-level seeding + whisperx.align().
    logger.warning(
        "Only %d timed words extracted; falling back to segment-seeded alignment.",
        len(timed_words),
    )
    coarse = aligned["segments"]
    segments = _seed_segments_from_coarse_alignment(lines, coarse)
    if len(segments) != len(lines):
        segments = _seed_segments_by_coarse_durations(lines, coarse)
    vocal_start = float(coarse[0]["start"]) if coarse else 0.0
    fallback_aligned = whisperx.align(segments, align_model, metadata, audio, device, return_char_alignments=False)
    return _apply_intro_onset_floor(fallback_aligned["segments"], vocal_start)


def _apply_intro_onset_floor(segments: list[dict], vocal_start: float) -> list[dict]:
    """Clamp first aligned lyric onset to the detected vocal start."""
    if not segments:
        return segments

    floor = float(vocal_start)
    first = segments[0]
    start = float(first.get("start", floor))
    end = float(first.get("end", start))

    if start < floor:
        first["start"] = floor
        if end <= floor:
            first["end"] = floor + 0.12

    prev_start = float(segments[0].get("start", floor))
    for seg in segments[1:]:
        seg_start = float(seg.get("start", prev_start))
        seg_end = float(seg.get("end", seg_start))
        if seg_start < prev_start:
            seg_start = prev_start
            seg["start"] = seg_start
            if seg_end <= seg_start:
                seg["end"] = seg_start + 0.12
        prev_start = seg_start

    return segments


def _is_overclustered_seed_timing(segments: list[dict]) -> bool:
    """Return True when many adjacent lyric starts are unnaturally clustered."""
    if len(segments) < 20:
        return False

    starts = [float(seg.get("start", 0.0)) for seg in segments]
    starts.sort()
    if len(starts) < 2:
        return False

    span = starts[-1] - starts[0]
    if span < 45.0:
        return False

    gaps = [starts[i + 1] - starts[i] for i in range(len(starts) - 1)]
    tiny_gaps = sum(1 for g in gaps if g < 0.25)
    tiny_ratio = tiny_gaps / max(1, len(gaps))
    return tiny_ratio >= 0.55


def _needs_dense_timing_baseline(line_count: int, coarse_count: int) -> bool:
    """Return True when coarse timing resolution is too low for lyric line count.

    Triggers whenever there are too few coarse segments relative to the number
    of lyric lines, even at moderate ratios (e.g. 7 segments for 39 lines → 5.6).
    Lowered from (≥30 lines, ≥8.0 ratio) so cases like 39 lines / 7 coarse
    segments are also caught.
    """
    if line_count <= 0 or coarse_count <= 0:
        return False
    lines_per_coarse = line_count / coarse_count
    return line_count >= 20 and lines_per_coarse >= 4.0


def _seed_timing_penalty(segments: list[dict]) -> float:
    """Lower is better: penalize collapsed and implausibly jumpy seed timing."""
    if len(segments) < 3:
        return 1e9

    starts = [float(seg.get("start", 0.0)) for seg in segments]
    starts.sort()
    gaps = [starts[i + 1] - starts[i] for i in range(len(starts) - 1)]
    if not gaps:
        return 1e9

    tiny_ratio = sum(1 for g in gaps if g < 0.25) / len(gaps)
    huge_ratio = sum(1 for g in gaps if g > 18.0) / len(gaps)
    giant_ratio = sum(1 for g in gaps if g > 25.0) / len(gaps)
    max_gap = max(gaps)

    penalty = 0.0
    penalty += tiny_ratio * 9.0
    penalty += huge_ratio * 8.0
    penalty += giant_ratio * 12.0
    penalty += max(0.0, max_gap - 35.0) * 0.2
    return penalty


def _estimate_vocal_onset_from_audio(audio, *, sample_rate: int = 16000) -> float | None:
    """Estimate first sustained vocal activity using short-time RMS energy."""
    if audio is None:
        return None

    waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
    if waveform.size < sample_rate:
        return None

    frame_len = max(1, int(0.05 * sample_rate))
    hop = max(1, int(0.02 * sample_rate))
    starts = np.arange(0, max(1, waveform.size - frame_len + 1), hop)
    if starts.size == 0:
        return None

    rms = np.sqrt(
        np.array([
            np.mean(np.square(waveform[s : s + frame_len]), dtype=np.float64)
            for s in starts
        ])
        + 1e-12
    )
    if rms.size == 0:
        return None

    noise_floor = float(np.percentile(rms, 15))
    peak_level = float(np.percentile(rms, 98))
    if peak_level <= noise_floor:
        return None

    threshold = noise_floor + 0.20 * (peak_level - noise_floor)
    active = rms >= threshold

    def _first_run(min_seconds: float) -> float | None:
        run_frames = max(1, int(round(min_seconds / (hop / sample_rate))))
        run = 0
        for idx, is_active in enumerate(active):
            if is_active:
                run += 1
                if run >= run_frames:
                    onset_frame = idx - run_frames + 1
                    onset_sample = int(starts[max(0, onset_frame)])
                    return onset_sample / float(sample_rate)
            else:
                run = 0
        return None

    onset_fast = _first_run(0.35)
    onset_sustained = _first_run(0.50)

    if onset_fast is None and onset_sustained is None:
        return None
    if onset_fast is None:
        return onset_sustained
    if onset_sustained is None:
        return onset_fast

    if onset_sustained - onset_fast > 3.0:
        return onset_fast + 0.5 * (onset_sustained - onset_fast)

    return onset_fast
