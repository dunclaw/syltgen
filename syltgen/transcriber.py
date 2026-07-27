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

#: Keyword arguments passed to ``stable_whisper.alignment.align`` in the USLT
#: forced-alignment path.  Exposed as a module constant so the accuracy
#: benchmark (``tests/benchmark``) can sweep them without forking the pipeline.
DEFAULT_ALIGN_OPTIONS: dict = {
    "original_split": True,   # one output segment per USLT line
    "vad": True,
    "suppress_silence": True,
    "nonspeech_skip": None,   # do not skip long gaps; outros can follow a break
    "only_voice_freq": True,  # 200-5000 Hz, avoids matching instrumental content
}




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
    align_options: Optional[dict] = None,
    align_method: str = "auto",
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
    align_options:
        Optional overrides merged over :data:`DEFAULT_ALIGN_OPTIONS` for the
        forced-alignment path.  Used by the accuracy benchmark to sweep
        alignment settings; production callers should leave this ``None``.
    align_method:
        Strategy for the USLT path.  ``"align"`` uses stable-ts forced
        alignment only, ``"transcribe_match"`` transcribes the audio and maps
        the lyric lines onto the transcript with a global token alignment,
        ``"transcribe_greedy"`` uses the older per-line greedy matcher, and
        ``"auto"`` (the default) runs forced alignment and falls back to
        transcribe-and-match when the alignment looks broken.

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
            align_options=align_options,
            align_method=align_method,
        )
    else:
        logger.info("Full transcription mode (no lyrics provided).")
        segments = _transcribe(whisperx, audio, model_name, device, compute_type, language)

    return segments


_STABLE_TS_MODEL_CACHE: dict[tuple[str, str], object] = {}


def _stable_ts_transcribe(
    audio,
    model_name: str,
    device: str,
    language: str,
) -> list[dict]:
    """Transcribe using stable-ts (regularized Whisper timestamps).

    Returns segments in the same shape ``whisperx.align`` expects::

        [{"text": ..., "start": ..., "end": ..., "words": [...]}]

    stable-ts produces dramatically better segment timing than vanilla Whisper /
    faster-whisper because it uses silence-suppression and per-token log-prob
    regularization to keep words anchored to the audio they actually occurred in,
    instead of letting Whisper relocate phrases across long silent gaps.
    """
    import stable_whisper

    cache_key = (model_name, device)
    model = _STABLE_TS_MODEL_CACHE.get(cache_key)
    if model is None:
        logger.info("Loading stable-ts model '%s' on %s…", model_name, device)
        model = stable_whisper.load_model(model_name, device=device)
        _STABLE_TS_MODEL_CACHE[cache_key] = model

    # suppress_silence + vad pre-detection are the main features that fix the
    # "Whisper guessed words then put them on the wrong audio" failure.
    result = model.transcribe(
        audio,
        language=language,
        vad=True,
        suppress_silence=True,
        word_timestamps=True,
        verbose=None,
    )

    segments_out: list[dict] = []
    for seg in result.segments:
        words = []
        for w in (seg.words or []):
            if w.start is None or w.end is None:
                continue
            words.append({
                "word": w.word.strip(),
                "start": float(w.start),
                "end": float(w.end),
                "score": float(getattr(w, "probability", 1.0) or 1.0),
            })
        segments_out.append({
            "text": seg.text.strip(),
            "start": float(seg.start),
            "end": float(seg.end),
            "words": words,
        })
    return segments_out


def _transcribe(whisperx, audio, model_name, device, compute_type, language):
    """Run stable-ts transcription with regularized word-level timestamps."""
    segments = _stable_ts_transcribe(audio, model_name, device, language)

    if _looks_probably_instrumental(segments):
        logger.info("No credible lyrics detected; treating track as instrumental.")
        return []

    # Post-process: produce consistent, phrase-like line lengths.
    refined = _split_long_segments(segments)

    # Apply vocal onset floor: if the audio-derived onset is detectably later
    # than the first transcribed segment, clamp early segments forward.
    if refined:
        vocal_onset = _estimate_vocal_onset_from_audio(audio)
        if vocal_onset is not None:
            first_start = float(refined[0].get("start", 0.0))
            offset = vocal_onset - first_start
            if 5.0 < offset < 60.0:
                logger.debug(
                    "Onset floor %.2f s applied (first transcribed seg at %.2f s).",
                    vocal_onset,
                    first_start,
                )
                refined = _apply_intro_onset_floor(refined, vocal_onset)

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
    short_alpha_word_count = 0
    max_words_in_segment = 0
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
        max_words_in_segment = max(max_words_in_segment, len(meaningful_words))
        alpha_char_count += sum(len(w) for w in meaningful_words)
        normalized_words = [w.lower() for w in meaningful_words]
        unique_words.update(normalized_words)
        short_alpha_word_count += sum(1 for w in normalized_words if len(w) <= 2)
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
    short_token_ratio = short_alpha_word_count / max(1, alpha_word_count)
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
        timed_duration >= 20.0
        and textful_segments <= 5
        and alpha_word_count <= 14
        and lexical_density < 0.20
        and mean_logprob <= -0.6
        and symbolic_ratio >= 0.75
        and contentful_word_count <= 5
    ):
        return True

    if (
        timed_duration >= 120.0
        and textful_segments <= 12
        and alpha_word_count <= 28
        and lexical_density < 0.18
        and mean_logprob <= -0.85
        and (symbolic_ratio >= 0.34 or contentful_word_count <= 6)
    ):
        return True

    # Very long tracks with only repeated stop-words / scraps (e.g. "and and",
    # "the the", "so") are strong instrumental hallucination candidates.
    if (
        timed_duration >= 90.0
        and textful_segments <= 16
        and alpha_word_count <= 20
        and lexical_density < 0.10
        and mean_logprob <= -1.35
        and symbolic_ratio >= 0.85
        and contentful_word_count == 0
        and len(unique_words) <= 6
    ):
        return True

    # Sparse tiny-line outputs from separated stems (e.g. "Bye.", "you",
    # "Ooh ooh", "Hmm.") are usually instrumental bleed-through, not lyrics.
    if (
        timed_duration >= 4.0
        and textful_segments <= 10
        and max_words_in_segment <= 3
        and alpha_word_count <= 24
        and contentful_word_count <= 3
        and len(unique_words) <= 12
        and mean_logprob <= -0.45
    ):
        return True

    # Repetitive syllabic babble ("da-da-da", "y-y-y") from stems can yield
    # many alphabetic tokens, but almost all are very short and non-contentful.
    if (
        timed_duration >= 60.0
        and textful_segments <= 14
        and short_token_ratio >= 0.68
        and contentful_word_count <= 2
        and len(unique_words) <= 12
        and mean_logprob <= -0.20
    ):
        return True

    # Ultra-sparse outputs: a few isolated fragments ("you I'll be right back",
    # "I can't", etc.) across the entire track. Usually instrumental hallucinations
    # or non-lyrical content (dialogue, environmental noise) that mistakenly made it
    # through stem separation or initial detection. These produce essentially no
    # usable SYLT content anyway.
    if (
        textful_segments <= 5
        and alpha_word_count <= 15
        and mean_logprob <= -0.40
    ):
        return True

    # Long tracks that only produce a tiny amount of text are almost always
    # instrumental for our pipeline purposes, even when confidence is not very low.
    # This catches cases where Whisper emits a few dialogue-like scraps over many
    # minutes of music (e.g. "I can't", "you", "So, um um you").
    if (
        timed_duration >= 45.0
        and textful_segments <= 8
        and alpha_word_count <= 24
        and (contentful_word_count <= 8 or len(unique_words) <= 14)
    ):
        return True

    # Extremely sparse long-track output (only a couple of lines over >1.5 min)
    # is not useful lyric material and is almost always instrumental hallucination.
    if (
        timed_duration >= 90.0
        and textful_segments <= 3
        and alpha_word_count <= 26
    ):
        return True

    # Highly repetitive low-content hallucinations: many repeated words but almost
    # all from filler vocabulary (e.g., "a baby a baby a baby" repeated, "um um um").  
    # Real lyrics would have diverse vocabulary and meaningful word content.
    if (
        textful_segments <= 6
        and alpha_word_count >= 16  # More than 15 words (catches repetition)
        and len(unique_words) <= 6  # But very few unique word types
        and contentful_word_count == 0  # No actual semantic content
        and mean_logprob <= -0.50
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


def _line_token_lists(lines: list[str]) -> list[list[str]]:
    """Normalized word tokens for each lyric line."""
    return [
        [t for t in (_normalize_token(w) for w in re.findall(r"[A-Za-z']+", line)) if t]
        for line in lines
    ]


def _interpolate_unanchored_lines(
    anchors: list[dict],
    timed_words: list[dict],
    *,
    audio_onset: float | None,
) -> list[dict]:
    """Fill in timing for lines that could not be anchored to a transcript word.

    Unanchored lines are placed by linear interpolation between the nearest
    anchored lines on either side, so a line the transcriber never heard still
    lands between its neighbours instead of inheriting a neighbour's timestamp.
    """
    matched = [(i, a["start"]) for i, a in enumerate(anchors) if a["matched"]]

    if not matched:
        total = float(timed_words[-1]["end"]) if timed_words else 0.0
        first = float(timed_words[0]["start"]) if timed_words else 0.0
        span = max(0.0, total - first)
        count = max(1, len(anchors))
        for index, anchor in enumerate(anchors):
            anchor["start"] = first + span * index / count
            anchor["end"] = first + span * (index + 1) / count
        return anchors

    for index, anchor in enumerate(anchors):
        if anchor["matched"]:
            continue

        previous = next(
            ((j, t) for j, t in reversed(matched) if j < index), None
        )
        following = next(((j, t) for j, t in matched if j > index), None)

        if previous is None and following is None:
            anchor["start"] = float(timed_words[0]["start"]) if timed_words else 0.0
        elif previous is None:
            next_index, next_time = following  # type: ignore[misc]
            if audio_onset is not None and audio_onset < next_time:
                fraction = (index + 1) / max(1, next_index + 1)
                anchor["start"] = audio_onset + fraction * (next_time - audio_onset)
            else:
                anchor["start"] = max(0.0, next_time - (next_index - index) * 1.5)
        elif following is None:
            prev_index, prev_time = previous
            anchor["start"] = prev_time + (index - prev_index) * 2.0
        else:
            prev_index, prev_time = previous
            next_index, next_time = following  # type: ignore[misc]
            fraction = (index - prev_index) / max(1, next_index - prev_index)
            anchor["start"] = prev_time + fraction * (next_time - prev_time)

        anchor["end"] = anchor["start"] + 2.0

    return anchors


def _align_lines_by_global_token_match(
    uslt_lines: list[str],
    timed_words: list[dict],
    *,
    audio_onset: float | None = None,
    stats: dict | None = None,
) -> list[dict]:
    """Map lyric lines onto transcript words with a single global alignment.

    The whole lyric sheet and the whole transcript are treated as two token
    streams and matched in one pass with :class:`difflib.SequenceMatcher`.
    Because the matching blocks it returns are monotonically increasing, every
    line is placed in order and a stretch of audio the lyric sheet does not
    cover (an unlisted verse, a long ad-lib, a repeat the sheet writes once) is
    simply skipped instead of dragging the remaining lines out of position.

    This replaces a per-line greedy forward search, which could not recover
    once a single line locked onto the wrong words.
    """
    if not timed_words or not uslt_lines:
        return []

    token_lists = _line_token_lists(uslt_lines)

    ref_tokens: list[str] = []
    ref_line_index: list[int] = []
    for line_index, tokens in enumerate(token_lists):
        for token in tokens:
            ref_tokens.append(token)
            ref_line_index.append(line_index)

    if not ref_tokens:
        return []

    transcript_tokens = [_normalize_token(w["token"]) for w in timed_words]

    matcher = difflib.SequenceMatcher(None, ref_tokens, transcript_tokens, autojunk=False)

    starts: dict[int, float] = {}
    ends: dict[int, float] = {}
    for ref_i, trans_i, size in matcher.get_matching_blocks():
        for offset in range(size):
            line_index = ref_line_index[ref_i + offset]
            word = timed_words[trans_i + offset]
            starts.setdefault(line_index, float(word["start"]))
            ends[line_index] = float(word["end"])

    anchors = [
        {
            "text": line,
            "start": starts.get(index, -1.0),
            "end": ends.get(index, -1.0),
            "matched": index in starts,
        }
        for index, line in enumerate(uslt_lines)
    ]

    matched_count = sum(1 for a in anchors if a["matched"])
    if stats is not None:
        stats["anchored_ratio"] = matched_count / len(anchors) if anchors else 0.0
    logger.debug(
        "Global token match anchored %d/%d lyric lines (%d transcript words).",
        matched_count,
        len(anchors),
        len(timed_words),
    )

    anchors = _interpolate_unanchored_lines(
        anchors, timed_words, audio_onset=audio_onset
    )
    return [{"text": a["text"], "start": a["start"], "end": a["end"]} for a in anchors]


_MATCH_MIN_SCORE = 0.30  # Lines scoring below this are treated as unmatched.
# Tuning notes:
#   - Too low (e.g. 0.20): garbage matches lock in, lines anchor to phantom
#     pre-vocal Whisper words from stem-separator bleed-through.
#   - Too high (e.g. 0.45): legitimate but imperfect matches (Whisper mis-hearing
#     a few words) get rejected, and interpolation produces worse timing than
#     the imperfect match would have.
# 0.30 balances both.  The pre-onset word filter is the primary defense against
# phantom matches, not this threshold.


def _align_uslt_to_transcribed_words(
    uslt_lines: list[str],
    timed_words: list[dict],
    *,
    audio_onset: float | None = None,
) -> list[dict]:
    """Match USLT lyric lines to transcribed word timestamps.

    Two-pass algorithm:

    Pass 1 — greedy forward search with a generous look-ahead window.  For each
    USLT line we search up to ``_SEARCH_WINDOW`` words ahead of the current
    cursor for the best-matching word sequence.  If the best score is below
    ``_MATCH_MIN_SCORE`` (the line was not transcribed, or Whisper used
    significantly different wording) we mark the line as *unmatched* and
    **freeze the cursor** so subsequent lines can still find their real words.

    Pass 2 — interpolate timestamps for unmatched lines.  Each unmatched line's
    timestamp is linearly interpolated from the nearest matched anchor lines
    before and after it, so the display timing is still musically reasonable
    rather than clustered at the wrong position.
    """
    if not timed_words:
        return []

    n_words = len(timed_words)
    norm_trans = [_normalize_token(w["token"]) for w in timed_words]

    # ── Pass 1: greedy match with cursor freeze on poor scores ──────────────
    _SEARCH_WINDOW = 100  # words ahead to search; large enough for instrumental gaps

    anchors: list[dict] = []  # text, start, end, matched (bool)
    word_cursor = 0

    for line in uslt_lines:
        line_tokens = [
            _normalize_token(t)
            for t in re.findall(r"[A-Za-z']+", line)
            if t.strip()
        ]

        # Lines with no alphabetic content (e.g. pure symbols / music notes)
        # are left unmatched so cursor isn't perturbed.
        if not line_tokens or word_cursor >= n_words:
            anchors.append({"text": line, "start": -1.0, "end": -1.0, "matched": False})
            continue

        n_line = len(line_tokens)
        search_limit = min(n_words - word_cursor, max(n_line * 8, _SEARCH_WINDOW))

        start_floor = min(word_cursor, n_words - 1)
        best_score = -1.0
        best_start_idx = start_floor
        best_end_idx = min(start_floor + n_line - 1, n_words - 1)

        for skip in range(search_limit + 1):
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

                sm = difflib.SequenceMatcher(
                    None, line_tokens, norm_trans[start_idx : end_idx + 1]
                )
                ratio = sm.ratio()
                # Weight by coverage: fraction of the USLT line's tokens that
                # were actually matched.  This suppresses spurious matches where
                # only a single stop-word (e.g. "and") aligns while the rest of
                # the line is absent from the transcript.
                matched_from_line = sum(sz for _, _, sz in sm.get_matching_blocks())
                coverage = matched_from_line / n_line
                score = ratio * max(0.5, coverage)
                # Skip penalty: gently prefer earlier matches when quality is equal,
                # but not so strongly that a weak early match beats a strong later one.
                # 0.001 per word: at skip=100 the penalty is only 0.10, so a match
                # scoring 0.72 early cannot beat a match scoring 0.85 at skip=100+.
                score -= skip * 0.001

                if score > best_score:
                    best_score = score
                    best_start_idx = start_idx
                    best_end_idx = end_idx

        if best_score >= _MATCH_MIN_SCORE:
            anchors.append({
                "text": line,
                "start": float(timed_words[best_start_idx]["start"]),
                "end": float(timed_words[best_end_idx]["end"]),
                "matched": True,
            })
            matched_tokens = norm_trans[best_start_idx : best_end_idx + 1]
            logger.debug(
                "  [match  %.2f] @ %6.2fs  USLT=%-60r  TRANS=%r",
                best_score,
                float(timed_words[best_start_idx]["start"]),
                " ".join(line_tokens),
                " ".join(matched_tokens),
            )
            word_cursor = min(best_end_idx + 1, n_words)
        else:
            # No confident match — freeze cursor so downstream lines can still
            # find their real words.
            anchors.append({"text": line, "start": -1.0, "end": -1.0, "matched": False})
            # Show the best (rejected) candidate so we can see what Whisper heard.
            if best_score > 0:
                rejected_tokens = norm_trans[best_start_idx : best_end_idx + 1]
                logger.debug(
                    "  [NO-MATCH %.2f] best @ %6.2fs USLT=%-60r  TRANS=%r",
                    best_score,
                    float(timed_words[best_start_idx]["start"]),
                    " ".join(line_tokens),
                    " ".join(rejected_tokens),
                )
            else:
                logger.debug("  [NO-MATCH ----] no candidates USLT=%r", " ".join(line_tokens))

    # ── Pass 2: interpolate timestamps for unmatched lines ──────────────────
    matched_pairs = [(i, a["start"]) for i, a in enumerate(anchors) if a["matched"]]

    if not matched_pairs:
        # Absolute fallback: spread evenly over audio duration.
        total_dur = float(timed_words[-1]["end"])
        n = max(1, len(anchors))
        for k, a in enumerate(anchors):
            a["start"] = total_dur * k / n
            a["end"] = total_dur * (k + 1) / n
        return [{"text": a["text"], "start": a["start"], "end": a["end"]} for a in anchors]

    for i, anchor in enumerate(anchors):
        if anchor["matched"]:
            continue

        # Find nearest matched anchor before and after position i.
        prev_match: tuple[int, float] | None = None
        next_match: tuple[int, float] | None = None
        for j, t in matched_pairs:
            if j < i:
                prev_match = (j, t)   # keep updating → gets the closest one before i
            elif j > i and next_match is None:
                next_match = (j, t)   # first one after i
                break

        if prev_match is None and next_match is None:
            anchor["start"] = float(timed_words[0]["start"])
        elif prev_match is None:
            # Leading unmatched line(s) before the first anchored line.
            # Use the audio-detected vocal onset as a synthetic anchor at index -1
            # so we interpolate forward from the true vocal start instead of
            # backing up linearly from the first matched line (which produces
            # wildly wrong starts when the first matched line is many seconds
            # into the song, e.g. line 3 anchored at 34s would push line 1 to ~31s
            # via backward linear extrapolation).
            nj, nt = next_match  # type: ignore[misc]
            if audio_onset is not None and audio_onset < nt:
                # Interpolate between (i = -1, t = audio_onset) and (nj, nt).
                frac = (i - (-1)) / max(1, nj - (-1))
                anchor["start"] = audio_onset + frac * (nt - audio_onset)
            else:
                anchor["start"] = max(0.0, nt - (nj - i) * 1.5)
        elif next_match is None:
            pj, pt = prev_match  # type: ignore[misc]
            anchor["start"] = pt + (i - pj) * 2.0
        else:
            pj, pt = prev_match  # type: ignore[misc]
            nj, nt = next_match  # type: ignore[misc]
            frac = (i - pj) / max(1, nj - pj)
            anchor["start"] = pt + frac * (nt - pt)

        anchor["end"] = anchor["start"] + 2.0

    return [{"text": a["text"], "start": a["start"], "end": a["end"]} for a in anchors]


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


def _vad_clip_pre_lyric_audio(audio, sr: int = 16000) -> tuple:
    """Clip a short pre-lyric speech burst from the start of the audio.

    Some tracks have garbled speech, DJ drops, or other voice-like content
    before the actual song lyrics begin.  This fools the forced aligner into
    anchoring the first lyric lines there instead of at the true vocal entry.

    Strategy: run Silero VAD, detect the first contiguous speech cluster.
    If that cluster is short (<10 s) AND followed by a long silence gap (>5 s
    of near-zero VAD activity), it is almost certainly a pre-lyric artifact —
    clip the audio so the aligner cannot match against it.

    Returns ``(audio_slice, offset_seconds)``.  When no clip is needed,
    ``offset_seconds`` is 0.0 and the original array is returned unchanged.
    """
    try:
        import torch
        from stable_whisper.stabilization.silero_vad import load_silero_vad_model, compute_vad_probs
    except Exception:
        return audio, 0.0

    window = 512
    vad_threshold = 0.4
    silence_threshold = 0.1

    try:
        vad_model, _ = load_silero_vad_model(verbose=False)
        import numpy as np
        audio_tensor = torch.from_numpy(np.asarray(audio)).float()
        probs = compute_vad_probs(vad_model, audio_tensor, sr, window, progress=False)
    except Exception as exc:
        logger.debug("VAD clip skipped: %s", exc)
        return audio, 0.0

    frame_dur = window / sr

    # Find first speech frame.
    first_speech = None
    for i, p in enumerate(probs):
        if p > vad_threshold:
            first_speech = i
            break

    if first_speech is None:
        return audio, 0.0

    # Extend to find the end of the first speech cluster (allow gaps ≤ 10 frames ≈ 320 ms).
    cluster_end = first_speech
    silence_run = 0
    for i in range(first_speech, len(probs)):
        if probs[i] > vad_threshold:
            cluster_end = i
            silence_run = 0
        else:
            silence_run += 1
            if silence_run >= 10:
                break

    cluster_start_s = first_speech * frame_dur
    cluster_end_s = cluster_end * frame_dur
    cluster_dur = cluster_end_s - cluster_start_s

    # Count how long the silence after the cluster lasts (up to 20 s ahead).
    silence_frames = 0
    max_look = min(len(probs), cluster_end + int(20.0 / frame_dur))
    for i in range(cluster_end + 1, max_look):
        if probs[i] < silence_threshold:
            silence_frames += 1
        else:
            break
    silence_after_s = silence_frames * frame_dur

    logger.debug(
        "VAD pre-lyric check: cluster %.2f–%.2f s (%.1f s), silence after %.1f s",
        cluster_start_s, cluster_end_s, cluster_dur, silence_after_s,
    )

    MAX_CLUSTER = 10.0   # s — clusters longer than this are likely the actual vocals
    MIN_GAP = 5.0        # s — gap must be this long to confirm it's a pre-lyric burst

    if cluster_dur <= MAX_CLUSTER and silence_after_s >= MIN_GAP:
        clip_at_s = cluster_end_s + 1.0   # 1 s buffer after cluster ends
        clip_sample = int(clip_at_s * sr)
        logger.info(
            "Clipping %.1f s of pre-lyric speech (%.2f–%.2f s) from audio before alignment.",
            clip_at_s, cluster_start_s, cluster_end_s,
        )
        return audio[clip_sample:], clip_at_s

    return audio, 0.0


def _forced_align(
    whisperx,
    audio,
    unsynced_lyrics: str,
    device: str,
    language: str,
    *,
    model_name: str,
    compute_type: str,
    align_options: Optional[dict] = None,
    align_method: str = "auto",
):
    """Align USLT lyrics to audio using stable-ts forced alignment.

    Strategy: pass the exact USLT text to stable_whisper.align(), which performs
    true forced alignment — it knows the words and only finds when they occur in
    the audio.  This is far more accurate than transcription + fuzzy matching
    because it does not invent words, cannot drift, and correctly handles
    separator bleed-through (it ignores audio that doesn't match the text).

    ``original_split=True`` preserves the original line-break structure so each
    output segment corresponds 1-to-1 with an input USLT line.

    Forced alignment does, however, fail badly when the lyric text and the audio
    disagree — abbreviated lyric sheets, missing verses, long instrumental
    breaks, or lyrics for a different mix of the song.  In those cases it runs
    out of text before it runs out of audio and crams every remaining line into
    the first half of the track.  ``align_method="auto"`` detects that outcome
    and re-derives the timing from a transcription instead.
    """
    import stable_whisper

    lines = [line.strip() for line in unsynced_lyrics.splitlines() if line.strip()]
    if not lines:
        logger.warning("No lyrics text provided for alignment.")
        return []

    audio_duration = len(audio) / 16000.0
    logger.debug("Audio duration: %.1f s, %d USLT lines", audio_duration, len(lines))

    if align_method in ("transcribe_match", "transcribe_greedy"):
        logger.info("Using transcribe-and-match alignment (align_method=%s).", align_method)
        return _forced_align_transcription_fallback(
            audio,
            lines,
            device,
            language,
            model_name,
            matcher="greedy" if align_method == "transcribe_greedy" else "global",
        )

    # Load (or reuse cached) model.
    cache_key = (model_name, device)
    model = _STABLE_TS_MODEL_CACHE.get(cache_key)
    if model is None:
        logger.info("Loading stable-ts model '%s' on %s…", model_name, device)
        model = stable_whisper.load_model(model_name, device=device)
        _STABLE_TS_MODEL_CACHE[cache_key] = model

    # Join USLT lines with newlines so original_split=True produces one segment per line.
    lyrics_text = "\n".join(lines)
    logger.info("Running stable-ts forced alignment on %d lyric lines…", len(lines))

    # Clip any pre-lyric speech burst to prevent the aligner from anchoring
    # the first lyric lines to early voice-like artifacts in the intro.
    audio_to_align, time_offset = _vad_clip_pre_lyric_audio(audio)

    options = dict(DEFAULT_ALIGN_OPTIONS)
    if align_options:
        options.update(align_options)
    logger.debug("Alignment options: %s", options)

    try:
        result = stable_whisper.alignment.align(
            model,
            audio_to_align,
            lyrics_text,
            language=language,
            verbose=None,
            **options,
        )
    except Exception as exc:
        logger.warning("stable_whisper.align failed (%s); falling back to transcription.", exc)
        return _forced_align_transcription_fallback(audio, lines, device, language, model_name)

    if result is None or not result.segments:
        logger.warning("stable_whisper.align returned no segments; falling back.")
        return _forced_align_transcription_fallback(audio, lines, device, language, model_name)

    # Convert WhisperResult segments → our standard list-of-dicts format.
    # The number of segments should equal the number of lyric lines when
    # original_split=True, but guard against count mismatch just in case.
    segments_out: list[dict] = []
    for seg in result.segments:
        segments_out.append({
            "text": seg.text.strip(),
            "start": float(seg.start),
            "end": float(seg.end),
        })

    logger.debug(
        "stable-ts forced alignment produced %d segments for %d USLT lines.",
        len(segments_out),
        len(lines),
    )

    # If segment count matches line count exactly, restore the original USLT line
    # text (forced alignment may slightly rephrase; we trust the source lyrics).
    if len(segments_out) == len(lines):
        for seg, line in zip(segments_out, lines):
            seg["text"] = line
    else:
        logger.warning(
            "Segment count mismatch: %d segments vs %d lyric lines; "
            "using aligned text as-is.",
            len(segments_out),
            len(lines),
        )

    if logger.isEnabledFor(logging.DEBUG):
        for seg in segments_out[:15]:
            logger.debug("  %6.2fs  %r", seg["start"], seg["text"])

    # Add back the clip offset so timestamps are relative to the original audio.
    if time_offset > 0.0:
        for seg in segments_out:
            seg["start"] += time_offset
            seg["end"] += time_offset

    if align_method == "auto":
        signals = _alignment_failure_signals(result, segments_out, audio)
        if _looks_like_failed_alignment(signals):
            logger.info(
                "Forced alignment looks unreliable (%s); "
                "re-deriving timing from a transcription.",
                ", ".join(f"{k}={v:.2f}" for k, v in signals.items()),
            )
            fallback_stats: dict = {}
            fallback = _forced_align_transcription_fallback(
                audio, lines, device, language, model_name, stats=fallback_stats
            )
            segments_out = _choose_better_placement(
                segments_out,
                signals,
                fallback,
                fallback_stats,
                audio_duration=len(audio) / 16000.0 if audio is not None else 0.0,
            )
        else:
            logger.debug(
                "Alignment quality signals: %s",
                ", ".join(f"{k}={v:.2f}" for k, v in signals.items()),
            )

    return segments_out


def _choose_better_placement(
    aligned: list[dict],
    aligned_signals: dict[str, float],
    fallback: list[dict],
    fallback_stats: dict[str, float],
    *,
    audio_duration: float,
) -> list[dict]:
    """Keep whichever of the two candidate placements looks less broken.

    The forced alignment is only discarded when the transcription-derived
    result is demonstrably healthier.  Without this check a shaky alignment
    could be replaced by an outright guess — which happens on songs Whisper
    transcribes badly (overlapping duets, non-English lyrics), where the
    transcript simply has nothing to anchor the lyric lines to.
    """
    if not fallback:
        logger.info("Transcription fallback produced nothing; keeping alignment.")
        return aligned

    anchored = float(fallback_stats.get("anchored_ratio", 0.0))
    approximate = bool(fallback_stats.get("approximate", 1.0))

    if approximate or anchored < _MIN_FALLBACK_ANCHORED_RATIO:
        logger.info(
            "Transcription fallback anchored only %.0f%% of lines; "
            "keeping the forced alignment.",
            anchored * 100.0,
        )
        return aligned

    aligned_badness = _placement_badness(
        aligned,
        audio_duration,
        unplaced_word_ratio=aligned_signals.get("unplaced_word_ratio", 0.0),
    )
    fallback_badness = _placement_badness(
        fallback, audio_duration, unanchored_ratio=1.0 - anchored
    )

    if fallback_badness < aligned_badness:
        logger.info(
            "Using transcription-derived timing (badness %.2f vs %.2f, "
            "%.0f%% of lines anchored).",
            fallback_badness,
            aligned_badness,
            anchored * 100.0,
        )
        return fallback

    logger.info(
        "Transcription fallback scored no better (badness %.2f vs %.2f); "
        "keeping the forced alignment.",
        fallback_badness,
        aligned_badness,
    )
    return aligned


#: A word the aligner could not place at all gets (almost) zero duration.
_ZERO_WORD_DURATION = 0.02

#: Two consecutive lines starting this close together did not really get
#: separate timing; the aligner emitted them at the same instant.
_COLLAPSED_LINE_GAP = 0.05

#: Fraction of unplaced words above which forced alignment is not trustworthy.
_MAX_UNPLACED_WORD_RATIO = 0.10

#: Fraction of collapsed lines above which forced alignment is not trustworthy.
_MAX_COLLAPSED_LINE_RATIO = 0.10

#: Fraction of the audio that may pass after the last aligned line before the
#: result is judged to have run out of text early.  Songs legitimately end with
#: an instrumental outro, so this is deliberately generous.
_MAX_TAIL_FRACTION = 0.25


def _alignment_failure_signals(result, segments: list[dict], audio) -> dict[str, float]:
    """Reference-free indicators that forced alignment went wrong.

    When the lyric sheet and the audio disagree, the aligner consumes all its
    text too early: the leftover words get zero duration, consecutive lines
    collapse onto the same timestamp, and the aligned lines stop well before
    the end of the recording.  All three are observable without ground truth.
    """
    unplaced = 0
    total_words = 0
    for segment in getattr(result, "segments", []) or []:
        for word in getattr(segment, "words", None) or []:
            start = getattr(word, "start", None)
            end = getattr(word, "end", None)
            if start is None or end is None:
                continue
            total_words += 1
            if float(end) - float(start) <= _ZERO_WORD_DURATION:
                unplaced += 1

    collapsed = sum(
        1
        for previous, current in zip(segments, segments[1:])
        if current["start"] - previous["start"] <= _COLLAPSED_LINE_GAP
    )

    audio_duration = len(audio) / 16000.0 if audio is not None else 0.0
    last_end = max((float(s["end"]) for s in segments), default=0.0)
    tail_fraction = (
        max(0.0, audio_duration - last_end) / audio_duration
        if audio_duration > 0
        else 0.0
    )

    return {
        "unplaced_word_ratio": unplaced / total_words if total_words else 0.0,
        "collapsed_line_ratio": collapsed / max(1, len(segments) - 1),
        "tail_fraction": tail_fraction,
    }


def _looks_like_failed_alignment(signals: dict[str, float]) -> bool:
    """Decide whether to discard a forced-alignment result.

    A large unaligned tail on its own is not enough — plenty of songs end with
    a long instrumental outro — so it only counts when the aligner also shows
    signs of having run out of text (unplaced words or collapsed lines).
    """
    unplaced = signals.get("unplaced_word_ratio", 0.0)
    collapsed = signals.get("collapsed_line_ratio", 0.0)
    tail = signals.get("tail_fraction", 0.0)

    if unplaced >= _MAX_UNPLACED_WORD_RATIO:
        return True
    if collapsed >= _MAX_COLLAPSED_LINE_RATIO:
        return True
    return tail >= _MAX_TAIL_FRACTION and (unplaced >= 0.03 or collapsed >= 0.03)


#: A transcription-derived result whose lines mostly could not be matched to
#: transcript words is guesswork; the forced alignment is preferred over it
#: even when the alignment itself looked shaky.
_MIN_FALLBACK_ANCHORED_RATIO = 0.55


def _placement_badness(
    segments: list[dict],
    audio_duration: float,
    *,
    unplaced_word_ratio: float = 0.0,
    unanchored_ratio: float = 0.0,
) -> float:
    """Reference-free penalty score for a set of line timings (lower is better).

    Used to compare two candidate placements for the same song — a forced
    alignment and a transcription-derived one — without ground truth.  Every
    term measures a way a result can be self-evidently wrong: lines stacked on
    the same instant, lines running backwards, timing that stops long before
    the audio does, or words/lines that were never really placed at all.
    """
    if not segments:
        return float("inf")

    pairs = list(zip(segments, segments[1:]))
    divisor = max(1, len(pairs))
    collapsed = sum(
        1 for a, b in pairs if float(b["start"]) - float(a["start"]) <= _COLLAPSED_LINE_GAP
    ) / divisor
    backwards = sum(
        1 for a, b in pairs if float(b["start"]) < float(a["start"]) - 1e-6
    ) / divisor

    last_end = max((float(s["end"]) for s in segments), default=0.0)
    tail = (
        max(0.0, audio_duration - last_end) / audio_duration
        if audio_duration > 0
        else 0.0
    )
    excess_tail = max(0.0, tail - _MAX_TAIL_FRACTION) / (1.0 - _MAX_TAIL_FRACTION)

    return (
        3.0 * collapsed
        + 3.0 * backwards
        + 2.0 * unplaced_word_ratio
        + 2.0 * unanchored_ratio
        + 1.0 * excess_tail
    )


def _forced_align_transcription_fallback(
    audio,
    lines: list[str],
    device: str,
    language: str,
    model_name: str,
    *,
    matcher: str = "global",
    stats: dict | None = None,
) -> list[dict]:
    """Derive line timing by transcribing the audio and matching the lyric text.

    Used when forced alignment is unavailable or produced a broken result.
    Transcription word timings are independent of the lyric sheet, so sections
    the sheet does not cover cannot push the remaining lines off position.

    ``matcher`` selects the line-to-transcript mapping: ``"global"`` runs a
    single global token alignment, ``"greedy"`` keeps the older per-line
    forward search (retained for benchmarking).

    ``stats``, when given, is filled with reference-free quality indicators for
    the result so the caller can decide whether to trust it.
    """
    if stats is not None:
        stats.setdefault("anchored_ratio", 0.0)
        stats.setdefault("approximate", 1.0)

    raw_segments = _stable_ts_transcribe(audio, model_name, device, language)

    if _looks_probably_instrumental(raw_segments):
        logger.warning(
            "Transcription is very sparse or instrumental; "
            "assigning approximate timestamps to %d USLT lines.",
            len(lines),
        )
        if raw_segments:
            return _seed_segments_by_coarse_durations(lines, raw_segments)
        total_dur = len(audio) / 16000.0
        step = total_dur / max(1, len(lines))
        return [
            {"text": line, "start": i * step, "end": (i + 1) * step}
            for i, line in enumerate(lines)
        ]

    timed_words = _extract_all_timed_words(raw_segments)
    audio_onset = _estimate_vocal_onset_from_audio(audio)

    if audio_onset is not None and timed_words:
        first_word_start = timed_words[0]["start"]
        if audio_onset - first_word_start > 5.0:
            cutoff = audio_onset - 3.0
            kept = [w for w in timed_words if w["start"] >= cutoff]
            if len(kept) >= 5:
                logger.info(
                    "Onset %.1fs: discarding %d pre-onset phantom words.",
                    audio_onset, len(timed_words) - len(kept),
                )
                timed_words = kept

    if len(timed_words) >= 5:
        if matcher == "greedy":
            result = _align_uslt_to_transcribed_words(
                lines, timed_words, audio_onset=audio_onset
            )
        else:
            result = _align_lines_by_global_token_match(
                lines, timed_words, audio_onset=audio_onset, stats=stats
            )
        if result:
            if stats is not None:
                stats["approximate"] = 0.0
            floor = audio_onset if audio_onset is not None else timed_words[0]["start"]
            return _apply_intro_onset_floor(result, floor)

    coarse = raw_segments
    segments = _seed_segments_from_coarse_alignment(lines, coarse)
    if len(segments) != len(lines):
        segments = _seed_segments_by_coarse_durations(lines, coarse)
    vocal_start = float(coarse[0]["start"]) if coarse else 0.0
    return _apply_intro_onset_floor(segments, vocal_start)


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
