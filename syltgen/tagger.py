"""Write SYLT (synced lyrics) ID3 tags and .lrc sidecar files."""

import logging
import os
import re
import struct
from io import BytesIO
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# VLC and some other players are unreliable with SYLT/USLT stored in ID3v2.4.
# Keep rewritten lyric tags in ID3v2.3-compatible form for broader playback
# compatibility.
_ID3_SAVE_VERSION = 3
_MAX_COVER_ART_BYTES = 250_000


def _decode_text_bytes(raw_text: bytes, encoding: int) -> str:
    """Decode ID3 text bytes for the given encoding id."""
    try:
        if encoding == 0:
            text = raw_text.decode("latin-1")
        elif encoding == 1:
            text = raw_text.decode("utf-16")
        elif encoding == 2:
            text = raw_text.decode("utf-16-be")
        else:
            text = raw_text.decode("utf-8")
    except Exception:
        text = raw_text.decode("latin-1", errors="replace")
    return text.strip("\x00").strip()


def _skip_encoded_terminator(data: bytes, pos: int, encoding: int) -> int:
    """Advance past a null-terminated ID3 text string."""
    if encoding in (1, 2):
        while pos + 1 < len(data):
            if data[pos] == 0 and data[pos + 1] == 0:
                return pos + 2
            pos += 2
        return len(data)

    while pos < len(data) and data[pos] != 0:
        pos += 1
    return min(pos + 1, len(data))


def _extract_encoded_string(data: bytes, pos: int, encoding: int, *, limit: int) -> tuple[bytes, int]:
    """Read one null-terminated encoded string, stopping before *limit*."""
    text_start = pos
    if encoding in (1, 2):
        text_end = text_start
        while text_end + 1 < limit:
            if data[text_end] == 0 and data[text_end + 1] == 0:
                break
            text_end += 2
        return data[text_start:text_end], min(text_end + 2, len(data))

    text_end = text_start
    while text_end < limit and data[text_end] != 0:
        text_end += 1
    return data[text_start:text_end], min(text_end + 1, len(data))


def _read_id3_tag_data(mp3_path: Path) -> Optional[tuple[int, bytes]]:
    """Return ``(major_version, tag_data)`` for the ID3 tag, if present."""
    try:
        data = mp3_path.read_bytes()
    except OSError:
        return None

    if len(data) < 10 or data[:3] != b"ID3":
        return None

    major_ver = data[3]
    sz = data[6:10]
    tag_size = (sz[0] << 21) | (sz[1] << 14) | (sz[2] << 7) | sz[3]
    return major_ver, data[10 : 10 + tag_size]


def _iter_raw_id3_frames(mp3_path: Path):
    """Yield ``(frame_id, body)`` pairs from the raw ID3 tag."""
    tag_info = _read_id3_tag_data(mp3_path)
    if not tag_info:
        return

    major_ver, tag_data = tag_info
    pos = 0
    while pos + 10 <= len(tag_data):
        frame_id = tag_data[pos : pos + 4]
        if frame_id == b"\x00\x00\x00\x00":
            break
        if major_ver == 4:
            b = tag_data[pos + 4 : pos + 8]
            frame_size = (b[0] << 21) | (b[1] << 14) | (b[2] << 7) | b[3]
        else:
            frame_size = struct.unpack(">I", tag_data[pos + 4 : pos + 8])[0]
        body = tag_data[pos + 10 : pos + 10 + frame_size]
        yield frame_id, body
        pos += 10 + frame_size
        if frame_size == 0:
            break


# ---------------------------------------------------------------------------
# SYLT tag writer
# ---------------------------------------------------------------------------


def write_uslt_tag(
    mp3_path: str | os.PathLike,
    lyrics: str,
    *,
    language: str = "eng",
    description: str = "Lyrics",
) -> None:
    """Embed a well-formed USLT frame into the ID3 tags of *mp3_path*."""
    try:
        from mutagen.id3 import ID3, USLT, Encoding, ID3NoHeaderError
    except ImportError as exc:
        raise ImportError("mutagen is not installed. Run: pip install mutagen") from exc

    mp3_path = Path(mp3_path)

    try:
        tags = ID3(str(mp3_path))
    except ID3NoHeaderError:
        from mutagen.id3 import ID3
        tags = ID3()

    tags.setall(
        "USLT",
        [
            USLT(
                encoding=Encoding.UTF16,
                lang=language,
                desc=description,
                text=lyrics,
            )
        ],
    )
    tags.save(str(mp3_path), v2_version=_ID3_SAVE_VERSION)
    logger.info("USLT tag written to '%s' (%d chars).", mp3_path.name, len(lyrics))

def write_sylt_tag(mp3_path: str | os.PathLike, segments: list[dict], language: str = "eng") -> None:
    """
    Embed a SYLT frame into the ID3 tags of *mp3_path*.

    Parameters
    ----------
    mp3_path:
        Path to the MP3 file to tag (typically the *output* copy).
    segments:
        List of segment dicts with at minimum ``"start"`` (seconds) and
        ``"text"`` keys, as returned by :func:`~syltgen.transcriber.transcribe_and_align`.
    language:
        ISO 639-2 three-letter language code (default ``"eng"``).
    """
    try:
        from mutagen.id3 import ID3, SYLT, Encoding, ID3NoHeaderError
    except ImportError as exc:
        raise ImportError("mutagen is not installed. Run: pip install mutagen") from exc

    mp3_path = Path(mp3_path)

    sync_data = _segments_to_sylt(segments)

    try:
        tags = ID3(str(mp3_path))
    except ID3NoHeaderError:
        from mutagen.id3 import ID3
        tags = ID3()

    tags.setall(
        "SYLT",
        [
            SYLT(
                encoding=Encoding.UTF16,
                lang=language,
                format=2,   # 2 = milliseconds
                type=1,     # 1 = lyrics
                text=sync_data,
            )
        ],
    )
    tags.save(str(mp3_path), v2_version=_ID3_SAVE_VERSION)
    logger.info("SYLT tag written to '%s' (%d lines).", mp3_path.name, len(sync_data))


def _segments_to_sylt(segments: list[dict]) -> list[tuple[str, int]]:
    """Convert WhisperX segment list to SYLT ``(text, ms)`` tuples."""
    result = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        ms_start = int(seg["start"] * 1000)
        result.append((text, ms_start))
    return result


def segments_to_plain_lyrics(segments: list[dict]) -> str:
    """Convert timed segments to plain multi-line lyrics text for USLT."""
    lines = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def _resample_cover_art_to_size(
    image_bytes: bytes,
    *,
    max_bytes: int,
) -> tuple[bytes, str] | None:
    """Return resized JPEG bytes that fit ``max_bytes``, or ``None`` if unchanged."""
    try:
        from PIL import Image
    except ImportError:
        logger.warning("Pillow is required to resample oversized cover art (pip install pillow).")
        return None

    try:
        with Image.open(BytesIO(image_bytes)) as img:
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            elif img.mode == "L":
                img = img.convert("RGB")

            resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
            max_sides = (1800, 1600, 1400, 1200, 1000, 900, 800, 700, 600, 500, 400)
            qualities = (90, 84, 78, 72, 68, 64, 60, 56, 52)

            best: bytes | None = None
            for max_side in max_sides:
                work = img.copy()
                work.thumbnail((max_side, max_side), resampling)

                for quality in qualities:
                    out = BytesIO()
                    work.save(out, format="JPEG", quality=quality, optimize=True, progressive=True)
                    candidate = out.getvalue()

                    if best is None or len(candidate) < len(best):
                        best = candidate

                    if len(candidate) <= max_bytes:
                        return candidate, "image/jpeg"

            if best is not None and len(best) < len(image_bytes):
                return best, "image/jpeg"
            return None
    except Exception:
        logger.exception("Could not resample cover art image bytes.")
        return None


def shrink_large_cover_art(
    mp3_path: str | os.PathLike,
    *,
    max_bytes: int = _MAX_COVER_ART_BYTES,
) -> bool:
    """Shrink APIC cover art when larger than ``max_bytes``.

    Returns ``True`` if any cover art frame was replaced with a smaller image.
    """
    try:
        from mutagen.id3 import ID3
    except ImportError as exc:
        raise ImportError("mutagen is not installed. Run: pip install mutagen") from exc

    mp3_path = Path(mp3_path)
    try:
        tags = ID3(str(mp3_path))
    except Exception:
        return False

    apic_frames = tags.getall("APIC")
    if not apic_frames:
        return False

    changed = False
    for frame in apic_frames:
        payload = getattr(frame, "data", b"") or b""
        if len(payload) <= max_bytes:
            continue

        resized = _resample_cover_art_to_size(payload, max_bytes=max_bytes)
        if not resized:
            continue

        resized_bytes, mime = resized
        if len(resized_bytes) >= len(payload):
            continue

        frame.data = resized_bytes
        frame.mime = mime
        changed = True
        logger.info(
            "Shrunk cover art in '%s' from %d KB to %d KB.",
            mp3_path.name,
            len(payload) // 1024,
            len(resized_bytes) // 1024,
        )

    if changed:
        tags.save(str(mp3_path), v2_version=_ID3_SAVE_VERSION)
    return changed


# ---------------------------------------------------------------------------
# .lrc file writer
# ---------------------------------------------------------------------------

def write_lrc_file(lrc_path: str | os.PathLike, segments: list[dict]) -> None:
    """
    Write a standard ``.lrc`` sidecar file alongside the audio file.

    LRC format: ``[MM:SS.xx] lyric line``
    """
    lrc_path = Path(lrc_path)
    lines = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        timestamp = _seconds_to_lrc_timestamp(seg["start"])
        lines.append(f"[{timestamp}]{text}")

    lrc_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("LRC file written to '%s'.", lrc_path.name)


def read_lrc_file(lrc_path: str | os.PathLike) -> Optional[list[dict]]:
    """Read timed LRC lines and return segments with ``start`` and ``text``."""
    lrc_path = Path(lrc_path)
    if not lrc_path.exists():
        return None

    segments: list[dict] = []
    time_pat = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]")

    for raw_line in lrc_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        stamps = list(time_pat.finditer(line))
        if not stamps:
            continue

        text = time_pat.sub("", line).strip()
        if not text:
            continue

        for stamp in stamps:
            minutes = int(stamp.group(1))
            seconds = float(stamp.group(2))
            start = (minutes * 60) + seconds
            segments.append({"start": start, "text": text})

    if not segments:
        logger.debug("No timed lyric lines found in LRC '%s'.", lrc_path.name)
        return None

    segments.sort(key=lambda s: s["start"])
    logger.debug("Read %d timed lines from LRC '%s'.", len(segments), lrc_path.name)
    return segments


def _seconds_to_lrc_timestamp(seconds: float) -> str:
    """Convert fractional seconds to ``MM:SS.xx`` LRC timestamp string."""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_uslt_lyrics(mp3_path: str | os.PathLike) -> Optional[str]:
    """
    Return unsynced lyrics from the USLT ID3 tag, or ``None`` if not present.
    """
    try:
        from mutagen.id3 import ID3
    except ImportError:
        return None

    try:
        tags = ID3(str(mp3_path))
    except Exception:
        return None

    uslt_frames = tags.getall("USLT")
    if uslt_frames:
        text = uslt_frames[0].text
        logger.debug("Found USLT lyrics in '%s' (%d chars).", Path(mp3_path).name, len(text))
        return text

    raw_text = _read_uslt_raw(Path(mp3_path))
    if raw_text:
        logger.debug("Found raw USLT lyrics in '%s' (%d chars).", Path(mp3_path).name, len(raw_text))
        return raw_text

    logger.debug(
        "No USLT frame in '%s'. Available frames: %s",
        Path(mp3_path).name,
        ", ".join(sorted(tags.keys())) or "(none)",
    )
    return None


def read_sylt_tag(mp3_path: str | os.PathLike) -> Optional[list[dict]]:
    """Return SYLT data as a segments list (dicts with ``start`` in seconds and
    ``text``), or ``None`` if no SYLT frame is present.

    Tries mutagen first; falls back to raw ID3 byte parsing for frames that
    mutagen silently drops (e.g. UTF-16 SYLT written by Windows Media Player).
    """
    mp3_path = Path(mp3_path)

    # --- mutagen fast path ---------------------------------------------------
    try:
        from mutagen.id3 import ID3
        tags = ID3(str(mp3_path))
        frames = tags.getall("SYLT")
        if frames:
            result = []
            for frame in frames:
                for text, ms in frame.text:
                    t = text.strip()
                    if t:
                        result.append({"start": ms / 1000.0, "text": t})
            if result:
                return sorted(result, key=lambda s: s["start"])
    except Exception:
        pass

    # --- raw fallback --------------------------------------------------------
    return _read_sylt_raw(mp3_path)


def _read_sylt_raw(mp3_path: Path) -> Optional[list[dict]]:
    """Parse the first SYLT frame directly from raw ID3 bytes."""
    for frame_id, body in _iter_raw_id3_frames(mp3_path):
        if frame_id == b"SYLT" and len(body) >= 6:
            segments = _decode_sylt_body(body)
            if segments:
                return segments

    return None


def _decode_sylt_body(body: bytes) -> Optional[list[dict]]:
    """Decode raw SYLT frame body into a segments list."""
    encoding = body[0]
    # body[1:4] = language, body[4] = timestamp_fmt, body[5] = content_type

    # Skip the description string (null-terminated, encoding-aware)
    pos = _skip_encoded_terminator(body, 6, encoding)

    result = []
    while pos + 4 < len(body):
        # Read null-terminated text entry
        raw_text, pos = _extract_encoded_string(body, pos, encoding, limit=len(body) - 4)

        if pos + 4 > len(body):
            break

        ms = struct.unpack(">I", body[pos: pos + 4])[0]
        pos += 4

        text = _decode_text_bytes(raw_text, encoding)
        if text:
            result.append({"start": ms / 1000.0, "text": text})

    return result if result else None


def _read_uslt_raw(mp3_path: Path) -> Optional[str]:
    """Parse the first USLT frame directly from raw ID3 bytes."""
    for frame_id, body in _iter_raw_id3_frames(mp3_path):
        if frame_id == b"USLT" and len(body) >= 4:
            text = _decode_uslt_body(body)
            if text:
                return text
    return None


def _decode_uslt_body(body: bytes) -> Optional[str]:
    """Decode raw USLT frame body into a lyrics string."""
    encoding = body[0]
    pos = _skip_encoded_terminator(body, 4, encoding)
    if pos > len(body):
        return None
    text = _decode_text_bytes(body[pos:], encoding)
    return text or None


def has_sylt_tag(mp3_path: str | os.PathLike) -> bool:
    """Return ``True`` if *mp3_path* already contains a SYLT frame (parseable
    or not). Uses ``read_sylt_tag`` which includes a raw byte fallback."""
    return read_sylt_tag(mp3_path) is not None


def has_uslt_tag(mp3_path: str | os.PathLike) -> bool:
    """Return ``True`` if *mp3_path* already contains a USLT frame (parseable
    or not). Uses ``read_uslt_lyrics`` which includes a raw byte fallback."""
    return read_uslt_lyrics(mp3_path) is not None
