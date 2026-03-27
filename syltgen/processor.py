"""Core per-song processing pipeline: separate → transcribe → tag."""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from .separator import separate_vocals, DEFAULT_MODEL as DEFAULT_SEP_MODEL
from .transcriber import transcribe_and_align, DEFAULT_WHISPER_MODEL, DEFAULT_DEVICE, DEFAULT_COMPUTE_TYPE
from .tagger import (
    write_sylt_tag,
    write_uslt_tag,
    write_lrc_file,
    read_uslt_lyrics,
    read_sylt_tag,
    read_lrc_file,
    segments_to_plain_lyrics,
    shrink_large_cover_art,
)

logger = logging.getLogger(__name__)

_MAX_COVER_ART_BYTES = 250_000


def _is_viable_manual_lrc(segments: list[dict]) -> bool:
    """Return True when manual LRC has enough lyric substance to trust."""
    if len(segments) >= 2:
        return True

    alpha_chars = 0
    for seg in segments:
        text = str(seg.get("text", ""))
        alpha_chars += sum(1 for ch in text if ch.isalpha())

    return alpha_chars >= 24


def _is_instrumental_or_classical_genre(mp3_path: Path) -> bool:
    """Return True when metadata strongly indicates non-lyrical content."""
    try:
        from mutagen.id3 import ID3
        tags = ID3(str(mp3_path))
    except Exception:
        return False

    genres: list[str] = []
    for frame in tags.getall("TCON"):
        text = frame.text
        if isinstance(text, str):
            genres.append(text)
        elif isinstance(text, (list, tuple)):
            genres.extend(str(x) for x in text)

    artists: list[str] = []
    for frame in tags.getall("TPE1"):
        text = frame.text
        if isinstance(text, str):
            artists.append(text)
        elif isinstance(text, (list, tuple)):
            artists.extend(str(x) for x in text)

    albums: list[str] = []
    for frame in tags.getall("TALB"):
        text = frame.text
        if isinstance(text, str):
            albums.append(text)
        elif isinstance(text, (list, tuple)):
            albums.extend(str(x) for x in text)

    if not genres:
        return False

    normalized = " | ".join(genres).lower()
    if ("instrumental" in normalized) or ("classical" in normalized):
        return True

    meta_blob = " | ".join(genres + artists + albums).lower()
    if "christmas" in normalized and any(token in meta_blob for token in ("piano", "instrumental", "karaoke", "unknown artist")):
        return True

    return False


def _shrink_cover_art_if_needed(mp3_path: Path) -> None:
    """Shrink oversized cover art on output files for better compatibility."""
    try:
        shrink_large_cover_art(mp3_path, max_bytes=_MAX_COVER_ART_BYTES)
    except Exception:
        logger.exception("Could not normalize cover art for '%s'.", mp3_path.name)


def process_song(
    mp3_path: str | os.PathLike,
    output_dir: str | os.PathLike,
    *,
    force: bool = False,
    write_lrc: bool = True,
    sep_model: str = DEFAULT_SEP_MODEL,
    whisper_model: str = DEFAULT_WHISPER_MODEL,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
    language: str = "en",
) -> Optional[Path]:
    """
    Full pipeline for a single MP3 file.

    1. Copy the MP3 to *output_dir* (skip if it already has SYLT and ``force``
       is ``False``).
    2. Separate vocals with audio-separator / Demucs.
    3. Transcribe and align with WhisperX (using USLT lyrics if available).
    4. Write SYLT tag to the output copy.
    5. Optionally write a ``.lrc`` sidecar file.

    Returns the path to the tagged output file, or ``None`` if the file was
    skipped.
    """
    mp3_path = Path(mp3_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dest_path = output_dir / mp3_path.name
    input_lrc_path = mp3_path.with_suffix(".lrc")

    source_sylt_segments = read_sylt_tag(mp3_path)

    dest_prepared = False
    existing_unsynced = None

    try:
        # --- Guard: skip / re-encode files that already have SYLT / USLT --------
        if not force:
            existing_segments = source_sylt_segments
            existing_unsynced = read_uslt_lyrics(mp3_path)
            if existing_segments is not None:
                # Check whether mutagen could parse it natively (well-formed).
                try:
                    from mutagen.id3 import ID3
                    _mt = ID3(str(mp3_path))
                    _well_formed = bool(_mt.getall("SYLT"))
                    _uslt_well_formed = bool(_mt.getall("USLT"))
                except Exception:
                    _well_formed = False
                    _uslt_well_formed = False

                if _well_formed and (existing_unsynced is None or _uslt_well_formed):
                    logger.info("SKIP '%s' – already has a well-formed SYLT tag.", mp3_path.name)
                    return None

                # Source already has synchronized lyrics. Re-encode malformed lyric
                # frames into a clean output copy instead of recomputing them.
                logger.info(
                    "Re-encoding lyric tags from '%s' (%d SYLT lines).",
                    mp3_path.name, len(existing_segments),
                )
                shutil.copy2(str(mp3_path), str(dest_path))
                write_sylt_tag(dest_path, existing_segments)
                if existing_unsynced is not None:
                    write_uslt_tag(dest_path, existing_unsynced)
                else:
                    write_uslt_tag(dest_path, segments_to_plain_lyrics(existing_segments))
                _shrink_cover_art_if_needed(dest_path)
                stale_lrc = dest_path.with_suffix(".lrc")
                if stale_lrc.exists():
                    stale_lrc.unlink()
                    logger.debug(
                        "Removed stale LRC '%s' for SYLT correctness-only update.",
                        stale_lrc.name,
                    )
                return dest_path

            if existing_unsynced is not None:
                try:
                    from mutagen.id3 import ID3
                    _mt = ID3(str(mp3_path))
                    _uslt_well_formed = bool(_mt.getall("USLT"))
                except Exception:
                    _uslt_well_formed = False

                if not _uslt_well_formed:
                    logger.info("Re-encoding non-standard USLT from '%s'.", mp3_path.name)
                    shutil.copy2(str(mp3_path), str(dest_path))
                    write_uslt_tag(dest_path, existing_unsynced)
                    dest_prepared = True

        # --- Manual correction path: use input-side LRC when source has no SYLT --
        if source_sylt_segments is None and input_lrc_path.exists():
            manual_segments = read_lrc_file(input_lrc_path)
            if manual_segments and _is_viable_manual_lrc(manual_segments):
                logger.info(
                    "Using manual input LRC '%s' for '%s' (source MP3 has no SYLT).",
                    input_lrc_path.name,
                    mp3_path.name,
                )
                shutil.copy2(str(mp3_path), str(dest_path))
                write_sylt_tag(dest_path, manual_segments)
                write_uslt_tag(dest_path, segments_to_plain_lyrics(manual_segments))
                _shrink_cover_art_if_needed(dest_path)
                if write_lrc:
                    shutil.copy2(str(input_lrc_path), str(dest_path.with_suffix(".lrc")))
                return dest_path
            if manual_segments:
                logger.info(
                    "Ignoring low-content manual LRC '%s' for '%s'; continuing with normal pipeline.",
                    input_lrc_path.name,
                    mp3_path.name,
                )

        logger.info("Processing '%s'…", mp3_path.name)

        if _is_instrumental_or_classical_genre(mp3_path):
            logger.info(
                "SKIP '%s' – genre tag indicates instrumental/classical.",
                mp3_path.name,
            )
            if dest_path.exists():
                dest_path.unlink()
                logger.debug("Removed stale output '%s'.", dest_path.name)
            stale_lrc = dest_path.with_suffix(".lrc")
            if stale_lrc.exists():
                stale_lrc.unlink()
                logger.debug("Removed stale LRC '%s'.", stale_lrc.name)
            return None

        # --- Step 0: copy to output directory -----------------------------------
        if not dest_prepared:
            shutil.copy2(str(mp3_path), str(dest_path))
            logger.debug("Copied to '%s'.", dest_path)
        else:
            logger.debug("Using pre-copied patched output '%s'.", dest_path)

        # Separate vocals for all tracks. Using clean vocal stems gives the
        # WhisperX coarse pass (and any full-transcription baseline) much better
        # temporal precision than running on the raw mix — this matters especially
        # for USLT forced-alignment where seed timing feeds directly into line
        # placement.
        unsynced = existing_unsynced if existing_unsynced is not None else read_uslt_lyrics(mp3_path)

        # --- Step 1: stem separation into a temp directory ----------------------
        with tempfile.TemporaryDirectory(prefix="syltgen_stems_") as tmp_stems:
            vocals_path = separate_vocals(mp3_path, tmp_stems, model_name=sep_model)

            if unsynced:
                logger.info("Found USLT lyrics – using forced alignment on separated vocals.")
                segments = transcribe_and_align(
                    vocals_path,
                    unsynced_lyrics=unsynced,
                    model_name=whisper_model,
                    device=device,
                    compute_type=compute_type,
                    language=language,
                )
            else:
                logger.info("No USLT lyrics – using full transcription on separated vocals.")

                # --- Step 2: transcription --------------------------------------
                segments = transcribe_and_align(
                    vocals_path,
                    unsynced_lyrics=None,
                    model_name=whisper_model,
                    device=device,
                    compute_type=compute_type,
                    language=language,
                )

        if not segments:
            logger.info("SKIP '%s' – no credible lyrics detected (instrumental or transcription failed).", mp3_path.name)
            if dest_path.exists():
                dest_path.unlink()
                logger.debug("Removed output copy '%s'.", dest_path.name)
            stale_lrc = dest_path.with_suffix(".lrc")
            if stale_lrc.exists():
                stale_lrc.unlink()
                logger.debug("Removed stale LRC '%s'.", stale_lrc.name)
            return None

        # --- Step 3: write normalized lyric tags to the output copy -------------
        # Always rewrite USLT as well so outputs do not keep legacy/mixed lyric
        # frame encodings from source files.
        write_sylt_tag(dest_path, segments)
        normalized_uslt = existing_unsynced if existing_unsynced is not None else segments_to_plain_lyrics(segments)
        write_uslt_tag(dest_path, normalized_uslt)
        _shrink_cover_art_if_needed(dest_path)

        # --- Step 4: optional .lrc sidecar --------------------------------------
        if write_lrc:
            lrc_path = dest_path.with_suffix(".lrc")
            write_lrc_file(lrc_path, segments)

        logger.info("Done: '%s'.", dest_path.name)
        return dest_path
    except Exception:
        # Never leave a partially processed copied MP3 in the output folder.
        if dest_path.exists():
            try:
                dest_path.unlink()
                logger.debug("Removed partial output '%s' after processing error.", dest_path.name)
            except Exception:
                logger.exception("Could not remove partial output '%s'.", dest_path.name)

        stale_lrc = dest_path.with_suffix(".lrc")
        if stale_lrc.exists():
            try:
                stale_lrc.unlink()
                logger.debug("Removed partial LRC '%s' after processing error.", stale_lrc.name)
            except Exception:
                logger.exception("Could not remove partial LRC '%s'.", stale_lrc.name)
        raise
