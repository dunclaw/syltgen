"""Executes syltgen code paths over benchmark files and caches the output.

The runner deliberately mirrors :mod:`syltgen.processor` rather than calling it:
``process_song`` short-circuits on files that already have SYLT (which every
ground-truth file does), and it writes tags to disk.  The benchmark needs the
*timing* the pipeline would produce, with no file mutation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

#: Code paths under test.
PATH_USLT = "uslt"
PATH_USLT_STEMS = "uslt_stems"
PATH_TRANSCRIBE = "transcribe"
ALL_PATHS = (PATH_USLT, PATH_USLT_STEMS, PATH_TRANSCRIBE)


@dataclass
class RunConfig:
    """Model/device settings shared by every benchmark run."""

    whisper_model: str = "large-v2"
    device: str = "auto"
    compute_type: str = "float16"
    language: str = "en"
    sep_model: str = "UVR-MDX-NET-Voc_FT.onnx"
    #: Overrides merged over ``syltgen.transcriber.DEFAULT_ALIGN_OPTIONS``.
    align_options: Optional[dict] = None
    #: ``auto`` / ``align`` / ``transcribe_match``.
    align_method: str = "auto"


def cache_key(mp3_path: Path, path_name: str, variant: str) -> str:
    digest = hashlib.sha1(str(mp3_path).encode("utf-8")).hexdigest()[:16]
    return f"{variant}__{path_name}__{digest}"


def _cache_file(cache_dir: Path, mp3_path: Path, path_name: str, variant: str) -> Path:
    return cache_dir / f"{cache_key(mp3_path, path_name, variant)}.json"


def run_uslt_path(mp3_path: Path, config: RunConfig) -> list[dict]:
    """Forced alignment of the file's own USLT text against the original audio."""
    from syltgen.tagger import read_uslt_lyrics
    from syltgen.transcriber import transcribe_and_align

    unsynced = read_uslt_lyrics(mp3_path)
    if not unsynced:
        raise ValueError(f"'{mp3_path.name}' has no USLT lyrics")

    return transcribe_and_align(
        mp3_path,
        unsynced_lyrics=unsynced,
        model_name=config.whisper_model,
        device=config.device,
        compute_type=config.compute_type,
        language=config.language,
        align_options=config.align_options,
        align_method=config.align_method,
    )


def run_uslt_stems_path(mp3_path: Path, config: RunConfig) -> list[dict]:
    """Forced alignment of the USLT text against separated vocal stems."""
    from syltgen.separator import separate_vocals
    from syltgen.tagger import read_uslt_lyrics
    from syltgen.transcriber import transcribe_and_align

    unsynced = read_uslt_lyrics(mp3_path)
    if not unsynced:
        raise ValueError(f"'{mp3_path.name}' has no USLT lyrics")

    with tempfile.TemporaryDirectory(prefix="syltgen_bench_stems_") as stems_dir:
        vocals_path = separate_vocals(mp3_path, stems_dir, model_name=config.sep_model)
        return transcribe_and_align(
            vocals_path,
            unsynced_lyrics=unsynced,
            model_name=config.whisper_model,
            device=config.device,
            compute_type=config.compute_type,
            language=config.language,
            align_options=config.align_options,
            align_method=config.align_method,
        )


def run_transcribe_path(mp3_path: Path, config: RunConfig) -> list[dict]:
    """Full transcription of separated vocals, as used when no USLT exists."""
    from syltgen.separator import separate_vocals
    from syltgen.transcriber import transcribe_and_align

    with tempfile.TemporaryDirectory(prefix="syltgen_bench_stems_") as stems_dir:
        vocals_path = separate_vocals(mp3_path, stems_dir, model_name=config.sep_model)
        return transcribe_and_align(
            vocals_path,
            unsynced_lyrics=None,
            model_name=config.whisper_model,
            device=config.device,
            compute_type=config.compute_type,
            language=config.language,
        )


def raw_transcript(mp3_path: Path, config: RunConfig) -> list[dict]:
    """Whisper segments *before* line splitting, including word timings.

    Caching this separately is what makes line-break tuning practical: the
    expensive part (separation + transcription) is identical for every splitter
    variant, so it is paid once and every later experiment is pure CPU.
    """
    from syltgen.separator import separate_vocals
    from syltgen.transcriber import _stable_ts_transcribe

    import whisperx

    with tempfile.TemporaryDirectory(prefix="syltgen_bench_stems_") as stems_dir:
        vocals_path = separate_vocals(mp3_path, stems_dir, model_name=config.sep_model)
        audio = whisperx.load_audio(str(vocals_path))
        return _stable_ts_transcribe(
            audio, config.whisper_model, _resolve_device(config.device), config.language
        )


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def raw_transcript_cached(
    mp3_path: Path,
    config: RunConfig,
    *,
    cache_dir: Path,
    refresh: bool = False,
) -> dict:
    """Cached :func:`raw_transcript`, in the same record shape as `run_path`."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_path = _cache_file(cache_dir, mp3_path, "raw", "transcript")
    if cached_path.exists() and not refresh:
        try:
            return json.loads(cached_path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Ignoring unreadable cache entry '%s'.", cached_path.name)

    started = time.perf_counter()
    try:
        record = {
            "path": str(mp3_path),
            "segments": raw_transcript(mp3_path, config),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - benchmark must survive bad files
        logger.exception("Raw transcription failed for '%s'.", mp3_path.name)
        record = {"path": str(mp3_path), "segments": [], "error": f"{type(exc).__name__}: {exc}"}

    record["elapsed"] = time.perf_counter() - started
    cached_path.write_text(json.dumps(record), encoding="utf-8")
    return record


_RUNNERS = {
    PATH_USLT: run_uslt_path,
    PATH_USLT_STEMS: run_uslt_stems_path,
    PATH_TRANSCRIBE: run_transcribe_path,
}


def run_path(
    mp3_path: Path,
    path_name: str,
    config: RunConfig,
    *,
    cache_dir: Optional[Path] = None,
    variant: str = "baseline",
    refresh: bool = False,
) -> dict:
    """Run one code path for one file, using the on-disk cache when possible.

    Returns a record with ``segments``, ``elapsed`` and ``error`` keys.  Errors
    are captured rather than raised so one bad file cannot abort a long run;
    they are also cached so reruns do not repeatedly pay for a known failure.
    """
    runner = _RUNNERS.get(path_name)
    if runner is None:
        raise ValueError(f"unknown code path '{path_name}'")

    cached_path: Optional[Path] = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_path = _cache_file(cache_dir, mp3_path, path_name, variant)
        if cached_path.exists() and not refresh:
            try:
                return json.loads(cached_path.read_text(encoding="utf-8"))
            except Exception:
                logger.warning("Ignoring unreadable cache entry '%s'.", cached_path.name)

    started = time.perf_counter()
    try:
        segments = runner(mp3_path, config)
        record = {
            "path": str(mp3_path),
            "code_path": path_name,
            "variant": variant,
            "segments": [
                {
                    "text": str(seg.get("text", "")),
                    "start": float(seg["start"]),
                    "end": float(seg.get("end", seg["start"])),
                }
                for seg in segments
            ],
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - benchmark must survive bad files
        logger.exception("Path '%s' failed for '%s'.", path_name, mp3_path.name)
        record = {
            "path": str(mp3_path),
            "code_path": path_name,
            "variant": variant,
            "segments": [],
            "error": f"{type(exc).__name__}: {exc}",
        }

    record["elapsed"] = time.perf_counter() - started

    if cached_path is not None:
        cached_path.write_text(json.dumps(record), encoding="utf-8")

    return record
