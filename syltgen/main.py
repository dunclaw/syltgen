"""
syltgen – CLI entry point.

Usage
-----
    python -m syltgen.main <input_dir> <output_dir> [options]

    # Process a whole directory
    python -m syltgen.main ./music ./music_tagged

    # Force-reprocess files that already have SYLT tags
    python -m syltgen.main ./music ./music_tagged --force

    # Use a more accurate Whisper model, run on CPU
    python -m syltgen.main ./music ./music_tagged --whisper-model medium --device cpu
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import colorlog

from .processor import process_song


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def _check_ffmpeg() -> bool:
    """Return True if ffmpeg is on PATH; log a helpful error and return False otherwise.

    On Windows the process may have been launched before FFmpeg was installed, so
    os.environ['PATH'] can be stale.  We read the current PATH directly from the
    Windows registry as a fallback and, if FFmpeg is found there, refresh
    os.environ['PATH'] for the rest of the process (so audio-separator etc. also work).
    """
    if shutil.which("ffmpeg") is not None:
        return True

    # --- Windows: try refreshing PATH from registry ---------------------------------
    try:
        import winreg  # only available on Windows
        machine_path = user_path = ""
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
        ) as k:
            machine_path, _ = winreg.QueryValueEx(k, "PATH")
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as k:
                user_path, _ = winreg.QueryValueEx(k, "PATH")
        except FileNotFoundError:
            pass
        refreshed_path = machine_path + ";" + user_path
        if shutil.which("ffmpeg", path=refreshed_path) is not None:
            # Propagate the refreshed PATH so subprocesses spawned later also find ffmpeg
            os.environ["PATH"] = refreshed_path
            return True
    except Exception:
        pass

    logging.getLogger(__name__).error(
        "FFmpeg is not installed or not on PATH.\n"
        "Install it with one of the following methods and then re-run syltgen:\n"
        "  winget : winget install --id Gyan.FFmpeg -e --source winget\n"
        "  choco  : choco install ffmpeg\n"
        "  manual : https://ffmpeg.org/download.html\n"
        "After installing, open a new terminal so that PATH is refreshed."
    )
    return False


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)-8s%(reset)s %(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
    )
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)

    # Third-party libraries can emit very noisy connection-level debug logs
    # when verbose mode enables the root logger at DEBUG.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="syltgen",
        description="Generate SYLT tags and .lrc files for MP3 files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing source MP3 files.")
    parser.add_argument("output_dir", type=Path, help="Directory for tagged output MP3s (and .lrc files).")

    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-process files that already contain a SYLT tag.",
    )
    parser.add_argument(
        "--no-lrc",
        action="store_true",
        default=False,
        help="Do not write .lrc sidecar files.",
    )
    parser.add_argument(
        "--sep-model",
        default="UVR-MDX-NET-Voc_FT.onnx",
        metavar="MODEL",
        help="audio-separator model filename for vocal separation (e.g. 'UVR-MDX-NET-Voc_FT.onnx', 'Kim_Vocal_2.onnx').",
    )
    parser.add_argument(
        "--whisper-model",
        default="large-v2",
        metavar="MODEL",
        help="Whisper model size: tiny, base, small, medium, large-v2 (default: large-v2 for best accuracy).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Inference device. 'auto' selects CUDA if available, otherwise CPU.",
    )
    parser.add_argument(
        "--compute-type",
        default="float16",
        choices=["int8", "float16", "float32"],
        help="Model compute type (float16 recommended for GPU; int8 for CPU speed).",
    )
    parser.add_argument(
        "--language",
        default="en",
        metavar="LANG",
        help="ISO 639-1 language code for transcription (e.g. 'en', 'fr').",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=False,
        help="Scan input_dir recursively for MP3 files.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Enable debug logging.",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    _setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    if not _check_ffmpeg():
        return 1

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.is_dir():
        logger.error("Input directory '%s' does not exist.", input_dir)
        return 1

    # Collect MP3 files
    glob_pattern = "**/*.mp3" if args.recursive else "*.mp3"
    mp3_files = sorted(input_dir.glob(glob_pattern))

    if not mp3_files:
        logger.warning("No MP3 files found in '%s'.", input_dir)
        return 0

    logger.info("Found %d MP3 file(s) in '%s'.", len(mp3_files), input_dir)

    processed, skipped, failed = 0, 0, 0
    failed_files: list[Path] = []

    for mp3_path in mp3_files:
        try:
            result = process_song(
                mp3_path,
                output_dir,
                force=args.force,
                write_lrc=not args.no_lrc,
                sep_model=args.sep_model,
                whisper_model=args.whisper_model,
                device=args.device,
                compute_type=args.compute_type,
                language=args.language,
            )
            if result is None:
                skipped += 1
            else:
                processed += 1
        except Exception as exc:
            logger.exception("Error processing '%s': %s", mp3_path.name, exc)
            failed += 1
            failed_files.append(mp3_path)

    logger.info("Complete – processed: %d  skipped: %d  failed: %d", processed, skipped, failed)
    if failed_files:
        logger.error("Failed files (%d):", len(failed_files))
        for failed_path in failed_files:
            logger.error("  - %s", failed_path)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
