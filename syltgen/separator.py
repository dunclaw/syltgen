"""Vocal stem separation using audio-separator (Demucs / UVR backend)."""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Default model – MDX-NET fine-tuned for vocals, good balance of speed/quality.
# Alternatives: "Kim_Vocal_2.onnx" (higher quality), "UVR-MDX-NET-Inst_HQ_3.onnx"
DEFAULT_MODEL = "UVR-MDX-NET-Voc_FT.onnx"


def separate_vocals(mp3_path: str | os.PathLike, output_dir: str | os.PathLike, model_name: str = DEFAULT_MODEL) -> Path:
    """
    Separate vocals from *mp3_path* and write stems to *output_dir*.

    Returns the path to the vocals stem file.
    """
    try:
        from audio_separator.separator import Separator
    except ImportError as exc:
        raise ImportError(
            "audio-separator is not installed. Run: pip install audio-separator[gpu]"
        ) from exc

    mp3_path = Path(mp3_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Separating vocals from '%s' using model '%s'…", mp3_path.name, model_name)

    sep = Separator(
        output_dir=str(output_dir),
        output_format="WAV",
        log_level=logging.WARNING,
    )
    sep.load_model(model_filename=model_name)
    output_files = sep.separate(str(mp3_path))

    # audio-separator returns [instrumental_path, vocals_path] conventionally,
    # but the ordering varies by model.  Find the vocals file by name fragment.
    vocals_file = _find_vocals_file(output_files, output_dir)
    logger.info("Vocals written to '%s'", vocals_file)
    return vocals_file


def _find_vocals_file(output_files: list[str], output_dir: Path) -> Path:
    """Return the vocals stem from the list of output file paths.

    audio-separator may return bare filenames or full paths depending on the
    version; normalise all entries to absolute paths under output_dir.
    """
    def resolve(f: str) -> Path:
        p = Path(f)
        return p if p.is_absolute() else output_dir / p

    for f in output_files:
        name_lower = Path(f).name.lower()
        if "vocal" in name_lower or "vocals" in name_lower:
            return resolve(f)

    # Fallback: return whichever file is NOT named 'instrumental'
    for f in output_files:
        name_lower = Path(f).name.lower()
        if "instrumental" not in name_lower and "music" not in name_lower:
            return resolve(f)

    # Last resort: just return the first file
    return resolve(output_files[0])
