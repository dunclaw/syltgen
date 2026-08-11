"""Ground-truth dataset discovery and deterministic sampling.

A library file is usable as ground truth when it carries **both** a SYLT frame
(the reference timing) and a USLT frame (the lyric text the production USLT
code path would consume).  Files where ``len(SYLT) == len(USLT lines)`` are
"strict" ground truth: forced alignment emits exactly one segment per USLT
line, so reference and hypothesis lines map 1-to-1.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

DEFAULT_LIBRARY = Path(os.environ.get("SYLTGEN_BENCH_LIBRARY", r"E:\Temp\OldMusic"))


@dataclass(frozen=True)
class GroundTruthItem:
    """One benchmarkable library file."""

    path: str
    sylt_lines: int
    uslt_lines: int
    first_start: float
    last_start: float

    @property
    def strict(self) -> bool:
        """True when SYLT and USLT line counts agree (1-to-1 mapping)."""
        return self.sylt_lines == self.uslt_lines

    @property
    def key(self) -> str:
        """Stable hash of the file path, used for deterministic sampling."""
        return hashlib.sha1(self.path.encode("utf-8")).hexdigest()


def _uslt_line_count(text: Optional[str]) -> int:
    if not text:
        return 0
    return sum(1 for line in text.splitlines() if line.strip())


def scan_library(library: Path) -> list[GroundTruthItem]:
    """Scan *library* for MP3s carrying both SYLT and USLT frames."""
    from syltgen.tagger import read_sylt_tag, read_uslt_lyrics

    items: list[GroundTruthItem] = []
    files = sorted(library.rglob("*.mp3"))
    logger.info("Scanning %d MP3 files under '%s'…", len(files), library)

    for index, path in enumerate(files, start=1):
        try:
            sylt = read_sylt_tag(path)
        except Exception:
            sylt = None
        if not sylt:
            continue
        try:
            uslt = read_uslt_lyrics(path)
        except Exception:
            uslt = None
        uslt_count = _uslt_line_count(uslt)
        if uslt_count == 0:
            continue

        items.append(
            GroundTruthItem(
                path=str(path),
                sylt_lines=len(sylt),
                uslt_lines=uslt_count,
                first_start=float(sylt[0]["start"]),
                last_start=float(sylt[-1]["start"]),
            )
        )
        if index % 250 == 0:
            logger.info("  scanned %d/%d…", index, len(files))

    logger.info("Found %d ground-truth candidates.", len(items))
    return items


def load_or_scan_index(
    library: Path = DEFAULT_LIBRARY,
    cache_path: Optional[Path] = None,
    *,
    refresh: bool = False,
) -> list[GroundTruthItem]:
    """Return the ground-truth index, scanning the library only when needed."""
    cache_path = cache_path or (Path(__file__).resolve().parent / ".cache" / "index.json")
    if cache_path.exists() and not refresh:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        return [GroundTruthItem(**row) for row in raw]

    items = scan_library(library)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps([asdict(i) for i in items], indent=1), encoding="utf-8"
    )
    return items


def sample(
    items: Iterable[GroundTruthItem],
    *,
    limit: Optional[int] = None,
    strict_only: bool = True,
    min_lines: int = 8,
) -> list[GroundTruthItem]:
    """Deterministically sample benchmark items.

    Sampling is by path hash rather than by ``random``, so growing or shrinking
    ``limit`` keeps the previously selected files in the set — results from
    different runs stay comparable and cached model output keeps its value.
    """
    pool = [
        item
        for item in items
        if item.sylt_lines >= min_lines and (item.strict or not strict_only)
    ]
    pool.sort(key=lambda item: item.key)
    if limit is not None:
        pool = pool[:limit]
    return pool
