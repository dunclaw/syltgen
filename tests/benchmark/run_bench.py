"""Benchmark CLI: run syltgen code paths over a library sample and score them.

Examples
--------
::

    # Scan the library once and cache the ground-truth index
    python -m tests.benchmark.run_bench index --refresh

    # Baseline: 120 files through both code paths
    python -m tests.benchmark.run_bench run --limit 120 --variant baseline

    # After a change, rerun and diff against the baseline
    python -m tests.benchmark.run_bench run --limit 120 --variant candidate
    python -m tests.benchmark.run_bench compare baseline candidate
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.benchmark import report  # noqa: E402
from tests.benchmark.dataset import (  # noqa: E402
    DEFAULT_LIBRARY,
    load_or_scan_index,
    sample,
)
from tests.benchmark.metrics import score_file  # noqa: E402
from tests.benchmark.runner import (  # noqa: E402
    ALL_PATHS,
    PATH_TRANSCRIBE,
    PATH_USLT,
    RunConfig,
    run_path,
)

logger = logging.getLogger("bench")

_BENCH_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = _BENCH_DIR / ".cache" / "segments"
DEFAULT_RESULTS_DIR = _BENCH_DIR / ".results"

#: Headline metrics shown by ``compare``.
COMPARE_KEYS = (
    "line_median_abs_error",
    "line_mean_of_medians",
    "line_median_jitter",
    "line_median_abs_offset",
    "line_gross_error_rate",
    "line_within_0.5s",
    "line_within_1s",
    "line_within_2s",
    "files_median_over_1s",
    "files_median_over_3s",
    "first_line_median_abs_error",
    "token_median_abs_error",
    "token_median_coverage",
    "token_within_1s",
)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(message)s",
    )
    for noisy in ("urllib3", "httpx", "httpcore", "numba", "matplotlib", "torio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _results_path(results_dir: Path, variant: str) -> Path:
    return results_dir / f"{variant}.json"


def cmd_index(args: argparse.Namespace) -> int:
    items = load_or_scan_index(args.library, refresh=args.refresh)
    strict = sum(1 for i in items if i.strict)
    print(f"ground-truth candidates : {len(items)}")
    print(f"strict (1:1 line count) : {strict}")
    selected = sample(items, limit=args.limit, min_lines=args.min_lines)
    print(f"selected for benchmark  : {len(selected)}")
    return 0


def _parse_align_overrides(pairs: list[str]) -> dict:
    """Parse ``key=value`` alignment overrides, coercing obvious literals."""
    import ast

    options: dict = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        key = key.strip()
        if not key or not _:
            raise ValueError(f"expected key=value, got '{pair}'")
        try:
            options[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            options[key] = raw
    return options


def cmd_run(args: argparse.Namespace) -> int:
    from syltgen.tagger import read_sylt_tag

    items = load_or_scan_index(args.library)
    selected = sample(items, limit=args.limit, min_lines=args.min_lines)
    if not selected:
        logger.error("No benchmark files selected.")
        return 1

    align_options = _parse_align_overrides(args.align or [])
    if align_options:
        logger.info("Alignment overrides: %s", align_options)

    config = RunConfig(
        whisper_model=args.whisper_model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        sep_model=args.sep_model,
        align_options=align_options or None,
        align_method=args.align_method,
    )
    paths = [p.strip() for p in args.paths.split(",") if p.strip()]
    for path_name in paths:
        if path_name not in ALL_PATHS:
            logger.error("Unknown code path '%s' (expected one of %s).", path_name, ALL_PATHS)
            return 1

    results: list[dict] = []
    total = len(selected) * len(paths)
    done = 0

    for item in selected:
        mp3_path = Path(item.path)
        reference = read_sylt_tag(mp3_path) or []
        stale = _reference_is_stale(mp3_path, reference)
        if stale:
            logger.info("Skipping scoring for '%s': %s.", mp3_path.name, stale)
        for path_name in paths:
            done += 1
            logger.info("[%d/%d] %s :: %s", done, total, path_name, mp3_path.name)
            record = run_path(
                mp3_path,
                path_name,
                config,
                cache_dir=args.cache_dir,
                variant=args.variant,
                refresh=args.refresh,
            )
            record["reference_lines"] = len(reference)
            if stale:
                record["reference_stale"] = stale
            record["score"] = (
                score_file(reference, record["segments"])
                if reference and record["segments"] and not stale
                else None
            )
            results.append(record)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = _results_path(args.results_dir, args.variant)
    out_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    logger.info("Wrote %d records to '%s'.", len(results), out_path)

    for path_name in paths:
        print()
        print(report.format_summary(report.aggregate(results, path_name)))
        print()
        print(report.format_worst(results, path_name))
    return 0


#: Reference SYLT whose last line starts after the audio ends cannot describe
#: this recording — the tag was copied from a different edit of the song.  Such
#: files are excluded from scoring rather than counted as huge errors.
_STALE_REFERENCE_MARGIN = 1.0


def _reference_is_stale(mp3_path: Path, reference: list[dict]) -> str:
    """Describe why *reference* cannot be ground truth for *mp3_path*, if so."""
    if not reference:
        return ""
    try:
        from mutagen.mp3 import MP3

        duration = float(MP3(str(mp3_path)).info.length)
    except Exception:
        return ""
    if duration <= 0:
        return ""
    last_start = float(reference[-1].get("start", 0.0))
    if last_start > duration + _STALE_REFERENCE_MARGIN:
        return (
            f"reference SYLT ends at {last_start:.1f}s but the audio is "
            f"only {duration:.1f}s long"
        )
    return ""


def cmd_report(args: argparse.Namespace) -> int:
    results = json.loads(
        _results_path(args.results_dir, args.variant).read_text(encoding="utf-8")
    )
    for path_name in ALL_PATHS:
        if not any(r.get("code_path") == path_name for r in results):
            continue
        print()
        print(report.format_summary(report.aggregate(results, path_name)))
        print()
        print(report.format_worst(results, path_name, limit=args.worst))
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Print reference vs hypothesis lines side by side for one file."""
    from syltgen.tagger import read_sylt_tag

    results = json.loads(
        _results_path(args.results_dir, args.variant).read_text(encoding="utf-8")
    )
    needle = args.file.lower()
    matches = [
        r
        for r in results
        if r.get("code_path") == args.path and needle in Path(r["path"]).name.lower()
    ]
    if not matches:
        logger.error("No '%s' result matching '%s'.", args.path, args.file)
        return 1

    record = matches[0]
    reference = read_sylt_tag(Path(record["path"])) or []
    print(Path(record["path"]).name)
    if record.get("error"):
        print(f"  ERROR: {record['error']}")
        return 0

    print(f"  {'ref':>8} {'hyp':>8} {'delta':>8}  text")
    for index in range(max(len(reference), len(record["segments"]))):
        ref = reference[index] if index < len(reference) else None
        hyp = record["segments"][index] if index < len(record["segments"]) else None
        ref_start = f"{ref['start']:8.2f}" if ref else " " * 8
        hyp_start = f"{hyp['start']:8.2f}" if hyp else " " * 8
        delta = (
            f"{hyp['start'] - ref['start']:8.2f}" if ref and hyp else " " * 8
        )
        text = (hyp or ref or {}).get("text", "")
        print(f"  {ref_start} {hyp_start} {delta}  {text[:70]}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    baseline = json.loads(
        _results_path(args.results_dir, args.baseline).read_text(encoding="utf-8")
    )
    candidate = json.loads(
        _results_path(args.results_dir, args.candidate).read_text(encoding="utf-8")
    )

    # Optionally treat a candidate code path as if it were the baseline's, so
    # e.g. "uslt on stems" can be diffed against "uslt on the mix".
    if args.candidate_path and args.baseline_path:
        candidate = [
            {**r, "code_path": args.baseline_path}
            for r in candidate
            if r.get("code_path") == args.candidate_path
        ]
        baseline = [r for r in baseline if r.get("code_path") == args.baseline_path]

    # Restrict both sides to the files they have in common so runs made with
    # different --limit values stay comparable.
    common = {(r["path"], r["code_path"]) for r in baseline} & {
        (r["path"], r["code_path"]) for r in candidate
    }
    baseline = [r for r in baseline if (r["path"], r["code_path"]) in common]
    candidate = [r for r in candidate if (r["path"], r["code_path"]) in common]

    for path_name in ALL_PATHS:
        base_summary = report.aggregate(baseline, path_name)
        cand_summary = report.aggregate(candidate, path_name)
        if not base_summary.get("files_scored") and not cand_summary.get("files_scored"):
            continue
        print()
        print(
            f"-- {path_name}: {args.baseline} -> {args.candidate} "
            f"({base_summary.get('files_scored', 0)} common files) " + "-" * 8
        )
        print(report.compare(base_summary, cand_summary, COMPARE_KEYS))
    return 0

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    sub = parser.add_subparsers(dest="command", required=True)

    index_cmd = sub.add_parser("index", help="Scan/refresh the ground-truth index.")
    index_cmd.add_argument("--refresh", action="store_true")
    index_cmd.add_argument("--limit", type=int, default=None)
    index_cmd.add_argument("--min-lines", type=int, default=8)
    index_cmd.set_defaults(func=cmd_index)

    run_cmd = sub.add_parser("run", help="Run code paths over the sample and score.")
    run_cmd.add_argument("--limit", type=int, default=100)
    run_cmd.add_argument("--min-lines", type=int, default=8)
    run_cmd.add_argument("--variant", default="baseline")
    run_cmd.add_argument("--paths", default=f"{PATH_USLT},{PATH_TRANSCRIBE}")
    run_cmd.add_argument(
        "--align",
        action="append",
        metavar="KEY=VALUE",
        help="Override a stable-ts align option, e.g. --align token_step=250 "
        "--align nonspeech_skip=5.0 (repeatable).",
    )
    run_cmd.add_argument(
        "--align-method",
        default="auto",
        choices=["auto", "align", "transcribe_match", "transcribe_greedy"],
        help="USLT timing strategy under test.",
    )
    run_cmd.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    run_cmd.add_argument("--refresh", action="store_true", help="Ignore cached output.")
    run_cmd.add_argument("--whisper-model", default="large-v2")
    run_cmd.add_argument("--device", default="auto")
    run_cmd.add_argument("--compute-type", default="float16")
    run_cmd.add_argument("--language", default="en")
    run_cmd.add_argument("--sep-model", default="UVR-MDX-NET-Voc_FT.onnx")
    run_cmd.set_defaults(func=cmd_run)

    report_cmd = sub.add_parser("report", help="Re-print a stored result set.")
    report_cmd.add_argument("variant", nargs="?", default="baseline")
    report_cmd.add_argument("--worst", type=int, default=15)
    report_cmd.set_defaults(func=cmd_report)

    inspect_cmd = sub.add_parser("inspect", help="Show ref vs hyp lines for one file.")
    inspect_cmd.add_argument("file", help="Substring of the file name.")
    inspect_cmd.add_argument("--variant", default="baseline")
    inspect_cmd.add_argument("--path", default="uslt", choices=list(ALL_PATHS))
    inspect_cmd.set_defaults(func=cmd_inspect)

    compare_cmd = sub.add_parser("compare", help="Diff two stored result sets.")
    compare_cmd.add_argument("baseline")
    compare_cmd.add_argument("candidate")
    compare_cmd.add_argument(
        "--baseline-path",
        choices=list(ALL_PATHS),
        help="Compare across code paths: the baseline code path to use.",
    )
    compare_cmd.add_argument(
        "--candidate-path",
        choices=list(ALL_PATHS),
        help="Compare across code paths: the candidate code path to remap.",
    )
    compare_cmd.set_defaults(func=cmd_compare)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
