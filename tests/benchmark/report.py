"""Aggregation and reporting of benchmark results."""

from __future__ import annotations

import statistics
from typing import Iterable, Optional, Sequence

from .metrics import TOLERANCES


def _finite(values: Iterable[Optional[float]]) -> list[float]:
    out: list[float] = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if number != number:  # NaN
            continue
        out.append(number)
    return out


def aggregate(results: Sequence[dict], code_path: str) -> dict:
    """Aggregate per-file scores for one code path into corpus-level numbers.

    Per-file medians are aggregated (rather than pooling every line) so that a
    single 200-line song cannot dominate the corpus score.
    """
    rows = [r for r in results if r.get("code_path") == code_path]
    scored = [r for r in rows if r.get("score") and not r.get("error")]

    line_rows = [r for r in scored if r["score"].get("lines")]
    token_rows = [r for r in scored if r["score"].get("tokens", {}).get("n_matched_tokens")]

    summary: dict = {
        "code_path": code_path,
        "files_total": len(rows),
        "files_scored": len(scored),
        "files_failed": sum(1 for r in rows if r.get("error")),
        "files_line_matched": len(line_rows),
    }

    medians = _finite(r["score"]["lines"]["median_abs_error"] for r in line_rows)
    if medians:
        summary["line_median_abs_error"] = statistics.median(medians)
        summary["line_mean_of_medians"] = statistics.fmean(medians)
        summary["line_worst_median"] = max(medians)
        summary["files_median_over_1s"] = sum(1 for m in medians if m > 1.0) / len(medians)
        summary["files_median_over_3s"] = sum(1 for m in medians if m > 3.0) / len(medians)

    first_errors = _finite(r["score"]["lines"]["first_line_error"] for r in line_rows)
    if first_errors:
        summary["first_line_median_abs_error"] = statistics.median(
            abs(e) for e in first_errors
        )
        summary["first_line_median_signed_error"] = statistics.median(first_errors)

    jitters = _finite(r["score"]["lines"]["jitter"] for r in line_rows)
    if jitters:
        summary["line_median_jitter"] = statistics.median(jitters)

    offsets = _finite(r["score"]["lines"]["median_signed_error"] for r in line_rows)
    if offsets:
        summary["line_median_offset"] = statistics.median(offsets)
        summary["line_median_abs_offset"] = statistics.median(abs(o) for o in offsets)

    gross = _finite(r["score"]["lines"]["gross_error_rate"] for r in line_rows)
    if gross:
        summary["line_gross_error_rate"] = statistics.fmean(gross)

    violations = [r["score"]["lines"]["monotonic_violations"] for r in line_rows]
    if violations:
        summary["files_with_monotonic_violations"] = sum(
            1 for v in violations if v
        ) / len(violations)

    for tol in TOLERANCES:
        key = f"{tol:g}s"
        values = _finite(r["score"]["lines"]["within"].get(key) for r in line_rows)
        if values:
            summary[f"line_within_{key}"] = statistics.fmean(values)

    token_medians = _finite(r["score"]["tokens"]["median_abs_error"] for r in token_rows)
    if token_medians:
        summary["token_median_abs_error"] = statistics.median(token_medians)
        summary["token_files_median_over_3s"] = sum(
            1 for m in token_medians if m > 3.0
        ) / len(token_medians)

    coverage = _finite(r["score"]["tokens"]["coverage"] for r in token_rows)
    if coverage:
        summary["token_median_coverage"] = statistics.median(coverage)

    for tol in TOLERANCES:
        key = f"{tol:g}s"
        values = _finite(r["score"]["tokens"]["within"].get(key) for r in token_rows)
        if values:
            summary[f"token_within_{key}"] = statistics.fmean(values)

    elapsed = _finite(r.get("elapsed") for r in rows)
    if elapsed:
        summary["median_seconds_per_file"] = statistics.median(elapsed)

    return summary


def format_summary(summary: dict) -> str:
    """Render one aggregated summary as an aligned text block."""
    lines = [f"-- {summary['code_path']} " + "-" * max(0, 60 - len(summary["code_path"]))]
    for key, value in summary.items():
        if key == "code_path":
            continue
        if isinstance(value, float):
            lines.append(f"  {key:<38} {value:>10.4f}")
        else:
            lines.append(f"  {key:<38} {value:>10}")
    return "\n".join(lines)


def worst_files(results: Sequence[dict], code_path: str, *, limit: int = 15) -> list[dict]:
    """Return the files with the largest median line error, worst first."""
    rows = [
        r
        for r in results
        if r.get("code_path") == code_path
        and r.get("score")
        and r["score"].get("lines")
        and not r.get("error")
    ]
    rows.sort(key=lambda r: r["score"]["lines"]["median_abs_error"], reverse=True)
    return rows[:limit]


def format_worst(results: Sequence[dict], code_path: str, *, limit: int = 15) -> str:
    """Render the worst-performing files for triage."""
    from pathlib import Path

    rows = worst_files(results, code_path, limit=limit)
    if not rows:
        return f"(no line-matched results for '{code_path}')"

    out = [f"-- worst {code_path} files " + "-" * 40]
    out.append(f"  {'median':>8} {'signed':>8} {'jitter':>8} {'first':>8}  file")
    for row in rows:
        score = row["score"]["lines"]
        out.append(
            f"  {score['median_abs_error']:>8.2f} {score['median_signed_error']:>8.2f} "
            f"{score['jitter']:>8.2f} {score['first_line_error']:>8.2f}  "
            f"{Path(row['path']).name}"
        )
    return "\n".join(out)


def compare(baseline: dict, candidate: dict, keys: Sequence[str]) -> str:
    """Render a baseline-vs-candidate delta table for the given metric keys."""
    out = [f"  {'metric':<38} {'baseline':>10} {'candidate':>10} {'delta':>10}"]
    for key in keys:
        base = baseline.get(key)
        cand = candidate.get(key)
        if base is None or cand is None:
            continue
        delta = float(cand) - float(base)
        out.append(f"  {key:<38} {float(base):>10.4f} {float(cand):>10.4f} {delta:>+10.4f}")
    return "\n".join(out)
