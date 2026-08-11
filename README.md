# syltgen

Generate synchronized lyric tags for MP3 collections with a practical, batch-first workflow.

`syltgen` writes:
- **SYLT** (synchronized lyric timestamps in ID3)
- **USLT** (plain lyric text)
- optional **.lrc** sidecar files

It is designed for real-world libraries: mixed tag quality, malformed legacy frames, instrumental tracks, and songs with pre-existing unsynced lyrics.

## Highlights

- Uses **WhisperX** + word-level alignment for precise phrase timing
- Supports **USLT-guided timing** using full-transcription word baselines
- Uses **audio-separator** vocal stems to improve alignment quality
- Re-encodes malformed lyric tags into consistent, player-friendly ID3 output
- Skips non-lyrical content with genre guards + instrumental heuristics
- Supports manual correction with input-side `.lrc` override

## Requirements

> Python **3.10–3.12** is required. WhisperX does not support Python 3.14+.

- Windows PowerShell recommended (project is currently optimized for Windows workflows)
- FFmpeg on PATH
- NVIDIA GPU strongly recommended for speed (CPU mode supported)

## Quick Start

### 1) Setup environment

```powershell
# GPU setup (default CUDA target)
.\setup.ps1

# CPU-only setup
.\setup.ps1 -Cuda cpu
```

### 2) Run

```powershell
python -m syltgen.main ./input ./output
```

## Typical Workflow

1. Put source MP3 files in `input/`
2. Run `syltgen`
3. Review output `.lrc`/SYLT timing
4. For manual fixes, place corrected `Song.lrc` beside `Song.mp3` in input and rerun

If the source song has no SYLT and has a viable same-name input `.lrc`, `syltgen` uses that manual timing directly.

## CLI Options

| Flag | Description |
|---|---|
| `--force` | Re-process files even if they already have SYLT |
| `--no-lrc` | Skip writing `.lrc` sidecars |
| `--whisper-model` | Whisper model (`tiny`, `base`, `small`, `medium`, `large-v2`) |
| `--sep-model` | Vocal-separation model file |
| `--device` | `auto`, `cuda`, `cpu` |
| `--compute-type` | `int8`, `float16`, `float32` |
| `--language` | Language code for transcription/alignment |
| `--recursive` | Scan input recursively |
| `-v`, `--verbose` | Verbose logging |

## Behavior Notes

- Existing well-formed SYLT tracks are skipped unless `--force` is set
- Files that fail processing are reported at the end of the run
- Instrumental/classical genre-tagged files are skipped entirely
- If no credible lyrics are detected, output artifacts are removed and file is skipped
- Oversized cover art can be automatically normalized during rewrite flows
- Both code paths run on separated vocal stems, and a forced alignment that
  shows signs of having run out of lyric text is replaced by transcription-derived
  timing (see *Accuracy benchmark*)
- When transcribing without a lyric sheet, line breaks are repaired so they do not
  fall after a word that requires a continuation; adjacent lines are merged and
  re-split at a boundary the grammar allows

## Accuracy Benchmark

`tests/benchmark/` measures line placement against the SYLT tags already present
in a library of MP3s. Files carrying both SYLT and a USLT sheet with the same
line count give a 1-to-1 reference.

```bash
pytest tests/                                    # unit tests only (no audio needed)
python -m tests.benchmark.run_bench index        # scan the library
python -m tests.benchmark.run_bench run --limit 100 --variant mine
python -m tests.benchmark.run_bench compare baseline mine
python -m tests.benchmark.run_bench linebreaks --limit 60   # line-break quality
```

`linebreaks` scores the transcription path's line grouping rather than its
timing. It caches Whisper output *before* splitting, so the splitter can be
re-tuned in seconds instead of re-transcribing. It reports the *dangling break
rate* — the share of line breaks that fall after a word which syntactically
demands a continuation ("...the feeling is | lightning") — alongside break
precision/recall against the human line breaks in each file's USLT sheet.

Point it at a different library with `--library` or `SYLTGEN_BENCH_LIBRARY`.
Model output is cached per variant under `tests/benchmark/.cache/`, so
re-scoring a finished run is instant.

## Debugging in VS Code

Predefined launch configs are included in `.vscode/launch.json` for:
- normal directory processing
- CPU-only processing
- forced re-tag runs
- current-file debug execution

## Stack

- `whisperx`
- `audio-separator`
- `mutagen`
- `pytorch`

---

If you plan to publish this project, see `CONTRIBUTING.md` for suggested contribution workflow.
