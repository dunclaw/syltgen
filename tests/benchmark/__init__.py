"""Accuracy benchmarking harness for syltgen lyric placement.

Runs library files through the two production code paths (USLT forced
alignment, and no-USLT full transcription) and scores the resulting line
timestamps against the reference SYLT already embedded in the source file.
"""
