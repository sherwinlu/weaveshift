# Changelog

All notable changes to WeaveShift will be documented in this file.

## Unreleased

### Added

- `Dockerfile`, `.dockerignore`, and `docs/docker.md` for running all three CLIs in a non-root `python:3.12-slim` container with `ffmpeg` and `libsndfile1` pre-installed (lean core image by default, optional `--build-arg INSTALL_DEMUCS=1` for Demucs).

### Fixed

- `weaveshift-cues` BPM detection now calls `librosa.feature.tempo` instead of `librosa.feature.rhythm.tempo`, which previously failed silently when `librosa.feature.rhythm` had not already been imported.
- `weaveshift-cues` manifest validation errors now raise `ValueError` instead of `SystemExit`, returning exit code `1` consistently through the shared CLI runner.

### Changed

- Consolidated shared CLI plumbing in `weaveshift._cli`, shared planner defaults and helpers in `weaveshift.plan`, shared DSP/validation helpers in `weaveshift.audio`, and generated cue CSV columns in `weaveshift.cues`.
- Trimmed unused top-level/facade Python re-exports (`weaveshift/__init__.py`, `weaveshift.two_tracks`) and deprecated compatibility shims (`validate_wav_*`, `render_transition_tail`, `apply_gain_ramp(..., step_ms=...)`, `TechnicalScoringConfig.version`, split report schema-version constants). CLI interfaces and JSON report schemas are unchanged.

## 1.0.0 - 2026-11-18

Initial release.

- Profile-driven continuous mix engine for WAV/MP3 setlists and two-track transitions (`weaveshift-setlist`, `weaveshift-two-tracks`).
- Candidate search and scoring from vocal safety, beat/onset evidence, energy, loudness, key/BPM compatibility, cue depth, and tail trimming; optional Demucs vocal analysis with a heuristic fallback (`--vocal-method auto|heuristic|demucs|none`).
- Eight built-in TOML scoring profiles (`edm`, `smooth`, `vocal_trance`, `vocal_trance_strict`, `lounge`, `jazz`, `heart`, `cinematic`) shipped as package data in `weaveshift/configs/`, plus custom profiles via `--scoring-config`.
- Transition modes including recommended, vocal_safe, beat_aligned, quick_cut, smooth, profile, vocal_ducked, long_blend, and natural; per-transition manual, handoff, candidate, and natural overrides from the setlist manifest.
- `weaveshift-cues`: video cue-sheet generation from CSV or JSON setlist manifests, with optional timeline reports.
- JSON mix reports (`.weaveshift_mix_report.json`, `.weaveshift_setlist_report.json`) and audition snippets.

### Security

- Track paths in setlists and cue manifests must resolve inside the manifest directory (or `--audio-root` for cues); `..`, absolute paths elsewhere, `~`, and symlinks that escape are rejected unless `--allow-outside-manifest-dir` is passed.
- `--scoring-config` is CLI-only; manifest settings use a fixed allowlist and cannot load arbitrary TOML paths or redirect output.
- Subprocess calls (`ffprobe`, `demucs.separate`) pass `--` before file paths; Demucs model and profile names are strictly validated.
