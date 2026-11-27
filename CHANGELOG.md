# Changelog

All notable changes to WeaveShift will be documented in this file.

## Unreleased

## 1.1.0 - 2026-11-26

### Added

- **Streaming Broadcast Audio QA (`weaveshift.broadcast_qa`):** Bounded-memory (~37 MiB peak RAM) ITU-R BS.1770-4 integrated LUFS, 4x-oversampled True Peak (`dBTP`), and transition splice-click diagnostics (with `O(log N)` beat/onset exclusion and an unclipped float32 FFmpeg decode fallback). Automatically runs after `weaveshift-setlist` and `weaveshift-two-tracks` renders, records `broadcast_qa` in JSON reports, and warns on `stderr` when True Peak exceeds `-0.5 dBTP` or splice clicks are detected.
- **Chapter & tracklist export (`weaveshift.chapters`, `--export-chapters`):** Exports YouTube chapters (`.chapters.txt`, with `<3` chapter / `<10s` warnings and automatic `H:MM:SS` over 1 hour), FFmpeg chapter metadata (`.ffmetadata`), 75-fps CUE sheets (`.cue`), and `mvdirector`-compatible tracklist JSON (`.tracklist.json`). Also supported via `--apply-manifest-settings` (`export_chapters`).
- **Tempo octave folding (`fold_tempo`, `fold_tempo_to_hint`):** Folds half- and double-time beat-tracker locks into `[70, 160]` BPM (or the closest octave to a manual BPM hint) across `weaveshift.audio` and `weaveshift.cues`, keeping beat grids aligned.
- **Docker support:** Added `Dockerfile`, `.dockerignore`, and `docs/docker.md` for running all three CLIs in a non-root `python:3.12-slim` container with `ffmpeg` and `libsndfile1` pre-installed (lean core image by default, optional `--build-arg INSTALL_DEMUCS=1` for Demucs).

### Changed

- **Key & vocal analysis hardening:** `estimate_key()` now sanitizes non-finite samples, filters out silent frames (`frame_rms > peak_rms * 0.01`), uses centered unit-vector Krumhansl-Schmuckler dot products, and enforces chroma contrast (`0.12`) and correlation (`0.25`) floors so unpitched noise/percussion returns `(None, None)`. Vocal activity extraction sanitizes non-finite audio and adds a `-60 dBFS` bleed floor.
- **Smooth raised-cosine soft ducking:** `apply_soft_duck()` now uses C¹-smooth raised-cosine attack/release ramps instead of piecewise-linear ramps.
- **Atomic file writes & Demucs cache staging:** WAV/MP3 exports, JSON reports, cue CSVs, chapter sidecars, and Demucs CLI cache commits write to sibling temp files (preserving readable permissions / `0o644`) before `os.replace()`. Demucs CLI fallback runs inside a temporary staging directory and honors `--no-mps`.
- **Documentation & codebase cleanup:** Split `README.md` into focused guides under `docs/` (`installation.md`, `docker.md`, `manifests-and-audio.md`, `profiles.md`, `overrides.md`). Consolidated shared CLI plumbing (`weaveshift._cli`), planner defaults (`weaveshift.plan`), DSP/validation helpers (`weaveshift.audio`), and cue CSV generation (`weaveshift.cues`), and trimmed unused internal compatibility shims while keeping CLI interfaces and JSON report schemas backwards-compatible.

### Fixed

- **DSP & rendering edge cases:** Clamped pydub `fade_out`/`fade_in` lengths to actual slice durations (preventing pydub length-doubling corruption), prevented Track B gaps/timeline drift on manual overrides where `a_cut_sec - a_fade_start_sec > overlap_sec`, scaled `apply_soft_duck()` attack/release windows proportionally on short segments, and guarded 1-sample WAVs in `moving_average()` and silent audio in `estimate_key()`.
- **Input, planning, and report validation:** Rejected empty/dot-only paths, directories, and zero-frame/invalid-stream WAVs; enforced finite in-range planning/manifest/override values (`search_*_sec`, `max_trim_a_sec`, `b_cue_max_sec`, `bpm`, `b_gain_db`, `soft_duck_db`, Track B duration bounds, and `len(tracks) >= 2`); enforced strict RFC 8259 JSON (`allow_nan=False`) in `write_report()`.
- **`weaveshift-cues` reliability:** Fixed BPM detection to call `librosa.feature.tempo` instead of `librosa.feature.rhythm.tempo`, and changed manifest validation errors to raise `ValueError` for consistent exit code `1` handling.

### Security

- **FFprobe / FFmpeg hardening:** Resolved audio paths to absolute paths before `ffprobe`/`pydub`/`ffmpeg` I/O, enforced `-protocol_whitelist file`, and added subprocess timeouts.
- **CSV & chapter injection protection:** Neutralized spreadsheet formula prefixes (`=`, `+`, `-`, `@`, `\t`, `\r`) in `weaveshift-cues` CSV output, and stripped control characters/newlines and escaped delimiters across YouTube, FFmpeg `;FFMETADATA1`, CUE, and tracklist JSON chapter exports.
- **Path & cache containment:** Rejected empty paths in `resolve_contained()` / `resolve_audio_path()`, blocked `..` basename fallback in `cues.py`, required `is_file()`, and verified cached/generated Demucs `vocals.wav` paths stay strictly inside `.weaveshift_demucs_cache`.

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
