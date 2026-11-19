# Changelog

All notable changes to WeaveShift will be documented in this file.

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
