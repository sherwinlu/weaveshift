# Changelog

All notable changes to WeaveShift will be documented in this file.

## Unreleased

## 1.1.0 - 2026-07-27

### Changed

- Renamed the project from FlowMix to WeaveShift: the `flowmix_*.py` modules are now `weaveshift_*.py`, the package/CLI entry points are `weaveshift`/`weaveshift-*`, and the PyPI package name changes accordingly. Purely a rename — no behavior changed.

### Security

- `--apply-manifest-settings` now applies only an allowlist of tuning keys (`transition_mode`, `profile`, `scoring_config`, `vocal_method`, `make_snippets`, `natural_pause_sec`, `search_a_sec`, `search_b_sec`, `max_trim_a_sec`, `b_cue_max_sec`). A manifest can no longer override CLI-only arguments such as the output path; unsupported keys are ignored with a warning.
- Built-in profile names are restricted to letters, digits, hyphens, and underscores, so a manifest-supplied `profile` value cannot traverse outside the bundled `configs/` directories. Custom profile files still load via `--scoring-config`.

### Fixed

- Manual and handoff transition overrides now reject non-finite (`NaN`/`Infinity`) values, negative `a_fade_start_sec`/`b_cue_sec`, and timing beyond either source track. A negative `b_cue_sec` previously sliced from the *end* of Track B and silently rendered the wrong audio.
- Manifest settings and track fields (`bpm`, `key`, `path`, `settings`) are validated with clear error messages instead of failing deep inside planning or rendering.
- Output stems containing dots (for example `-o "mix v1.2.wav"`) no longer lose part of their name in candidate, snippet, and report filenames.
- `weaveshift_cues.py` no longer crashes on degenerate, extremely short tracks when computing analysis frame spacing.
- Missing `ffprobe` now reports an actionable "install ffmpeg" error instead of a raw `FileNotFoundError`.

## 1.0.2 - 2026-07-02

### Added

- `natural` setlist transitions that preserve the outgoing track through its end, insert a configurable silent pause, and start the incoming track from source time zero without a fade or overlap.

## 1.0.1 - 2026-06-26

### Added

- Mixed WAV and MP3 **format** support for setlist and two-track mixing when sample rate and channel count match.
- MP3 output via `-o mix.mp3` (320 kbps). WAV remains the default when no suffix is given.
- MP3 cue-sheet analysis in `flowmix_cues.py` (renamed to `weaveshift_cues.py` in the later WeaveShift rebrand) using the same windowed decode path as mix rendering.
- Demo fixtures are generated locally via `examples/generate_fixtures.py` (synthetic audio only; not committed).
- Manual-only `handoff` transition overrides for cases where Track A should fade down before Track B takes over with a short, controlled overlap.

## 1.0.0 - Initial public release

Initial public release of FlowMix (renamed to WeaveShift in a later release).

### Added

- Continuous WAV mix rendering from ordered setlist manifests.
- Two-track transition rendering for auditioning individual handoffs.
- Profile-driven transition scoring with built-in TOML profiles:
  - `edm`
  - `vocal_trance`
  - `lounge`
  - `jazz`
  - `heart`
  - `cinematic`
- Transition candidate scoring using vocal safety, beat/onset evidence, energy, loudness, key/BPM compatibility, cue depth, and tail trimming.
- Optional Demucs vocal analysis with heuristic fallback.
- Per-transition manifest overrides, including manual timing overrides.
- Transition snippets and detailed JSON reports.
- MIT License.
