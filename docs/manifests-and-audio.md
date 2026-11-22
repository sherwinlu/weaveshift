# Manifests, CLI Usage & Audio Policy

## Multi-track setlist usage

Setlist paths may be `.wav`, `.mp3`, or a mix of both. Every track must share the same sample rate and channel count. Pick output format with `-o` (`.wav` or `.mp3`).

```bash
python -m weaveshift.setlist examples/setlist_example.json \
  -o output/weaveshift_demo.wav \
  --transition-mode profile \
  --profile edm \
  --vocal-method heuristic \
  --make-snippets
```

MP3 output uses the same command with a `.mp3` destination:

```bash
python -m weaveshift.setlist examples/setlist_example.json \
  -o output/weaveshift_demo.mp3 \
  --apply-manifest-settings
```

Available setlist transition modes:

```text
recommended, vocal_safe, beat_aligned, quick_cut, smooth, profile, vocal_ducked, long_blend, natural
```

* `recommended` selects the best technical candidate.
* `smooth` is the built-in smooth vocal-aware handoff.
* `profile` uses the selected TOML profile (see [Scoring Profiles](profiles.md)).
* `natural` preserves tracks end-to-end with a configurable silent pause (see [Per-Transition Overrides](overrides.md#natural-track-boundary)).

Pass `--export-chapters` to also write chapter and tracklist sidecars beside the output mix (`.chapters.txt` for YouTube, `.ffmetadata` for FFmpeg chapter embedding, `.cue` for DJ/media players, and `.tracklist.json` for downstream tools such as `mvdirector`).

## Two-track transition usage

```bash
python -m weaveshift.two_tracks "Track A.wav" "Track B.mp3" \
  -o output/transition.wav \
  --mode profile \
  --profile edm \
  --vocal-method heuristic \
  --make-snippets
```

Available two-track `--mode` values (alias `--transition-mode`, default `all`):

```text
all, vocal_safe, beat_aligned, quick_cut, smooth, profile, vocal_ducked, long_blend
```

Each rendered candidate appends `_<candidate>` to the `-o` base name (for example, `-o output/transition.wav` with `--mode profile` writes `output/transition_profile.wav` alongside `output/transition.weaveshift_mix_report.json`). You can also override detected tempo/key per track with `--a-bpm`, `--b-bpm`, `--a-key`, and `--b-key`.

## Vocal analysis methods

* `heuristic` — fast midrange/harmonic estimator; default for `weaveshift-setlist`.
* `demucs` — slower neural vocal separation (requires optional Demucs install; see [Installation](installation.md#optional-demucs-setup)).
* `auto` — tries Demucs and falls back to heuristic; default for `weaveshift-two-tracks`.
* `none` — skips vocal analysis entirely (vocal-collision scoring sees no vocals).

## Video cue sheets (`weaveshift-cues`)

`weaveshift-cues` (`python -m weaveshift.cues`) reads CSV or WeaveShift JSON manifests with `.wav`/`.mp3` tracks, writes a CSV cue sheet (`--out`, default `weaveshift_cues.csv`), and optionally maps cue timestamps onto a final mix using a `.weaveshift_setlist_report.json` or timeline CSV:

```bash
python -m weaveshift.cues examples/setlist_example.json \
  --timeline output/weaveshift_demo.weaveshift_setlist_report.json \
  --out output/weaveshift_cues.csv
```

Additional options: `--audio-root DIR` (extra allowed search directory for manifest audio paths), `--allow-outside-manifest-dir` (lift path containment), and `--skip-missing` (skip missing or unreadable tracks instead of failing; path containment errors are still rejected).

## Setlist manifest format

Each `path` may point to a WAV or MP3 file. Mixed formats are fine when decode parameters match.

```json
{
  "tracks": [
    {"path": "masters/A.wav", "title": "Track A", "bpm": 128, "key": "E major"},
    {"path": "downloads/B.mp3", "title": "Track B", "bpm": 130, "key": "A minor"}
  ],
  "settings": {
    "transition_mode": "profile",
    "profile": "edm",
    "vocal_method": "heuristic",
    "transition_overrides": [
      {
        "index": 1,
        "from": "Track A",
        "to": "Track B",
        "mode": "quick_cut"
      }
    ]
  }
}
```

Plain text setlists (one audio file path per line, `#` comments ignored) are also accepted. Relative paths are resolved relative to the manifest file, and `~` paths are expanded before path containment is checked (see below).

Other manifest `settings` keys apply only when you pass `--apply-manifest-settings`, and only keys on the supported allowlist are copied (see [Per-Transition Overrides & Manifest Settings](overrides.md#applying-other-manifest-settings)); unsupported keys are ignored with a warning. Without the flag, CLI arguments control those defaults, while **`transition_overrides` still apply**.

## Track path security rules

Track paths in a setlist (JSON or text) and in `weaveshift-cues` manifests must resolve **inside the manifest directory** (or `--audio-root` for cues). `..`, absolute paths elsewhere, expanded `~` paths, and symlinks that resolve outside are rejected. Pass `--allow-outside-manifest-dir` to opt in when your music library lives outside the project:

```bash
python -m weaveshift.setlist setlist.json --allow-outside-manifest-dir
```

This flag is CLI-only and cannot be set from a manifest.

## Audio format policy

WeaveShift accepts **WAV (`.wav`, `.wave`) and MP3 (`.mp3`) inputs in any combination** — a setlist can mix WAV and MP3 tracks. All tracks must share the same **sample rate** and **channel count**. When every input is WAV, subtype must also match (`PCM_16`, `PCM_24`, `PCM_32`, `FLOAT`, or `DOUBLE`).

**Output is one file**, either WAV or MP3 — pick with the `-o` suffix:

* `mix.wav` → lossless WAV (default when no suffix is given)
* `mix.mp3` → MP3 at 320 kbps

WAV export uses the first lossless WAV subtype found in the inputs (otherwise `PCM_16`). MP3 export ignores source subtype.

WAV masters are still recommended for final release quality; MP3 sources may add encoder delay or re-encoding artifacts at transitions.
