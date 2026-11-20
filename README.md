![tests](https://github.com/sherwinlu/weaveshift/actions/workflows/tests.yml/badge.svg)

# WeaveShift

Profile-driven continuous music mix engine for full-mix audio (WAV or MP3).

WeaveShift renders auditionable two-track transitions or builds a full continuous setlist mix from an ordered manifest. It is designed for repeatable, explainable, audio-aware transitions without manually editing every handoff.

The same engine can be tuned for EDM/workout mixes, vocal trance, lounge/background journeys, jazz/blues sequencing, cinematic background music, or emotional ballad sets.

## What it does

* Builds continuous mixes from full-mix WAV or MP3 sources (mixed `.wav`/`.mp3` setlists allowed when sample rate and channel count match).
* Exports a single output file as `.wav` (default) or `.mp3` (320 kbps) via the `-o` suffix.
* Preserves original timing and pitch by default.
* Scores transition candidates using vocal safety, beat/onset evidence, energy, loudness, key/BPM compatibility, cue depth, and tail trimming.
* Supports optional Demucs neural vocal separation with a fast heuristic fallback (`--vocal-method auto|heuristic|demucs|none`).
* Ships 8 built-in TOML scoring profiles (`edm`, `smooth`, `vocal_trance`, `vocal_trance_strict`, `lounge`, `jazz`, `heart`, `cinematic`) plus custom profiles via `--scoring-config`.
* Supports per-transition overrides (`quick_cut`, `natural`, `manual`, `handoff`, and more) for exact handoffs that sound better by ear.
* Exports audition snippets, detailed JSON mix reports, and video cue sheets (`weaveshift-cues`).

## Quick start

Requires Python 3.11+ and `ffmpeg` on `PATH` (see [Installation](docs/installation.md) for OS-specific steps and optional Demucs setup, or [Docker Usage](docs/docker.md) to run in a container without installing Python or FFmpeg locally).

```bash
git clone https://github.com/sherwinlu/weaveshift.git
cd weaveshift

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Generate synthetic demo audio (not checked into git):

```bash
python examples/generate_fixtures.py
```

Run a demo setlist mix:

```bash
python -m weaveshift.setlist examples/setlist_example.json \
  -o output/weaveshift_demo.wav \
  --apply-manifest-settings \
  --make-snippets
```

Or run a two-track transition test (appends `_<candidate>` to the `-o` base name, e.g. `output/transition_profile.wav`):

```bash
python -m weaveshift.two_tracks "Track A.wav" "Track B.mp3" \
  -o output/transition.wav \
  --mode profile \
  --profile edm \
  --vocal-method heuristic \
  --make-snippets
```

Use `--vocal-method heuristic` for fast drafting (default for setlist mode) and `--vocal-method demucs` for slower final QA with neural vocal separation (two-track mode defaults to `auto`).

## Documentation

| Guide | Covers |
| --- | --- |
| [Installation](docs/installation.md) | macOS, Windows, Ubuntu/Debian, Fedora setup, optional Demucs, editable install & CLI entry points |
| [Docker Usage](docs/docker.md) | Building and running the lean or Demucs container images, bind mounts, Linux UID/GID, GPU caveats |
| [Manifests, CLI Usage & Audio Policy](docs/manifests-and-audio.md) | Setlist/two-track/cues commands, transition modes, manifest format, track path security rules, WAV/MP3 policy |
| [Scoring Profiles](docs/profiles.md) | All 8 built-in profiles, profile selection guide, custom TOML schema, `--scoring-config` |
| [Per-Transition Overrides](docs/overrides.md) | `--apply-manifest-settings` allowlist, candidate, `natural`, `manual`, and `handoff` overrides |

## CLI entry points

After `pip install -e .`, three console commands are available (or invoke them with `python -m weaveshift.<module>`):

* `weaveshift-setlist` (`python -m weaveshift.setlist`) — build a continuous mix, snippets, and `.weaveshift_setlist_report.json`.
* `weaveshift-two-tracks` (`python -m weaveshift.two_tracks`) — rank and render candidate transitions between two tracks plus `.weaveshift_mix_report.json`.
* `weaveshift-cues` (`python -m weaveshift.cues`) — generate CSV video cue sheets from source tracks and an optional timeline report.

## Suggested workflow

1. Build and audition with `--vocal-method heuristic`.
2. Iterate on track order and profile settings.
3. Listen to transition snippets first.
4. Use manual or handoff overrides for any transition that sounds better by ear than the automatic candidate.
5. Run a final `--vocal-method demucs` pass if vocal-aware separation helps your material.
6. Use the full mix render (`.wav` or `.mp3`) as the release candidate only after snippets pass.

## Repository layout

```text
weaveshift/
  LICENSE
  README.md
  CHANGELOG.md
  CONTRIBUTING.md
  SECURITY.md
  requirements.txt
  requirements-demucs.txt
  requirements-test.txt
  pyproject.toml
  Dockerfile
  .dockerignore
  .gitignore

  docs/                       # detailed installation, Docker, profile, override, and manifest guides
  weaveshift/                 # core Python package
    __init__.py
    _cli.py                   # arguments and process plumbing shared by the CLIs
    _paths.py                 # manifest path containment
    _version.py               # single source of the package version
    audio.py                  # analysis and validation
    cues.py                   # video cue sheet generator (reads CSV/JSON, writes CSV)
    plan.py                   # plan/execute separation
    profiles.py               # TOML scoring profiles
    rendering.py              # transition rendering
    reports.py                # JSON report builders
    scoring.py                # candidate search and scoring
    setlist.py                # setlist CLI orchestration
    two_tracks.py             # two-track CLI orchestration
    configs/                  # built-in TOML scoring profiles (package data)

  examples/
    fixtures/                 # generated synthetic demo WAVs (see fixtures/README.md)
  tests/
```

## Attribution

WeaveShift was created by Sherwin Lu.

WeaveShift is released under the MIT License, so you may use it in personal, research, open-source, or commercial projects. If you use WeaveShift publicly, attribution is appreciated:

```text
WeaveShift by Sherwin Lu
```

## License

WeaveShift is released under the MIT License.

You may use, copy, modify, merge, publish, distribute, sublicense, and sell copies of WeaveShift, including in commercial products, as long as the MIT copyright and license notice are preserved.

See `LICENSE` for details.
