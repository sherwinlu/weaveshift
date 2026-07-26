![tests](https://github.com/sherwinlu/weaveshift/actions/workflows/tests.yml/badge.svg)

# WeaveShift

Profile-driven continuous music mix engine for full-mix audio (WAV or MP3).

WeaveShift can render auditionable two-track transitions or build a full continuous setlist mix from an ordered manifest. It is designed for people who want repeatable, explainable, audio-aware transitions without manually editing every handoff.

The same engine can be tuned for EDM/workout mixes, vocal trance, lounge/background journeys, jazz/blues sequencing, cinematic background music, or emotional love/heartbreak sets.

## What it does

* Builds continuous mixes from full-mix WAV or MP3 sources.
* Accepts mixed `.wav` and `.mp3` tracks in one setlist when sample rate and channel count match.
* Exports a single output file as `.wav` (default) or `.mp3` via the `-o` suffix.
* Preserves original timing and pitch by default.
* Renders two-track transition tests or full setlist mixes.
* Scores transition candidates using vocal safety, beat/onset evidence, energy, loudness, key/BPM compatibility, cue depth, and tail trimming.
* Supports optional Demucs vocal analysis with CLI fallback when `demucs.api` is unavailable.
* Exports transition snippets and a detailed JSON report.
* Supports TOML scoring profiles: `edm`, `vocal_trance`, `vocal_trance_strict`, `lounge`, `jazz`, `heart`, and `cinematic`.
* Supports per-transition overrides for exact handoffs that sound better by ear than the highest-scoring automatic candidate.

## Quick start

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

Run a demo setlist mix (uses generated fixtures under `examples/fixtures/`):

```bash
python weaveshift_setlist.py examples/setlist_example.json \
  -o output/weaveshift_demo.wav \
  --apply-manifest-settings \
  --make-snippets
```

Or pass settings on the command line:

```bash
python weaveshift_setlist.py examples/setlist_example.json \
  -o output/weaveshift_demo.wav \
  --transition-mode profile \
  --profile edm \
  --vocal-method heuristic \
  --make-snippets
```

MP3 output uses the same command with a `.mp3` destination:

```bash
python weaveshift_setlist.py examples/setlist_example.json \
  -o output/weaveshift_demo.mp3 \
  --apply-manifest-settings
```

Run a two-track transition test:

```bash
python weaveshift_two_tracks.py "Track A.wav" "Track B.mp3" \
  -o output/transition.wav \
  --mode profile \
  --profile edm \
  --vocal-method heuristic \
  --make-snippets
```

For fast drafting, use:

```bash
--vocal-method heuristic
```

For slower final QA with neural vocal separation, use:

```bash
--vocal-method demucs
```

## Requirements

WeaveShift is a command-line Python project. It needs:

* Python 3.11 or newer (CI tests 3.11 and 3.12; 3.13+ works when your audio stack supports it).
* `ffmpeg` available on your command line (required for MP3 input/output and pydub export).
* Install with `pip install -e .` or `pip install -r requirements.txt`.
* Optional: `pip install -e ".[demucs]"` or `pip install -r requirements-demucs.txt` for `--vocal-method demucs`.

## Install by operating system

### macOS

Install command-line tools and Python support:

```bash
xcode-select --install
brew install ffmpeg
```

Create a virtual environment and install WeaveShift:

```bash
cd weaveshift
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Confirm the required tools are visible:

```bash
python --version
ffmpeg -version
python -m pytest
```

If you use `pyenv`, this also works well:

```bash
pyenv install 3.12.8
pyenv local 3.12.8
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### Windows 10/11

Install these first:

* Python 3.11+ from python.org or the Microsoft Store.
* FFmpeg from winget, Chocolatey, or a manual download.
* Optional: Git for cloning/downloading repos.

Using `winget`:

```powershell
winget install Python.Python.3.12
winget install Gyan.FFmpeg
winget install Git.Git
```

Open a new PowerShell window so `python` and `ffmpeg` are on PATH. Then install WeaveShift:

```powershell
cd weaveshift
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Confirm the required tools are visible:

```powershell
python --version
ffmpeg -version
python -m pytest
```

If PowerShell blocks virtualenv activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then reopen PowerShell and activate again.

### Linux / Ubuntu / Debian

Install system tools:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip ffmpeg git
```

Create a virtual environment and install WeaveShift:

```bash
cd weaveshift
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Confirm the required tools are visible:

```bash
python --version
ffmpeg -version
python -m pytest
```

### Linux / Fedora

```bash
sudo dnf install -y python3 python3-pip ffmpeg git
cd weaveshift
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
```

## Optional Demucs setup

Demucs is optional. It is slower, but can improve vocal-aware transition choices.

Check whether Demucs is available:

```bash
python -m demucs.separate --help
```

If that command works, WeaveShift can use Demucs with:

```bash
--vocal-method demucs
```

If Demucs is not installed or fails, use:

```bash
--vocal-method heuristic
```

On Python 3.13+ or 3.14+, `torchaudio` may require TorchCodec for saving Demucs output. `requirements.txt` includes the conditional dependency, but if you installed dependencies manually and see a TorchCodec error, run:

```bash
python -m pip install torchcodec
```

Demucs CLI fallback caches generated vocals under `.weaveshift_demucs_cache/`. This can use substantial disk space during final QA runs. It is safe to delete the cache when you no longer need it.

## Lightweight test-only install

For CI or quick code checks:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Optional editable install

You can run WeaveShift directly from the source tree with `python weaveshift_setlist.py ...`.

For command-line entry points, install it in editable mode from the repo root:

```bash
python -m pip install -e .
```

Then run:

```bash
weaveshift-setlist examples/setlist_example.json \
  -o output/weaveshift_demo.wav \
  --apply-manifest-settings

weaveshift-two-tracks "Track A.wav" "Track B.wav" \
  -o output/transition.wav \
  --mode profile \
  --profile edm

weaveshift-cues examples/setlist_example.json --out output/weaveshift_cues.csv
```

## Multi-track setlist usage

Setlist paths may be `.wav`, `.mp3`, or a mix of both. Every track must share the same sample rate and channel count. Pick output format with `-o` (`.wav` or `.mp3`).

```bash
python weaveshift_setlist.py examples/setlist_example.json \
  -o output/weaveshift_demo.wav \
  --transition-mode profile \
  --profile edm \
  --vocal-method heuristic \
  --make-snippets
```

Use different built-in profiles:

```bash
# EDM / workout / running mix
python weaveshift_setlist.py examples/setlist_example.json \
  -o output/edm_hour.wav \
  --transition-mode profile \
  --profile edm \
  --make-snippets

# Vocal trance / euphoric vocal EDM
python weaveshift_setlist.py examples/setlist_example.json \
  -o output/vocal_trance_hour.wav \
  --transition-mode profile \
  --profile vocal_trance \
  --vocal-method demucs \
  --make-snippets

# Lounge / speakeasy / around-the-world background mix
python weaveshift_setlist.py examples/setlist_example.json \
  -o output/lounge_hour.wav \
  --transition-mode profile \
  --profile lounge \
  --make-snippets

# Jazz / blues / organic live-feel sequencing
python weaveshift_setlist.py examples/setlist_example.json \
  -o output/jazz_hour.wav \
  --transition-mode profile \
  --profile jazz \
  --make-snippets

# Love songs / heartbreak / ballad arc
python weaveshift_setlist.py examples/setlist_example.json \
  -o output/songs_of_the_heart.wav \
  --transition-mode profile \
  --profile heart \
  --make-snippets

# Cinematic / atmospheric continuous listening
python weaveshift_setlist.py examples/setlist_example.json \
  -o output/cinematic_mix.wav \
  --transition-mode profile \
  --profile cinematic \
  --make-snippets
```

Use a custom TOML scoring profile:

```bash
python weaveshift_setlist.py examples/setlist_example.json \
  -o output/custom_mix.wav \
  --transition-mode profile \
  --scoring-config configs/heart.toml
```

Available setlist transition modes:

```text
recommended, vocal_safe, beat_aligned, quick_cut, smooth, profile, vocal_ducked, long_blend, natural
```

`recommended` selects the best technical candidate. `smooth` is the built-in EDM-style smooth handoff from earlier development. `profile` uses the selected TOML profile.

`--profile` and `--scoring-config` only affect runs where the mode is `profile`:

* `--transition-mode profile` for setlist runs.
* `--mode profile` for two-track transitions.

WeaveShift prints a warning if profile options are supplied with another mode.

## Per-transition overrides

Setlist runs can override individual transitions from the manifest `settings.transition_overrides` array. This is useful when the overall profile works well, but one specific junction needs a different transition style or exact handoff.

**`transition_overrides` are always honored** when they appear in the manifest JSON, regardless of CLI flags. Use `--apply-manifest-settings` to also copy other manifest `settings` keys (such as `transition_mode`, `profile`, and `vocal_method`) onto CLI defaults.

Transition indexes are 1-based junction numbers:

```text
index 1 = Track 1 → Track 2
index 2 = Track 2 → Track 3
index 6 = Track 6 → Track 7
```

Manifest settings behavior:

```bash
--apply-manifest-settings
```

When this flag is set, WeaveShift copies a fixed allowlist of manifest `settings` keys onto CLI defaults: `transition_mode`, `profile`, `scoring_config`, `vocal_method`, `make_snippets`, `natural_pause_sec`, `search_a_sec`, `search_b_sec`, `max_trim_a_sec`, and `b_cue_max_sec`. Explicit CLI values always win, and any other settings key (such as `output`) is ignored with a warning. **`transition_overrides` do not require this flag** — they are applied whenever present in the manifest.

### Candidate-mode override

Use a candidate-mode override when you want WeaveShift to force one transition to use a specific candidate style while still letting WeaveShift calculate the timing, cue, overlap, and gain for that candidate.

```json
{
  "tracks": [
    {"path": "A.wav", "title": "Track A", "bpm": 128},
    {"path": "B.wav", "title": "Track B", "bpm": 130},
    {"path": "C.wav", "title": "Track C", "bpm": 132}
  ],
  "settings": {
    "transition_mode": "profile",
    "profile": "vocal_trance",
    "vocal_method": "demucs",
    "transition_overrides": [
      {
        "index": 2,
        "from": "Track B",
        "to": "Track C",
        "mode": "quick_cut"
      }
    ]
  }
}
```

Supported override modes are the same candidate names used by setlist transitions:

```text
recommended, vocal_safe, beat_aligned, quick_cut, smooth, profile, vocal_ducked, long_blend
```

Setlist overrides also support the semantic `natural` and manual-only `handoff` modes described below.

The optional `from` and `to` fields can identify a junction when `index` is omitted; otherwise they are labels for humans and reports. WeaveShift prefers `index` when both are supplied.

### Natural track boundary

Use `natural` when the outgoing song should play through its final sample, followed by a normal silent gap, before the next song starts from source time zero. No fade, overlap, cue, or transition gain is applied. This behaves like consecutive tracks on a CD or in a streaming queue.

Apply it to every junction from the command line:

```bash
python weaveshift_setlist.py setlist.json \
  -o natural-listening-mix.wav \
  --transition-mode natural \
  --natural-pause-sec 1.5
```

Or use it for one junction in `settings.transition_overrides`:

```json
{
  "index": 3,
  "from": "Song A",
  "to": "Song B",
  "mode": "natural",
  "pause_sec": 1.5
}
```

`pause_sec` must be between 0 and 300 seconds and defaults to `--natural-pause-sec` (1.0 second by default). Silence already present at either end of the source files is preserved in addition to this pause.

### Manual parameter override

Use a manual override when your ears found an exact transition that works and you want the full setlist render to reproduce it. Manual overrides bypass candidate timing for that junction and force the specific fade, cue, overlap, and incoming gain values.

```json
{
  "settings": {
    "transition_mode": "profile",
    "profile": "vocal_trance",
    "vocal_method": "demucs",
    "transition_overrides": [
      {
        "index": 6,
        "from": "You Never Broke Me",
        "to": "Je Cherche La Lumière",
        "mode": "manual",
        "a_fade_start_sec": 253.531,
        "a_cut_sec": 256.531,
        "b_cue_sec": 3.883,
        "overlap_sec": 3.0,
        "b_gain_db": 3.01
      }
    ]
  }
}
```

Manual override fields:

| Field              | Meaning                                                |
| ------------------ | ------------------------------------------------------ |
| `index`            | 1-based transition number in the setlist               |
| `mode`             | Must be `manual` for exact parameter overrides         |
| `a_fade_start_sec` | Source time in Track A where the outgoing fade starts  |
| `a_cut_sec`        | Source time in Track A where Track A is fully cut      |
| `b_cue_sec`        | Source time in Track B where the incoming track begins |
| `overlap_sec`      | Length of the crossfade/overlap in seconds             |
| `b_gain_db`        | Gain applied to Track B during the transition          |

Validation keeps manual overrides inside the source material: all values must be finite numbers, `a_fade_start_sec` and `b_cue_sec` must be >= 0, `a_cut_sec` must be greater than `a_fade_start_sec` and within Track A, `b_cue_sec` must be within Track B, and `overlap_sec` must be > 0. WeaveShift rejects the manifest with a clear error rather than rendering from the wrong part of a track.

Manual overrides are especially helpful when a long crossfade causes a perceptual “switch forward, then switch back” effect. In that case, forcing a shorter overlap and a deeper Track B cue can sound cleaner than the technically highest-scoring candidate.

### Handoff override

Use a `handoff` override when Track A should fade down first and Track B should take over only near the end of that fade. This avoids muddy long blends while still avoiding hard silence or a click.

```json
{
  "settings": {
    "transition_overrides": [
      {
        "index": 9,
        "from": "Golden & Free",
        "to": "Half of Everything",
        "mode": "handoff",
        "a_fade_start_sec": 162.8,
        "a_cut_sec": 166.06,
        "b_cue_sec": 1.5,
        "takeover_overlap_sec": 0.25,
        "b_entry_gain_db": -1.0,
        "b_fade_in_sec": 0.75
      }
    ]
  }
}
```

Handoff fields:

| Field                  | Meaning                                                       |
| ---------------------- | ------------------------------------------------------------- |
| `index`                | 1-based transition number in the setlist                      |
| `mode`                 | Must be `handoff`                                             |
| `a_fade_start_sec`     | Source time in Track A where the outgoing fade starts         |
| `a_cut_sec`            | Source time in Track A where Track A is fully cut             |
| `b_cue_sec`            | Source time in Track B where the incoming track begins        |
| `takeover_overlap_sec` | How long Track B starts before Track A is fully gone          |
| `b_entry_gain_db`      | Gain applied to Track B entry                                 |
| `b_fade_in_sec`        | Track B fade-in duration; may continue after Track A is gone  |

Validation keeps handoffs short and intentional: `takeover_overlap_sec` must be 0.05-1.0 seconds, `b_fade_in_sec` must be 0.1-4.0 seconds and at least the takeover overlap, and `b_entry_gain_db` must be between -6.0 and +3.0 dB. As with manual overrides, all values must be finite, `a_fade_start_sec` and `b_cue_sec` must be >= 0, and the timing must fall within the source tracks.

### Example setlist command with overrides

```bash
python weaveshift_setlist.py examples/setlist_example.json \
  -o output/final_mix_with_overrides.wav \
  --transition-mode profile \
  --profile vocal_trance \
  --vocal-method demucs \
  --apply-manifest-settings \
  --make-snippets
```

When an override is applied, WeaveShift prints it in the transition log, for example:

```text
Using transition override for #6: You Never Broke Me → Je Cherche La Lumière = manual
Selected manual: score=0.926; new song starts in mix at 26:30, using Track B cue 0:04; overlap 3.0s, B gain +3.0 dB
```

The JSON report also includes the top-level `transition_overrides` array and marks the overridden transition notes with `transition override: ...`.

## Two-track transition usage

```bash
python weaveshift_two_tracks.py "Track A.wav" "Track B.wav" \
  -o output/transition.wav \
  --mode profile \
  --profile edm \
  --make-snippets
```

## Scoring profiles

Built-in profiles live under `configs/`:

* `edm.toml` — beat-aware, momentum-focused workout/running mixes.
* `vocal_trance.toml` — beat-aware euphoric vocal trance with stronger vocal and breakdown protection.
* `lounge.toml` — cinematic lounge, speakeasy, and global nightclub background mixes.
* `jazz.toml` — organic jazz/blues/live-feel sequencing with less beat-grid emphasis.
* `heart.toml` — love songs, heartbreak ballads, and emotional continuous listening.
* `cinematic.toml` — narrative/atmospheric continuous listening, soundtrack-like sets, and gentle background music.

Profiles control weights and limits such as vocal collision, beat alignment, overlap preference, gain adjustment penalty, tail-trim penalty, cue-depth penalty, and maximum allowed incoming gain.

In `profile` mode, profile limits also shape candidate generation. For example, `max_b_cue_sec`, `max_tail_trim_sec`, and preferred overlap ranges affect which transition candidates are searched, not just how already-found candidates are re-ranked.

### Profile selection guide

| Profile        | Best for                                                  | Transition personality                                                      |
| -------------- | --------------------------------------------------------- | --------------------------------------------------------------------------- |
| `edm`          | Running mixes, workout EDM, club sets                     | Momentum, beat alignment, energy continuity                                 |
| `vocal_trance` | Vocal trance, euphoric EDM, melodic running mixes         | Beat-aware but more protective of vocals, breakdowns, and reverb tails      |
| `lounge`       | Speakeasy, global nightclub, cafe/lounge background music | Gentle room-to-room flow, ambience, low gain jumps                          |
| `jazz`         | Jazz, blues, live-feel recordings                         | Phrase/decay preservation, low beat-grid pressure                           |
| `heart`        | Love songs, heartbreak ballads, emotional arcs            | Strong vocal protection, final chord preservation, emotional breathing room |
| `cinematic`    | Soundtrack-style, ambient, atmospheric, narrative mixes   | Long gentle fades, mood continuity, minimal abruptness                      |

### Profile files

A profile is a TOML file with three sections:

```toml
name = "my_profile"
description = "My custom transition personality."

[weights]
vocal_collision = 1.5
beat_alignment = 1.0
energy = 1.0
loudness = 1.0
compatibility = 1.0
placement = 1.0
overlap_preference = 1.0
gain_penalty = 1.0
tail_trim_penalty = 1.0
cue_depth_penalty = 1.0

[preferences]
target_overlap_sec = 8.0
preferred_overlap_min_sec = 4.0
preferred_overlap_max_sec = 12.0
prefer_beat_alignment = true
prefer_vocal_safety = true

[limits]
max_abs_b_gain_db = 3.0
max_tail_trim_sec = 5.0
max_b_cue_sec = 12.0
max_vocal_collision_score = 0.25
min_beat_alignment = 0.35
```

Run a custom file with:

```bash
python weaveshift_setlist.py examples/setlist_example.json \
  -o output/custom_profile_mix.wav \
  --transition-mode profile \
  --scoring-config configs/vocal_trance.toml \
  --make-snippets
```

Profile search bounds are intentionally profile-driven. For example, `vocal_trance.toml` can search deeper incoming cues than `edm.toml`, while `heart.toml` and `jazz.toml` can restrict tail trimming and vocal overlap more aggressively.

To change those behaviors, edit the TOML file or pass a custom profile with `--scoring-config`.

Built-in profile names passed to `--profile` (or via manifest settings) may only contain letters, digits, hyphens, and underscores. To load a profile from an arbitrary file path, use `--scoring-config` instead.

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

Relative paths are resolved relative to the manifest file. `~` paths are expanded.

Other manifest `settings` keys apply only when you pass `--apply-manifest-settings`, and only keys on the supported allowlist are copied (see [Per-transition overrides](#per-transition-overrides) for the full list); unsupported keys are ignored with a warning. Without the flag, CLI arguments control those defaults; **`transition_overrides` still apply** as described in [Per-transition overrides](#per-transition-overrides).

## Audio format policy

WeaveShift accepts **WAV and MP3 inputs in any combination** — a setlist can mix `.wav` and `.mp3` tracks. All tracks must share the same **sample rate** and **channel count**. When every input is WAV, subtype must also match (PCM_16, PCM_24, and so on).

**Output is one file**, either WAV or MP3 — pick with the `-o` suffix:

- `mix.wav` → lossless WAV (default when no suffix is given)
- `mix.mp3` → MP3 at 320 kbps

WAV export uses the first lossless WAV subtype found in the inputs (otherwise PCM_16). MP3 export ignores source subtype.

WAV masters are still recommended for final release quality; MP3 sources may add encoder delay or re-encoding artifacts at transitions.

`weaveshift_cues.py` accepts the same `.wav` / `.mp3` inputs for video cue-sheet generation.

## Suggested workflow

1. Build and audition with `--vocal-method heuristic`.
2. Iterate on track order and profile settings.
3. Listen to transition snippets first.
4. Use manual overrides for any transition that sounds better by ear than the automatic candidate.
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
  .gitignore

  weaveshift_setlist.py       # setlist CLI
  weaveshift_two_tracks.py    # two-track CLI
  weaveshift_cues.py          # video cue sheet helper
  weaveshift_profiles.py      # TOML scoring profiles
  weaveshift_audio.py         # analysis and validation
  weaveshift_scoring.py       # candidate search and scoring
  weaveshift_rendering.py     # transition rendering
  weaveshift_reports.py       # JSON/CSV report builders
  weaveshift_plan.py          # plan/execute separation

  configs/
  examples/
    fixtures/              # generated synthetic demo WAVs (see fixtures/README.md)
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
