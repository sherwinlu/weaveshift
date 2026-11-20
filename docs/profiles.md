# Scoring Profiles

Built-in profiles ship inside the package under `weaveshift/configs/` and are selected with `--profile` when running in `profile` mode:

* `--transition-mode profile` for setlist runs (`weaveshift-setlist`).
* `--mode profile` for two-track transitions (`weaveshift-two-tracks`).

WeaveShift prints a warning if a non-default `--profile` or `--scoring-config` is supplied with another mode.

## Built-in profiles

| Profile               | Best for                                                      | Transition personality                                                                          |
| --------------------- | ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `edm`                 | Running mixes, workout EDM, club sets                         | Momentum, beat alignment, energy continuity (2–8s overlaps)                                     |
| `smooth`              | Balanced vocal-aware blends across general electronic sets    | 6–10s blends, modest gain (≤ 2.5 dB), shallow cues, strong vocal protection                     |
| `vocal_trance`        | Vocal trance, euphoric EDM, melodic running mixes             | Beat-aware with stronger vocal, breakdown, and reverb-tail protection (4–12s overlaps)          |
| `vocal_trance_strict` | Vocal-heavy trance where vocal overlap must be tightly capped | Stricter vocal collision cap (0.10), tighter gain (≤ 2.0 dB), shorter 4–6s overlaps, deeper cue |
| `lounge`              | Speakeasy, global nightclub, cafe/lounge background music     | Gentle room-to-room flow, ambience, low gain jumps (6–15s overlaps, low beat-grid emphasis)     |
| `jazz`                | Jazz, blues, live-feel recordings                             | Phrase/decay preservation, minimal tail trimming, low beat-grid pressure                        |
| `heart`               | Love songs, heartbreak ballads, emotional arcs                | Strong vocal protection, final-chord preservation, emotional breathing room                     |
| `cinematic`           | Soundtrack-style, ambient, atmospheric, narrative mixes       | Long gentle fades (6–18s), mood continuity, minimal abruptness                                  |

Profiles control weights and limits such as vocal collision, beat alignment, overlap preference, gain adjustment penalty, tail-trim penalty, cue-depth penalty, and maximum allowed incoming gain.

In `profile` mode, profile limits also shape candidate generation: `max_b_cue_sec`, `max_tail_trim_sec`, and preferred overlap ranges affect which transition candidates are searched, not just how already-found candidates are re-ranked.

## Usage examples

```bash
# EDM / workout / running mix
python -m weaveshift.setlist examples/setlist_example.json \
  -o output/edm_hour.wav \
  --transition-mode profile \
  --profile edm \
  --make-snippets

# Vocal trance / euphoric vocal EDM
python -m weaveshift.setlist examples/setlist_example.json \
  -o output/vocal_trance_hour.wav \
  --transition-mode profile \
  --profile vocal_trance \
  --vocal-method demucs \
  --make-snippets

# Lounge / speakeasy / around-the-world background mix
python -m weaveshift.setlist examples/setlist_example.json \
  -o output/lounge_hour.wav \
  --transition-mode profile \
  --profile lounge \
  --make-snippets

# Jazz / blues / organic live-feel sequencing
python -m weaveshift.setlist examples/setlist_example.json \
  -o output/jazz_hour.wav \
  --transition-mode profile \
  --profile jazz \
  --make-snippets

# Love songs / heartbreak / ballad arc
python -m weaveshift.setlist examples/setlist_example.json \
  -o output/songs_of_the_heart.wav \
  --transition-mode profile \
  --profile heart \
  --make-snippets

# Cinematic / atmospheric continuous listening
python -m weaveshift.setlist examples/setlist_example.json \
  -o output/cinematic_mix.wav \
  --transition-mode profile \
  --profile cinematic \
  --make-snippets
```

## Custom TOML profile files

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

Load a custom profile with `--scoring-config` (which overrides `--profile`):

```bash
python -m weaveshift.setlist examples/setlist_example.json \
  -o output/custom_profile_mix.wav \
  --transition-mode profile \
  --scoring-config weaveshift/configs/vocal_trance.toml \
  --make-snippets
```

Profile search bounds are intentionally profile-driven. For example, `vocal_trance.toml` can search deeper incoming cues than `edm.toml`, while `heart.toml` and `jazz.toml` restrict tail trimming and vocal overlap more aggressively.

## Profile name validation

Built-in profile names passed to `--profile` (or via manifest settings) may only contain letters, digits, hyphens, and underscores. To load a profile from an arbitrary file path, pass `--scoring-config` on the CLI instead (`scoring_config` cannot be set from a manifest).
