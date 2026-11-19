# Per-Transition Overrides & Manifest Settings

Setlist runs can override individual transitions from the manifest `settings.transition_overrides` array. This is useful when the overall profile works well, but one specific junction needs a different transition style or exact handoff.

**`transition_overrides` are always honored** when they appear in the manifest JSON, regardless of CLI flags.

Transition indexes are 1-based junction numbers:

```text
index 1 = Track 1 → Track 2
index 2 = Track 2 → Track 3
index 6 = Track 6 → Track 7
```

## Applying other manifest settings

```bash
--apply-manifest-settings
```

When this flag is set, WeaveShift copies a fixed allowlist of manifest `settings` keys onto CLI defaults:

* `transition_mode`
* `profile`
* `vocal_method`
* `make_snippets`
* `natural_pause_sec`
* `search_a_sec`
* `search_b_sec`
* `max_trim_a_sec`
* `b_cue_max_sec`

Explicit CLI values always win, and any other settings key (such as `output` or `scoring_config`) is ignored with a warning. **`transition_overrides` do not require this flag** — they are applied whenever present in the manifest.

## Candidate-mode override

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

Supported candidate override modes:

```text
recommended, vocal_safe, beat_aligned, quick_cut, smooth, profile, vocal_ducked, long_blend
```

Setlist overrides also support the semantic `natural`, `manual`, and `handoff` modes described below.

The optional `from` and `to` fields can identify a junction when `index` is omitted; otherwise they are labels for humans and reports. WeaveShift prefers `index` when both are supplied.

## Natural track boundary

Use `natural` when the outgoing song should play through its final sample, followed by a normal silent gap, before the next song starts from source time zero. No fade, overlap, cue, or transition gain is applied. This behaves like consecutive tracks on a CD or in a streaming queue.

Apply it to every junction from the command line:

```bash
python -m weaveshift.setlist setlist.json \
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

`pause_sec` must be between `0` and `300` seconds and defaults to `--natural-pause-sec` (`1.0` second by default). Silence already present at either end of the source files is preserved in addition to this pause.

## Manual parameter override

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

| Field              | Meaning                                                                                                         |
| ------------------ | --------------------------------------------------------------------------------------------------------------- |
| `index`            | 1-based transition number in the setlist                                                                        |
| `mode`             | Must be `manual` for exact parameter overrides                                                                  |
| `a_fade_start_sec` | Source time in Track A where the outgoing fade starts                                                           |
| `a_cut_sec`        | Source time in Track A where Track A is fully cut                                                               |
| `b_cue_sec`        | Source time in Track B where the incoming track begins                                                          |
| `overlap_sec`      | Length of the crossfade/overlap in seconds                                                                      |
| `b_gain_db`        | Gain applied to Track B during the transition                                                                   |
| `soft_duck_db`     | Optional (manual overrides only) dip in dB (≤ 0) applied to the ducked track; bare `soft_duck_db` targets `a`   |
| `soft_duck_target` | Optional `a`, `b`, or `none`: which track is ducked (`a` or `b` requires a negative `soft_duck_db`)             |

Validation keeps manual overrides inside the source material: all values must be finite numbers, `a_fade_start_sec` and `b_cue_sec` must be `>= 0`, `a_cut_sec` must be greater than `a_fade_start_sec` and within Track A, `b_cue_sec` must be within Track B, and `overlap_sec` must be `> 0`. WeaveShift rejects the manifest with a clear error rather than rendering from the wrong part of a track.

Manual overrides are especially helpful when a long crossfade causes a perceptual "switch forward, then switch back" effect. In that case, forcing a shorter overlap and a deeper Track B cue can sound cleaner than the technically highest-scoring candidate.

## Handoff override

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

Validation keeps handoffs short and intentional: `takeover_overlap_sec` must be `0.05`–`1.0` seconds, `b_fade_in_sec` must be `0.1`–`4.0` seconds and at least `takeover_overlap_sec`, and `b_entry_gain_db` must be between `-6.0` and `+3.0` dB. As with manual overrides, all values must be finite, `a_fade_start_sec` and `b_cue_sec` must be `>= 0`, and timing must fall within the source tracks.

## Running with overrides

```bash
python -m weaveshift.setlist examples/setlist_example.json \
  -o output/final_mix_with_overrides.wav \
  --transition-mode profile \
  --profile vocal_trance \
  --vocal-method demucs \
  --apply-manifest-settings \
  --make-snippets
```

When an override is applied, WeaveShift logs it during rendering:

```text
Using transition override for #6: You Never Broke Me → Je Cherche La Lumière = manual
Selected manual: score=0.926; new song starts in mix at 26:30, using Track B cue 0:04; overlap 3.0s, B gain +3.0 dB
```

The JSON report also includes the top-level `transition_overrides` array and marks the overridden transition notes with `transition override: ...`.
