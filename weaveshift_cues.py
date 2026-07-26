#!/usr/bin/env python3
"""
weaveshift_cues.py

Generate candidate video cue points for a long-form WeaveShift / EDM running mix.

What it does:
- Reads a track manifest CSV.
- Optionally reads a WeaveShift timeline/report CSV containing final mix placement.
- Analyzes each source audio file with librosa.
- Suggests cue candidates:
    first_lift_candidate
    main_peak_candidate
    second_peak_candidate
    breakdown_candidate
    transition_out
- Exports a reviewable CSV for video production.

This is NOT a perfect "drop detector." It is a practical cue helper:
use the output to jump to likely moments, then confirm by ear.

Typical usage:

    python weaveshift_cues.py tracks.csv --timeline weaveshift_report.csv --out cue_sheet.csv

Minimum tracks.csv columns:
    path,title

Helpful optional columns:
    bpm,key,visual_mood

Example tracks.csv:
    path,title,bpm,key,visual_mood
    audio/demo_track_01.wav,Demo Track 01,110,E major / 12B,warm-up
    audio/demo_track_02.mp3,Demo Track 02,126,E major / 12B,build

Supported audio inputs: `.wav`, `.wave`, and `.mp3` (file formats only — use masters you have rights to).

Timeline CSV optional columns. The script tries several common names:
    title
    track_start_sec OR track_source_zero_in_mix_sec OR start_sec
    transition_out_sec OR mix_transition_start_sec OR end_sec
    overlap_sec

If no timeline is supplied, final_mix timestamps will be blank and source-relative
timestamps will still be produced.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import librosa
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: librosa\n\n"
        "Install with:\n"
        "  python -m pip install librosa soundfile numpy\n"
    ) from exc

from weaveshift_audio import load_mono_analysis_audio, lossy_input_note, validate_audio_input


logger = logging.getLogger(__name__)


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class Track:
    path: str
    title: str
    bpm: Optional[float] = None
    key: str = ""
    visual_mood: str = ""


@dataclass
class TimelineRow:
    title: str
    start_sec: Optional[float] = None
    transition_out_sec: Optional[float] = None
    overlap_sec: Optional[float] = None


@dataclass
class CueResult:
    title: str
    path: str
    bpm_manifest: Optional[float]
    bpm_detected: Optional[float]
    key: str
    duration_sec: float
    final_start_sec: Optional[float]
    transition_out_sec: Optional[float]
    first_lift_sec: Optional[float]
    main_peak_sec: Optional[float]
    second_peak_sec: Optional[float]
    breakdown_sec: Optional[float]
    visual_mood: str
    confidence: str
    notes: str


# -----------------------------
# Utility
# -----------------------------

def parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    # Accept mm:ss or hh:mm:ss.
    if ":" in s:
        return timestamp_to_sec(s)
    try:
        return float(s)
    except ValueError:
        return None


def timestamp_to_sec(ts: str) -> Optional[float]:
    ts = ts.strip()
    if not ts:
        return None
    parts = ts.split(":")
    try:
        parts_f = [float(p) for p in parts]
    except ValueError:
        return None
    if len(parts_f) == 2:
        m, s = parts_f
        return m * 60 + s
    if len(parts_f) == 3:
        h, m, s = parts_f
        return h * 3600 + m * 60 + s
    return None


def sec_to_timestamp(sec: Optional[float]) -> str:
    if sec is None or math.isnan(sec):
        return ""
    sec = max(0.0, float(sec))
    total = int(round(sec))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def normalize_title(title: str) -> str:
    s = title.lower()
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def get_first(row: Dict[str, str], names: Iterable[str]) -> str:
    lower_map = {k.lower().strip(): v for k, v in row.items()}
    for name in names:
        key = name.lower().strip()
        if key in lower_map:
            return lower_map[key]
    return ""


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_tracks(path: Path) -> List[Track]:
    rows = read_csv_dicts(path)
    tracks: List[Track] = []
    for row in rows:
        audio_path = get_first(row, ["path", "file", "filepath", "audio_path", "track_path"])
        title = get_first(row, ["title", "track", "name", "track_title"])
        if not audio_path:
            raise SystemExit("Track manifest row is missing a path/file/audio_path column.")
        if not title:
            title = Path(audio_path).stem
        tracks.append(
            Track(
                path=audio_path,
                title=title,
                bpm=parse_float(get_first(row, ["bpm", "tempo"])) ,
                key=get_first(row, ["key", "camelot_key", "musical_key"]),
                visual_mood=get_first(row, ["visual_mood", "mood", "visual", "prompt_mood"]),
            )
        )
    return tracks


def read_timeline(path: Optional[Path]) -> Dict[str, TimelineRow]:
    if path is None:
        return {}

    rows = read_csv_dicts(path)
    out: Dict[str, TimelineRow] = {}

    for row in rows:
        title = get_first(row, ["title", "track", "name", "track_title", "track_b_title"])
        if not title:
            # Some WeaveShift transition rows may describe pairwise transitions.
            title = get_first(row, ["incoming_title", "to_title", "b_title"])
        if not title:
            continue

        start = parse_float(get_first(row, [
            "track_source_zero_in_mix_sec",
            "track_start_sec",
            "start_sec",
            "mix_start_sec",
            "final_start_sec",
            "start",
        ]))

        transition_out = parse_float(get_first(row, [
            "transition_out_sec",
            "mix_transition_start_sec",
            "transition_start_sec",
            "next_transition_start_sec",
            "end_sec",
            "end",
        ]))

        overlap = parse_float(get_first(row, [
            "overlap_sec",
            "selected_overlap_sec",
            "transition_overlap_sec",
        ]))

        out[normalize_title(title)] = TimelineRow(
            title=title,
            start_sec=start,
            transition_out_sec=transition_out,
            overlap_sec=overlap,
        )

    return out


def resolve_audio_path(audio_path: str, manifest_path: Path, audio_root: Optional[Path]) -> Path:
    p = Path(audio_path)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        if audio_root:
            candidates.append(audio_root / p)
            candidates.append(audio_root / p.name)
        candidates.append(manifest_path.parent / p)
        candidates.append(Path.cwd() / p)

    for c in candidates:
        if c.exists():
            return c

    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Audio file not found for {audio_path}. Tried:\n  {tried}")


# -----------------------------
# Cue analysis
# -----------------------------

def median_step_sec(times: np.ndarray, default: float = 0.5) -> float:
    """Median frame spacing in seconds; falls back for degenerate short tracks.

    np.median(np.diff(times)) is NaN for a single-frame track, which would
    crash the int(round(...)) window-size math downstream.
    """
    if len(times) > 1:
        step = float(np.median(np.diff(times)))
        if math.isfinite(step) and step > 0:
            return step
    return default


def moving_average(x: np.ndarray, n: int) -> np.ndarray:
    n = max(1, int(n))
    if n == 1:
        return x.astype(float)
    kernel = np.ones(n, dtype=float) / n
    return np.convolve(x, kernel, mode="same")


def robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad < 1e-9:
        std = np.std(x) + 1e-9
        return (x - med) / std
    return 0.6745 * (x - med) / mad


def pick_local_peaks(
    times: np.ndarray,
    score: np.ndarray,
    min_time: float,
    max_time: float,
    min_distance_sec: float,
    count: int,
) -> List[float]:
    """Simple local-max picker without scipy."""
    if len(times) < 3:
        return []

    mask = (times >= min_time) & (times <= max_time)
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        return []

    candidates = []
    for i in idxs:
        if i <= 0 or i >= len(score) - 1:
            continue
        if score[i] >= score[i - 1] and score[i] >= score[i + 1]:
            candidates.append((float(score[i]), float(times[i])))

    candidates.sort(reverse=True, key=lambda x: x[0])

    chosen: List[float] = []
    for _, t in candidates:
        if all(abs(t - existing) >= min_distance_sec for existing in chosen):
            chosen.append(t)
        if len(chosen) >= count:
            break

    return sorted(chosen)


def first_sustained_energy_lift(
    times: np.ndarray,
    energy: np.ndarray,
    duration_sec: float,
) -> Optional[float]:
    """Find the first point where smoothed energy meaningfully exceeds intro baseline."""
    if len(times) == 0:
        return None

    intro_end = min(max(20.0, duration_sec * 0.12), 45.0)
    search_start = min(8.0, duration_sec * 0.05)
    search_end = max(search_start + 5.0, duration_sec * 0.55)

    intro_mask = times <= intro_end
    if not np.any(intro_mask):
        baseline = float(np.percentile(energy, 35))
    else:
        baseline = float(np.percentile(energy[intro_mask], 60))

    high = float(np.percentile(energy, 72))
    threshold = baseline + 0.45 * (high - baseline)

    # Require a small sustained region above threshold.
    frame_step = median_step_sec(times)
    sustain_frames = max(3, int(round(4.0 / max(frame_step, 0.1))))

    mask = (times >= search_start) & (times <= search_end)
    idxs = np.where(mask)[0]
    for i in idxs:
        j = min(len(energy), i + sustain_frames)
        if j <= i:
            continue
        if np.mean(energy[i:j] >= threshold) >= 0.75:
            return float(times[i])

    return None


def find_breakdown_candidate(
    times: np.ndarray,
    energy: np.ndarray,
    duration_sec: float,
) -> Optional[float]:
    """Find a low-energy valley after the track is underway."""
    if len(times) == 0 or duration_sec < 90:
        return None

    min_time = max(45.0, duration_sec * 0.25)
    max_time = duration_sec * 0.82
    mask = (times >= min_time) & (times <= max_time)
    if not np.any(mask):
        return None

    idxs = np.where(mask)[0]
    # Pick the lowest smoothed energy point, but avoid tiny local gaps.
    i = int(idxs[np.argmin(energy[idxs])])
    return float(times[i])


def analyze_track(audio_file: Path, bpm_hint: Optional[float]) -> Tuple[float, Optional[float], Dict[str, Optional[float]], str, str]:
    # Avoid librosa.load(): it unconditionally touches audioread (broken on Python 3.13+).
    # Read WAV via soundfile and MP3 via pydub/ffmpeg through weaveshift_audio instead.
    y_native, sr_native, _duration_probe = load_mono_analysis_audio(audio_file)
    sr = 22050
    y = librosa.resample(y_native, orig_sr=sr_native, target_sr=sr) if sr_native != sr else y_native
    duration_sec = float(librosa.get_duration(y=y, sr=sr))

    hop_length = 512

    # Energy.
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    frame_step = median_step_sec(times)
    rms_db = librosa.power_to_db(np.maximum(rms, 1e-10) ** 2, ref=np.max)
    energy = moving_average(rms_db, max(3, int(round(2.5 / frame_step))))

    # Onset / rhythmic intensity.
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=hop_length)
    onset_smooth = moving_average(onset_env, max(3, int(round(2.0 / median_step_sec(onset_times)))))

    # Align onset score to energy frames if needed.
    onset_interp = np.interp(times, onset_times, onset_smooth)

    # Spectral brightness / flux proxy.
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    centroid = np.interp(times, librosa.frames_to_time(np.arange(len(centroid)), sr=sr, hop_length=hop_length), centroid)
    centroid_z = robust_z(moving_average(centroid, max(3, int(round(3.0 / frame_step)))))

    energy_z = robust_z(energy)
    onset_z = robust_z(onset_interp)

    # Candidate "drop/peak" score:
    # high loudness + high onset/rhythm + brightness. This is a cue candidate, not truth.
    score = 0.55 * energy_z + 0.35 * onset_z + 0.10 * centroid_z
    score = moving_average(score, max(3, int(round(1.5 / frame_step))))

    # BPM detection.
    bpm_detected = None
    try:
        tempo_raw = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop_length, aggregate=np.median)
        if len(tempo_raw):
            bpm_detected = float(tempo_raw[0])
            if bpm_hint:
                candidates = [bpm_detected, bpm_detected * 2, bpm_detected / 2]
                bpm_detected = min(candidates, key=lambda x: abs(x - bpm_hint))
    except Exception as exc:
        logger.warning(
            "BPM detection failed for %s; continuing without detected tempo. stage=bpm reason=%s",
            audio_file,
            type(exc).__name__,
        )
        bpm_detected = None

    # Avoid first/last few seconds and avoid fade-out regions.
    peak_min = max(20.0, duration_sec * 0.12)
    peak_max = max(peak_min + 10.0, duration_sec * 0.90)
    peak_candidates = pick_local_peaks(
        times=times,
        score=score,
        min_time=peak_min,
        max_time=peak_max,
        min_distance_sec=35.0,
        count=4,
    )

    first_lift = first_sustained_energy_lift(times, energy, duration_sec)

    main_peak = peak_candidates[0] if peak_candidates else None
    second_peak = None

    if peak_candidates:
        # Prefer a second peak later than main_peak. If not, choose another distinct candidate.
        later = [p for p in peak_candidates if main_peak is not None and p > main_peak + 35.0]
        if later:
            second_peak = later[0]
        elif len(peak_candidates) > 1:
            second_peak = peak_candidates[1]

    breakdown = find_breakdown_candidate(times, energy, duration_sec)

    confidence_bits = []
    if main_peak is not None:
        # Crude confidence from peak score percentile.
        score_at_peak = float(np.interp(main_peak, times, score))
        pctl = float(np.mean(score <= score_at_peak))
        if pctl > 0.92:
            confidence_bits.append("high")
        elif pctl > 0.80:
            confidence_bits.append("medium")
        else:
            confidence_bits.append("low")
    else:
        confidence_bits.append("low")

    notes = []
    if bpm_hint and bpm_detected and abs(bpm_detected - bpm_hint) > 4:
        notes.append(f"Detected BPM {bpm_detected:.1f} differs from manifest BPM {bpm_hint:.1f}; use manifest for video grid.")
    if duration_sec < 120:
        notes.append("Short track; second_peak may be unreliable.")
    if main_peak is None:
        notes.append("No strong peak found; mark manually by ear.")

    cues = {
        "first_lift": first_lift,
        "main_peak": main_peak,
        "second_peak": second_peak,
        "breakdown": breakdown,
    }

    return duration_sec, bpm_detected, cues, confidence_bits[0], " ".join(notes)


def add_final_time(final_start: Optional[float], source_time: Optional[float]) -> Optional[float]:
    if final_start is None or source_time is None:
        return None
    return final_start + source_time


# -----------------------------
# Output
# -----------------------------

def write_results(path: Path, results: List[CueResult]) -> None:
    fieldnames = [
        "title",
        "path",
        "bpm_manifest",
        "bpm_detected",
        "key",
        "duration_sec",
        "duration_timestamp",

        "final_start_sec",
        "final_start_timestamp",

        "first_lift_source_sec",
        "first_lift_source_timestamp",
        "first_lift_final_sec",
        "first_lift_final_timestamp",

        "main_peak_source_sec",
        "main_peak_source_timestamp",
        "main_peak_final_sec",
        "main_peak_final_timestamp",

        "second_peak_source_sec",
        "second_peak_source_timestamp",
        "second_peak_final_sec",
        "second_peak_final_timestamp",

        "breakdown_source_sec",
        "breakdown_source_timestamp",
        "breakdown_final_sec",
        "breakdown_final_timestamp",

        "transition_out_sec",
        "transition_out_timestamp",

        "visual_mood",
        "confidence",
        "notes",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            first_final = add_final_time(r.final_start_sec, r.first_lift_sec)
            main_final = add_final_time(r.final_start_sec, r.main_peak_sec)
            second_final = add_final_time(r.final_start_sec, r.second_peak_sec)
            breakdown_final = add_final_time(r.final_start_sec, r.breakdown_sec)

            writer.writerow({
                "title": r.title,
                "path": r.path,
                "bpm_manifest": "" if r.bpm_manifest is None else f"{r.bpm_manifest:.2f}",
                "bpm_detected": "" if r.bpm_detected is None else f"{r.bpm_detected:.2f}",
                "key": r.key,
                "duration_sec": f"{r.duration_sec:.3f}",
                "duration_timestamp": sec_to_timestamp(r.duration_sec),

                "final_start_sec": "" if r.final_start_sec is None else f"{r.final_start_sec:.3f}",
                "final_start_timestamp": sec_to_timestamp(r.final_start_sec),

                "first_lift_source_sec": "" if r.first_lift_sec is None else f"{r.first_lift_sec:.3f}",
                "first_lift_source_timestamp": sec_to_timestamp(r.first_lift_sec),
                "first_lift_final_sec": "" if first_final is None else f"{first_final:.3f}",
                "first_lift_final_timestamp": sec_to_timestamp(first_final),

                "main_peak_source_sec": "" if r.main_peak_sec is None else f"{r.main_peak_sec:.3f}",
                "main_peak_source_timestamp": sec_to_timestamp(r.main_peak_sec),
                "main_peak_final_sec": "" if main_final is None else f"{main_final:.3f}",
                "main_peak_final_timestamp": sec_to_timestamp(main_final),

                "second_peak_source_sec": "" if r.second_peak_sec is None else f"{r.second_peak_sec:.3f}",
                "second_peak_source_timestamp": sec_to_timestamp(r.second_peak_sec),
                "second_peak_final_sec": "" if second_final is None else f"{second_final:.3f}",
                "second_peak_final_timestamp": sec_to_timestamp(second_final),

                "breakdown_source_sec": "" if r.breakdown_sec is None else f"{r.breakdown_sec:.3f}",
                "breakdown_source_timestamp": sec_to_timestamp(r.breakdown_sec),
                "breakdown_final_sec": "" if breakdown_final is None else f"{breakdown_final:.3f}",
                "breakdown_final_timestamp": sec_to_timestamp(breakdown_final),

                "transition_out_sec": "" if r.transition_out_sec is None else f"{r.transition_out_sec:.3f}",
                "transition_out_timestamp": sec_to_timestamp(r.transition_out_sec),

                "visual_mood": r.visual_mood,
                "confidence": r.confidence,
                "notes": r.notes,
            })


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate candidate WeaveShift video cue points from source tracks."
    )
    parser.add_argument(
        "manifest",
        type=Path,
        help="Track manifest CSV. Minimum columns: path,title. Paths may be .wav or .mp3.",
    )
    parser.add_argument(
        "--timeline",
        type=Path,
        default=None,
        help="Optional WeaveShift timeline/report CSV with final mix track starts and transition markers.",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=None,
        help="Optional root directory for audio files in the manifest.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("weaveshift_cues.csv"),
        help="Output cue CSV path. Default: weaveshift_cues.csv",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip missing audio files instead of failing.",
    )

    args = parser.parse_args()

    tracks = read_tracks(args.manifest)
    timeline = read_timeline(args.timeline)

    results: List[CueResult] = []

    for idx, track in enumerate(tracks, start=1):
        print(f"[{idx}/{len(tracks)}] Analyzing: {track.title}")

        try:
            audio_file = resolve_audio_path(track.path, args.manifest, args.audio_root)
        except FileNotFoundError as exc:
            if args.skip_missing:
                print(f"  SKIP: {exc}")
                continue
            raise

        validate_audio_input(str(audio_file), track.title)
        note = lossy_input_note(audio_file)
        if note:
            print(f"  {note}")

        trow = timeline.get(normalize_title(track.title), TimelineRow(title=track.title))

        try:
            duration, bpm_detected, cues, confidence, notes = analyze_track(audio_file, track.bpm)
        except Exception as exc:
            if args.skip_missing:
                print(f"  SKIP analysis error: {exc}")
                continue
            raise

        results.append(
            CueResult(
                title=track.title,
                path=str(audio_file),
                bpm_manifest=track.bpm,
                bpm_detected=bpm_detected,
                key=track.key,
                duration_sec=duration,
                final_start_sec=trow.start_sec,
                transition_out_sec=trow.transition_out_sec,
                first_lift_sec=cues.get("first_lift"),
                main_peak_sec=cues.get("main_peak"),
                second_peak_sec=cues.get("second_peak"),
                breakdown_sec=cues.get("breakdown"),
                visual_mood=track.visual_mood,
                confidence=confidence,
                notes=notes,
            )
        )

    write_results(args.out, results)

    print()
    print(f"Wrote cue sheet: {args.out}")
    print()
    print("Review tips:")
    print("- Treat main_peak as the first place to check for the drop/chorus.")
    print("- If a cue is off, move it by ear in your editor.")
    print("- For video editors, useful audio_start values are usually main_peak_final_sec - 4 to main_peak_final_sec.")
    print("- Use transition_out_timestamp for visual crossfades between track worlds.")


if __name__ == "__main__":
    main()
