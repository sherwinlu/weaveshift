"""Report serialization and human-readable candidate summaries."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Mapping, Sequence

from weaveshift_audio import (
    AudioAnalysis,
    NaturalTransition,
    SetlistTransition,
    TransitionCandidate,
    format_timestamp,
)

TWO_TRACK_REPORT_SCHEMA_VERSION = "1.0.0"
SETLIST_REPORT_SCHEMA_VERSION = "1.0.0"

TWO_TRACK_REPORT_REQUIRED_KEYS = (
    "app",
    "schema_version",
    "recommended_candidate",
    "mode",
    "ranked_candidates",
    "track_a",
    "track_b",
    "outputs",
)

SETLIST_REPORT_REQUIRED_KEYS = (
    "app",
    "schema_version",
    "output_file",
    "tracks",
    "transitions",
)

RANKED_CANDIDATE_REQUIRED_KEYS = (
    "rank",
    "name",
    "score",
    "recommendation",
    "verdict",
    "transition",
    "component_scores",
    "notes",
)

TRANSITION_KEYS = (
    "a_fade_start_sec",
    "a_cut_sec",
    "b_cue_sec",
    "overlap_sec",
    "b_gain_db",
    "trim_a_tail_sec",
    "new_song_starts_in_mix_sec",
    "new_song_starts_in_mix_timestamp",
    "track_b_source_cue_timestamp",
)


SETLIST_TRANSITION_KEYS = (
    "index",
    "selected_candidate",
    "mix_transition_start_sec",
    "source_a_fade_start_sec",
    "source_b_cue_sec",
    "overlap_sec",
    "ranked_candidates",
)


def validate_report_schema(report: Mapping[str, Any], *, kind: str = "two_track") -> None:
    """Raise ValueError when a report dict is missing required fields."""
    if kind == "two_track":
        required = TWO_TRACK_REPORT_REQUIRED_KEYS
    elif kind == "setlist":
        required = SETLIST_REPORT_REQUIRED_KEYS
    else:
        raise ValueError(f"Unknown report kind: {kind}")

    missing = [key for key in required if key not in report]
    if missing:
        raise ValueError(f"Report missing required keys: {', '.join(missing)}")

    ranked = report.get("ranked_candidates")
    if ranked is not None:
        for i, item in enumerate(ranked):
            item_missing = [key for key in RANKED_CANDIDATE_REQUIRED_KEYS if key not in item]
            if item_missing:
                raise ValueError(f"ranked_candidates[{i}] missing keys: {', '.join(item_missing)}")
            transition = item.get("transition") or {}
            transition_missing = [key for key in TRANSITION_KEYS if key not in transition]
            if transition_missing:
                raise ValueError(f"ranked_candidates[{i}].transition missing keys: {', '.join(transition_missing)}")

    if kind == "setlist":
        transitions = report.get("transitions")
        if transitions is not None:
            for i, item in enumerate(transitions):
                item_missing = [key for key in SETLIST_TRANSITION_KEYS if key not in item]
                if item_missing:
                    raise ValueError(f"transitions[{i}] missing keys: {', '.join(item_missing)}")


def build_two_track_report(
    *,
    app_name: str,
    mode: str,
    recommended: str | None,
    ranked: Sequence[Mapping[str, Any]],
    track_a: AudioAnalysis,
    track_b: AudioAnalysis,
    outputs: Sequence[Mapping[str, Any]],
    wav_format_validation: Mapping[str, Any],
    profile: str | None,
    scoring_config: str | None,
) -> Dict[str, Any]:
    report = {
        "app": app_name,
        "schema_version": TWO_TRACK_REPORT_SCHEMA_VERSION,
        "wav_format_validation": dict(wav_format_validation),
        "input_policy": "WAV and MP3 full-mix inputs. Output is a single .wav or .mp3 file. Stem analysis/rendering deferred.",
        "recommended_candidate": recommended,
        "mode": mode,
        "profile": profile,
        "scoring_config": scoring_config,
        "ranked_candidates": list(ranked),
        "notes": [
            "No time-stretching is applied. Finished masters keep original tempo and pitch.",
            "WAV and MP3 inputs may be mixed in one setlist when sample rate and channel count match.",
            "Output format is chosen from the destination suffix (.wav or .mp3), not both.",
            "Demucs vocal separation is used when available/requested; heuristic fallback is used otherwise.",
            "Stem-file processing is intentionally deferred to a later version.",
            "Precomputes stereo RMS and short-term LUFS curves in memory; scoring does not read audio from disk.",
            "Forces an explicit ffmpeg PCM codec at export time matching the validated source WAV subtype.",
            "Widens analysis windows when needed so local RMS curves cover max trim/cue plus maximum overlap.",
            "Post-overlap recovery uses a sample-smooth NumPy gain envelope.",
            "Candidates are ranked best-to-worst by the technical scoring matrix, but final transition choice should still be auditioned.",
        ],
        "track_a": serialize_analysis(track_a),
        "track_b": serialize_analysis(track_b),
        "outputs": list(outputs),
    }
    validate_report_schema(report, kind="two_track")
    return report


def build_setlist_report(
    *,
    app_name: str,
    transition_mode: str,
    profile: str | None,
    scoring_config: str | None,
    vocal_method: str,
    output_file: str,
    output_duration_sec: float,
    output_duration_timestamp: str,
    final_gain_db: float,
    wav_format_validation: Mapping[str, Any],
    tracks: Sequence[Mapping[str, Any]],
    track_source_zero_in_mix_sec: Sequence[float],
    transitions: Sequence[Mapping[str, Any]],
    transition_overrides: Any,
    snippet_files: Sequence[str],
) -> Dict[str, Any]:
    report = {
        "app": app_name,
        "schema_version": SETLIST_REPORT_SCHEMA_VERSION,
        "input_policy": "WAV and MP3 full-mix inputs. Output is a single .wav or .mp3 file. Stem analysis/rendering deferred.",
        "transition_mode": transition_mode,
        "transition_overrides": transition_overrides,
        "profile": profile,
        "scoring_config": scoring_config,
        "vocal_method": vocal_method,
        "output_file": output_file,
        "output_duration_sec": output_duration_sec,
        "output_duration_timestamp": output_duration_timestamp,
        "final_gain_db": final_gain_db,
        "wav_format_validation": dict(wav_format_validation),
        "tracks": list(tracks),
        "track_source_zero_in_mix_sec": list(track_source_zero_in_mix_sec),
        "transitions": list(transitions),
        "snippet_files": list(snippet_files),
    }
    validate_report_schema(report, kind="setlist")
    return report

def serialize_analysis(a: AudioAnalysis) -> dict[str, Any]:
    """JSON-safe analysis summary for reports (curves omitted; they are large and in-memory only)."""
    d = asdict(a)
    d.pop("energy_curve", None)
    d.pop("loudness_curve", None)
    d["vocal_segments"] = [asdict(v) for v in a.vocal_segments]
    if a.energy_curve is not None:
        d["energy_curve_points"] = int(len(a.energy_curve.get("times", [])))
    if a.loudness_curve is not None:
        d["loudness_curve_points"] = int(len(a.loudness_curve.get("times", [])))
    return d


def candidate_verdict(c: SetlistTransition) -> str:
    """Human-readable recommendation note for ranked reports."""
    if isinstance(c, NaturalTransition):
        return f"Natural track boundary with {c.pause_sec:.1f}s of silence."
    cautions = []
    if c.vocal_collision_score > 0.25:
        cautions.append("possible vocal overlap")
    if c.beat_alignment_score < 0.55:
        cautions.append("loose beat alignment")
    if c.trim_a_tail_sec > 3.0:
        cautions.append(f"trims {c.trim_a_tail_sec:.1f}s from Track A")
    if abs(c.b_gain_db) > 3.0:
        cautions.append(f"large B gain adjustment {c.b_gain_db:+.1f} dB")
    if c.compatibility_score < 0.55:
        cautions.append("limited tempo/key compatibility")
    if not cautions:
        return "Best technical balance; audition this first."
    return "Audition carefully: " + "; ".join(cautions) + "."


def ranked_candidate_summary(candidates: List[TransitionCandidate]) -> list[dict[str, Any]]:
    """Return candidates ranked best-to-worst by total score, with component scores."""
    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
    summary = []
    for i, c in enumerate(ranked, start=1):
        summary.append({
            "rank": i,
            "name": c.name,
            "score": c.score,
            "recommendation": "recommended" if i == 1 else "alternate",
            "verdict": candidate_verdict(c),
            "transition": {
                "a_fade_start_sec": c.a_fade_start_sec,
                "a_cut_sec": c.a_cut_sec,
                "b_cue_sec": c.b_cue_sec,
                "overlap_sec": c.overlap_sec,
                "b_gain_db": c.b_gain_db,
                "trim_a_tail_sec": c.trim_a_tail_sec,
                "new_song_starts_in_mix_sec": c.a_fade_start_sec,
                "new_song_starts_in_mix_timestamp": format_timestamp(c.a_fade_start_sec),
                "track_b_source_cue_timestamp": format_timestamp(c.b_cue_sec),
            },
            "component_scores": {
                "vocal_collision_risk": c.vocal_collision_score,
                "beat_alignment": c.beat_alignment_score,
                "energy": c.energy_score,
                "placement": c.placement_score,
                "loudness": c.loudness_score,
                "compatibility": c.compatibility_score,
            },
            "notes": c.notes,
        })
    return summary
