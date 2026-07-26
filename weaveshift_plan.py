"""Transition planning: separate analysis/scoring from render/export."""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from pydub import AudioSegment

from weaveshift_profiles import ScoringProfile, load_scoring_profile
from weaveshift_audio import (
    MAX_OVERLAP_SEC,
    AudioAnalysis,
    NaturalTransition,
    SetlistTransition,
    TransitionCandidate,
    compatibility_fields,
    analyze_audio,
    audio_info,
    load_audio_segment,
    lossy_input_note,
    validate_audio_input,
    validate_audio_pair_compatibility,
    validate_loaded_segment_parity,
)
from weaveshift_scoring import choose_candidates, profile_search_parameters, score_candidate
from weaveshift_reports import ranked_candidate_summary


MAX_NATURAL_PAUSE_SEC = 300.0


@dataclass
class TrackSpec:
    path: str
    title: str
    bpm: Optional[float] = None
    key: Optional[str] = None


@dataclass
class TwoTrackMixPlan:
    """Analysis and candidate selection for a two-track mix."""

    track_a_path: Path
    track_b_path: Path
    track_a_wav_info: Dict[str, object]
    track_b_wav_info: Dict[str, object]
    track_a_analysis: AudioAnalysis
    track_b_analysis: AudioAnalysis
    seg_a: AudioSegment
    seg_b: AudioSegment
    candidates: List[TransitionCandidate]
    ranked: list[dict[str, object]]
    recommended_name: Optional[str]
    scoring_profile: Optional[ScoringProfile]
    effective_search_a_sec: float
    effective_search_b_sec: float


@dataclass
class JunctionPlan:
    """Candidate selection and mix timeline mapping for one setlist junction."""

    index: int
    from_title: str
    to_title: str
    track_a_analysis: Optional[AudioAnalysis]
    track_b_analysis: Optional[AudioAnalysis]
    candidates: List[TransitionCandidate]
    ranked: list[dict[str, object]]
    selected: SetlistTransition
    override_mode: Optional[str] = None
    mix_transition_start_sec: float = 0.0
    next_track_source_zero_in_mix_sec: float = 0.0


@dataclass
class SetlistMixPlan:
    """Full setlist transition plan before stitching/render."""

    tracks: List[TrackSpec]
    junctions: List[JunctionPlan] = field(default_factory=list)
    scoring_profile: Optional[ScoringProfile] = None
    wav_infos: List[Dict[str, object]] = field(default_factory=list)
    effective_search_a_sec: float = 35.0
    effective_search_b_sec: float = 35.0
    source_zero_in_mix_sec: List[float] = field(default_factory=list)
    manifest_settings: dict[str, object] = field(default_factory=dict)
    segments: List[AudioSegment] = field(default_factory=list)


def _effective_search_windows(
    search_a_sec: float,
    search_b_sec: float,
    max_trim_a_sec: float,
    b_cue_max_sec: float,
    scoring_profile: Optional[ScoringProfile],
) -> tuple[float, float]:
    profile_trim_sec, profile_b_cue_sec, profile_overlaps, _ = profile_search_parameters(
        max_trim_a_sec, b_cue_max_sec, scoring_profile
    )
    max_overlap_sec = max(profile_overlaps) if profile_overlaps else MAX_OVERLAP_SEC
    effective_search_a_sec = max(float(search_a_sec), float(profile_trim_sec) + max_overlap_sec)
    effective_search_b_sec = max(float(search_b_sec), float(profile_b_cue_sec) + max_overlap_sec)
    return effective_search_a_sec, effective_search_b_sec


def filter_candidates_by_mode(candidates: List[TransitionCandidate], mode: str) -> List[TransitionCandidate]:
    if mode == "all" or not candidates:
        return list(candidates)
    filtered = [c for c in candidates if c.name == mode]
    return filtered if filtered else candidates[:1]


def select_candidate(candidates: List[TransitionCandidate], mode: str, transition_index: int) -> TransitionCandidate:
    if not candidates:
        raise ValueError(f"No valid transition candidates generated for transition #{transition_index}.")
    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
    if mode in ("recommended", "best"):
        return ranked[0]
    matches = [c for c in candidates if c.name == mode]
    if matches:
        return matches[0]
    fallback = ranked[0]
    print(
        f"Warning: transition #{transition_index} requested mode '{mode}' was not generated; "
        f"falling back to '{fallback.name}'.",
        file=sys.stderr,
    )
    return fallback


def transition_override(settings: dict[str, object], index: int, from_title: str, to_title: str) -> dict[str, object] | None:
    """Return a per-transition override object from manifest settings, if present."""
    overrides = settings.get("transition_overrides") or []
    if not isinstance(overrides, list):
        return None
    for item in overrides:
        if not isinstance(item, dict):
            continue
        if item.get("index") == index:
            return item
        if item.get("from") == from_title and item.get("to") == to_title:
            return item
    return None


def coerce_float_override(item: dict[str, object], key: str, fallback: float) -> float:
    value = item.get(key)
    if value is None:
        return fallback
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"Invalid manual transition override value for {key}: {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid manual transition override value for {key}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Manual transition override value for {key} must be finite. Got: {value!r}")
    return result


def _validate_override_source_bounds(
    kind: str,
    transition_index: int,
    a_fade_start: float,
    a_cut: float,
    b_cue: float,
    track_a: AudioAnalysis,
    track_b: AudioAnalysis,
) -> None:
    """Reject override timing outside the source tracks.

    Negative cue/fade values are especially dangerous: pydub interprets a
    negative slice start as "from the end of the segment," so a typo like
    b_cue_sec=-3 would silently render the wrong part of Track B.
    """
    if a_fade_start < 0:
        raise ValueError(
            f"Transition #{transition_index} {kind} override a_fade_start_sec must be >= 0. Got {a_fade_start}."
        )
    if b_cue < 0:
        raise ValueError(f"Transition #{transition_index} {kind} override b_cue_sec must be >= 0. Got {b_cue}.")
    if a_cut > track_a.duration_sec + 0.05:
        raise ValueError(
            f"Transition #{transition_index} {kind} override a_cut_sec is beyond the end of Track A "
            f"({track_a.duration_sec:.3f}s). Got {a_cut}."
        )
    if b_cue >= track_b.duration_sec:
        raise ValueError(
            f"Transition #{transition_index} {kind} override b_cue_sec is beyond the end of Track B "
            f"({track_b.duration_sec:.3f}s). Got {b_cue}."
        )


def _override_mode(item: dict[str, object]) -> str:
    return str(item.get("mode") or item.get("candidate") or item.get("transition_mode") or "manual")


def build_natural_transition(
    track_a_duration_sec: float,
    pause_sec: float,
    transition_index: int,
) -> NaturalTransition:
    """Create a no-fade, no-overlap transition after Track A's natural end."""
    if not math.isfinite(pause_sec) or not 0 <= pause_sec <= MAX_NATURAL_PAUSE_SEC:
        raise ValueError(
            f"Transition #{transition_index} natural pause_sec must be between 0 and "
            f"{MAX_NATURAL_PAUSE_SEC:g} seconds. Got {pause_sec}."
        )
    end_sec = round(track_a_duration_sec, 3)
    pause_sec = round(pause_sec, 3)
    return NaturalTransition(
        boundary_sec=end_sec,
        pause_sec=pause_sec,
        notes=[
            "natural transition: preserves Track A through its end",
            f"inserts {pause_sec:.3f}s of silence before Track B",
            "starts Track B from source time zero without a fade or gain change",
        ],
    )


def apply_manual_transition_override(
    base: TransitionCandidate,
    item: dict[str, object],
    transition_index: int,
    track_a: AudioAnalysis,
    track_b: AudioAnalysis,
) -> TransitionCandidate:
    """Create a candidate using exact transition parameters from the manifest."""
    a_fade_start = coerce_float_override(item, "a_fade_start_sec", base.a_fade_start_sec)
    a_cut = coerce_float_override(item, "a_cut_sec", base.a_cut_sec)
    b_cue = coerce_float_override(item, "b_cue_sec", base.b_cue_sec)
    overlap = coerce_float_override(item, "overlap_sec", base.overlap_sec)
    b_gain = coerce_float_override(item, "b_gain_db", base.b_gain_db)

    if overlap <= 0:
        raise ValueError(f"Transition #{transition_index} manual override overlap_sec must be > 0. Got {overlap}.")
    if a_cut <= a_fade_start:
        raise ValueError(
            f"Transition #{transition_index} manual override requires a_cut_sec > a_fade_start_sec. "
            f"Got {a_cut} <= {a_fade_start}."
        )
    _validate_override_source_bounds("manual", transition_index, a_fade_start, a_cut, b_cue, track_a, track_b)

    rescored = score_candidate(track_a, track_b, a_cut, b_cue, overlap, "candidate")
    trim_a_tail_sec = max(
        0.0,
        coerce_float_override(item, "trim_a_tail_sec", max(0.0, track_a.duration_sec - a_cut)),
    )

    notes = list(rescored.notes) + [
        "manual transition override",
        f"manual a_fade_start_sec={a_fade_start:.3f}",
        f"manual a_cut_sec={a_cut:.3f}",
        f"manual b_cue_sec={b_cue:.3f}",
        f"manual overlap_sec={overlap:.3f}",
        f"manual b_gain_db={b_gain:+.2f}",
        "component scores recomputed for manual timing",
    ]
    if abs((a_cut - a_fade_start) - overlap) > 0.01:
        notes.append(
            "manual warning: a_cut_sec - a_fade_start_sec differs from overlap_sec; "
            "this is allowed but may not be a standard crossfade shape"
        )

    return TransitionCandidate(
        name=str(item.get("name") or item.get("mode") or "manual"),
        score=rescored.score,
        a_fade_start_sec=a_fade_start,
        a_cut_sec=a_cut,
        b_cue_sec=b_cue,
        overlap_sec=overlap,
        b_gain_db=b_gain,
        trim_a_tail_sec=trim_a_tail_sec,
        vocal_collision_score=rescored.vocal_collision_score,
        beat_alignment_score=rescored.beat_alignment_score,
        energy_score=rescored.energy_score,
        placement_score=rescored.placement_score,
        loudness_score=rescored.loudness_score,
        perceptual_loudness_score=rescored.perceptual_loudness_score,
        compatibility_score=rescored.compatibility_score,
        notes=notes,
        soft_duck_db=coerce_float_override(item, "soft_duck_db", base.soft_duck_db),
        soft_duck_target=str(item.get("soft_duck_target", base.soft_duck_target)),
    )


def apply_handoff_transition_override(
    base: TransitionCandidate,
    item: dict[str, object],
    transition_index: int,
    track_a: AudioAnalysis,
    track_b: AudioAnalysis,
) -> TransitionCandidate:
    """Create a semantic handoff candidate from exact manifest parameters."""
    a_fade_start = coerce_float_override(item, "a_fade_start_sec", base.a_fade_start_sec)
    a_cut = coerce_float_override(item, "a_cut_sec", base.a_cut_sec)
    b_cue = coerce_float_override(item, "b_cue_sec", base.b_cue_sec)
    takeover_overlap = coerce_float_override(item, "takeover_overlap_sec", base.overlap_sec)
    b_entry_gain = coerce_float_override(item, "b_entry_gain_db", base.b_gain_db)
    b_fade_in = coerce_float_override(item, "b_fade_in_sec", max(takeover_overlap, base.overlap_sec))

    if a_cut <= a_fade_start:
        raise ValueError(
            f"Transition #{transition_index} handoff override requires a_cut_sec > a_fade_start_sec. "
            f"Got {a_cut} <= {a_fade_start}."
        )
    _validate_override_source_bounds("handoff", transition_index, a_fade_start, a_cut, b_cue, track_a, track_b)
    if not 0.05 <= takeover_overlap <= 1.0:
        raise ValueError(
            f"Transition #{transition_index} handoff override takeover_overlap_sec must be between 0.05 and 1.0. "
            f"Got {takeover_overlap}."
        )
    if not 0.1 <= b_fade_in <= 4.0:
        raise ValueError(
            f"Transition #{transition_index} handoff override b_fade_in_sec must be between 0.1 and 4.0. "
            f"Got {b_fade_in}."
        )
    if b_fade_in < takeover_overlap:
        raise ValueError(
            f"Transition #{transition_index} handoff override requires b_fade_in_sec >= takeover_overlap_sec. "
            f"Got {b_fade_in} < {takeover_overlap}."
        )
    if not -6.0 <= b_entry_gain <= 3.0:
        raise ValueError(
            f"Transition #{transition_index} handoff override b_entry_gain_db must be between -6.0 and 3.0. "
            f"Got {b_entry_gain}."
        )

    # Use the short takeover window for the comparable technical score. The
    # renderer keeps the longer outgoing fade and incoming fade-in shape.
    rescored = score_candidate(track_a, track_b, a_cut, b_cue, takeover_overlap, "candidate")
    trim_a_tail_sec = max(0.0, track_a.duration_sec - a_cut)
    notes = list(rescored.notes) + [
        "handoff transition: fades Track A down before Track B takeover",
        "uses short takeover overlap to avoid mud",
        f"applies {b_entry_gain:+.1f} dB to Track B entry",
        f"handoff a_fade_start_sec={a_fade_start:.3f}",
        f"handoff a_cut_sec={a_cut:.3f}",
        f"handoff b_cue_sec={b_cue:.3f}",
        f"handoff takeover_overlap_sec={takeover_overlap:.3f}",
        f"handoff b_fade_in_sec={b_fade_in:.3f}",
    ]

    return TransitionCandidate(
        name="handoff",
        score=rescored.score,
        a_fade_start_sec=a_fade_start,
        a_cut_sec=a_cut,
        b_cue_sec=b_cue,
        overlap_sec=takeover_overlap,
        b_gain_db=b_entry_gain,
        trim_a_tail_sec=trim_a_tail_sec,
        vocal_collision_score=rescored.vocal_collision_score,
        beat_alignment_score=rescored.beat_alignment_score,
        energy_score=rescored.energy_score,
        placement_score=rescored.placement_score,
        loudness_score=rescored.loudness_score,
        perceptual_loudness_score=rescored.perceptual_loudness_score,
        compatibility_score=rescored.compatibility_score,
        notes=notes,
        takeover_overlap_sec=takeover_overlap,
        b_fade_in_sec=b_fade_in,
        b_entry_gain_db=b_entry_gain,
    )


def candidate_b_start_offset_sec(cand: SetlistTransition) -> float:
    if isinstance(cand, NaturalTransition):
        return cand.boundary_sec
    if cand.name == "handoff":
        overlap = cand.takeover_overlap_sec if cand.takeover_overlap_sec is not None else cand.overlap_sec
        return cand.a_cut_sec - overlap
    return cand.a_fade_start_sec


def validate_setlist_formats(tracks: Sequence[TrackSpec]) -> tuple[List[Dict[str, object]], List[AudioSegment]]:
    """Validate all inputs and load decoded segments before expensive analysis."""
    infos: List[Dict[str, object]] = []
    segments: List[AudioSegment] = []
    first_info: Optional[Dict[str, object]] = None
    first_seg: Optional[AudioSegment] = None

    for idx, spec in enumerate(tracks, start=1):
        p = validate_audio_input(spec.path, f"Track {idx}")
        note = lossy_input_note(p)
        if note:
            print(note, file=sys.stderr)
        info = audio_info(p)
        if first_info is None:
            first_info = info
        else:
            fields = compatibility_fields(first_info, info)
            mismatches = [(f, first_info[f], info[f]) for f in fields if first_info[f] != info[f]]
            if mismatches:
                details = "\n".join(f"  - {field}: first={a_val}, track {idx}={b_val}" for field, a_val, b_val in mismatches)
                raise ValueError(
                    "All setlist audio formats must match. Refusing to chain tracks rather than silently resampling.\n"
                    + details
                )
        infos.append(info)

        seg = load_audio_segment(p)
        if first_seg is None:
            first_seg = seg
        else:
            validate_loaded_segment_parity(first_seg, seg)
        segments.append(seg)

    return infos, segments


def plan_setlist_mix(
    tracks: Sequence[TrackSpec],
    *,
    manifest_settings: dict[str, object] | None = None,
    transition_mode: str = "recommended",
    profile: str = "edm",
    scoring_config: Optional[str] = None,
    search_a_sec: float = 35.0,
    search_b_sec: float = 35.0,
    max_trim_a_sec: float = 8.0,
    b_cue_max_sec: float = 10.0,
    natural_pause_sec: float = 1.0,
    vocal_method: str = "heuristic",
    prefer_mps: bool = True,
    assembled_mix_len_ms: Optional[int] = None,
) -> SetlistMixPlan:
    """Analyze each junction and return a render-ready setlist plan."""
    settings = dict(manifest_settings or {})
    scoring_profile = load_scoring_profile(profile, scoring_config) if transition_mode == "profile" else None
    effective_search_a_sec, effective_search_b_sec = _effective_search_windows(
        search_a_sec, search_b_sec, max_trim_a_sec, b_cue_max_sec, scoring_profile
    )

    wav_infos, segments = validate_setlist_formats(tracks)
    source_zero_in_mix_sec: List[float] = [0.0]
    junctions: List[JunctionPlan] = []
    mix_len_ms = assembled_mix_len_ms if assembled_mix_len_ms is not None else len(segments[0])

    for i in range(len(tracks) - 1):
        a_spec, b_spec = tracks[i], tracks[i + 1]
        override_item = transition_override(settings, i + 1, a_spec.title, b_spec.title)
        override_mode = None
        if override_item:
            override_mode = _override_mode(override_item)
        selection_mode = override_mode or transition_mode
        if selection_mode == "natural":
            pause_sec = natural_pause_sec
            if override_item:
                pause_sec = coerce_float_override(override_item, "pause_sec", natural_pause_sec)
            selected = build_natural_transition(len(segments[i]) / 1000.0, pause_sec, i + 1)
            a_analysis = None
            b_analysis = None
            candidates: List[TransitionCandidate] = []
            ranked: list[dict[str, object]] = []
        else:
            a_analysis = analyze_audio(
                a_spec.path,
                role="a",
                window_sec=effective_search_a_sec,
                manual_bpm=a_spec.bpm,
                manual_key=a_spec.key,
                vocal_method=vocal_method,
                prefer_mps=prefer_mps,
            )
            b_analysis = analyze_audio(
                b_spec.path,
                role="b",
                window_sec=effective_search_b_sec,
                manual_bpm=b_spec.bpm,
                manual_key=b_spec.key,
                vocal_method=vocal_method,
                prefer_mps=prefer_mps,
            )
            candidates = choose_candidates(
                a_analysis, b_analysis, max_trim_a_sec, b_cue_max_sec, scoring_profile=scoring_profile
            )
            base_selection_mode = "recommended" if override_mode in ("manual", "handoff") else selection_mode
            selected = select_candidate(candidates, base_selection_mode, i + 1)
            if override_item and override_mode == "handoff":
                selected = apply_handoff_transition_override(selected, override_item, i + 1, a_analysis, b_analysis)
            elif override_item and (
                override_mode == "manual"
                or any(k in override_item for k in ("a_fade_start_sec", "a_cut_sec", "b_cue_sec", "overlap_sec", "b_gain_db"))
            ):
                selected = apply_manual_transition_override(selected, override_item, i + 1, a_analysis, b_analysis)
            ranked = ranked_candidate_summary(candidates)

        current_a_zero = source_zero_in_mix_sec[i]
        mix_transition_start_sec = current_a_zero + selected.a_fade_start_sec
        mix_transition_start_ms = int(round(mix_transition_start_sec * 1000))
        if mix_transition_start_ms < 0 or mix_transition_start_ms > mix_len_ms:
            raise ValueError(
                f"Transition #{i+1} maps outside the assembled mix: {mix_transition_start_sec:.3f}s. "
                "This usually means a prior transition mapping failed."
            )

        pause_sec = selected.pause_sec if isinstance(selected, NaturalTransition) else 0.0
        b_start_in_mix_sec = current_a_zero + candidate_b_start_offset_sec(selected) + pause_sec
        next_zero = b_start_in_mix_sec - selected.b_cue_sec
        if next_zero < -1e-6:
            raise ValueError(
                f"Transition #{i+1} maps Track B source zero before the start of the assembled mix: "
                f"next_zero={next_zero:.3f}s, transition_start={mix_transition_start_sec:.3f}s, "
                f"b_cue={selected.b_cue_sec:.3f}s. Reduce --b-cue-max-sec or use a later Track A transition point."
            )
        source_zero_in_mix_sec.append(next_zero)

        junctions.append(
            JunctionPlan(
                index=i + 1,
                from_title=a_spec.title,
                to_title=b_spec.title,
                track_a_analysis=a_analysis,
                track_b_analysis=b_analysis,
                candidates=candidates,
                ranked=ranked,
                selected=selected,
                override_mode=override_mode,
                mix_transition_start_sec=round(mix_transition_start_sec, 3),
                next_track_source_zero_in_mix_sec=round(next_zero, 3),
            )
        )
        mix_len_ms = int(round(next_zero * 1000)) + len(segments[i + 1])

    return SetlistMixPlan(
        tracks=list(tracks),
        junctions=junctions,
        scoring_profile=scoring_profile,
        wav_infos=wav_infos,
        effective_search_a_sec=effective_search_a_sec,
        effective_search_b_sec=effective_search_b_sec,
        source_zero_in_mix_sec=source_zero_in_mix_sec,
        manifest_settings=settings,
        segments=segments,
    )


def plan_two_track_mix(
    track_a: str,
    track_b: str,
    *,
    mode: str = "all",
    profile: str = "edm",
    scoring_config: Optional[str] = None,
    search_a_sec: float = 35.0,
    search_b_sec: float = 35.0,
    max_trim_a_sec: float = 8.0,
    b_cue_max_sec: float = 10.0,
    a_bpm: Optional[float] = None,
    b_bpm: Optional[float] = None,
    a_key: Optional[str] = None,
    b_key: Optional[str] = None,
    vocal_method: str = "auto",
    prefer_mps: bool = True,
) -> TwoTrackMixPlan:
    """Analyze both tracks, score candidates, and return a render-ready plan."""
    track_a_path = validate_audio_input(track_a, "Track A")
    track_b_path = validate_audio_input(track_b, "Track B")
    for p in (track_a_path, track_b_path):
        note = lossy_input_note(p)
        if note:
            print(note, file=sys.stderr)
    a_wav_info, b_wav_info = validate_audio_pair_compatibility(track_a_path, track_b_path)

    scoring_profile = load_scoring_profile(profile, scoring_config) if mode == "profile" else None
    effective_search_a_sec, effective_search_b_sec = _effective_search_windows(
        search_a_sec, search_b_sec, max_trim_a_sec, b_cue_max_sec, scoring_profile
    )

    seg_a = load_audio_segment(track_a_path)
    seg_b = load_audio_segment(track_b_path)
    validate_loaded_segment_parity(seg_a, seg_b)

    a_analysis = analyze_audio(
        str(track_a_path),
        role="a",
        window_sec=effective_search_a_sec,
        manual_bpm=a_bpm,
        manual_key=a_key,
        vocal_method=vocal_method,
        prefer_mps=prefer_mps,
    )
    b_analysis = analyze_audio(
        str(track_b_path),
        role="b",
        window_sec=effective_search_b_sec,
        manual_bpm=b_bpm,
        manual_key=b_key,
        vocal_method=vocal_method,
        prefer_mps=prefer_mps,
    )

    candidates = choose_candidates(
        a_analysis, b_analysis, max_trim_a_sec, b_cue_max_sec, scoring_profile=scoring_profile
    )
    if not candidates:
        raise ValueError(
            "No valid transition candidates were generated. Track A may be too short for the requested "
            "search/overlap settings, or the candidate limits are too restrictive."
        )

    candidates = filter_candidates_by_mode(candidates, mode)
    ranked = ranked_candidate_summary(candidates)
    recommended = ranked[0]["name"] if ranked else None

    return TwoTrackMixPlan(
        track_a_path=track_a_path,
        track_b_path=track_b_path,
        track_a_wav_info=a_wav_info,
        track_b_wav_info=b_wav_info,
        track_a_analysis=a_analysis,
        track_b_analysis=b_analysis,
        seg_a=seg_a,
        seg_b=seg_b,
        candidates=candidates,
        ranked=ranked,
        recommended_name=recommended,
        scoring_profile=scoring_profile,
        effective_search_a_sec=effective_search_a_sec,
        effective_search_b_sec=effective_search_b_sec,
    )
