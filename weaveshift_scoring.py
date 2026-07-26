"""Candidate search and scoring for WeaveShift transitions."""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from weaveshift_profiles import ScoringProfile
from weaveshift_audio import (
    AudioAnalysis,
    TransitionCandidate,
    OVERLAP_LENGTHS,
    clamp,
    camelot_compat,
    energy_db_from_curve,
    lufs_from_curve,
    overlap_vocal_collision,
    vocal_active_fraction,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TechnicalScoringConfig:
    """Versioned technical scoring weights for base candidate ranking."""

    version: str = "1.0.0"
    beat_distance_default_sec: float = 2.0
    beat_alignment_divisor_sec: float = 1.0
    perceptual_lufs_blend: float = 0.70
    perceptual_rms_blend: float = 0.30
    perceptual_lufs_tolerance_db: float = 8.0
    rms_tolerance_db: float = 10.0
    b_gain_min_db: float = -4.0
    b_gain_max_db: float = 4.0
    energy_perceptual_weight: float = 0.65
    energy_rms_weight: float = 0.35
    loudness_gain_ok_db: float = 2.0
    loudness_gain_warn_db: float = 3.2
    loudness_score_ok: float = 1.0
    loudness_score_warn: float = 0.75
    loudness_score_high_gain: float = 0.55
    placement_trim_none_sec: float = 0.3
    placement_trim_short_sec: float = 4.0
    placement_trim_medium_sec: float = 8.0
    placement_score_none: float = 1.0
    placement_score_short: float = 0.92
    placement_score_medium: float = 0.72
    placement_score_long: float = 0.45
    b_cue_late_sec: float = 8.0
    b_cue_mid_sec: float = 4.0
    b_cue_late_multiplier: float = 0.78
    b_cue_mid_multiplier: float = 0.90
    tempo_default_score: float = 0.65
    tempo_close_bpm: float = 2.0
    tempo_near_bpm: float = 6.0
    tempo_far_bpm: float = 12.0
    tempo_close_score: float = 1.0
    tempo_near_score: float = 0.86
    tempo_far_score: float = 0.68
    tempo_distant_score: float = 0.45
    compat_tempo_weight: float = 0.60
    compat_key_weight: float = 0.40
    overlap_bpm_jump_bpm: float = 12.0
    overlap_short_sec: float = 4.0
    overlap_long_sec: float = 6.0
    overlap_bpm_jump_bonus: float = 0.08
    overlap_compat_bonus: float = 0.06
    overlap_compat_min: float = 0.82
    overlap_too_short_sec: float = 1.0
    overlap_too_short_penalty: float = 0.06
    vocal_weight: float = 0.34
    beat_alignment_weight: float = 0.17
    energy_weight: float = 0.15
    placement_weight: float = 0.12
    loudness_weight: float = 0.10
    compatibility_weight: float = 0.12
    vocal_density_penalty_weight: float = 0.20
    vocal_collision_note_threshold: float = 0.25
    trim_note_threshold_sec: float = 4.0
    b_cue_note_threshold_sec: float = 4.0
    b_gain_note_threshold_db: float = 2.5
    compatibility_note_threshold: float = 0.55


DEFAULT_TECHNICAL_SCORING = TechnicalScoringConfig()

def nearest_distance(x: float, arr: Sequence[float], default: float = 999.0) -> float:
    if not arr:
        return default
    return min(abs(x - v) for v in arr)


def build_candidate_times(a: AudioAnalysis, b: AudioAnalysis, max_trim_a_sec: float, b_cue_max_sec: float) -> Tuple[List[float], List[float]]:
    # A cut points near the end: actual end, beats/onsets in last max_trim window, plus a few grid points.
    a_end = a.duration_sec
    a_min = max(0.0, a_end - max_trim_a_sec)
    a_cuts = {round(a_end, 3)}
    for t in a.beats_sec + a.onsets_sec:
        if a_min <= t <= a_end:
            a_cuts.add(round(float(t), 3))
    for trim in [0, 1, 2, 3, 4, 6, 8, 10, max_trim_a_sec]:
        t = a_end - trim
        if a_min <= t <= a_end:
            a_cuts.add(round(t, 3))

    # B cue points: 0, early beats/onsets before cue limit.
    b_cues = {0.0}
    for t in b.beats_sec + b.onsets_sec:
        if 0.0 <= t <= min(b_cue_max_sec, b.duration_sec - 1.0):
            b_cues.add(round(float(t), 3))
    # Avoid too many: cluster by 0.25 sec.
    def cluster(vals, prefer_end=False):
        out = []
        for v in sorted(vals, reverse=prefer_end):
            if not out or abs(v - out[-1]) > 0.35:
                out.append(v)
        return sorted(out)

    def limit(vals, max_n, prefer_end=False):
        vals = sorted(vals)
        if len(vals) <= max_n:
            return vals
        # Keep a diverse subset plus the musically important edges.
        idx = np.linspace(0, len(vals) - 1, max_n).round().astype(int)
        limited = [vals[int(i)] for i in idx]
        if prefer_end and vals[-1] not in limited:
            limited[-1] = vals[-1]
        if not prefer_end and vals[0] not in limited:
            limited[0] = vals[0]
        return sorted(set(limited))

    return limit(cluster(a_cuts, prefer_end=True), 22, prefer_end=True), limit(cluster(b_cues), 22, prefer_end=False)


def score_candidate(
    a: AudioAnalysis,
    b: AudioAnalysis,
    a_cut: float,
    b_cue: float,
    overlap: float,
    name: str,
    *,
    scoring: TechnicalScoringConfig = DEFAULT_TECHNICAL_SCORING,
) -> TransitionCandidate:
    a_fade_start = a_cut - overlap
    if a_fade_start < 0 or a_cut > a.duration_sec or b_cue + overlap > b.duration_sec:
        return TransitionCandidate(
            name=name, score=-999.0,
            a_fade_start_sec=round(a_fade_start, 3), a_cut_sec=round(a_cut, 3),
            b_cue_sec=round(b_cue, 3), overlap_sec=round(overlap, 3),
            b_gain_db=0.0, trim_a_tail_sec=0.0,
            vocal_collision_score=1.0, beat_alignment_score=0.0, energy_score=0.0,
            placement_score=0.0, loudness_score=0.0, perceptual_loudness_score=0.0, compatibility_score=0.0,
            notes=["invalid window"],
        )

    trim = max(0.0, a.duration_sec - a_cut)
    collision = overlap_vocal_collision(a.vocal_segments, b.vocal_segments, a_fade_start, b_cue, overlap)
    a_vocal_frac = vocal_active_fraction(a.vocal_segments, a_fade_start, a_cut)
    b_vocal_frac = vocal_active_fraction(b.vocal_segments, b_cue, b_cue + overlap)

    # Beat cue score: B cue near a detected beat/onset; A cut near a beat/onset; lower is better.
    b_beat_dist = min(
        nearest_distance(b_cue, b.beats_sec, scoring.beat_distance_default_sec),
        nearest_distance(b_cue, b.onsets_sec, scoring.beat_distance_default_sec),
    )
    a_cut_dist = min(
        nearest_distance(a_cut, a.beats_sec, scoring.beat_distance_default_sec),
        nearest_distance(a_cut, a.onsets_sec, scoring.beat_distance_default_sec),
    )
    beat_alignment = clamp(1.0 - ((b_beat_dist + a_cut_dist) / scoring.beat_alignment_divisor_sec), 0.0, 1.0)

    # Energy continuity and loudness/gain.
    # 1.0.0 uses precomputed in-memory local stereo RMS windows at the actual
    # transition. No disk I/O or mono-downmixed loudness occurs inside scoring.
    a_ref_db = energy_db_from_curve(a.energy_curve, a_fade_start, overlap)
    b_ref_db = energy_db_from_curve(b.energy_curve, b_cue, overlap)
    rms_diff = a_ref_db - b_ref_db

    a_lufs = lufs_from_curve(a.loudness_curve, a_fade_start, overlap)
    b_lufs = lufs_from_curve(b.loudness_curve, b_cue, overlap)
    if a_lufs is not None and b_lufs is not None:
        perceptual_diff = a_lufs - b_lufs
        perceptual_loudness_score = clamp(1.0 - abs(perceptual_diff) / scoring.perceptual_lufs_tolerance_db, 0.0, 1.0)
        diff = scoring.perceptual_lufs_blend * perceptual_diff + scoring.perceptual_rms_blend * rms_diff
    else:
        perceptual_diff = rms_diff
        perceptual_loudness_score = clamp(1.0 - abs(rms_diff) / scoring.rms_tolerance_db, 0.0, 1.0)
        diff = rms_diff

    b_gain = clamp(diff, scoring.b_gain_min_db, scoring.b_gain_max_db)
    rms_energy_score = clamp(1.0 - abs(rms_diff) / scoring.rms_tolerance_db, 0.0, 1.0)
    energy_score = clamp(
        scoring.energy_perceptual_weight * perceptual_loudness_score + scoring.energy_rms_weight * rms_energy_score,
        0.0,
        1.0,
    )
    if abs(b_gain) <= scoring.loudness_gain_ok_db:
        loudness_score = scoring.loudness_score_ok
    elif abs(b_gain) <= scoring.loudness_gain_warn_db:
        loudness_score = scoring.loudness_score_warn
    else:
        loudness_score = scoring.loudness_score_high_gain

    if trim <= scoring.placement_trim_none_sec:
        placement = scoring.placement_score_none
    elif trim <= scoring.placement_trim_short_sec:
        placement = scoring.placement_score_short
    elif trim <= scoring.placement_trim_medium_sec:
        placement = scoring.placement_score_medium
    else:
        placement = scoring.placement_score_long

    if b_cue > scoring.b_cue_late_sec:
        placement *= scoring.b_cue_late_multiplier
    elif b_cue > scoring.b_cue_mid_sec:
        placement *= scoring.b_cue_mid_multiplier

    tempo_score = scoring.tempo_default_score
    if a.bpm and b.bpm:
        bpm_diff = abs(float(a.bpm) - float(b.bpm))
        if bpm_diff <= scoring.tempo_close_bpm:
            tempo_score = scoring.tempo_close_score
        elif bpm_diff <= scoring.tempo_near_bpm:
            tempo_score = scoring.tempo_near_score
        elif bpm_diff <= scoring.tempo_far_bpm:
            tempo_score = scoring.tempo_far_score
        else:
            tempo_score = scoring.tempo_distant_score
    compat = scoring.compat_tempo_weight * tempo_score + scoring.compat_key_weight * camelot_compat(a.camelot, b.camelot)

    overlap_bonus = 0.0
    if a.bpm and b.bpm and abs(a.bpm - b.bpm) > scoring.overlap_bpm_jump_bpm and overlap <= scoring.overlap_short_sec:
        overlap_bonus += scoring.overlap_bpm_jump_bonus
    if compat > scoring.overlap_compat_min and overlap >= scoring.overlap_long_sec:
        overlap_bonus += scoring.overlap_compat_bonus
    if overlap < scoring.overlap_too_short_sec:
        overlap_bonus -= scoring.overlap_too_short_penalty

    vocal_score = 1.0 - collision
    vocal_density_penalty = scoring.vocal_density_penalty_weight * min(a_vocal_frac, b_vocal_frac)

    score = (
        scoring.vocal_weight * vocal_score
        + scoring.beat_alignment_weight * beat_alignment
        + scoring.energy_weight * energy_score
        + scoring.placement_weight * placement
        + scoring.loudness_weight * loudness_score
        + scoring.compatibility_weight * compat
        + overlap_bonus
        - vocal_density_penalty
    )

    notes = []
    if collision > scoring.vocal_collision_note_threshold:
        notes.append("vocal collision risk")
    if trim > scoring.trim_note_threshold_sec:
        notes.append(f"trims {trim:.1f}s from Track A tail")
    if b_cue > scoring.b_cue_note_threshold_sec:
        notes.append(f"cues Track B at {b_cue:.1f}s, not from the very start")
    if abs(b_gain) > scoring.b_gain_note_threshold_db:
        notes.append(f"applies {b_gain:+.1f} dB to Track B")
    if compat < scoring.compatibility_note_threshold:
        notes.append("tempo/key compatibility is limited; shorter cinematic transition may be safer")
    if not notes:
        notes.append("balanced candidate")

    return TransitionCandidate(
        name=name,
        score=round(float(score), 4),
        a_fade_start_sec=round(a_fade_start, 3),
        a_cut_sec=round(a_cut, 3),
        b_cue_sec=round(b_cue, 3),
        overlap_sec=round(overlap, 3),
        b_gain_db=round(float(b_gain), 2),
        trim_a_tail_sec=round(trim, 3),
        vocal_collision_score=round(collision, 4),
        beat_alignment_score=round(beat_alignment, 4),
        energy_score=round(energy_score, 4),
        placement_score=round(placement, 4),
        loudness_score=round(loudness_score, 4),
        perceptual_loudness_score=round(perceptual_loudness_score, 4),
        compatibility_score=round(compat, 4),
        notes=notes,
    )




def profile_candidate_score(c: TransitionCandidate, profile: ScoringProfile) -> float:
    """Configurable style score used by --transition-mode profile.

    It starts from the technical candidate score, then applies the TOML profile's
    weights/preferences to favor different musical behaviors for EDM, lounge,
    jazz/blues, or heart/ballad sets.
    """
    score = 0.55 * float(c.score)
    target = float(profile.pref("target_overlap_sec", 4.0))
    ov_min = float(profile.pref("preferred_overlap_min_sec", 2.0))
    ov_max = float(profile.pref("preferred_overlap_max_sec", 8.0))
    if ov_min <= c.overlap_sec <= ov_max:
        score += 0.18 * profile.weight("overlap_preference", 1.0)
    else:
        # Gentle distance penalty from the profile's target overlap.
        score -= min(0.30, abs(c.overlap_sec - target) / max(1.0, target) * 0.24) * profile.weight("overlap_preference", 1.0)

    max_gain = float(profile.limit("max_abs_b_gain_db", 4.0))
    gain_abs = abs(c.b_gain_db)
    if gain_abs <= min(1.5, max_gain):
        score += 0.10
    elif gain_abs > max_gain:
        score -= 0.28 * profile.weight("gain_penalty", 1.0)
    else:
        score -= (gain_abs / max(0.1, max_gain)) * 0.08 * profile.weight("gain_penalty", 1.0)

    max_tail = float(profile.limit("max_tail_trim_sec", 8.0))
    if c.trim_a_tail_sec > max_tail:
        score -= 0.22 * profile.weight("tail_trim_penalty", 1.0)
    else:
        score -= (c.trim_a_tail_sec / max(0.1, max_tail)) * 0.05 * profile.weight("tail_trim_penalty", 1.0)

    max_cue = float(profile.limit("max_b_cue_sec", 12.0))
    if c.b_cue_sec > max_cue:
        score -= 0.18 * profile.weight("cue_depth_penalty", 1.0)
    else:
        score -= (c.b_cue_sec / max(0.1, max_cue)) * 0.05 * profile.weight("cue_depth_penalty", 1.0)

    min_beat = float(profile.limit("min_beat_alignment", 0.0))
    if c.beat_alignment_score < min_beat:
        score -= 0.16 * profile.weight("beat_alignment", 1.0)

    score += 0.08 * c.energy_score * profile.weight("energy", 1.0)
    score += 0.06 * c.loudness_score * profile.weight("loudness", 1.0)
    score += 0.05 * c.compatibility_score * profile.weight("compatibility", 1.0)
    score += 0.04 * c.placement_score * profile.weight("placement", 1.0)
    score += 0.05 * c.beat_alignment_score * profile.weight("beat_alignment", 1.0)
    score -= 0.25 * c.vocal_collision_score * profile.weight("vocal_collision", 1.0)

    if profile.pref("prefer_vocal_safety", True) and c.vocal_collision_score > 0.05 and c.overlap_sec >= 6.0:
        score -= 0.18 * profile.weight("vocal_collision", 1.0)
    if not profile.pref("prefer_beat_alignment", True) and c.beat_alignment_score < 0.45:
        # Jazz/lounge/heart may intentionally accept loose beat grids.
        score += 0.06
    return round(float(score), 4)

def smooth_candidate_score(c: TransitionCandidate) -> float:
    """Style score for EDM/setlist handoffs that should feel less sudden.

    Smooth mode is intentionally taste-weighted, but 1.0.0 makes it
    vocal-aware. The first smooth pass proved that longer fades alone can still
    create muddy transitions when both tracks have vocal activity. Smooth now
    prefers a longer handoff only when the overlap is clean; when vocal risk is
    present, it backs off toward a shorter/cleaner or ducked handoff.
    """
    score = 0.55 * float(c.score)

    # Prefer 6-10s handoffs, accept 4s, avoid abrupt 2-3s cuts unless they are
    # the cleanest vocal-safe option. Long blends are a reward, not a mandate.
    if 6.0 <= c.overlap_sec <= 10.0:
        score += 0.18
    elif 4.0 <= c.overlap_sec < 6.0:
        score += 0.10
    elif 10.0 < c.overlap_sec <= 12.0:
        score += 0.08
    elif c.overlap_sec < 4.0:
        score -= 0.16

    # Hard boosts can make an otherwise long transition feel like the next song
    # suddenly appears. Keep this penalty strong, but not absolute: a +4 dB
    # transition can still work if the overlap is otherwise clean.
    gain_abs = abs(c.b_gain_db)
    if gain_abs <= 1.5:
        score += 0.12
    elif gain_abs <= 2.5:
        score += 0.05
    elif gain_abs <= 3.2:
        score -= 0.08
    else:
        score -= 0.34

    # Prefer the incoming track to arrive from the very beginning or a shallow cue.
    if c.b_cue_sec <= 2.0:
        score += 0.12
    elif c.b_cue_sec <= 4.0:
        score += 0.06
    elif c.b_cue_sec <= 6.0:
        score -= 0.04
    elif c.b_cue_sec <= 8.0:
        score -= 0.12
    else:
        score -= 0.22

    # Avoid making the outgoing song feel chopped.
    if c.trim_a_tail_sec <= 2.0:
        score += 0.08
    elif c.trim_a_tail_sec <= 4.0:
        score += 0.03
    elif c.trim_a_tail_sec <= 6.0:
        score -= 0.08
    else:
        score -= 0.22

    # 1.0.0: vocal-mud guardrails. If there is any meaningful vocal collision,
    # a long blend is risky. This specifically targets the type of transition
    # where both voices smear together during a 6-8s overlap.
    if c.vocal_collision_score > 0.04 and c.overlap_sec >= 6.0:
        score -= 0.16
    if c.vocal_collision_score > 0.06 and c.overlap_sec >= 6.0:
        score -= 0.18
    if c.vocal_collision_score > 0.04 and c.b_gain_db >= 3.5:
        score -= 0.14

    # Keep loose beat alignment from winning long blends. Short clean handoffs
    # may be fine with looser beat matching, but a 6-12s blend should land on
    # stronger rhythmic evidence.
    if c.beat_alignment_score < 0.45 and c.overlap_sec >= 4.0:
        score -= 0.14
    if c.beat_alignment_score < 0.45 and c.overlap_sec >= 8.0:
        score -= 0.10

    # Keep the underlying technical qualities, but do not let them overpower taste.
    score += 0.08 * c.energy_score
    score += 0.06 * c.loudness_score
    score += 0.04 * c.compatibility_score
    score += 0.05 * c.beat_alignment_score
    score -= 0.35 * c.vocal_collision_score
    return round(float(score), 4)

def candidate_too_close(c: TransitionCandidate, selected: Sequence[TransitionCandidate], tolerance_sec: float = 0.5) -> bool:
    """True if c's transition timing is within tolerance of any already-selected candidate.

    Single source of truth for "is this a meaningfully distinct transition,"
    used both when picking the named pools (vocal_safe/beat_aligned/quick_cut/
    long_blend) and when picking vocal_ducked, so the two selection paths
    cannot silently drift apart the way OVERLAP_LENGTHS/MAX_OVERLAP_SEC once did.
    """
    return any(
        abs(c.a_fade_start_sec - s.a_fade_start_sec) < tolerance_sec
        and abs(c.b_cue_sec - s.b_cue_sec) < tolerance_sec
        and abs(c.overlap_sec - s.overlap_sec) < tolerance_sec
        for s in selected
    )


def profile_search_parameters(max_trim_a_sec: float, b_cue_max_sec: float, scoring_profile: Optional[ScoringProfile] = None) -> Tuple[float, float, List[float], float]:
    """Return candidate-generation bounds for the active scoring profile.

    WeaveShift profiles are not just post-hoc ranking weights. In profile mode,
    the profile's limits and preferred overlap range also shape the candidate
    search space so a profile such as vocal_trance can actually search deeper
    Track B cue points, while heart/jazz profiles can restrict tail trimming.
    """
    effective_trim = float(max_trim_a_sec)
    effective_b_cue = float(b_cue_max_sec)
    overlap_lengths = list(OVERLAP_LENGTHS)
    max_vocal_collision = 0.35

    if scoring_profile is not None:
        effective_trim = max(0.0, float(scoring_profile.limit("max_tail_trim_sec", effective_trim)))
        effective_b_cue = max(0.0, float(scoring_profile.limit("max_b_cue_sec", effective_b_cue)))
        ov_min = float(scoring_profile.pref("preferred_overlap_min_sec", min(OVERLAP_LENGTHS)))
        ov_max = float(scoring_profile.pref("preferred_overlap_max_sec", max(OVERLAP_LENGTHS)))
        bounded = [ov for ov in OVERLAP_LENGTHS if ov_min <= ov <= ov_max]
        if not bounded:
            logger.warning(
                "Profile '%s' overlap range [%.1f, %.1f]s excludes all standard overlaps %s; using full overlap grid.",
                scoring_profile.name,
                ov_min,
                ov_max,
                OVERLAP_LENGTHS,
            )
        overlap_lengths = bounded or list(OVERLAP_LENGTHS)
        max_vocal_collision = float(scoring_profile.limit("max_vocal_collision_score", max_vocal_collision))

    return effective_trim, effective_b_cue, overlap_lengths, max_vocal_collision


def choose_candidates(a: AudioAnalysis, b: AudioAnalysis, max_trim_a_sec: float, b_cue_max_sec: float, scoring_profile: Optional[ScoringProfile] = None) -> List[TransitionCandidate]:
    effective_trim, effective_b_cue, overlap_lengths, profile_max_vocal_collision = profile_search_parameters(max_trim_a_sec, b_cue_max_sec, scoring_profile)
    a_cuts, b_cues = build_candidate_times(a, b, effective_trim, effective_b_cue)
    all_cands: List[TransitionCandidate] = []
    for a_cut in a_cuts:
        for b_cue in b_cues:
            for ov in overlap_lengths:
                if a_cut - ov < 0 or b_cue + ov > b.duration_sec:
                    continue
                all_cands.append(score_candidate(a, b, a_cut, b_cue, ov, "candidate"))
    valid = [c for c in all_cands if c.score > -100]
    valid.sort(key=lambda c: c.score, reverse=True)

    # Pick distinct named candidates.
    selected: List[TransitionCandidate] = []
    def add_named(label: str, pool: List[TransitionCandidate]):
        for c in pool:
            if not candidate_too_close(c, selected):
                c.name = label
                selected.append(c)
                return

    # Best overall vocal-safe.
    add_named("vocal_safe", valid)
    # Beat aligned: strong beat score, not too much collision.
    beat_pool = sorted([c for c in valid if c.beat_alignment_score >= 0.65 and c.vocal_collision_score <= 0.30], key=lambda c: (c.beat_alignment_score, c.score), reverse=True)
    add_named("beat_aligned", beat_pool or valid)
    # Quick cut: <=3 sec transition, useful for big tempo/key jumps.
    quick_pool = sorted([c for c in valid if c.overlap_sec <= 3.0], key=lambda c: c.score, reverse=True)
    add_named("quick_cut", quick_pool or valid)

    # Smooth: taste-weighted setlist/EDM handoff. Prefer 6-10s blends with
    # modest gain and shallow Track B cue; avoid the +4 dB/deep-cue behavior
    # that made forced long_blend sound sudden. This is selected explicitly via
    # --mode smooth or --transition-mode smooth; recommended mode remains the
    # original technical ranking.
    smooth_pool = sorted(
        [c for c in valid if c.vocal_collision_score <= 0.25],
        key=smooth_candidate_score,
        reverse=True,
    )
    for c in smooth_pool:
        if candidate_too_close(c, selected):
            continue
        sc = copy.deepcopy(c)
        sc.name = "smooth"
        sc.notes = list(sc.notes) + [
            f"smooth-mode taste score {smooth_candidate_score(sc):.3f}: favors clean vocal overlap, moderate/long overlap, modest B gain, shallow B cue, limited tail trim, and usable beat alignment"
        ]
        selected.append(sc)
        break

    # Configurable profile-selected candidate. This generalizes smooth mode for
    # EDM, lounge, jazz/blues, and emotional ballad setlists.
    if scoring_profile is not None:
        profile_pool = sorted(
            [c for c in valid if c.vocal_collision_score <= profile_max_vocal_collision],
            key=lambda c: profile_candidate_score(c, scoring_profile),
            reverse=True,
        )
        for c in profile_pool:
            if candidate_too_close(c, selected):
                continue
            pc = copy.deepcopy(c)
            pc.name = "profile"
            pc.notes = list(pc.notes) + [
                f"profile score {profile_candidate_score(pc, scoring_profile):.3f} using '{scoring_profile.name}' profile: {scoring_profile.description}"
            ]
            selected.append(pc)
            break

    # Soft-ducked: allow a musically strong candidate with minor vocal overlap,
    # then render a gentle full-mix duck during the overlap. This is not stem
    # ducking; it is a conservative bridge before true stem-aware rendering.
    # Uses the same too-close de-duplication as add_named() so this candidate
    # is only offered when it represents a genuinely distinct transition from
    # whatever has already been selected (e.g. vocal_safe), rather than
    # re-rendering the exact same cut with a small extra dip applied.
    duck_pool = sorted([c for c in valid if 0.05 < c.vocal_collision_score <= 0.42 and c.beat_alignment_score >= 0.45], key=lambda c: (c.score + 0.04 * c.beat_alignment_score), reverse=True)
    for c in duck_pool:
        if candidate_too_close(c, selected):
            continue
        dc = copy.deepcopy(c)
        dc.name = "vocal_ducked"
        dc.soft_duck_db = round(-min(3.0, max(1.25, 1.0 + dc.vocal_collision_score * 5.0)), 2)
        dc.soft_duck_target = "a"
        dc.notes = list(dc.notes) + [f"soft-ducks Track A by {dc.soft_duck_db:.1f} dB during the overlap"]
        selected.append(dc)
        break

    # Long blend: >=6 sec, if compatible.
    long_pool = sorted([c for c in valid if c.overlap_sec >= 6.0 and c.compatibility_score >= 0.65 and c.vocal_collision_score <= 0.20], key=lambda c: c.score, reverse=True)
    if long_pool:
        add_named("long_blend", long_pool)

    return selected
