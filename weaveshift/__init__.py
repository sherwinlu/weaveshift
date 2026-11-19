"""WeaveShift: Profile-driven continuous music mix engine."""
from __future__ import annotations

from weaveshift.audio import (
    AudioAnalysis,
    NaturalTransition,
    SetlistTransition,
    TransitionCandidate,
    VocalSegment,
    analyze_audio,
    validate_audio_input,
    validate_audio_output,
    validate_audio_pair_compatibility,
)
from weaveshift.plan import (
    JunctionPlan,
    SetlistMixPlan,
    TrackSpec,
    TwoTrackMixPlan,
    plan_setlist_mix,
    plan_two_track_mix,
)
from weaveshift.profiles import BUILTIN_PROFILES, ScoringProfile, builtin_profile_path, load_scoring_profile
from weaveshift.rendering import (
    TransitionAudio,
    apply_final_peak_guard,
    apply_gain_ramp,
    apply_soft_duck,
    build_transition_audio,
    export_audio,
    render_candidate,
)
from weaveshift.reports import (
    build_setlist_report,
    build_two_track_report,
    candidate_verdict,
    ranked_candidate_summary,
    serialize_analysis,
    validate_report_schema,
)
from weaveshift.scoring import (
    DEFAULT_TECHNICAL_SCORING,
    TechnicalScoringConfig,
    choose_candidates,
    profile_candidate_score,
    score_candidate,
    smooth_candidate_score,
)

from weaveshift._version import APP_VERSION

APP_NAME = f"WeaveShift {APP_VERSION}"

__all__ = [
    "APP_NAME",
    "APP_VERSION",
    "AudioAnalysis",
    "BUILTIN_PROFILES",
    "DEFAULT_TECHNICAL_SCORING",
    "JunctionPlan",
    "NaturalTransition",
    "ScoringProfile",
    "SetlistMixPlan",
    "SetlistTransition",
    "TechnicalScoringConfig",
    "TrackSpec",
    "TransitionAudio",
    "TransitionCandidate",
    "TwoTrackMixPlan",
    "VocalSegment",
    "analyze_audio",
    "apply_final_peak_guard",
    "apply_gain_ramp",
    "apply_soft_duck",
    "build_setlist_report",
    "build_transition_audio",
    "build_two_track_report",
    "builtin_profile_path",
    "candidate_verdict",
    "choose_candidates",
    "export_audio",
    "load_scoring_profile",
    "plan_setlist_mix",
    "plan_two_track_mix",
    "profile_candidate_score",
    "ranked_candidate_summary",
    "render_candidate",
    "score_candidate",
    "serialize_analysis",
    "smooth_candidate_score",
    "validate_audio_input",
    "validate_audio_output",
    "validate_audio_pair_compatibility",
    "validate_report_schema",
]
