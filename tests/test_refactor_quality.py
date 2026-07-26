import numpy as np

from weaveshift_audio import AudioAnalysis, VocalSegment
from weaveshift_rendering import build_transition_audio, render_candidate, render_transition_tail
from weaveshift_reports import (
    build_two_track_report,
    validate_report_schema,
)
from weaveshift_scoring import DEFAULT_TECHNICAL_SCORING, TechnicalScoringConfig, score_candidate
from weaveshift_two_tracks import TransitionCandidate, ranked_candidate_summary


def analysis(role: str, *, duration=30.0, beats=None, vocals=None, bpm=128.0, camelot="12B", energy_db=-14.0):
    times = np.array([0.0, 10.0, 20.0, 30.0], dtype=float)
    curve = {"times": times, "dbfs": np.full_like(times, energy_db)}
    return AudioAnalysis(
        path=f"{role}.wav",
        duration_sec=duration,
        bpm=bpm,
        key="E major",
        camelot=camelot,
        rms_dbfs=-18.0,
        peak_dbfs=-3.0,
        beats_sec=list(beats or []),
        onsets_sec=[],
        vocal_segments=list(vocals or []),
        vocal_method="heuristic",
        analysis_window_start_sec=0.0,
        analysis_window_duration_sec=duration,
        energy_curve=curve,
        loudness_curve=curve,
    )


def test_golden_candidate_ranking_prefers_clean_vocal_handoff():
    """Stable ranking fixture: low vocal collision should beat high collision at same overlap."""
    a = analysis("a", duration=30, beats=[24.0, 26.0, 28.0, 30.0], vocals=[VocalSegment(20.0, 22.0)])
    b = analysis("b", duration=30, beats=[0.0, 2.0, 4.0], vocals=[VocalSegment(0.0, 1.0)])

    clean = score_candidate(a, b, a_cut=28.0, b_cue=0.0, overlap=4.0, name="clean")
    muddy = score_candidate(a, b, a_cut=24.0, b_cue=0.0, overlap=4.0, name="muddy")

    ranked = ranked_candidate_summary([muddy, clean])
    assert ranked[0]["name"] == "clean"
    assert ranked[0]["score"] > ranked[1]["score"]
    assert clean.vocal_collision_score < muddy.vocal_collision_score


def test_technical_scoring_config_is_versioned_and_overridable():
    custom = TechnicalScoringConfig(version="test-1", vocal_weight=0.50, beat_alignment_weight=0.10)
    assert custom.version == "test-1"
    assert DEFAULT_TECHNICAL_SCORING.version == "1.0.0"

    a = analysis("a", duration=30, beats=[24.0])
    b = analysis("b", duration=30, beats=[0.0])
    default_score = score_candidate(a, b, 24.0, 0.0, 4.0, "default").score
    custom_score = score_candidate(a, b, 24.0, 0.0, 4.0, "custom", scoring=custom).score
    assert default_score != custom_score


def test_report_schema_validation_accepts_two_track_report():
    a = analysis("a")
    b = analysis("b")
    ranked = ranked_candidate_summary(
        [
            TransitionCandidate(
                name="vocal_safe",
                score=0.9,
                a_fade_start_sec=20.0,
                a_cut_sec=24.0,
                b_cue_sec=0.0,
                overlap_sec=4.0,
                b_gain_db=0.0,
                trim_a_tail_sec=1.0,
                vocal_collision_score=0.0,
                beat_alignment_score=0.9,
                energy_score=0.9,
                placement_score=0.9,
                loudness_score=0.9,
                perceptual_loudness_score=0.9,
                compatibility_score=0.9,
                notes=["ok"],
            )
        ]
    )
    report = build_two_track_report(
        app_name="WeaveShift test",
        mode="all",
        recommended="vocal_safe",
        ranked=ranked,
        track_a=a,
        track_b=b,
        outputs=[],
        wav_format_validation={"track_a": {}, "track_b": {}},
        profile=None,
        scoring_config=None,
    )
    validate_report_schema(report, kind="two_track")
    assert report["schema_version"] == "1.0.0"


def test_plan_setlist_mix_monkeypatched(tmp_path, monkeypatch):
    import soundfile as sf
    from weaveshift_plan import TrackSpec, plan_setlist_mix
    import weaveshift_plan as plan_mod

    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    sr = 48000
    duration = 4.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    for path, freq in ((a, 440.0), (b, 660.0)):
        y = 0.05 * np.column_stack([np.sin(2 * np.pi * freq * t), np.sin(2 * np.pi * freq * t)])
        sf.write(path, y, sr, subtype="PCM_16")

    fake = AudioAnalysis(
        path="x.wav",
        duration_sec=4.0,
        bpm=128.0,
        key="E major",
        camelot="12B",
        rms_dbfs=-20.0,
        peak_dbfs=-6.0,
        beats_sec=[0.0, 1.0, 2.0, 3.0, 4.0],
        onsets_sec=[],
        vocal_segments=[],
        vocal_method="heuristic",
        analysis_window_start_sec=0.0,
        analysis_window_duration_sec=4.0,
    )
    cand = TransitionCandidate(
        name="vocal_safe",
        score=0.91,
        a_fade_start_sec=2.0,
        a_cut_sec=3.0,
        b_cue_sec=0.0,
        overlap_sec=1.0,
        b_gain_db=0.0,
        trim_a_tail_sec=1.0,
        vocal_collision_score=0.0,
        beat_alignment_score=1.0,
        energy_score=0.9,
        placement_score=0.9,
        loudness_score=0.9,
        perceptual_loudness_score=0.9,
        compatibility_score=0.9,
        notes=["ok"],
    )

    monkeypatch.setattr(plan_mod, "analyze_audio", lambda path, role, **_: fake)
    monkeypatch.setattr(plan_mod, "choose_candidates", lambda *_a, **_k: [cand])

    mix_plan = plan_setlist_mix(
        [TrackSpec(path=str(a), title="A"), TrackSpec(path=str(b), title="B")],
        search_a_sec=1.0,
        search_b_sec=1.0,
        vocal_method="heuristic",
        prefer_mps=False,
    )
    assert len(mix_plan.junctions) == 1
    assert mix_plan.junctions[0].selected.name == "vocal_safe"
    assert mix_plan.segments[0].frame_rate == 48000


def test_plan_two_track_mix_monkeypatched(tmp_path, monkeypatch):
    import soundfile as sf
    from weaveshift_plan import plan_two_track_mix
    import weaveshift_plan as plan_mod

    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    sr = 48000
    duration = 4.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    for path, freq in ((a, 440.0), (b, 660.0)):
        y = 0.05 * np.column_stack([np.sin(2 * np.pi * freq * t), np.sin(2 * np.pi * freq * t)])
        sf.write(path, y, sr, subtype="PCM_16")

    fake = AudioAnalysis(
        path="x.wav",
        duration_sec=4.0,
        bpm=128.0,
        key="E major",
        camelot="12B",
        rms_dbfs=-20.0,
        peak_dbfs=-6.0,
        beats_sec=[0.0, 1.0, 2.0, 3.0, 4.0],
        onsets_sec=[],
        vocal_segments=[],
        vocal_method="heuristic",
        analysis_window_start_sec=0.0,
        analysis_window_duration_sec=4.0,
    )
    cand = TransitionCandidate(
        name="vocal_safe",
        score=0.91,
        a_fade_start_sec=2.0,
        a_cut_sec=3.0,
        b_cue_sec=0.0,
        overlap_sec=1.0,
        b_gain_db=0.0,
        trim_a_tail_sec=1.0,
        vocal_collision_score=0.0,
        beat_alignment_score=1.0,
        energy_score=0.9,
        placement_score=0.9,
        loudness_score=0.9,
        perceptual_loudness_score=0.9,
        compatibility_score=0.9,
        notes=["ok"],
    )

    monkeypatch.setattr(plan_mod, "analyze_audio", lambda path, role, **_: fake)
    monkeypatch.setattr(plan_mod, "choose_candidates", lambda *_a, **_k: [cand])

    mix_plan = plan_two_track_mix(
        str(a), str(b), search_a_sec=1.0, search_b_sec=1.0, vocal_method="heuristic", prefer_mps=False
    )
    assert mix_plan.recommended_name == "vocal_safe"
    assert mix_plan.candidates[0].name == "vocal_safe"
    assert mix_plan.track_a_wav_info["subtype"] == "PCM_16"


def test_build_transition_audio_matches_tail_and_full(tmp_path):
    from pydub import AudioSegment
    import soundfile as sf

    sr = 48000
    duration = 4.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    for name, freq in (("a", 440.0), ("b", 660.0)):
        y = 0.05 * np.column_stack([np.sin(2 * np.pi * freq * t), np.sin(2 * np.pi * freq * t)])
        sf.write(tmp_path / f"{name}.wav", y, sr, subtype="PCM_16")

    seg_a = AudioSegment.from_file(str(tmp_path / "a.wav"), format="wav")
    seg_b = AudioSegment.from_file(str(tmp_path / "b.wav"), format="wav")
    cand = TransitionCandidate(
        name="test",
        score=0.9,
        a_fade_start_sec=2.0,
        a_cut_sec=3.0,
        b_cue_sec=0.0,
        overlap_sec=1.0,
        b_gain_db=0.0,
        trim_a_tail_sec=1.0,
        vocal_collision_score=0.0,
        beat_alignment_score=1.0,
        energy_score=0.9,
        placement_score=0.9,
        loudness_score=0.9,
        perceptual_loudness_score=0.9,
        compatibility_score=0.9,
        notes=["ok"],
    )

    built = build_transition_audio(seg_a, seg_b, cand)
    tail = render_transition_tail(seg_a, seg_b, cand)
    assert len(tail) == len(built.body)
    assert len(built.full) == len(built.prefix) + len(built.body)

    out = tmp_path / "out.wav"
    render_candidate(seg_a, seg_b, cand, out, output_subtype="PCM_16")
    assert out.exists()
