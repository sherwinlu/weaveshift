from pathlib import Path

import numpy as np
import soundfile as sf
from pydub import AudioSegment

from weaveshift_profiles import ScoringProfile
from weaveshift_two_tracks import (
    AudioAnalysis,
    TransitionCandidate,
    VocalSegment,
    apply_gain_ramp,
    apply_soft_duck,
    build_candidate_times,
    camelot_compat,
    candidate_verdict,
    choose_candidates,
    format_timestamp,
    ranked_candidate_summary,
    render_candidate,
    score_candidate,
    serialize_analysis,
    validate_loaded_segment_parity,
    validate_wav_output,
)


def analysis(
    role: str,
    *,
    duration=30.0,
    beats=None,
    onsets=None,
    vocals=None,
    bpm=128.0,
    camelot="12B",
):
    return AudioAnalysis(
        path=f"{role}.wav",
        duration_sec=duration,
        bpm=bpm,
        key="E major",
        camelot=camelot,
        rms_dbfs=-18.0,
        peak_dbfs=-3.0,
        beats_sec=list(beats or []),
        onsets_sec=list(onsets or []),
        vocal_segments=list(vocals or []),
        vocal_method="heuristic",
        analysis_window_start_sec=0.0,
        analysis_window_duration_sec=duration,
    )


def candidate(**overrides):
    data = dict(
        name="candidate",
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
        notes=["balanced candidate"],
    )
    data.update(overrides)
    return TransitionCandidate(**data)


def sine_segment(path: Path, duration=4.0, sr=48000, freq=440.0, amp=0.05):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = amp * np.column_stack([np.sin(2 * np.pi * freq * t), np.sin(2 * np.pi * freq * t)])
    sf.write(path, y, sr, subtype="PCM_16")
    return AudioSegment.from_file(str(path), format="wav")


def test_build_candidate_times_clusters_and_limits_edges():
    a = analysis("a", duration=30, beats=[22.0, 22.1, 25.0, 29.8], onsets=[24.9])
    b = analysis("b", duration=30, beats=[0.1, 0.2, 3.0, 9.5, 12.0], onsets=[3.1])

    a_cuts, b_cues = build_candidate_times(a, b, max_trim_a_sec=8.0, b_cue_max_sec=10.0)

    assert a_cuts[-1] == 30.0
    assert len([t for t in a_cuts if 21.9 <= t <= 22.2]) == 1
    assert b_cues[0] == 0.0
    assert 12.0 not in b_cues


def test_score_candidate_uses_energy_curve_for_gain_and_notes():
    a = analysis("a", duration=30, beats=[24.0], onsets=[])
    b = analysis("b", duration=30, beats=[0.0], onsets=[])
    a.energy_curve = {"times": np.array([20.0, 22.0, 24.0]), "dbfs": np.array([-14.0, -14.0, -14.0])}
    b.energy_curve = {"times": np.array([0.0, 2.0, 4.0]), "dbfs": np.array([-20.0, -20.0, -20.0])}

    c = score_candidate(a, b, a_cut=24.0, b_cue=0.0, overlap=4.0, name="candidate")

    assert c.score > 0
    assert c.b_gain_db == 4.0
    assert "applies +4.0 dB to Track B" in c.notes


def test_choose_candidates_returns_distinct_named_and_profile_options():
    a = analysis("a", duration=30, beats=[18, 20, 22, 24, 26, 28, 30], onsets=[21, 25, 29])
    b = analysis("b", duration=30, beats=[0, 2, 4, 6, 8, 10], onsets=[1, 3, 5])
    profile = ScoringProfile(
        name="wide",
        description="wide overlaps",
        preferences={"preferred_overlap_min_sec": 4.0, "preferred_overlap_max_sec": 10.0, "target_overlap_sec": 6.0},
        limits={"max_tail_trim_sec": 8.0, "max_b_cue_sec": 10.0, "max_vocal_collision_score": 0.35},
    )

    selected = choose_candidates(a, b, 8.0, 10.0, scoring_profile=profile)
    names = {c.name for c in selected}

    assert {"vocal_safe", "beat_aligned", "quick_cut", "smooth", "profile"} <= names
    assert len({(c.a_fade_start_sec, c.b_cue_sec, c.overlap_sec) for c in selected}) == len(selected)


def test_vocal_overlap_can_produce_ducked_candidate():
    a = analysis("a", duration=30, beats=[20, 22, 24, 26, 28, 30], vocals=[VocalSegment(20.0, 24.0)])
    b = analysis("b", duration=30, beats=[0, 2, 4, 6, 8], vocals=[VocalSegment(2.0, 6.0)])

    selected = choose_candidates(a, b, 8.0, 8.0)
    ducked = [c for c in selected if c.name == "vocal_ducked"]

    assert ducked
    assert ducked[0].soft_duck_target == "a"
    assert ducked[0].soft_duck_db < 0


def test_render_candidate_writes_output_and_snippet(tmp_path):
    seg_a = sine_segment(tmp_path / "a.wav", duration=4.0, freq=440)
    seg_b = sine_segment(tmp_path / "b.wav", duration=4.0, freq=660)
    out = tmp_path / "out.wav"
    snippet = tmp_path / "snippet.wav"
    c = candidate(a_fade_start_sec=2.0, a_cut_sec=3.0, b_cue_sec=0.0, overlap_sec=1.0)

    render_candidate(seg_a, seg_b, c, out, snippet, output_subtype="PCM_16")

    assert out.exists()
    assert snippet.exists()
    assert sf.info(str(out)).subtype == "PCM_16"
    assert sf.info(str(out)).duration > 4.0


def test_audio_helpers_and_report_serialization(tmp_path):
    seg = sine_segment(tmp_path / "a.wav", duration=1.0)
    ramped = apply_gain_ramp(seg, -12.0, 0.0)
    ducked = apply_soft_duck(seg, -3.0, attack_ms=50, release_ms=50)

    assert len(ramped) == len(seg)
    assert len(ducked) == len(seg)
    assert validate_wav_output(str(tmp_path / "mix")) == tmp_path / "mix.wav"
    assert format_timestamp(65.2) == "1:05"
    assert camelot_compat("12B", "12A") > camelot_compat("12B", "7A")
    assert "possible vocal overlap" in candidate_verdict(candidate(vocal_collision_score=0.4))
    assert ranked_candidate_summary([candidate(name="b", score=0.5), candidate(name="a", score=0.9)])[0]["name"] == "a"
    assert serialize_analysis(analysis("a", vocals=[VocalSegment(1.0, 2.0)]))["vocal_segments"][0]["start_sec"] == 1.0


def test_validate_loaded_segment_parity_rejects_decoder_mismatch(tmp_path):
    seg_a = sine_segment(tmp_path / "a.wav", sr=48000)
    seg_b = sine_segment(tmp_path / "b.wav", sr=44100)

    try:
        validate_loaded_segment_parity(seg_a, seg_b)
    except ValueError as exc:
        assert "Decoded AudioSegment formats do not match" in str(exc)
    else:
        raise AssertionError("expected decoded segment parity failure")
