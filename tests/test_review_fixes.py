import json

import numpy as np
import pytest

import weaveshift_audio
import weaveshift_plan
import weaveshift_setlist
from weaveshift_audio import AudioAnalysis, TransitionCandidate, analyze_audio, analyze_vocals
from weaveshift_plan import (
    TrackSpec,
    apply_handoff_transition_override,
    apply_manual_transition_override,
    build_natural_transition,
    plan_setlist_mix,
)
from weaveshift_reports import build_setlist_report, build_two_track_report, serialize_analysis
from weaveshift_scoring import profile_search_parameters
from weaveshift_profiles import ScoringProfile
from tests.audio_helpers import write_click_track, write_stereo_wav


def analysis(role: str, *, duration=30.0, beats=None, bpm=128.0):
    times = np.array([0.0, 10.0, 20.0, 30.0], dtype=float)
    curve = {"times": times, "dbfs": np.full_like(times, -14.0)}
    return AudioAnalysis(
        path=f"{role}.wav",
        duration_sec=duration,
        bpm=bpm,
        key="E major",
        camelot="12B",
        rms_dbfs=-18.0,
        peak_dbfs=-3.0,
        beats_sec=list(beats or []),
        onsets_sec=[],
        vocal_segments=[],
        vocal_method="heuristic",
        analysis_window_start_sec=max(0.0, duration - 35.0) if role == "a" else 0.0,
        analysis_window_duration_sec=min(35.0, duration),
        energy_curve=curve,
        loudness_curve=curve,
    )


def test_serialize_analysis_is_json_serializable():
    a = analysis("a")
    payload = serialize_analysis(a)
    json.dumps(payload)
    assert "energy_curve" not in payload
    assert payload["energy_curve_points"] == 4


def test_natural_transition_rejects_negative_or_non_finite_pause():
    for pause in (-0.1, 300.001, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="pause_sec must be between 0 and 300 seconds"):
            build_natural_transition(30.0, pause, 1)


def test_build_two_track_report_round_trips_through_json():
    a = analysis("a")
    b = analysis("b")
    from weaveshift_reports import ranked_candidate_summary

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
    json.loads(json.dumps(report))


def test_build_setlist_report_validates_transition_shape():
    report = build_setlist_report(
        app_name="WeaveShift Setlist",
        transition_mode="recommended",
        profile=None,
        scoring_config=None,
        vocal_method="heuristic",
        output_file="/tmp/mix.wav",
        output_duration_sec=120.0,
        output_duration_timestamp="2:00",
        final_gain_db=0.0,
        wav_format_validation={"reference": {}, "tracks": []},
        tracks=[{"path": "a.wav", "title": "A"}],
        track_source_zero_in_mix_sec=[0.0, 10.0],
        transitions=[
            {
                "index": 1,
                "selected_candidate": "vocal_safe",
                "mix_transition_start_sec": 20.0,
                "source_a_fade_start_sec": 20.0,
                "source_b_cue_sec": 0.0,
                "overlap_sec": 4.0,
                "ranked_candidates": [],
            }
        ],
        transition_overrides=None,
        snippet_files=[],
    )
    assert report["schema_version"] == "1.0.0"
    json.dumps(report)


def test_analyze_audio_track_a_uses_outro_offset_for_beats(monkeypatch, tmp_path):
    wav = write_click_track(tmp_path / "long.wav", duration=240.0, bpm=128.0)
    captured = {}

    real_sf_read = weaveshift_audio.sf.read

    def fake_sf_read(path, *args, start=0, frames=-1, **kwargs):
        # Only the windowed tempo/beat read (called positionally with start=...) is of
        # interest here; the main stereo read at the top of analyze_audio also calls
        # sf.read but isn't what this test is checking.
        if start:
            sr = weaveshift_audio.sf.info(path).samplerate
            captured["offset"] = start / sr
            captured["duration"] = frames / sr
        return real_sf_read(path, *args, start=start, frames=frames, **kwargs)

    monkeypatch.setattr(weaveshift_audio.sf, "read", fake_sf_read)
    monkeypatch.setattr(weaveshift_audio, "analyze_vocals", lambda *args, **kwargs: ([], "heuristic"))

    result = analyze_audio(
        str(wav),
        role="a",
        window_sec=35.0,
        manual_bpm=128.0,
        manual_key=None,
        vocal_method="heuristic",
        prefer_mps=False,
    )
    assert captured["offset"] == pytest.approx(205.0, abs=0.05)
    assert captured["duration"] == pytest.approx(35.0, abs=0.05)
    assert result.analysis_window_start_sec == pytest.approx(205.0, abs=0.05)


def test_manual_override_recomputes_component_scores():
    a = analysis("a", duration=30, beats=[24.0, 26.0, 28.0, 30.0])
    b = analysis("b", duration=30, beats=[0.0, 2.0, 4.0])
    base = TransitionCandidate(
        name="vocal_safe",
        score=0.5,
        a_fade_start_sec=10.0,
        a_cut_sec=14.0,
        b_cue_sec=0.0,
        overlap_sec=4.0,
        b_gain_db=0.0,
        trim_a_tail_sec=16.0,
        vocal_collision_score=0.9,
        beat_alignment_score=0.1,
        energy_score=0.1,
        placement_score=0.1,
        loudness_score=0.1,
        perceptual_loudness_score=0.1,
        compatibility_score=0.1,
        notes=["stale"],
    )
    overridden = apply_manual_transition_override(
        base,
        {
            "mode": "manual",
            "a_fade_start_sec": 26.0,
            "a_cut_sec": 30.0,
            "b_cue_sec": 0.0,
            "overlap_sec": 4.0,
            "b_gain_db": 0.0,
        },
        1,
        a,
        b,
    )
    assert overridden.vocal_collision_score != base.vocal_collision_score
    assert overridden.trim_a_tail_sec == pytest.approx(0.0, abs=0.01)
    assert "component scores recomputed" in " ".join(overridden.notes)


def test_handoff_override_validates_fade_in_covers_takeover_overlap():
    a = analysis("a", duration=30, beats=[24.0, 26.0, 28.0, 30.0])
    b = analysis("b", duration=30, beats=[0.0, 2.0, 4.0])
    base = TransitionCandidate(
        name="vocal_safe",
        score=0.5,
        a_fade_start_sec=10.0,
        a_cut_sec=14.0,
        b_cue_sec=0.0,
        overlap_sec=4.0,
        b_gain_db=0.0,
        trim_a_tail_sec=16.0,
        vocal_collision_score=0.9,
        beat_alignment_score=0.1,
        energy_score=0.1,
        placement_score=0.1,
        loudness_score=0.1,
        perceptual_loudness_score=0.1,
        compatibility_score=0.1,
        notes=[],
    )

    with pytest.raises(ValueError, match="b_fade_in_sec >= takeover_overlap_sec"):
        apply_handoff_transition_override(
            base,
            {
                "mode": "handoff",
                "a_fade_start_sec": 20.0,
                "a_cut_sec": 24.0,
                "b_cue_sec": 1.5,
                "takeover_overlap_sec": 0.5,
                "b_entry_gain_db": -1.0,
                "b_fade_in_sec": 0.25,
            },
            1,
            a,
            b,
        )


def test_analyze_vocals_auto_falls_back_to_heuristic(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("no demucs")

    monkeypatch.setattr(weaveshift_audio, "vocal_segments_demucs", boom)
    y = np.zeros((2, 48000), dtype=np.float32)
    segs, method = analyze_vocals("x.wav", y, 48000, 0.0, 1.0, "auto", False)
    assert method == "heuristic"
    assert segs == []


def test_analyze_vocals_demucs_mode_reraises(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("no demucs")

    monkeypatch.setattr(weaveshift_audio, "vocal_segments_demucs", boom)
    y = np.zeros((2, 48000), dtype=np.float32)
    with pytest.raises(RuntimeError, match="no demucs"):
        analyze_vocals("x.wav", y, 48000, 0.0, 1.0, "demucs", False)


def test_profile_overlap_fallback_logs_warning(caplog):
    profile = ScoringProfile(
        name="broken",
        preferences={"preferred_overlap_min_sec": 20.0, "preferred_overlap_max_sec": 25.0},
    )
    with caplog.at_level("WARNING"):
        _, _, overlaps, _ = profile_search_parameters(8.0, 10.0, profile)
    assert overlaps == [2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    assert any("excludes all standard overlaps" in r.message for r in caplog.records)


def test_three_track_setlist_plan_monkeypatched(tmp_path, monkeypatch):
    paths = []
    for i, freq in enumerate((440.0, 550.0, 660.0), start=1):
        p = write_stereo_wav(tmp_path / f"t{i}.wav", duration=6.0, freq=freq)
        paths.append(p)

    fake = analysis("a", duration=6.0, beats=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    cand = TransitionCandidate(
        name="vocal_safe",
        score=0.91,
        a_fade_start_sec=4.0,
        a_cut_sec=5.0,
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

    monkeypatch.setattr(weaveshift_plan, "analyze_audio", lambda path, role, **_: fake)
    monkeypatch.setattr(weaveshift_plan, "choose_candidates", lambda *_a, **_k: [cand])

    tracks = [TrackSpec(path=str(p), title=f"T{i}") for i, p in enumerate(paths, start=1)]
    plan = plan_setlist_mix(tracks, search_a_sec=2.0, search_b_sec=2.0, vocal_method="heuristic", prefer_mps=False)
    assert len(plan.junctions) == 2
    assert plan.source_zero_in_mix_sec == [0.0, pytest.approx(4.0), pytest.approx(8.0)]


def test_three_track_setlist_execute_writes_mix(tmp_path, monkeypatch):
    paths = []
    for i, freq in enumerate((440.0, 550.0, 660.0), start=1):
        p = write_stereo_wav(tmp_path / f"t{i}.wav", duration=6.0, freq=freq)
        paths.append(p)

    manifest = tmp_path / "setlist.json"
    manifest.write_text(
        json.dumps({"tracks": [{"path": p.name, "title": f"T{i}"} for i, p in enumerate(paths, start=1)]}),
        encoding="utf-8",
    )
    out = tmp_path / "mix.wav"

    fake = analysis("a", duration=6.0, beats=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    cand = TransitionCandidate(
        name="vocal_safe",
        score=0.91,
        a_fade_start_sec=4.0,
        a_cut_sec=5.0,
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

    monkeypatch.setattr(weaveshift_plan, "analyze_audio", lambda path, role, **_: fake)
    monkeypatch.setattr(weaveshift_plan, "choose_candidates", lambda *_a, **_k: [cand])

    from types import SimpleNamespace

    weaveshift_setlist.build_continuous_mix(
        SimpleNamespace(
            setlist=str(manifest),
            output=str(out),
            transition_mode="recommended",
            make_snippets=False,
            profile="edm",
            scoring_config=None,
            vocal_method="heuristic",
            no_mps=True,
            search_a_sec=2.0,
            search_b_sec=2.0,
            max_trim_a_sec=2.0,
            b_cue_max_sec=2.0,
            apply_manifest_settings=False,
        )
    )
    assert out.exists()
    report = json.loads((tmp_path / "mix.weaveshift_1_0_0_setlist_report.json").read_text(encoding="utf-8"))
    assert len(report["transitions"]) == 2
    assert len(report["track_source_zero_in_mix_sec"]) == 3
