"""Cleanup wired into the two-track and setlist planners via --cleanup."""
from pathlib import Path

import numpy as np
import soundfile as sf

from weaveshift_audio import AudioAnalysis
from weaveshift_cleanup import CleanupOptions
from weaveshift_two_tracks import TransitionCandidate


def write_tone(path: Path, *, sr=48000, duration=4.0, freq=440.0, amp=0.1):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = amp * np.column_stack([np.sin(2 * np.pi * freq * t)] * 2)
    sf.write(path, y, sr, subtype="PCM_16")


def fake_analysis():
    return AudioAnalysis(
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


def fake_candidate():
    return TransitionCandidate(
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


def test_plan_two_track_mix_with_cleanup_analyzes_a_cleaned_copy(tmp_path, monkeypatch):
    import weaveshift_plan as plan_mod

    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    write_tone(a, freq=440.0)
    write_tone(b, freq=660.0)

    seen_paths = []

    def fake_analyze_audio(path, role, **_):
        seen_paths.append(Path(path))
        return fake_analysis()

    cand = fake_candidate()
    monkeypatch.setattr(plan_mod, "analyze_audio", fake_analyze_audio)
    monkeypatch.setattr(plan_mod, "choose_candidates", lambda *_a, **_k: [cand])

    cleanup = CleanupOptions(song_type="custom", intensity="light", loud="match")
    mix_plan = plan_mod.plan_two_track_mix(
        str(a), str(b), search_a_sec=1.0, search_b_sec=1.0, vocal_method="heuristic", prefer_mps=False,
        cleanup=cleanup,
    )

    assert mix_plan.recommended_name == "vocal_safe"
    # analyze_audio was called on cleaned temp copies, not the original files.
    assert len(seen_paths) == 2
    for p in seen_paths:
        assert p != a and p != b
        assert p.name.endswith("_cleaned.wav")
    # original-file format info is unaffected by the cleanup substitution.
    assert mix_plan.track_a_path == a
    assert mix_plan.track_a_wav_info["subtype"] == "PCM_16"
    # segments were loaded (from the cleaned copies) with the same transport-level format.
    assert mix_plan.seg_a.frame_rate == 48000


def test_plan_two_track_mix_without_cleanup_analyzes_the_original_files(tmp_path, monkeypatch):
    import weaveshift_plan as plan_mod

    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    write_tone(a, freq=440.0)
    write_tone(b, freq=660.0)

    seen_paths = []

    def fake_analyze_audio(path, role, **_):
        seen_paths.append(Path(path))
        return fake_analysis()

    monkeypatch.setattr(plan_mod, "analyze_audio", fake_analyze_audio)
    monkeypatch.setattr(plan_mod, "choose_candidates", lambda *_a, **_k: [fake_candidate()])

    plan_mod.plan_two_track_mix(
        str(a), str(b), search_a_sec=1.0, search_b_sec=1.0, vocal_method="heuristic", prefer_mps=False,
    )

    assert set(seen_paths) == {a, b}


def test_plan_setlist_mix_with_cleanup_analyzes_cleaned_copies(tmp_path, monkeypatch):
    import weaveshift_plan as plan_mod
    from weaveshift_plan import TrackSpec

    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    write_tone(a, freq=440.0)
    write_tone(b, freq=660.0)

    seen_paths = []

    def fake_analyze_audio(path, role, **_):
        seen_paths.append(Path(path))
        return fake_analysis()

    monkeypatch.setattr(plan_mod, "analyze_audio", fake_analyze_audio)
    monkeypatch.setattr(plan_mod, "choose_candidates", lambda *_a, **_k: [fake_candidate()])

    cleanup = CleanupOptions(song_type="custom", intensity="light", loud="match")
    mix_plan = plan_mod.plan_setlist_mix(
        [TrackSpec(path=str(a), title="A"), TrackSpec(path=str(b), title="B")],
        search_a_sec=1.0,
        search_b_sec=1.0,
        vocal_method="heuristic",
        prefer_mps=False,
        cleanup=cleanup,
    )

    assert len(mix_plan.junctions) == 1
    assert len(seen_paths) == 2
    for p in seen_paths:
        assert p != a and p != b
    # the returned plan still reports the original track specs.
    assert [t.path for t in mix_plan.tracks] == [str(a), str(b)]
    assert mix_plan.segments[0].frame_rate == 48000
