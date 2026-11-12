"""Exercises the --cleanup CLI flags themselves (argparse wiring, the
--cleanup-song-type fallback onto --profile, and --cleanup-report-dir),
as opposed to tests/test_cleanup_integration.py which calls
weaveshift_plan.plan_two_track_mix / plan_setlist_mix directly with a
hand-built CleanupOptions.
"""
import json

import numpy as np
import soundfile as sf

import weaveshift_plan
import weaveshift_setlist
import weaveshift_two_tracks
from weaveshift_cleanup import SONG_TYPES
from weaveshift_two_tracks import AudioAnalysis, TransitionCandidate


def write_wav(path, duration=4.0, sr=48000, freq=440.0):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.05 * np.column_stack([np.sin(2 * np.pi * freq * t), np.sin(2 * np.pi * freq * t)])
    sf.write(path, y, sr, subtype="PCM_16")


def fake_analysis(path, role):
    return AudioAnalysis(
        path=path,
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


def fake_candidate(name="vocal_safe"):
    return TransitionCandidate(
        name=name,
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


def test_two_tracks_cli_without_cleanup_flag_defaults_to_no_cleanup(tmp_path):
    """--cleanup is opt-in: build_parser()'s own default must be False."""
    args = weaveshift_two_tracks.build_parser().parse_args(["a.wav", "b.wav"])
    assert args.cleanup is False
    assert args.cleanup_song_type is None
    assert args.cleanup_intensity == "balanced"
    assert args.cleanup_loud == "louder"
    assert args.cleanup_report_dir is None


def test_two_tracks_cli_cleanup_flag_cleans_tracks_and_resolves_song_type_from_profile(tmp_path, monkeypatch):
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    write_wav(a, freq=440.0)
    write_wav(b, freq=660.0)
    out = tmp_path / "mix.wav"
    report_dir = tmp_path / "cleanup_reports"

    monkeypatch.setattr(weaveshift_plan, "analyze_audio", lambda path, role, **_: fake_analysis(path, role))
    monkeypatch.setattr(weaveshift_plan, "choose_candidates", lambda *_a, **_k: [fake_candidate()])

    args = weaveshift_two_tracks.build_parser().parse_args(
        [
            str(a),
            str(b),
            "-o",
            str(out),
            "--mode",
            "profile",
            "--profile",
            "vocal_trance",
            "--vocal-method",
            "heuristic",
            "--no-mps",
            "--search-a-sec",
            "1.0",
            "--search-b-sec",
            "1.0",
            "--cleanup",
            "--cleanup-report-dir",
            str(report_dir),
        ]
    )

    weaveshift_two_tracks.run(args)

    produced = list(tmp_path.glob("mix_*.wav"))
    assert produced, "two_tracks CLI with --cleanup should still render an output file"
    for f in produced:
        assert sf.info(str(f)).samplerate == 48000

    report_a = json.loads((report_dir / "track_a_cleanup_report.json").read_text())
    report_b = json.loads((report_dir / "track_b_cleanup_report.json").read_text())
    # --cleanup-song-type was not passed, so it must fall back to --profile
    # ("vocal_trance" is a real cleanup Song Type, not just a scoring profile).
    assert report_a["cleanup_params"]["mono_freq"] == SONG_TYPES["vocal_trance"].mono_freq
    assert report_b["cleanup_params"]["mono_freq"] == SONG_TYPES["vocal_trance"].mono_freq
    assert "what_changed" in report_a and "what_changed" in report_b


def test_two_tracks_cli_explicit_cleanup_song_type_overrides_profile_fallback(tmp_path, monkeypatch):
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    write_wav(a, freq=440.0)
    write_wav(b, freq=660.0)
    out = tmp_path / "mix.wav"
    report_dir = tmp_path / "cleanup_reports"

    monkeypatch.setattr(weaveshift_plan, "analyze_audio", lambda path, role, **_: fake_analysis(path, role))
    monkeypatch.setattr(weaveshift_plan, "choose_candidates", lambda *_a, **_k: [fake_candidate()])

    args = weaveshift_two_tracks.build_parser().parse_args(
        [
            str(a),
            str(b),
            "-o",
            str(out),
            "--vocal-method",
            "heuristic",
            "--no-mps",
            "--search-a-sec",
            "1.0",
            "--search-b-sec",
            "1.0",
            "--cleanup",
            "--cleanup-song-type",
            "cinematic",
            "--cleanup-report-dir",
            str(report_dir),
        ]
    )
    # --profile stays at its "edm" default here; explicit --cleanup-song-type must win.
    assert args.profile == "edm"

    weaveshift_two_tracks.run(args)

    report_a = json.loads((report_dir / "track_a_cleanup_report.json").read_text())
    assert report_a["cleanup_params"]["mono_freq"] == SONG_TYPES["cinematic"].mono_freq


def test_setlist_cli_cleanup_flag_cleans_every_track_and_writes_reports(tmp_path, monkeypatch):
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    write_wav(a, freq=440.0)
    write_wav(b, freq=660.0)
    setlist_file = tmp_path / "setlist.txt"
    setlist_file.write_text(f"{a}\n{b}\n", encoding="utf-8")
    out = tmp_path / "mix.wav"
    report_dir = tmp_path / "cleanup_reports"

    monkeypatch.setattr(weaveshift_plan, "analyze_audio", lambda path, role, **_: fake_analysis(path, role))
    monkeypatch.setattr(weaveshift_plan, "choose_candidates", lambda *_a, **_k: [fake_candidate()])

    args = weaveshift_setlist.build_parser().parse_args(
        [
            str(setlist_file),
            "-o",
            str(out),
            "--vocal-method",
            "heuristic",
            "--no-mps",
            "--search-a-sec",
            "1.0",
            "--search-b-sec",
            "1.0",
            "--cleanup",
            "--cleanup-song-type",
            "jazz",
            "--cleanup-report-dir",
            str(report_dir),
        ]
    )

    weaveshift_setlist.build_continuous_mix(args)

    assert out.exists()
    assert sf.info(str(out)).samplerate == 48000

    report_1 = json.loads((report_dir / "track_1_cleanup_report.json").read_text())
    report_2 = json.loads((report_dir / "track_2_cleanup_report.json").read_text())
    assert report_1["cleanup_params"]["mono_freq"] == SONG_TYPES["jazz"].mono_freq
    assert report_2["cleanup_params"]["mono_freq"] == SONG_TYPES["jazz"].mono_freq


def test_setlist_cli_without_cleanup_flag_does_not_write_reports(tmp_path, monkeypatch):
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    write_wav(a, freq=440.0)
    write_wav(b, freq=660.0)
    setlist_file = tmp_path / "setlist.txt"
    setlist_file.write_text(f"{a}\n{b}\n", encoding="utf-8")
    out = tmp_path / "mix.wav"
    report_dir = tmp_path / "cleanup_reports"

    monkeypatch.setattr(weaveshift_plan, "analyze_audio", lambda path, role, **_: fake_analysis(path, role))
    monkeypatch.setattr(weaveshift_plan, "choose_candidates", lambda *_a, **_k: [fake_candidate()])

    args = weaveshift_setlist.build_parser().parse_args(
        [
            str(setlist_file),
            "-o",
            str(out),
            "--vocal-method",
            "heuristic",
            "--no-mps",
            "--search-a-sec",
            "1.0",
            "--search-b-sec",
            "1.0",
        ]
    )
    assert args.cleanup is False

    weaveshift_setlist.build_continuous_mix(args)

    assert out.exists()
    assert not report_dir.exists()
