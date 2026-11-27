"""Tests for streaming Broadcast Audio QA (BS.1770-4 LUFS, 4x True Peak, splice clicks)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import pytest
import soundfile as sf

import weaveshift.broadcast_qa as bqa_mod
import weaveshift.plan as plan_mod
import weaveshift.setlist as setlist_mod
from weaveshift.audio import AudioAnalysis, NaturalTransition, TransitionCandidate
from weaveshift.broadcast_qa import (
    BroadcastAudioReport,
    analyze_broadcast_audio,
    format_broadcast_qa_warnings,
    integrated_lufs,
    run_broadcast_qa,
    scan_splice_clicks,
    transition_qa_targets,
    true_peak_dbtp,
)
from weaveshift.two_tracks import main as two_tracks_main
from tests.audio_helpers import write_stereo_mp3
from tests.cli_helpers import setlist_args


# -----------------------------
# BS.1770-4 LUFS & 4x True Peak
# -----------------------------

def test_integrated_lufs_matches_pyloudnorm_on_stereo_tone():
    sr = 48000
    t = np.linspace(0.0, 2.0, sr * 2, endpoint=False)
    mono = 0.1 * np.sin(2.0 * np.pi * 1000.0 * t)
    stereo = np.column_stack([mono, mono])

    ours = integrated_lufs(stereo, sr)
    ref = float(pyln.Meter(sr).integrated_loudness(stereo))
    assert ours == pytest.approx(ref, abs=0.15)


def test_lufs_and_true_peak_handle_silence_short_and_non_finite():
    sr = 48000
    assert integrated_lufs(np.zeros((sr, 2)), sr) == -70.0
    assert true_peak_dbtp(np.zeros((sr, 2))) == -120.0
    assert integrated_lufs(np.full((sr, 2), np.nan), sr) == -70.0
    assert true_peak_dbtp(np.full((sr, 2), np.inf)) == -120.0
    # Short (<400 ms) active audio still produces a finite LUFS value.
    short = 0.2 * np.ones((int(sr * 0.1), 2), dtype=np.float64)
    val = integrated_lufs(short, sr)
    assert -30.0 < val < 0.0


def test_true_peak_detects_intersample_overshoot():
    # Classic quarter-rate shifted cosine [+A, +A, -A, -A, ...]:
    # sample peak is A (-3.01 dBFS), while continuous inter-sample peak is sqrt(2) * A (~0.0 dBTP).
    amp = 1.0 / np.sqrt(2.0)
    pattern = np.array([amp, amp, -amp, -amp], dtype=np.float64)
    signal = np.tile(pattern, 200)
    sample_peak_db = 20.0 * np.log10(amp)
    tp_db = true_peak_dbtp(signal, oversample=4)
    assert tp_db > sample_peak_db + 2.5


# -----------------------------
# Splice-click detection
# -----------------------------

def test_scan_splice_clicks_detects_step_click_and_respects_onset_exclusion():
    sr = 48000
    t = np.linspace(0.0, 2.0, sr * 2, endpoint=False)
    clean = 0.1 * np.sin(2.0 * np.pi * 220.0 * t)
    assert scan_splice_clicks(clean, sr, boundary_times=[1.0]) == []

    # Inject a sharp splice discontinuity (step/impulse) at t = 1.0s.
    clicked = clean.copy()
    clicked[sr] += 0.65
    found = scan_splice_clicks(clicked, sr, boundary_times=[1.0])
    assert len(found) == 1
    assert abs(found[0] - 1.0) <= 0.005

    # Full-file scan (boundary_times=None) also finds it.
    found_full = scan_splice_clicks(clicked, sr)
    assert any(abs(c - 1.0) <= 0.005 for c in found_full)

    # Excluding a known beat/drum onset at 1.0s suppresses the flag.
    assert scan_splice_clicks(clicked, sr, boundary_times=[1.0], drum_onset_times=[1.0]) == []

    # Unsorted boundary_times must still detect all clicks (no earlier click dropped).
    two_clicks = clean.copy()
    two_clicks[int(0.5 * sr)] += 0.65
    two_clicks[int(1.5 * sr)] += 0.65
    found_two = scan_splice_clicks(two_clicks, sr, boundary_times=[1.5, 0.5])
    assert len(found_two) == 2
    assert abs(found_two[0] - 0.5) <= 0.005
    assert abs(found_two[1] - 1.5) <= 0.005


def test_broadcast_qa_rejects_3d_arrays_and_caps_oversample_and_handles_non_finite_warnings():
    arr3d = np.zeros((10, 2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="1-D or 2-D"):
        integrated_lufs(arr3d, 48000)
    with pytest.raises(ValueError, match="1-D or 2-D"):
        true_peak_dbtp(arr3d)
    with pytest.raises(ValueError, match="1-D or 2-D"):
        scan_splice_clicks(arr3d, 48000)

    # Oversample factor is safely bounded (no DoS on huge values, safe on negative values).
    assert np.isfinite(true_peak_dbtp(np.ones(100), oversample=-4))
    assert np.isfinite(true_peak_dbtp(np.ones(100), oversample=10000))

    # Non-finite warning values do not raise or emit bogus warnings.
    assert format_broadcast_qa_warnings(BroadcastAudioReport(float("nan"), float("nan"), [float("nan")])) == []


# -----------------------------
# Streaming parity & MP3 fallback
# -----------------------------

def test_streaming_analysis_matches_in_memory_and_catches_chunk_edge_click(tmp_path):
    sr = 48000
    dur = 3.0
    t = np.linspace(0.0, dur, int(sr * dur), endpoint=False)
    y = 0.15 * np.sin(2.0 * np.pi * 440.0 * t)
    stereo = np.column_stack([y, y])
    # Place a sharp splice click right on the 1.0s chunk boundary.
    stereo[sr, :] += 0.7

    wav = tmp_path / "stream.wav"
    sf.write(wav, stereo, sr, subtype="FLOAT")

    report = analyze_broadcast_audio(wav, boundary_times=[1.0], chunk_seconds=1.0)
    assert report.integrated_lufs == pytest.approx(integrated_lufs(stereo, sr), abs=0.05)
    assert report.true_peak_dbtp == pytest.approx(true_peak_dbtp(stereo), abs=0.02)
    assert any(abs(c - 1.0) <= 0.005 for c in report.splice_clicks)


def test_streaming_sanitizes_nan_without_poisoning_filter_state(tmp_path):
    sr = 48000
    t = np.linspace(0.0, 2.0, sr * 2, endpoint=False)
    stereo = np.column_stack([0.1 * np.sin(2.0 * np.pi * 1000.0 * t)] * 2)
    stereo[100, 0] = np.nan
    wav = tmp_path / "nan_sample.wav"
    sf.write(wav, stereo, sr, subtype="FLOAT")

    report = analyze_broadcast_audio(wav, chunk_seconds=1.0)
    assert np.isfinite(report.integrated_lufs)
    assert report.integrated_lufs > -30.0
    assert np.isfinite(report.true_peak_dbtp)


def test_analyze_broadcast_audio_mp3_ffmpeg_fallback(tmp_path, monkeypatch):
    mp3 = write_stereo_mp3(tmp_path / "clip.mp3", duration=2.0, sr=48000)

    real_soundfile = bqa_mod._analyze_broadcast_soundfile
    seen_fallback = []

    def fail_on_mp3(path: Path, **kwargs):
        if path.suffix.lower() == ".mp3":
            seen_fallback.append(path)
            raise sf.LibsndfileError(1, "simulated missing mp3 decoder")
        return real_soundfile(path, **kwargs)

    monkeypatch.setattr(bqa_mod, "_analyze_broadcast_soundfile", fail_on_mp3)

    report = analyze_broadcast_audio(mp3)
    assert len(seen_fallback) == 1
    assert np.isfinite(report.integrated_lufs)
    assert np.isfinite(report.true_peak_dbtp)


# -----------------------------
# Warnings, helpers & CLI integration
# -----------------------------

def test_warnings_targets_and_diagnostic_wrapper(tmp_path):
    clean = BroadcastAudioReport(integrated_lufs=-14.0, true_peak_dbtp=-1.0, splice_clicks=[])
    assert format_broadcast_qa_warnings(clean) == []

    hot = BroadcastAudioReport(integrated_lufs=-8.0, true_peak_dbtp=0.25, splice_clicks=[12.345])
    warnings = format_broadcast_qa_warnings(hot)
    assert any("0.25 dBTP" in w for w in warnings)
    assert any("12.345s" in w for w in warnings)

    # Diagnostic wrapper returns None instead of raising on missing file.
    assert run_broadcast_qa(tmp_path / "does_not_exist.wav") is None

    # Natural & handoff target mapping
    nat = NaturalTransition(boundary_sec=10.0, pause_sec=1.5)
    bounds, onsets = transition_qa_targets(nat, a_zero_sec=0.0, b_zero_sec=11.5)
    assert bounds == [10.0, 11.5]
    assert onsets == []


def test_setlist_and_two_tracks_record_broadcast_qa_in_reports(tmp_path, monkeypatch):
    sr = 48000
    t = np.linspace(0.0, 6.0, sr * 6, endpoint=False)
    for name, freq in (("a.wav", 220.0), ("b.wav", 330.0)):
        y = 0.08 * np.column_stack([np.sin(2.0 * np.pi * freq * t), np.sin(2.0 * np.pi * freq * t)])
        sf.write(tmp_path / name, y, sr, subtype="PCM_16")

    # 1. Two-tracks CLI
    pair_out = tmp_path / "pair.wav"
    rc = two_tracks_main(
        [
            str(tmp_path / "a.wav"),
            str(tmp_path / "b.wav"),
            "-o",
            str(pair_out),
            "--mode",
            "quick_cut",
            "--vocal-method",
            "none",
        ]
    )
    assert rc == 0
    pair_report = json.loads((tmp_path / "pair.weaveshift_mix_report.json").read_text(encoding="utf-8"))
    bqa_out = pair_report["outputs"][0]["broadcast_qa"]
    assert "integrated_lufs" in bqa_out and bqa_out["integrated_lufs"] > -40.0
    assert "true_peak_dbtp" in bqa_out and bqa_out["true_peak_dbtp"] <= 0.0
    assert isinstance(bqa_out["splice_clicks"], list)

    # 2. Setlist CLI
    manifest = tmp_path / "setlist.json"
    manifest.write_text(
        json.dumps(
            {
                "tracks": [
                    {"path": "a.wav", "title": "A"},
                    {"path": "b.wav", "title": "B"},
                ]
            }
        ),
        encoding="utf-8",
    )
    fake = AudioAnalysis(
        path="x.wav",
        duration_sec=6.0,
        bpm=128.0,
        key="A minor",
        camelot="8A",
        rms_dbfs=-20.0,
        peak_dbfs=-6.0,
        beats_sec=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        onsets_sec=[],
        vocal_segments=[],
        vocal_method="none",
        analysis_window_start_sec=0.0,
        analysis_window_duration_sec=6.0,
    )
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
    monkeypatch.setattr(plan_mod, "analyze_audio", lambda path, role, **_: fake)
    monkeypatch.setattr(plan_mod, "choose_candidates", lambda *_a, **_k: [cand])

    mix_out = tmp_path / "mix.wav"
    setlist_mod.build_continuous_mix(setlist_args(manifest, mix_out))
    setlist_report = json.loads((tmp_path / "mix.weaveshift_setlist_report.json").read_text(encoding="utf-8"))
    bqa_mix = setlist_report["broadcast_qa"]
    assert bqa_mix["integrated_lufs"] > -40.0
    assert bqa_mix["true_peak_dbtp"] <= 0.0
    assert isinstance(bqa_mix["splice_clicks"], list)
