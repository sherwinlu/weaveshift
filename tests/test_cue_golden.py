from pathlib import Path

import pytest

from weaveshift_cues import analyze_track

from tests.audio_helpers import cue_fixture_track, write_stereo_mp3


def assert_in_range(value, lo, hi, label):
    assert value is not None, f"{label} should not be None"
    assert lo <= value <= hi, f"{label}={value:.2f}s expected in [{lo}, {hi}]"


def test_analyze_track_golden_fixture_cue_ranges(tmp_path):
    wav = cue_fixture_track(tmp_path / "fixture.wav", duration=180.0)
    duration_sec, bpm_detected, cues, confidence, _notes = analyze_track(wav, bpm_hint=128.0)

    assert duration_sec == pytest.approx(180.0, abs=1.0)
    assert confidence in {"high", "medium", "low"}

    assert_in_range(cues["first_lift"], 14.0, 24.0, "first_lift")
    assert_in_range(cues["main_peak"], 40.0, 56.0, "main_peak")
    assert_in_range(cues["breakdown"], 110.0, 132.0, "breakdown")

    if bpm_detected is not None:
        assert 90.0 <= bpm_detected <= 170.0


def test_analyze_track_short_track_sets_notes(tmp_path):
    wav = cue_fixture_track(tmp_path / "short.wav", duration=90.0)
    duration_sec, _bpm, cues, _confidence, notes = analyze_track(wav, bpm_hint=None)
    assert duration_sec < 120
    assert "Short track" in notes
    assert cues["main_peak"] is None or cues["main_peak"] >= 20.0


def test_analyze_track_reads_mp3_source(tmp_path):
    mp3 = write_stereo_mp3(tmp_path / "fixture.mp3", duration=90.0, sr=22050)
    duration_sec, _bpm, cues, confidence, _notes = analyze_track(mp3, bpm_hint=128.0)

    assert duration_sec == pytest.approx(90.0, abs=2.0)
    assert confidence in {"high", "medium", "low"}
    assert cues["main_peak"] is None or cues["main_peak"] >= 0.0
