from pathlib import Path

import numpy as np
import pytest

from weaveshift_audio import (
    camelot_compat,
    compute_local_energy_curve,
    compute_local_lufs_curve,
    dbfs_from_samples,
    energy_db_from_curve,
    lufs_from_curve,
    parse_manual_key,
    validate_loaded_segment_parity,
    validate_wav_input,
    validate_wav_output,
    validate_wav_pair_compatibility,
    analyze_audio,
    vocal_segments_heuristic,
)
from pydub import AudioSegment

from tests.audio_helpers import write_stereo_wav


def test_validate_wav_input_rejects_missing_and_bad_suffix(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_wav_input(str(tmp_path / "missing.wav"), "Track A")

    fake = tmp_path / "track.mp3"
    fake.write_bytes(b"ID3")
    with pytest.raises(ValueError, match="must be a WAV file"):
        validate_wav_input(str(fake), "Track A")


def test_validate_wav_input_rejects_non_wave_header(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"NOTAVALIDWAVFILE")
    with pytest.raises(ValueError, match="not a RIFF/RF64 WAVE file"):
        validate_wav_input(str(bad), "Track A")


def test_validate_wav_pair_compatibility_detects_samplerate_mismatch(tmp_path):
    a = write_stereo_wav(tmp_path / "a.wav", sr=48000)
    b = write_stereo_wav(tmp_path / "b.wav", sr=44100)
    validate_wav_input(str(a), "Track A")
    validate_wav_input(str(b), "Track B")
    with pytest.raises(ValueError, match="WAV formats do not match"):
        validate_wav_pair_compatibility(a, b)


def test_validate_wav_output_normalizes_suffix(tmp_path):
    assert validate_wav_output(str(tmp_path / "mix")) == tmp_path / "mix.wav"
    with pytest.raises(ValueError):
        validate_wav_output(str(tmp_path / "mix.mp3"))


def test_curve_helpers_and_key_parsing():
    curve = {"times": np.array([0.0, 1.0, 2.0]), "dbfs": np.array([-20.0, -14.0, -10.0])}
    assert energy_db_from_curve(curve, 0.0, 1.0) > -20.0
    assert energy_db_from_curve(None, 0.0, 1.0) == -40.0

    lufs_curve = {"times": np.array([0.0, 1.0]), "lufs": np.array([-18.0, -16.0])}
    assert lufs_from_curve(lufs_curve, 0.0, 1.0) is not None
    assert lufs_from_curve(None, 0.0, 1.0) is None

    key, cam = parse_manual_key("E major")
    assert key == "E major"
    assert cam == "12B"
    assert camelot_compat("12B", "12A") > camelot_compat("12B", "7A")


def test_compute_local_energy_and_lufs_curves():
    sr = 48000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    quiet = 0.01 * np.sin(2 * np.pi * 440 * t)
    loud = 0.20 * np.sin(2 * np.pi * 440 * t)
    y = np.column_stack([np.concatenate([quiet, loud]), np.concatenate([quiet, loud])]).astype(np.float32)
    curve = compute_local_energy_curve(y, sr, offset_sec=0.0)
    assert len(curve["times"]) > 0
    assert float(curve["dbfs"][-1]) > float(curve["dbfs"][0])

    lufs = compute_local_lufs_curve(y, sr, offset_sec=0.0)
    if lufs["times"].size:
        assert lufs["lufs"].size == lufs["times"].size


def test_dbfs_from_samples_handles_empty_and_signal():
    assert dbfs_from_samples(np.array([], dtype=np.float32))[0] == -120.0
    rms_db, peak_db = dbfs_from_samples(np.array([0.1, -0.1], dtype=np.float32))
    assert rms_db > -40.0
    assert peak_db > rms_db - 1.0


def test_analyze_audio_attaches_curves_on_short_wav(tmp_path):
    wav = write_stereo_wav(tmp_path / "track.wav", duration=6.0, freq=220.0, amp=0.08)
    analysis = analyze_audio(
        str(wav),
        role="b",
        window_sec=5.0,
        manual_bpm=128.0,
        manual_key="E major",
        vocal_method="heuristic",
        prefer_mps=False,
    )
    assert analysis.duration_sec == pytest.approx(6.0, abs=0.05)
    assert analysis.bpm == 128.0
    assert analysis.key == "E major"
    assert analysis.energy_curve is not None
    assert analysis.loudness_curve is not None
    assert len(analysis.energy_curve["times"]) > 0


def test_vocal_segments_heuristic_finds_midrange_activity():
    sr = 22050
    duration = 4.0
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    y = np.zeros((1, n), dtype=np.float32)
    vocal = 0.25 * np.sin(2 * np.pi * 350.0 * t)
    y[0, int(1.0 * sr): int(2.5 * sr)] = vocal[int(1.0 * sr): int(2.5 * sr)]
    segs = vocal_segments_heuristic(y, sr, offset_sec=0.0)
    assert segs
    assert any(seg.start_sec <= 1.5 and seg.end_sec >= 2.0 for seg in segs)


def test_validate_loaded_segment_parity_rejects_mismatch(tmp_path):
    seg_a = AudioSegment.from_file(str(write_stereo_wav(tmp_path / "a.wav", sr=48000)), format="wav")
    seg_b = AudioSegment.from_file(str(write_stereo_wav(tmp_path / "b.wav", sr=44100)), format="wav")
    with pytest.raises(ValueError, match="Decoded AudioSegment formats do not match"):
        validate_loaded_segment_parity(seg_a, seg_b)
