from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from weaveshift_cleanup import (
    INTENSITY,
    LOUD_MODES,
    SONG_TYPES,
    CleanupOptions,
    analyze_track,
    auto_params,
    apply_cleanup,
    cleaned_temp_copy,
    clean_audio,
    clean_file,
    main as cleanup_main,
)


def make_problem_track(sr=48000, duration=12.0, seed=0):
    """Stereo signal with a mud peak (~300 Hz), a fizz peak (~7.2 kHz), and
    a wide low end -- the things AUTO is supposed to notice and correct."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    l = 0.25 * np.sin(2 * np.pi * 300 * t) + 0.12 * np.sin(2 * np.pi * 60 * t) + 0.05 * rng.standard_normal(len(t))
    r = 0.25 * np.sin(2 * np.pi * 300 * t) + 0.12 * np.sin(2 * np.pi * 65 * t) + 0.05 * rng.standard_normal(len(t))
    fizz = 0.08 * np.sin(2 * np.pi * 7200 * t)
    l = l + fizz
    r = r + fizz * 0.95
    y = np.clip(np.column_stack([l, r]), -0.98, 0.98)
    return y, sr


def make_clean_tone(sr=48000, duration=2.0, freq=440.0, amp=0.1):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = amp * np.column_stack([np.sin(2 * np.pi * freq * t)] * 2)
    return y.astype(np.float64), sr


def make_drifting_track(sr=48000, duration=30.0, freq=220.0, quiet_amp=0.05, loud_amp=0.2):
    """A track that is quiet for its first half and clearly louder for its
    second half -- long enough (>= 20 s) for the leveler's drift check to
    engage."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    amp = np.where(t < duration / 2.0, quiet_amp, loud_amp)
    y = amp * np.sin(2 * np.pi * freq * t)
    return np.column_stack([y, y]).astype(np.float64), sr


def test_analyze_track_finds_the_mud_and_fizz_peaks():
    y, sr = make_problem_track()
    metrics = analyze_track(y, sr)
    assert abs(metrics.mud_peak_hz - 300.0) < 20.0
    assert abs(metrics.fizz_peak_hz - 7200.0) < 150.0
    assert metrics.mud_excess_db > 5.0
    assert metrics.fizz_excess_db > 5.0


def test_auto_params_respects_song_type_caps():
    y, sr = make_problem_track()
    metrics = analyze_track(y, sr)
    for song_type, st in SONG_TYPES.items():
        params = auto_params(metrics, song_type=song_type, intensity="strong")
        assert 0.0 <= params.mud_depth_db <= st.mud_cap_db + 1e-6
        assert 0.0 <= params.fizz_depth_db <= st.fizz_cap_db + 1e-6
        assert 0.0 <= params.air_gain_db <= st.air_cap_db + 1e-6


def test_intensity_scales_the_correction_up():
    y, sr = make_problem_track()
    metrics = analyze_track(y, sr)
    light = auto_params(metrics, song_type="custom", intensity="light")
    strong = auto_params(metrics, song_type="custom", intensity="strong")
    assert strong.mud_depth_db >= light.mud_depth_db
    assert strong.fizz_depth_db >= light.fizz_depth_db
    # The leveler's correction amount must scale with intensity too, the
    # same way mud/fizz/air do (regression test: auto_params() used to
    # always return level_range_db=st.level_cap_db regardless of intensity).
    assert strong.level_range_db > light.level_range_db
    balanced = auto_params(metrics, song_type="custom", intensity="balanced")
    assert light.level_range_db < balanced.level_range_db < strong.level_range_db


def test_apply_cleanup_reduces_mono_low_end_width():
    y, sr = make_problem_track(duration=6.0)
    metrics = analyze_track(y, sr)
    params = auto_params(metrics, song_type="edm", intensity="strong")
    assert params.mono_amt > 0.0

    out = apply_cleanup(y, sr, params)
    assert out.shape == y.shape
    assert np.all(np.isfinite(out))

    after = analyze_track(out, sr)
    assert after.low_width_db < metrics.low_width_db


def test_apply_cleanup_is_a_no_op_shape_wise_for_mono_input():
    y, sr = make_clean_tone()
    mono = y[:, :1]
    metrics = analyze_track(mono, sr)
    params = auto_params(metrics, song_type="custom")
    out = apply_cleanup(mono, sr, params)
    assert out.shape == mono.shape
    assert np.all(np.isfinite(out))


def test_leveler_engages_on_loudness_drift_and_turns_down_the_loud_half():
    y, sr = make_drifting_track()
    metrics = analyze_track(y, sr)
    assert metrics.loudness_drift_lu > 1.0

    params = auto_params(metrics, song_type="custom", intensity="balanced")
    assert params.level_amt > 0.0

    out = apply_cleanup(y, sr, params)
    half = len(y) // 2

    def rms_db(block):
        return 20.0 * np.log10(np.sqrt(np.mean(block.astype(np.float64) ** 2)) + 1e-12)

    before_gap = rms_db(y[half:]) - rms_db(y[:half])
    after_gap = rms_db(out[half:]) - rms_db(out[:half])
    assert after_gap < before_gap - 0.3  # the loud half was pulled down relative to the quiet half


def test_leveler_stays_off_for_short_tracks_even_with_a_big_level_jump():
    y, sr = make_drifting_track(duration=10.0)  # below the 20 s engage threshold
    metrics = analyze_track(y, sr)
    params = auto_params(metrics, song_type="custom", intensity="balanced")
    assert params.level_amt == 0.0


def test_true_peak_ceiling_is_respected_even_with_strong_loudness_boost():
    y, sr = make_problem_track(duration=6.0)
    for loud in LOUD_MODES:
        result = clean_audio(y, sr, song_type="edm", intensity="strong", loud=loud)
        assert np.all(np.isfinite(result.samples))
        assert result.output_true_peak_db <= -1.0 + 0.3  # small tolerance for the offline oversampled estimate


def test_match_loud_mode_lands_close_to_original_loudness():
    y, sr = make_clean_tone(duration=5.0, amp=0.2)
    result = clean_audio(y, sr, song_type="custom", intensity="balanced", loud="match")
    assert abs(result.output_lufs - result.metrics_before.integrated_lufs) < 1.5


def test_streaming_loud_mode_targets_minus_14_lufs_when_headroom_allows():
    y, sr = make_clean_tone(duration=5.0, amp=0.1)
    result = clean_audio(y, sr, song_type="custom", intensity="balanced", loud="streaming")
    assert abs(result.output_lufs - (-14.0)) < 0.5


def test_clean_audio_rejects_unknown_song_type():
    y, sr = make_clean_tone()
    with pytest.raises(ValueError, match="song type"):
        clean_audio(y, sr, song_type="not_a_real_song_type")


def test_clean_file_round_trips_through_disk(tmp_path):
    y, sr = make_problem_track(duration=4.0)
    src = tmp_path / "in.wav"
    sf.write(src, y, sr, subtype="PCM_16")

    out = tmp_path / "out.wav"
    result = clean_file(src, out, song_type="edm", intensity="balanced", loud="louder")
    assert out.exists()

    written, written_sr = sf.read(out)
    assert written_sr == sr
    assert written.shape[0] == pytest.approx(y.shape[0], abs=4)
    assert np.all(np.isfinite(written))
    assert len(result.changed) >= 2


def test_clean_file_supports_mp3_output(tmp_path):
    y, sr = make_clean_tone(duration=3.0)
    src = tmp_path / "in.wav"
    sf.write(src, y, sr, subtype="PCM_16")

    out = tmp_path / "out.mp3"
    clean_file(src, out, song_type="custom")
    assert out.exists()
    assert out.stat().st_size > 0


def test_cleaned_temp_copy_writes_wav_and_optional_report(tmp_path):
    y, sr = make_problem_track(duration=3.0)
    src = tmp_path / "in.wav"
    sf.write(src, y, sr, subtype="PCM_16")

    workdir = tmp_path / "work"
    workdir.mkdir()
    report_dir = tmp_path / "reports"
    opts = CleanupOptions(song_type="jazz", intensity="balanced", loud="match", report_dir=report_dir)

    cleaned_path = cleaned_temp_copy(src, workdir, opts, "track_1")
    assert cleaned_path.suffix == ".wav"
    assert cleaned_path.exists()

    report_path = report_dir / "track_1_cleanup_report.json"
    assert report_path.exists()
    import json

    report = json.loads(report_path.read_text())
    assert report["cleanup_params"]["ceiling_db"] == -1.0
    assert "what_changed" in report


def test_cli_main_writes_output_and_report(tmp_path):
    y, sr = make_problem_track(duration=3.0)
    src = tmp_path / "in.wav"
    sf.write(src, y, sr, subtype="PCM_16")
    out = tmp_path / "out.wav"
    report = tmp_path / "report.json"

    code = cleanup_main([str(src), "-o", str(out), "--song-type", "lounge", "--report", str(report)])
    assert code == 0
    assert out.exists()
    assert report.exists()


def test_cli_main_reports_missing_input_as_error(tmp_path, capsys):
    code = cleanup_main([str(tmp_path / "missing.wav")])
    assert code == 1
    captured = capsys.readouterr()
    assert "Error" in captured.err


def test_song_types_and_intensities_are_both_nonempty():
    assert set(SONG_TYPES) >= {"custom", "edm", "vocal_trance", "lounge", "jazz", "heart", "cinematic"}
    assert set(INTENSITY) == {"light", "balanced", "strong"}
