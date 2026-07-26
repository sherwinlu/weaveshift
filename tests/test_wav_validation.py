from pathlib import Path
import numpy as np
import soundfile as sf
import pytest

from weaveshift_two_tracks import validate_wav_input, validate_wav_pair_compatibility, OVERLAP_LENGTHS, MAX_OVERLAP_SEC


def write_wav(path: Path, sr=48000, subtype="PCM_16", channels=2):
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
    y = 0.01 * np.sin(2 * np.pi * 440 * t)
    if channels == 2:
        y = np.column_stack([y, y])
    sf.write(path, y, sr, subtype=subtype)


def test_overlap_constant_is_source_of_truth():
    assert MAX_OVERLAP_SEC == max(OVERLAP_LENGTHS)


def test_validate_real_wav(tmp_path):
    p = tmp_path / "a.wav"
    write_wav(p)
    assert validate_wav_input(str(p), "Track A") == p


def test_reject_mislabeled_wav(tmp_path):
    p = tmp_path / "fake.wav"
    p.write_bytes(b"not a wave")
    with pytest.raises(ValueError):
        validate_wav_input(str(p), "Track A")


def test_reject_mismatched_sample_rate(tmp_path):
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    write_wav(a, sr=48000)
    write_wav(b, sr=44100)
    with pytest.raises(ValueError):
        validate_wav_pair_compatibility(a, b)


def test_missing_file_cli_exits_cleanly():
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, "weaveshift_two_tracks.py", "missing_a.wav", "missing_b.wav"],
        cwd=str(Path(__file__).resolve().parents[1]),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    assert "Error:" in result.stderr
    assert "Traceback" not in result.stderr


def test_export_wav_matching_subtype_preserves_pcm24(tmp_path):
    """Regression test for the v4.3.1 bit-depth fix.

    ffmpeg/pydub decode 24-bit PCM WAV into 32-bit-padded in-memory samples,
    so AudioSegment.sample_width reads as 4 even for a genuinely PCM_24
    source. export_wav_matching_subtype() must force the ffmpeg encoder
    explicitly rather than trusting the in-memory width.
    """
    from pydub import AudioSegment
    from weaveshift_two_tracks import export_wav_matching_subtype

    src = tmp_path / "src24.wav"
    write_wav(src, subtype="PCM_24")
    assert sf.info(str(src)).subtype == "PCM_24"

    seg = AudioSegment.from_file(str(src), format="wav")
    # This is the exact trap the bug hid behind: pydub already reports this
    # as 32-bit-wide, even though the source file is genuinely PCM_24.
    assert seg.sample_width == 4

    out = tmp_path / "out24.wav"
    export_wav_matching_subtype(seg, out, "PCM_24")
    assert sf.info(str(out)).subtype == "PCM_24"


def test_export_wav_matching_subtype_preserves_pcm16(tmp_path):
    from pydub import AudioSegment
    from weaveshift_two_tracks import export_wav_matching_subtype

    src = tmp_path / "src16.wav"
    write_wav(src, subtype="PCM_16")
    seg = AudioSegment.from_file(str(src), format="wav")
    out = tmp_path / "out16.wav"
    export_wav_matching_subtype(seg, out, "PCM_16")
    assert sf.info(str(out)).subtype == "PCM_16"


def test_candidate_too_close_deduplicates_vocal_ducked():
    """Regression test for the v4.3.1 vocal_ducked de-duplication fix.

    candidate_too_close() is the single shared predicate used both for the
    named pools (vocal_safe/beat_aligned/quick_cut/long_blend) and for
    vocal_ducked, so this exercises the actual production logic rather than
    a re-implementation of it.
    """
    from dataclasses import replace
    from weaveshift_two_tracks import TransitionCandidate, candidate_too_close

    base = TransitionCandidate(
        name="candidate", score=0.9, a_fade_start_sec=40.0, a_cut_sec=48.0,
        b_cue_sec=5.0, overlap_sec=8.0, b_gain_db=0.0, trim_a_tail_sec=2.0,
        vocal_collision_score=0.10, beat_alignment_score=0.9, energy_score=0.9,
        placement_score=0.9, perceptual_loudness_score=0.9, loudness_score=0.9,
        compatibility_score=0.9, notes=["balanced candidate"],
    )
    same_timing = replace(base, score=0.85)
    distinct = replace(base, score=0.80, a_fade_start_sec=20.0, a_cut_sec=26.0, b_cue_sec=1.0, overlap_sec=6.0)

    selected = [base]
    assert candidate_too_close(same_timing, selected) is True
    assert candidate_too_close(distinct, selected) is False


def test_two_track_cli_short_track_fails_clearly_and_creates_no_false_success(tmp_path):
    import subprocess
    import sys

    a = tmp_path / "too_short_a.wav"
    b = tmp_path / "normal_b.wav"
    write_wav(a, sr=48000, channels=2)
    # overwrite with a deliberately very short Track A (< candidate overlap)
    t = np.linspace(0, 1.5, int(48000 * 1.5), endpoint=False)
    y = 0.01 * np.column_stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 440 * t)])
    sf.write(a, y, 48000, subtype="PCM_16")
    t2 = np.linspace(0, 8.0, int(48000 * 8.0), endpoint=False)
    y2 = 0.01 * np.column_stack([np.sin(2 * np.pi * 220 * t2), np.sin(2 * np.pi * 220 * t2)])
    sf.write(b, y2, 48000, subtype="PCM_16")

    out = tmp_path / "new_output_dir" / "out.wav"
    result = subprocess.run(
        [sys.executable, "weaveshift_two_tracks.py", str(a), str(b), "-o", str(out), "--vocal-method", "heuristic"],
        cwd=str(Path(__file__).resolve().parents[1]),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    assert "No valid transition candidates" in result.stderr
    assert "Traceback" not in result.stderr
    assert not out.parent.exists() or not any(out.parent.glob("*.wav"))


def test_heuristic_vocal_length_guard_uses_sample_count_not_channel_count(monkeypatch):
    from weaveshift_two_tracks import vocal_segments_heuristic, librosa

    called = {"hpss": False}

    def fake_hpss(_):
        called["hpss"] = True
        return _, _

    monkeypatch.setattr(librosa.effects, "hpss", fake_hpss)
    # 0.30 seconds of stereo audio has y.size > sr//2 but only 0.30s of samples.
    # The guard should treat this as shorter than 0.5s and return before HPSS.
    y = np.zeros((2, int(48000 * 0.30)), dtype=np.float32)
    assert vocal_segments_heuristic(y, 48000, 0.0) == []
    assert called["hpss"] is False
