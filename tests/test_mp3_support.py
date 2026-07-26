from pathlib import Path

import pytest
import soundfile as sf
from pydub import AudioSegment

from weaveshift_audio import (
    analyze_audio,
    audio_info,
    load_audio_segment,
    read_audio_window,
    resolve_wav_export_subtype,
    validate_audio_input,
    validate_audio_output,
    validate_audio_pair_compatibility,
    validate_loaded_segment_parity,
)
from weaveshift_plan import TrackSpec, validate_setlist_formats
from weaveshift_rendering import export_audio
from tests.audio_helpers import write_stereo_mp3, write_stereo_wav


def test_validate_audio_input_accepts_mp3(tmp_path):
    mp3 = write_stereo_mp3(tmp_path / "track.mp3", duration=3.0)
    assert validate_audio_input(str(mp3), "Track A") == mp3
    info = audio_info(mp3)
    assert info["subtype"] == "MP3"
    assert info["lossless"] is False
    assert info["samplerate"] == 48000
    assert info["channels"] == 2


def test_validate_audio_output_accepts_mp3_suffix(tmp_path):
    assert validate_audio_output(str(tmp_path / "mix.mp3")) == tmp_path / "mix.mp3"
    assert validate_audio_output(str(tmp_path / "mix")) == tmp_path / "mix.wav"


def test_mixed_wav_mp3_pair_requires_matching_sr(tmp_path):
    wav = write_stereo_wav(tmp_path / "a.wav", sr=48000)
    mp3 = write_stereo_mp3(tmp_path / "b.mp3", sr=44100)
    validate_audio_input(str(wav), "Track A")
    validate_audio_input(str(mp3), "Track B")
    with pytest.raises(ValueError, match="audio formats do not match"):
        validate_audio_pair_compatibility(wav, mp3)


def test_wav_mp3_pair_ok_when_sr_and_channels_match(tmp_path):
    wav = write_stereo_wav(tmp_path / "a.wav", sr=48000)
    mp3 = write_stereo_mp3(tmp_path / "b.mp3", sr=48000)
    a_info, b_info = validate_audio_pair_compatibility(wav, mp3)
    assert a_info["samplerate"] == b_info["samplerate"] == 48000


def test_mixed_pcm24_wav_and_mp3_decode_with_matching_sr_channels(tmp_path):
    wav = write_stereo_wav(tmp_path / "a.wav", sr=48000, subtype="PCM_24", duration=2.0)
    mp3 = write_stereo_mp3(tmp_path / "b.mp3", sr=48000, duration=2.0)
    seg_a = load_audio_segment(wav)
    seg_b = load_audio_segment(mp3)
    validate_loaded_segment_parity(seg_a, seg_b)
    assert seg_a.sample_width != seg_b.sample_width


def test_mixed_wav_mp3_setlist_formats(tmp_path):
    wav = write_stereo_wav(tmp_path / "a.wav", sr=48000, subtype="PCM_24", duration=2.0)
    mp3 = write_stereo_mp3(tmp_path / "b.mp3", sr=48000, duration=2.0)
    tracks = [TrackSpec(path=str(wav), title="A"), TrackSpec(path=str(mp3), title="B")]
    infos, segments = validate_setlist_formats(tracks)
    assert len(segments) == 2
    assert infos[0]["lossless"] is True
    assert infos[1]["lossless"] is False
    assert resolve_wav_export_subtype(infos) == "PCM_24"


def test_resolve_wav_export_subtype_falls_back_when_only_mp3():
    assert resolve_wav_export_subtype([{"lossless": False, "subtype": "MP3"}]) == "PCM_16"


def test_all_wav_setlist_rejects_subtype_mismatch(tmp_path):
    a = write_stereo_wav(tmp_path / "a.wav", sr=48000, subtype="PCM_16", duration=2.0)
    b = write_stereo_wav(tmp_path / "b.wav", sr=48000, subtype="PCM_24", duration=2.0)
    tracks = [TrackSpec(path=str(a), title="A"), TrackSpec(path=str(b), title="B")]
    with pytest.raises(ValueError, match="audio formats must match"):
        validate_setlist_formats(tracks)


def test_validate_audio_input_rejects_fake_mp3(tmp_path):
    bad = tmp_path / "fake.mp3"
    bad.write_bytes(b"not-an-mp3")
    with pytest.raises(ValueError, match="could not be read as a valid audio file"):
        validate_audio_input(str(bad), "Track A")


def test_two_track_cli_writes_mp3_output(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    a = write_stereo_wav(tmp_path / "a.wav", duration=8.0, freq=220.0, amp=0.08)
    b = write_stereo_wav(tmp_path / "b.wav", duration=8.0, freq=330.0, amp=0.08)
    out = tmp_path / "mix.mp3"
    result = subprocess.run(
        [
            sys.executable,
            "weaveshift_two_tracks.py",
            str(a),
            str(b),
            "-o",
            str(out),
            "--mode",
            "quick_cut",
            "--vocal-method",
            "heuristic",
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    rendered = tmp_path / "mix_quick_cut.mp3"
    assert rendered.exists()
    assert rendered.stat().st_size > 0


def test_analyze_audio_reads_mp3_window(tmp_path):
    mp3 = write_stereo_mp3(tmp_path / "track.mp3", duration=6.0)
    analysis = analyze_audio(
        str(mp3),
        role="b",
        window_sec=5.0,
        manual_bpm=128.0,
        manual_key="E major",
        vocal_method="heuristic",
        prefer_mps=False,
    )
    assert analysis.duration_sec == pytest.approx(6.0, abs=0.2)
    assert analysis.energy_curve is not None


def test_export_audio_mp3_roundtrip(tmp_path):
    wav = write_stereo_wav(tmp_path / "src.wav", duration=2.0)
    seg = AudioSegment.from_file(str(wav), format="wav")
    out = tmp_path / "mix.mp3"
    export_audio(seg, out, "PCM_16")
    assert out.exists()
    assert out.stat().st_size > 0
    y, sr = read_audio_window(str(out), 0.0, 1.0)
    assert y.shape[0] > 0
    assert sr > 0


def test_export_audio_wav_still_preserves_subtype(tmp_path):
    from weaveshift_rendering import export_wav_matching_subtype

    src = write_stereo_wav(tmp_path / "src.wav", subtype="PCM_24")
    seg = AudioSegment.from_file(str(src), format="wav")
    out = tmp_path / "out.wav"
    export_wav_matching_subtype(seg, out, "PCM_24")
    assert sf.info(str(out)).subtype == "PCM_24"
