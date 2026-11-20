"""Regression tests from white-box, black-box, stress, and adversarial QA passes."""
from __future__ import annotations

import csv
import json
import math
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from pydub import AudioSegment

import weaveshift.audio as audio_mod
from weaveshift._cli import write_report
from weaveshift._paths import resolve_contained
from weaveshift.audio import (
    AudioAnalysis,
    TransitionCandidate,
    estimate_key,
    format_timestamp,
    validate_audio_input,
    validate_audio_output,
)
from weaveshift.cues import (
    CueResult,
    _read_json_timeline,
    analyze_track,
    moving_average,
    resolve_audio_path,
    sec_to_timestamp,
    timestamp_to_sec,
    write_results,
)
from weaveshift.plan import (
    TrackSpec,
    apply_handoff_transition_override,
    apply_manual_transition_override,
    plan_setlist_mix,
    plan_two_track_mix,
)
from weaveshift.profiles import ScoringProfile, load_scoring_profile
from weaveshift.rendering import apply_soft_duck, build_transition_audio
from weaveshift.scoring import profile_candidate_score
from weaveshift.setlist import load_setlist


def _make_analysis(role: str, duration: float = 30.0) -> AudioAnalysis:
    times = np.array([0.0, 10.0, 20.0, duration], dtype=float)
    curve = {"times": times, "dbfs": np.full_like(times, -14.0)}
    return AudioAnalysis(
        path=f"{role}.wav",
        duration_sec=duration,
        bpm=128.0,
        key="E major",
        camelot="12B",
        rms_dbfs=-18.0,
        peak_dbfs=-3.0,
        beats_sec=[],
        onsets_sec=[],
        vocal_segments=[],
        vocal_method="none",
        analysis_window_start_sec=0.0,
        analysis_window_duration_sec=min(35.0, duration),
        energy_curve=curve,
        loudness_curve=curve,
    )


def _make_candidate(**overrides) -> TransitionCandidate:
    base = dict(
        name="vocal_safe",
        score=0.9,
        a_fade_start_sec=20.0,
        a_cut_sec=24.0,
        b_cue_sec=0.0,
        overlap_sec=4.0,
        b_gain_db=0.0,
        trim_a_tail_sec=6.0,
        vocal_collision_score=0.0,
        beat_alignment_score=0.9,
        energy_score=0.9,
        placement_score=0.9,
        loudness_score=0.9,
        perceptual_loudness_score=0.9,
        compatibility_score=0.9,
        notes=[],
    )
    base.update(overrides)
    return TransitionCandidate(**base)


# -----------------------------
# Input & output path validation
# -----------------------------

@pytest.mark.parametrize("raw", ["", "   ", ".", "..", "/"])
def test_validate_audio_output_rejects_empty_or_dot_paths(raw):
    with pytest.raises(ValueError):
        validate_audio_output(raw)


def test_validate_audio_output_rejects_existing_directories(tmp_path):
    with pytest.raises(ValueError, match="is a directory"):
        validate_audio_output(str(tmp_path))
    dir_with_wav_ext = tmp_path / "mix.wav"
    dir_with_wav_ext.mkdir()
    with pytest.raises(ValueError, match="is a directory"):
        validate_audio_output(str(dir_with_wav_ext))
    with pytest.raises(ValueError, match="is a directory"):
        validate_audio_output(str(tmp_path / "mix"))


def test_validate_audio_input_rejects_directory_and_zero_frame_wav(tmp_path):
    dir_wav = tmp_path / "folder.wav"
    dir_wav.mkdir()
    with pytest.raises(ValueError, match="is not a file"):
        validate_audio_input(str(dir_wav), "Track A")

    zero_wav = tmp_path / "zero.wav"
    sf.write(str(zero_wav), np.zeros((0, 2), dtype=np.float32), 48000, subtype="PCM_16")
    with pytest.raises(ValueError, match="zero duration"):
        validate_audio_input(str(zero_wav), "Track A")


def test_resolve_contained_rejects_empty_path(tmp_path):
    with pytest.raises(ValueError, match="must not be empty"):
        resolve_contained("", tmp_path)
    with pytest.raises(ValueError, match="must not be empty"):
        resolve_contained("   ", tmp_path)


# -----------------------------
# Non-finite / numeric safety
# -----------------------------

def test_timestamp_helpers_handle_non_finite_and_negative_inputs():
    assert format_timestamp(float("nan")) == "unknown"
    assert format_timestamp(float("inf")) == "unknown"
    assert sec_to_timestamp(float("inf")) == ""
    assert sec_to_timestamp(float("-inf")) == ""
    assert timestamp_to_sec("nan:00") is None
    assert timestamp_to_sec("inf:10") is None
    assert timestamp_to_sec("-1:10") is None


def test_write_report_rejects_nan_values(tmp_path):
    with pytest.raises(ValueError):
        write_report(tmp_path / "mix", ".report.json", {"bpm": float("nan")})


def test_estimate_key_silent_audio_no_runtime_warning():
    silent = np.zeros(22050 * 2, dtype=np.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        key, camelot = estimate_key(silent, 22050)
    assert (key, camelot) == (None, None)
    assert not any(issubclass(w.category, RuntimeWarning) for w in caught)


# -----------------------------
# Profile TOML validation
# -----------------------------

def test_load_scoring_profile_rejects_directory_and_non_table_sections(tmp_path):
    with pytest.raises(ValueError, match="not a file"):
        load_scoring_profile(config_path=str(tmp_path))

    bad_toml = tmp_path / "bad.toml"
    bad_toml.write_text('name = "bad"\nweights = "not-a-table"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="must be a TOML table"):
        load_scoring_profile(config_path=str(bad_toml))


def test_scoring_profile_ignores_non_finite_bool_and_non_numeric_values():
    prof = ScoringProfile(
        name="weird",
        weights={"energy": float("nan"), "loudness": True},  # type: ignore[dict-item]
        preferences={"target_overlap_sec": "nope", "preferred_overlap_min_sec": float("inf")},
        limits={"max_abs_b_gain_db": float("-inf")},
    )
    assert prof.weight("energy", 1.5) == 1.5
    assert prof.weight("loudness", 1.2) == 1.2
    assert prof.pref_number("target_overlap_sec", 4.0) == 4.0
    assert prof.pref_number("preferred_overlap_min_sec", 2.0) == 2.0
    assert prof.limit("max_abs_b_gain_db", 4.0) == 4.0
    score = profile_candidate_score(_make_candidate(), prof)
    assert math.isfinite(score)


# -----------------------------
# Planning & override bounds
# -----------------------------

@pytest.mark.parametrize(
    "kwargs",
    [
        {"search_a_sec": -1.0},
        {"search_b_sec": float("nan")},
        {"max_trim_a_sec": -2.0},
        {"b_cue_max_sec": float("inf")},
        {"a_bpm": -120.0},
        {"b_bpm": float("nan")},
    ],
)
def test_plan_two_track_mix_rejects_invalid_planning_params(tmp_path, kwargs):
    with pytest.raises(ValueError):
        plan_two_track_mix(str(tmp_path / "a.wav"), str(tmp_path / "b.wav"), **kwargs)


def test_plan_setlist_mix_requires_two_tracks():
    with pytest.raises(ValueError, match="at least two tracks"):
        plan_setlist_mix([TrackSpec(path="a.wav", title="A")])


@pytest.mark.parametrize("bpm_json", ["NaN", "-120", "0", "Infinity"])
def test_load_setlist_rejects_non_positive_or_non_finite_bpm(tmp_path, bpm_json):
    manifest = tmp_path / "setlist.json"
    manifest.write_text(
        f'{{"tracks": [{{"path": "a.wav", "bpm": {bpm_json}}}, {{"path": "b.wav"}}]}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="positive finite number"):
        load_setlist(str(manifest))


def test_override_bounds_for_gain_duck_and_track_b_duration():
    a, b = _make_analysis("a", 30.0), _make_analysis("b", 10.0)
    base = _make_candidate()
    with pytest.raises(ValueError, match="between -24 and 24"):
        apply_manual_transition_override(base, {"b_gain_db": 50.0}, 1, a, b)
    with pytest.raises(ValueError, match=">= -60 dB"):
        apply_manual_transition_override(base, {"soft_duck_db": -90.0}, 1, a, b)
    with pytest.raises(ValueError, match="exceeds Track B duration"):
        apply_manual_transition_override(base, {"b_cue_sec": 8.0, "overlap_sec": 4.0}, 1, a, b)
    with pytest.raises(ValueError, match="exceeds Track B duration"):
        apply_handoff_transition_override(
            base,
            {"b_cue_sec": 8.0, "takeover_overlap_sec": 0.5, "b_fade_in_sec": 3.0},
            1,
            a,
            b,
        )
    # Within the 0.05s tolerance at the end of Track A still scores validly (not -999)
    ok = apply_manual_transition_override(
        base,
        {"a_fade_start_sec": 26.0, "a_cut_sec": 30.02, "b_cue_sec": 0.0, "overlap_sec": 4.0},
        1,
        a,
        b,
    )
    assert ok.score > -100


# -----------------------------
# DSP & rendering edge cases
# -----------------------------

def test_build_transition_audio_clamps_fade_out_and_avoids_track_b_gap():
    seg_a = AudioSegment.silent(duration=10000, frame_rate=48000)
    seg_b = AudioSegment.silent(duration=10000, frame_rate=48000)

    # Case 1: a_slice (1s) < overlap (4s) must NOT trigger pydub fade_out length corruption
    short_slice_cand = _make_candidate(a_fade_start_sec=8.0, a_cut_sec=9.0, b_cue_sec=0.0, overlap_sec=4.0)
    rendered_short = build_transition_audio(seg_a, seg_b, short_slice_cand)
    assert len(rendered_short.full) == 18000

    # Case 2: a_slice (6s) > overlap (2s) must keep Track B continuous without a 4s gap
    long_slice_cand = _make_candidate(a_fade_start_sec=4.0, a_cut_sec=10.0, b_cue_sec=0.0, overlap_sec=2.0)
    rendered_long = build_transition_audio(seg_a, seg_b, long_slice_cand)
    assert len(rendered_long.full) == 14000


def test_apply_soft_duck_short_segment_releases_cleanly():
    sr = 48000
    # 100ms constant-amplitude signal (shorter than default 250ms attack)
    samples = np.full(int(sr * 0.1), 16000, dtype=np.int16)
    seg = AudioSegment(samples.tobytes(), frame_rate=sr, sample_width=2, channels=1)
    ducked = apply_soft_duck(seg, -6.0)
    arr = np.array(ducked.get_array_of_samples())
    assert len(arr) == len(samples)
    # Must start near 1.0, dip in the middle, and release back near 1.0 at the end
    assert arr[0] > 15500
    assert arr[len(arr) // 2] < 12000
    assert arr[-1] > 15500


# -----------------------------
# Cues edge cases & CSV injection
# -----------------------------

def test_moving_average_and_one_sample_wav_in_cues(tmp_path):
    x = np.array([1.0], dtype=float)
    assert len(moving_average(x, 10)) == 1

    one_sample = tmp_path / "one.wav"
    sf.write(str(one_sample), np.array([[0.1, 0.1]], dtype=np.float32), 22050, subtype="PCM_16")
    duration, _bpm, cues, _conf, _notes = analyze_track(one_sample, bpm_hint=-50.0)
    assert duration > 0
    assert isinstance(cues, dict)


def test_read_json_timeline_ignores_non_list_fields(tmp_path):
    tl = tmp_path / "tl.json"
    tl.write_text(json.dumps({"tracks": "bad", "track_source_zero_in_mix_sec": 42}), encoding="utf-8")
    assert _read_json_timeline(tl) == {}


def test_cues_csv_neutralizes_formula_injection(tmp_path):
    out = tmp_path / "cues.csv"
    write_results(
        out,
        [
            CueResult(
                title="=CMD|' /C calc'!A0",
                path="/tmp/a.wav",
                bpm_manifest=128.0,
                bpm_detected=128.0,
                key="+1+1",
                duration_sec=60.0,
                final_start_sec=0.0,
                transition_out_sec=50.0,
                first_lift_sec=10.0,
                main_peak_sec=30.0,
                second_peak_sec=None,
                breakdown_sec=None,
                visual_mood="@SUM(1+1)",
                confidence="high",
                notes="-2+3",
            )
        ],
    )
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert rows[0]["title"].startswith("'=")
    assert rows[0]["key"].startswith("'+")
    assert rows[0]["visual_mood"].startswith("'@")
    assert rows[0]["notes"].startswith("'-")


def test_cues_resolve_audio_path_blocks_traversal_and_directories(tmp_path):
    manifest_dir = tmp_path / "set"
    root = tmp_path / "root"
    manifest_dir.mkdir()
    root.mkdir()
    (root / "secret.wav").write_bytes(b"x")
    manifest = manifest_dir / "tracks.csv"

    with pytest.raises(ValueError, match="must not be empty"):
        resolve_audio_path("", manifest, root)

    # Traversal '../outside/secret.wav' must NOT silently fall back to root/secret.wav
    with pytest.raises(FileNotFoundError):
        resolve_audio_path("../outside/secret.wav", manifest, root)

    # '.' must not resolve to a directory
    with pytest.raises(FileNotFoundError):
        resolve_audio_path(".", manifest, root)


# -----------------------------
# FFprobe & Demucs cache security
# -----------------------------

def test_ffprobe_enforces_protocol_whitelist_and_timeout(tmp_path, monkeypatch):
    seen_cmd = []
    seen_timeout = []

    def fake_run(cmd, **kwargs):
        seen_cmd.append(cmd)
        seen_timeout.append(kwargs.get("timeout"))
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 30))

    monkeypatch.setattr(audio_mod.subprocess, "run", fake_run)
    sample = tmp_path / "concat:a.mp3"
    sample.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="timed out"):
        audio_mod._ffprobe_audio_metadata(sample)

    assert "-protocol_whitelist" in seen_cmd[0]
    assert seen_cmd[0][seen_cmd[0].index("-protocol_whitelist") + 1] == "file"
    assert Path(seen_cmd[0][-1]).is_absolute()
    assert seen_timeout[0] == 30


def test_demucs_cache_ignores_symlink_escape(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "song.wav"
    src.write_bytes(b"wav")
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"secret")

    cache_key = audio_mod._audio_cache_key(str(src))
    track_dir = tmp_path / ".weaveshift_demucs_cache" / cache_key / "htdemucs" / "song"
    track_dir.mkdir(parents=True)
    (track_dir / "vocals.wav").symlink_to(outside)

    called = []

    def fake_run(cmd, check=True):
        called.append(cmd)
        # Do not write a valid file; verify it did NOT return the symlinked outside file
        raise RuntimeError("cli invoked")

    monkeypatch.setattr(audio_mod.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="cli invoked"):
        audio_mod._demucs_cli_vocals_path(str(src), "htdemucs")
    assert len(called) == 1
