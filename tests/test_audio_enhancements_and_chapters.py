"""Tests for tempo folding, key/vocal hardening, cosine ducking, atomic I/O, and chapter export."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from pydub import AudioSegment

import weaveshift.audio as audio_mod
import weaveshift.plan as plan_mod
import weaveshift.setlist as setlist_mod
from weaveshift._paths import atomic_copy, atomic_write_text
from weaveshift.audio import (
    AudioAnalysis,
    TransitionCandidate,
    _demucs_cli_vocals_path,
    _vocal_segments_from_mono,
    estimate_key,
    fold_tempo,
    fold_tempo_to_hint,
    vocal_segments_heuristic,
)
from weaveshift.chapters import (
    MixChapter,
    build_tracklist_document,
    chapters_from_setlist_report,
    export_chapters,
    format_cue_sheet,
    format_ffmetadata_chapters,
    format_youtube_chapters,
)
from weaveshift.rendering import apply_soft_duck, export_audio
from tests.cli_helpers import setlist_args


# -----------------------------
# Tempo octave folding
# -----------------------------

def test_fold_tempo_halves_double_time_and_thins_beats():
    tempo, beats = fold_tempo(192.0, [0.0, 0.3125, 0.625, 0.9375, 1.25])
    assert tempo == pytest.approx(96.0)
    assert beats == [0.0, 0.625, 1.25]


def test_fold_tempo_doubles_half_time_and_interpolates_midpoints():
    tempo, beats = fold_tempo(60.0, [0.0, 1.0, 2.0])
    assert tempo == pytest.approx(120.0)
    assert beats == [0.0, 0.5, 1.0, 1.5, 2.0]


def test_fold_tempo_preserves_in_range_and_handles_degenerate_inputs():
    assert fold_tempo(128.0, [0.5, 1.0]) == (128.0, [0.5, 1.0])
    assert fold_tempo(64.0) == (128.0, [])
    assert fold_tempo(0.0, [0.5, 1.0]) == (0.0, [0.5, 1.0])
    assert fold_tempo(-120.0, [0.5]) == (-120.0, [0.5])
    tempo_nan, beats_nan = fold_tempo(float("nan"), [0.5])
    assert math.isnan(tempo_nan) and beats_nan == [0.5]
    # Extreme near-zero tempo must stay bounded (at most 4 doublings -> <= 16x beats).
    t_small, b_small = fold_tempo(1e-6, [0.0, 1.0])
    assert t_small == pytest.approx(1e-6 * 16)
    assert len(b_small) == 17
    # Non-finite beats are stripped rather than multiplied during octave doubling.
    t_clean, b_clean = fold_tempo(64.0, [0.0, float("nan"), float("inf"), 1.0])
    assert t_clean == pytest.approx(128.0)
    assert b_clean == [0.0, 0.5, 1.0]


def test_fold_tempo_to_hint_matches_nearest_octave():
    t_up, b_up = fold_tempo_to_hint(64.0, 128.0, [0.0, 1.0])
    assert t_up == pytest.approx(128.0)
    assert b_up == [0.0, 0.5, 1.0]

    t_down, b_down = fold_tempo_to_hint(256.0, 128.0, [0.0, 0.25, 0.5, 0.75, 1.0])
    assert t_down == pytest.approx(128.0)
    assert b_down == [0.0, 0.5, 1.0]

    # Invalid hint falls back to default musical range.
    t_fallback, _ = fold_tempo_to_hint(64.0, -10.0)
    assert t_fallback == pytest.approx(128.0)


# -----------------------------
# Key estimation & vocal hardening
# -----------------------------

def test_estimate_key_returns_none_for_unpitched_noise_and_nan_audio():
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 0.2, size=22050 * 3).astype(np.float32)
    assert estimate_key(noise, 22050) == (None, None)

    bad = np.full(22050 * 2, np.nan, dtype=np.float32)
    assert estimate_key(bad, 22050) == (None, None)


def test_estimate_key_detects_tonal_chord_despite_leading_and_trailing_silence():
    sr = 22050
    t = np.linspace(0.0, 2.0, sr * 2, endpoint=False)
    # C major triad (C4, E4, G4) surrounded by 2s of silence on each side.
    chord = (
        0.3 * np.sin(2 * np.pi * 261.63 * t)
        + 0.25 * np.sin(2 * np.pi * 329.63 * t)
        + 0.25 * np.sin(2 * np.pi * 392.00 * t)
    ).astype(np.float32)
    silence = np.zeros(sr * 2, dtype=np.float32)
    signal = np.concatenate([silence, chord, silence])

    key, camelot = estimate_key(signal, sr)
    assert key == "C major"
    assert camelot == "8B"


def test_vocal_segments_rejects_near_silent_bleed_and_handles_nan():
    sr = 22050
    bleed = np.full(sr * 2, 1e-4, dtype=np.float32)
    assert _vocal_segments_from_mono(bleed, sr, start_sec=0.0, threshold_db=-80.0) == []

    nan_audio = np.full(sr * 2, np.nan, dtype=np.float32)
    assert _vocal_segments_from_mono(nan_audio, sr, start_sec=0.0, threshold_db=-38.0) == []
    assert vocal_segments_heuristic(nan_audio, sr, offset_sec=0.0) == []


# -----------------------------
# Atomic Demucs CLI & file helpers
# -----------------------------

def test_demucs_cli_runs_in_temp_dir_honors_no_mps_and_caches_atomically(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "track.wav"
    sf.write(src, np.zeros((22050, 2), dtype=np.float32), 22050, subtype="PCM_16")

    seen_cmds: list[list[str]] = []

    def fake_run(cmd, check=True):
        seen_cmds.append(list(cmd))
        out_idx = cmd.index("-o") + 1
        out_dir = Path(cmd[out_idx]) / "htdemucs" / "track"
        out_dir.mkdir(parents=True, exist_ok=True)
        sf.write(out_dir / "vocals.wav", np.zeros((1024, 2), dtype=np.float32), 22050, subtype="PCM_16")

    monkeypatch.setattr(audio_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(audio_mod, "get_torch_device", lambda prefer_mps=True: "cpu")

    cached_path = _demucs_cli_vocals_path(str(src), "htdemucs", prefer_mps=False)
    assert cached_path.is_file()
    assert ".weaveshift_demucs_cache" in cached_path.parts
    assert len(seen_cmds) == 1
    assert "-d" in seen_cmds[0] and seen_cmds[0][seen_cmds[0].index("-d") + 1] == "cpu"
    # Demucs -o target must be a temporary staging directory, not the live cache dir.
    out_arg = Path(seen_cmds[0][seen_cmds[0].index("-o") + 1])
    assert ".weaveshift_demucs_cache" not in out_arg.parts

    # Second call uses cache without re-running Demucs.
    cached_again = _demucs_cli_vocals_path(str(src), "htdemucs", prefer_mps=False)
    assert cached_again == cached_path
    assert len(seen_cmds) == 1


def test_atomic_helpers_and_export_preserve_existing_file_on_failure(tmp_path, monkeypatch):
    target = tmp_path / "existing.txt"
    target.write_text("original", encoding="utf-8")

    # Successful atomic write & copy preserve readable mode (not restrictive 0o600 mkstemp default)
    import stat

    atomic_write_text(target, "updated")
    assert target.read_text(encoding="utf-8") == "updated"

    fresh = tmp_path / "fresh.txt"
    atomic_write_text(fresh, "hello")
    assert stat.S_IMODE(fresh.stat().st_mode) & 0o444 == 0o444

    copied = tmp_path / "copied.txt"
    atomic_copy(target, copied)
    assert copied.read_text(encoding="utf-8") == "updated"
    assert stat.S_IMODE(copied.stat().st_mode) & 0o444 == 0o444

    # Failed audio export must preserve existing destination and clean up temp files.
    wav_dest = tmp_path / "mix.wav"
    wav_dest.write_bytes(b"KEEP_ME")
    seg = AudioSegment.silent(duration=500, frame_rate=48000)

    def boom(*_args, **_kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(seg, "export", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        export_audio(seg, wav_dest, "PCM_16")

    assert wav_dest.read_bytes() == b"KEEP_ME"
    assert not list(tmp_path.glob(".weaveshift_export_*"))


# -----------------------------
# Raised-cosine soft duck
# -----------------------------

def test_apply_soft_duck_uses_smooth_raised_cosine_ramps():
    sr = 48000
    samples = np.full(int(sr * 1.2), 20000, dtype=np.int16)
    seg = AudioSegment(samples.tobytes(), frame_rate=sr, sample_width=2, channels=1)

    ducked = apply_soft_duck(seg, -6.0, attack_ms=200, release_ms=200)
    arr = np.array(ducked.get_array_of_samples(), dtype=np.float64)

    # Raised cosine starts flat (slope ~0), so 10% into attack it has moved < 3% of the total drop
    # (a linear ramp would have moved 10%).
    attack_n = int(sr * 0.2)
    drop_total = 20000.0 - float(arr[attack_n + 100])
    drop_at_10pct = 20000.0 - float(arr[attack_n // 10])
    assert drop_total > 8000.0
    assert drop_at_10pct / drop_total < 0.04
    assert arr[-1] > 19900.0


# -----------------------------
# Chapter & tracklist exporters
# -----------------------------

def test_chapters_from_setlist_report_tracks_crossfade_handoff_and_natural():
    report = {
        "output_duration_sec": 300.0,
        "tracks": [
            {"path": "a.wav", "title": "Opening", "bpm": 124.0, "key": "A minor"},
            {"path": "b.wav", "title": "Peak Hour", "bpm": 128.0, "key": "C major"},
            {"path": "c.wav", "title": "Closing", "bpm": 126.0, "key": "G major"},
        ],
        "track_source_zero_in_mix_sec": [0.0, 95.0, 200.0],
        "transitions": [
            {
                "index": 1,
                "mix_transition_start_sec": 100.0,
                "next_track_source_zero_in_mix_sec": 95.0,
                "source_b_cue_sec": 5.0,
            },
            {
                "index": 2,
                "mix_transition_start_sec": 199.0,
                "next_track_source_zero_in_mix_sec": 200.0,
                "source_b_cue_sec": 0.0,
            },
        ],
    }

    chapters = chapters_from_setlist_report(report)
    assert [(c.start_sec, c.end_sec, c.title) for c in chapters] == [
        (0.0, 100.0, "Opening"),
        (100.0, 200.0, "Peak Hour"),
        (200.0, 300.0, "Closing"),
    ]
    assert chapters[0].bpm == 124.0
    assert chapters[1].key == "C major"


def test_youtube_chapters_validation_hours_and_sanitization():
    short_chapters = [
        MixChapter(0, 0.0, 5.0, "Line1\nInjected"),
        MixChapter(1, 5.0, 20.0, "Second"),
    ]
    text, warnings = format_youtube_chapters(short_chapters, total_duration_sec=20.0)
    assert text == "00:00 Line1 Injected\n00:05 Second\n"
    assert any("at least 3 chapters" in w for w in warnings)
    assert any(">= 10s" in w for w in warnings)

    long_chapters = [
        MixChapter(0, 0.0, 1800.0, "Part 1", artist="DJ"),
        MixChapter(1, 1800.0, 3605.0, "Part 2"),
        MixChapter(2, 3605.0, 4200.0, "Part 3"),
    ]
    long_text, long_warnings = format_youtube_chapters(long_chapters, total_duration_sec=4200.0)
    assert long_warnings == []
    assert long_text.splitlines() == [
        "0:00:00 DJ - Part 1",
        "0:30:00 Part 2",
        "1:00:05 Part 3",
    ]


def test_ffmetadata_cue_sheet_and_tracklist_document_formatting():
    chapters = [
        MixChapter(0, 0.0, 59.999, 'Track=1; "Quote" #1\nNext', artist="A\\B", bpm=128.0, key="E major"),
        MixChapter(1, 59.999, 120.0, "Track 2", bpm=130.0),
    ]

    ffmeta = format_ffmetadata_chapters(chapters, title="Set #1=Live")
    assert ffmeta.startswith(";FFMETADATA1\ntitle=Set \\#1\\=Live\n")
    assert "START=0\nEND=59999\n" in ffmeta
    assert "START=59999\nEND=120000\n" in ffmeta
    assert r"title=A\\B - Track\=1\; \"Quote\" \#1 Next" not in ffmeta  # quotes unescaped in ffmeta
    assert r'title=A\\B - Track\=1\; "Quote" \#1 Next' in ffmeta

    cue = format_cue_sheet(chapters, "my_mix.mp3", title='Mix "Live"')
    assert 'TITLE "Mix \'Live\'"' in cue
    assert 'FILE "my_mix.mp3" MP3' in cue
    assert "INDEX 01 00:00:00" in cue
    # 59.999s rounds to 4500 frames -> 01:00:00 (no invalid 00:59:75 overflow)
    assert "INDEX 01 01:00:00" in cue
    assert "REM BPM 128.00" in cue
    assert 'REM KEY "E major"' in cue

    doc = build_tracklist_document(chapters, title="My Set")
    assert doc["source"] == "weaveshift"
    assert len(doc["tracks"]) == 2
    assert doc["tracks"][0]["start"] == 0.0
    assert doc["tracks"][0]["end"] == 59.999
    assert doc["tracks"][0]["bpm"] == 128.0
    assert doc["tracks"][0]["key"] == "E major"


def test_export_chapters_preserves_dotted_stems(tmp_path):
    chapters = [
        MixChapter(0, 0.0, 15.0, "A"),
        MixChapter(1, 15.0, 30.0, "B"),
        MixChapter(2, 30.0, 45.0, "C"),
    ]
    base = tmp_path / "mix v1.2"
    written, warnings = export_chapters(chapters, base, "mix v1.2.wav", 45.0, title="mix v1.2")
    assert warnings == []
    assert written["youtube"] == tmp_path / "mix v1.2.chapters.txt"
    assert written["ffmetadata"] == tmp_path / "mix v1.2.ffmetadata"
    assert written["cue"] == tmp_path / "mix v1.2.cue"
    assert written["tracklist"] == tmp_path / "mix v1.2.tracklist.json"
    for p in written.values():
        assert p.is_file()


def test_setlist_build_exports_chapter_sidecars(tmp_path, monkeypatch):
    sr = 48000
    t = np.linspace(0.0, 4.0, sr * 4, endpoint=False)
    for name, freq in (("a.wav", 440.0), ("b.wav", 660.0)):
        y = 0.03 * np.column_stack([np.sin(2 * np.pi * freq * t), np.sin(2 * np.pi * freq * t)])
        sf.write(tmp_path / name, y, sr, subtype="PCM_16")

    manifest = tmp_path / "setlist.json"
    manifest.write_text(
        json.dumps(
            {
                "tracks": [
                    {"path": "a.wav", "title": "Track One", "bpm": 124, "key": "A minor"},
                    {"path": "b.wav", "title": "Track Two", "bpm": 126, "key": "C major"},
                ],
                "settings": {"export_chapters": True},
            }
        ),
        encoding="utf-8",
    )

    fake = AudioAnalysis(
        path="x.wav",
        duration_sec=4.0,
        bpm=124.0,
        key="A minor",
        camelot="8A",
        rms_dbfs=-20.0,
        peak_dbfs=-6.0,
        beats_sec=[0.0, 1.0, 2.0, 3.0, 4.0],
        onsets_sec=[],
        vocal_segments=[],
        vocal_method="heuristic",
        analysis_window_start_sec=0.0,
        analysis_window_duration_sec=4.0,
    )
    cand = TransitionCandidate(
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
    monkeypatch.setattr(plan_mod, "analyze_audio", lambda path, role, **_: fake)
    monkeypatch.setattr(plan_mod, "choose_candidates", lambda *_a, **_k: [cand])

    out = tmp_path / "demo.wav"
    args = setlist_args(manifest, out, apply_manifest_settings=True)
    setlist_mod.build_continuous_mix(args)

    report = json.loads((tmp_path / "demo.weaveshift_setlist_report.json").read_text(encoding="utf-8"))
    assert set(report["chapter_files"].keys()) == {"youtube", "ffmetadata", "cue", "tracklist"}
    for path_str in report["chapter_files"].values():
        assert Path(path_str).is_file()

    yt = Path(report["chapter_files"]["youtube"]).read_text(encoding="utf-8")
    assert "00:00 Track One" in yt
    assert "00:02 Track Two" in yt

    tl = json.loads(Path(report["chapter_files"]["tracklist"]).read_text(encoding="utf-8"))
    assert tl["tracks"][0]["title"] == "Track One"
    assert tl["tracks"][1]["start"] == pytest.approx(2.0, abs=0.05)


def test_chapters_handle_non_finite_timestamps_over_hour_and_blank_metadata():
    bad_chapters = [
        MixChapter(0, float("nan"), float("inf"), "\n\t"),
        MixChapter(1, float("inf"), 10.0, "Second\nLine"),
    ]
    yt_txt, _ = format_youtube_chapters(bad_chapters, float("inf"))
    assert len(yt_txt.splitlines()) == 2
    ffmeta = format_ffmetadata_chapters(bad_chapters)
    assert ";FFMETADATA1" in ffmeta
    cue = format_cue_sheet(bad_chapters, "\n\t")
    assert 'FILE "mix.wav" WAVE' in cue
    assert 'TITLE "Track 01"' in cue
    doc = build_tracklist_document(bad_chapters)
    # Must be strict JSON-serializable without NaN/Inf and with cleaned single-line titles.
    dumped = json.dumps(doc, allow_nan=False)
    assert '"Track 01"' in dumped
    assert '"Second Line"' in dumped

    # Chapter >= 1h triggers H:MM:SS formatting even if total_duration_sec is underestimated.
    over_hour = [
        MixChapter(0, 0.0, 3665.0, "Hour One"),
        MixChapter(1, 3665.0, 3700.0, "Hour Two"),
    ]
    yt_hr, _ = format_youtube_chapters(over_hour, total_duration_sec=100.0)
    assert yt_hr.splitlines() == ["0:00:00 Hour One", "1:01:05 Hour Two"]
