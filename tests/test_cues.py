import csv

import numpy as np

from weaveshift.cues import (
    CueResult,
    TimelineRow,
    add_final_time,
    find_breakdown_candidate,
    first_sustained_energy_lift,
    moving_average,
    normalize_title,
    parse_float,
    pick_local_peaks,
    read_timeline,
    read_tracks,
    resolve_audio_path,
    robust_z,
    sec_to_timestamp,
    timestamp_to_sec,
    write_results,
)


def test_timestamp_and_title_helpers():
    assert timestamp_to_sec("1:02") == 62
    assert timestamp_to_sec("1:02:03") == 3723
    assert timestamp_to_sec("bad") is None
    assert sec_to_timestamp(3723) == "1:02:03"
    assert parse_float("2:30") == 150
    assert parse_float("128.5") == 128.5
    assert parse_float("") is None
    assert normalize_title("Track One (Extended Mix)!") == "track one"
    assert add_final_time(100.0, 12.5) == 112.5
    assert add_final_time(None, 12.5) is None


def test_read_tracks_and_timeline_accept_column_aliases(tmp_path):
    manifest = tmp_path / "tracks.csv"
    manifest.write_text(
        "file,track,tempo,musical_key,mood\n"
        "audio/a.wav,Song A,128,E major,bright\n",
        encoding="utf-8",
    )
    timeline = tmp_path / "timeline.csv"
    timeline.write_text(
        "track_b_title,track_source_zero_in_mix_sec,mix_transition_start_sec,selected_overlap_sec\n"
        "Song A,1:00,2:30,8\n",
        encoding="utf-8",
    )

    tracks = read_tracks(manifest)
    rows = read_timeline(timeline)

    assert tracks[0].path == "audio/a.wav"
    assert tracks[0].title == "Song A"
    assert tracks[0].bpm == 128
    assert tracks[0].key == "E major"
    assert rows["song a"] == TimelineRow("Song A", start_sec=60, transition_out_sec=150, overlap_sec=8)


def test_resolve_audio_path_checks_audio_root_and_manifest_dir(tmp_path):
    manifest = tmp_path / "tracks.csv"
    manifest.write_text("path,title\nsong.wav,Song\n", encoding="utf-8")
    root = tmp_path / "audio"
    root.mkdir()
    audio = root / "song.wav"
    audio.write_bytes(b"placeholder")

    assert resolve_audio_path("nested/song.wav", manifest, root) == audio


def test_cue_signal_helpers_pick_energy_moments():
    times = np.arange(0, 120, 1.0)
    energy = np.zeros_like(times, dtype=float)
    energy[15:45] = 4.0
    energy[80] = -5.0
    score = np.zeros_like(times, dtype=float)
    score[30] = 10.0
    score[70] = 8.0

    assert moving_average(np.array([0, 3, 0]), 3)[1] == 1
    assert np.isclose(np.median(robust_z(np.array([1.0, 2.0, 3.0]))), 0.0)
    assert pick_local_peaks(times, score, 20, 90, min_distance_sec=20, count=2) == [30.0, 70.0]
    assert first_sustained_energy_lift(times, energy, duration_sec=120) == 6.0
    assert find_breakdown_candidate(times, energy, duration_sec=120) == 80.0


def test_write_results_outputs_source_and_final_timestamps(tmp_path):
    out = tmp_path / "cues.csv"
    write_results(
        out,
        [
            CueResult(
                title="Song",
                path="song.wav",
                bpm_manifest=128.0,
                bpm_detected=127.8,
                key="E major",
                duration_sec=185.0,
                final_start_sec=60.0,
                transition_out_sec=180.0,
                first_lift_sec=15.0,
                main_peak_sec=45.0,
                second_peak_sec=None,
                breakdown_sec=120.0,
                visual_mood="bright",
                confidence="high",
                notes="ok",
            )
        ],
    )

    with out.open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))

    assert row["duration_timestamp"] == "03:05"
    assert row["main_peak_final_sec"] == "105.000"
    assert row["main_peak_final_timestamp"] == "01:45"
    assert row["second_peak_final_timestamp"] == ""


def test_read_tracks_and_timeline_support_weaveshift_json_formats(tmp_path):
    import json

    manifest = tmp_path / "setlist.json"
    manifest.write_text(
        json.dumps(
            {
                "tracks": [
                    {"path": "fixtures/track_01.wav", "title": "Demo 01", "bpm": 110, "key": "F major", "visual_mood": "warm-up"},
                    {"path": "fixtures/track_02.wav", "title": "Demo 02", "bpm": 126, "key": "G major"},
                ]
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "demo.weaveshift_setlist_report.json"
    report.write_text(
        json.dumps(
            {
                "tracks": [
                    {"path": "fixtures/track_01.wav", "title": "Demo 01"},
                    {"path": "fixtures/track_02.wav", "title": "Demo 02"},
                ],
                "track_source_zero_in_mix_sec": [0.0, 52.5],
                "transitions": [
                    {
                        "index": 1,
                        "selected_candidate": "vocal_safe",
                        "mix_transition_start_sec": 54.0,
                        "overlap_sec": 6.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    tracks = read_tracks(manifest)
    timeline = read_timeline(report)

    assert len(tracks) == 2
    assert tracks[0].title == "Demo 01"
    assert tracks[0].bpm == 110.0
    assert tracks[0].visual_mood == "warm-up"
    assert timeline["demo 01"] == TimelineRow("Demo 01", start_sec=0.0, transition_out_sec=54.0, overlap_sec=6.0)
    assert timeline["demo 02"] == TimelineRow("Demo 02", start_sec=52.5, transition_out_sec=None, overlap_sec=None)


def _assert_bpm_octave(bpm: float | None, expected: float) -> None:
    assert bpm is not None
    assert min(abs(bpm - expected), abs(bpm * 2 - expected), abs(bpm / 2 - expected)) < 4.0


def test_analyze_track_detects_tempo(tmp_path):
    from tests.audio_helpers import write_click_track
    from weaveshift.cues import analyze_track

    audio = write_click_track(tmp_path / "clicks.wav", duration=30.0, bpm=128.0)
    _duration, bpm_detected, _cues, _confidence, _notes = analyze_track(audio, bpm_hint=128.0)
    assert bpm_detected is not None
    assert abs(bpm_detected - 128.0) < 4.0

    _duration, bpm_raw, _cues, _confidence, _notes = analyze_track(audio, bpm_hint=None)
    _assert_bpm_octave(bpm_raw, 128.0)


def test_analyze_track_detects_tempo_in_fresh_process_without_rhythm_preimport(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    from tests.audio_helpers import write_click_track

    audio = write_click_track(tmp_path / "clicks.wav", duration=30.0, bpm=128.0)
    script = (
        "import sys; from pathlib import Path; "
        "from weaveshift.cues import analyze_track; "
        "assert 'librosa.feature.rhythm' not in sys.modules; "
        "_, bpm, _, _, _ = analyze_track(Path(sys.argv[1]), bpm_hint=None); "
        "assert bpm is not None, 'expected detected BPM'; "
        "print(bpm)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script, str(audio)],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    _assert_bpm_octave(float(proc.stdout.strip()), 128.0)


def test_read_tracks_and_main_reject_malformed_manifests_with_value_error(tmp_path, capsys):
    import pytest
    from weaveshift.cues import main as cues_main

    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("title\nOnlyTitle\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing a path/file/audio_path"):
        read_tracks(bad_csv)

    empty_json = tmp_path / "empty.json"
    empty_json.write_text('{"tracks": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty 'tracks' list"):
        read_tracks(empty_json)

    bad_entry = tmp_path / "bad_entry.json"
    bad_entry.write_text('{"tracks": [123]}', encoding="utf-8")
    with pytest.raises(ValueError, match="string or an object"):
        read_tracks(bad_entry)

    rc = cues_main([str(empty_json), "--out", str(tmp_path / "out.csv")])
    assert rc == 1
    assert "Error: JSON manifest must contain a non-empty 'tracks' list." in capsys.readouterr().err
