import csv

import numpy as np

from weaveshift_cues import (
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
