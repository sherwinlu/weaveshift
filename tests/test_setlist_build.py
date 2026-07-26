import json
from types import SimpleNamespace

import numpy as np
import soundfile as sf

import weaveshift_plan
import weaveshift_setlist
from weaveshift_two_tracks import AudioAnalysis, TransitionCandidate


def write_wav(path, duration=4.0, sr=48000, freq=440.0):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.03 * np.column_stack([np.sin(2 * np.pi * freq * t), np.sin(2 * np.pi * freq * t)])
    sf.write(path, y, sr, subtype="PCM_16")


def fake_analysis(path, role):
    return AudioAnalysis(
        path=path,
        duration_sec=4.0,
        bpm=128.0,
        key="E major",
        camelot="12B",
        rms_dbfs=-20.0,
        peak_dbfs=-6.0,
        beats_sec=[0.0, 1.0, 2.0, 3.0, 4.0],
        onsets_sec=[],
        vocal_segments=[],
        vocal_method="heuristic",
        analysis_window_start_sec=0.0,
        analysis_window_duration_sec=4.0,
    )


def base_candidate(name="vocal_safe"):
    return TransitionCandidate(
        name=name,
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
        notes=["balanced candidate"],
    )


def args_for(setlist, output):
    return SimpleNamespace(
        setlist=str(setlist),
        output=str(output),
        transition_mode="recommended",
        make_snippets=True,
        profile="edm",
        scoring_config=None,
        vocal_method="heuristic",
        no_mps=True,
        search_a_sec=1.0,
        search_b_sec=1.0,
        max_trim_a_sec=2.0,
        b_cue_max_sec=2.0,
        natural_pause_sec=1.0,
        apply_manifest_settings=False,
    )


def test_build_continuous_mix_applies_manual_override_and_writes_report(tmp_path, monkeypatch):
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    write_wav(a, freq=440)
    write_wav(b, freq=660)
    manifest = tmp_path / "setlist.json"
    manifest.write_text(
        json.dumps(
            {
                "tracks": [
                    {"path": "a.wav", "title": "Song A"},
                    {"path": "b.wav", "title": "Song B"},
                ],
                "settings": {
                    "transition_overrides": [
                        {
                            "index": 1,
                            "mode": "manual",
                            "a_fade_start_sec": 1.5,
                            "a_cut_sec": 2.5,
                            "b_cue_sec": 0.25,
                            "overlap_sec": 1.0,
                            "b_gain_db": -1.5,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "mix.wav"

    monkeypatch.setattr(weaveshift_plan, "analyze_audio", lambda path, role, **_: fake_analysis(path, role))
    monkeypatch.setattr(weaveshift_plan, "choose_candidates", lambda *_args, **_kwargs: [base_candidate()])

    weaveshift_setlist.build_continuous_mix(args_for(manifest, out))

    report_path = tmp_path / "mix.weaveshift_1_0_0_setlist_report.json"
    snippet_dir = tmp_path / "mix_transition_snippets"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert out.exists()
    assert sf.info(str(out)).subtype == "PCM_16"
    assert report["transitions"][0]["selected_candidate"] == "manual"
    assert report["transitions"][0]["source_a_fade_start_sec"] == 1.5
    assert report["transitions"][0]["source_b_cue_sec"] == 0.25
    assert report["track_source_zero_in_mix_sec"] == [0.0, 1.25]
    assert len(report["snippet_files"]) == 1
    assert snippet_dir.exists()


def test_build_continuous_mix_applies_handoff_override_and_writes_report(tmp_path, monkeypatch):
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    write_wav(a, freq=440)
    write_wav(b, freq=660)
    manifest = tmp_path / "setlist.json"
    manifest.write_text(
        json.dumps(
            {
                "tracks": [
                    {"path": "a.wav", "title": "Golden & Free"},
                    {"path": "b.wav", "title": "Half of Everything"},
                ],
                "settings": {
                    "transition_overrides": [
                        {
                            "index": 1,
                            "mode": "handoff",
                            "a_fade_start_sec": 1.5,
                            "a_cut_sec": 2.5,
                            "b_cue_sec": 0.25,
                            "takeover_overlap_sec": 0.25,
                            "b_entry_gain_db": -1.0,
                            "b_fade_in_sec": 0.75,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "mix.wav"

    monkeypatch.setattr(weaveshift_plan, "analyze_audio", lambda path, role, **_: fake_analysis(path, role))
    monkeypatch.setattr(weaveshift_plan, "choose_candidates", lambda *_args, **_kwargs: [base_candidate()])

    weaveshift_setlist.build_continuous_mix(args_for(manifest, out))

    report_path = tmp_path / "mix.weaveshift_1_0_0_setlist_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    transition = report["transitions"][0]

    assert transition["selected_candidate"] == "handoff"
    assert transition["source_a_fade_start_sec"] == 1.5
    assert transition["source_a_cut_sec"] == 2.5
    assert transition["source_b_cue_sec"] == 0.25
    assert transition["takeover_overlap_sec"] == 0.25
    assert transition["b_fade_in_sec"] == 0.75
    assert transition["b_entry_gain_db"] == -1.0
    assert report["track_source_zero_in_mix_sec"] == [0.0, 2.0]
    assert "handoff transition: fades Track A down before Track B takeover" in transition["notes"]


def test_build_continuous_mix_natural_override_preserves_tracks_and_inserts_pause(tmp_path, monkeypatch):
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    write_wav(a, freq=440)
    write_wav(b, freq=660)
    manifest = tmp_path / "setlist.json"
    manifest.write_text(
        json.dumps(
            {
                "tracks": [
                    {"path": "a.wav", "title": "Song A"},
                    {"path": "b.wav", "title": "Song B"},
                ],
                "settings": {
                    "transition_overrides": [
                        {"index": 1, "mode": "natural", "pause_sec": 1.25}
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "mix.wav"
    args = args_for(manifest, out)
    args.make_snippets = False

    monkeypatch.setattr(
        weaveshift_plan,
        "analyze_audio",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("natural mode must not analyze audio")),
    )
    monkeypatch.setattr(
        weaveshift_plan,
        "choose_candidates",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("natural mode must not score crossfades")),
    )

    weaveshift_setlist.build_continuous_mix(args)

    source_a, _ = sf.read(a)
    source_b, _ = sf.read(b)
    audio, sample_rate = sf.read(out)
    report = json.loads((tmp_path / "mix.weaveshift_1_0_0_setlist_report.json").read_text(encoding="utf-8"))
    transition = report["transitions"][0]

    assert len(audio) == int(sample_rate * 9.25)
    assert np.max(np.abs(audio[int(sample_rate * 4.0):int(sample_rate * 5.25)])) == 0.0
    np.testing.assert_allclose(audio[:len(source_a)], source_a, atol=1 / 32768)
    np.testing.assert_allclose(audio[-len(source_b):], source_b, atol=1 / 32768)
    assert transition["selected_candidate"] == "natural"
    assert transition["score"] is None
    assert transition["ranked_candidates"] == []
    assert transition["source_a_cut_sec"] == 4.0
    assert transition["source_b_cue_sec"] == 0.0
    assert transition["overlap_sec"] == 0.0
    assert transition["pause_sec"] == 1.25
    assert transition["trim_a_tail_sec"] == 0.0
    assert report["track_source_zero_in_mix_sec"] == [0.0, 5.25]


def test_global_natural_mode_uses_cli_pause_without_analysis(tmp_path, monkeypatch):
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    write_wav(a, freq=440)
    write_wav(b, freq=660)
    manifest = tmp_path / "setlist.json"
    manifest.write_text(json.dumps({"tracks": ["a.wav", "b.wav"]}), encoding="utf-8")
    out = tmp_path / "global.wav"
    args = args_for(manifest, out)
    args.transition_mode = "natural"
    args.natural_pause_sec = 0.5
    args.make_snippets = False

    monkeypatch.setattr(
        weaveshift_plan,
        "analyze_audio",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("natural mode must not analyze audio")),
    )
    monkeypatch.setattr(
        weaveshift_plan,
        "choose_candidates",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("natural mode must not score crossfades")),
    )

    weaveshift_setlist.build_continuous_mix(args)

    report = json.loads((tmp_path / "global.weaveshift_1_0_0_setlist_report.json").read_text(encoding="utf-8"))
    assert sf.info(out).duration == 8.5
    assert report["transitions"][0]["pause_sec"] == 0.5
    assert report["track_source_zero_in_mix_sec"] == [0.0, 4.5]


def test_mixed_mode_plan_analyzes_only_scored_junctions(tmp_path, monkeypatch):
    paths = []
    for index, freq in enumerate((440, 550, 660), start=1):
        path = tmp_path / f"track-{index}.wav"
        write_wav(path, freq=freq)
        paths.append(path)

    calls = []

    def record_analysis(path, role, **_kwargs):
        calls.append((path, role))
        return fake_analysis(path, role)

    monkeypatch.setattr(weaveshift_plan, "analyze_audio", record_analysis)
    monkeypatch.setattr(weaveshift_plan, "choose_candidates", lambda *_args, **_kwargs: [base_candidate()])

    plan = weaveshift_plan.plan_setlist_mix(
        [weaveshift_plan.TrackSpec(path=str(path), title=f"Track {i}") for i, path in enumerate(paths, start=1)],
        manifest_settings={"transition_overrides": [{"index": 1, "mode": "natural", "pause_sec": 0.5}]},
        search_a_sec=1.0,
        search_b_sec=1.0,
        prefer_mps=False,
    )

    assert calls == [(str(paths[1]), "a"), (str(paths[2]), "b")]
    assert plan.junctions[0].track_a_analysis is None
    assert plan.junctions[0].track_b_analysis is None
    assert plan.junctions[0].ranked == []
    assert plan.junctions[1].track_a_analysis is not None
    assert plan.source_zero_in_mix_sec == [0.0, 4.5, 6.5]
