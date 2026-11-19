"""Security and edge-case hardening tests.

Covers the manifest settings allowlist, transition override bounds checking,
built-in profile name validation, and output naming for dotted stems.
"""
import json
from pathlib import Path

import numpy as np
import pytest

import weaveshift.plan as weaveshift_plan
import weaveshift.setlist as weaveshift_setlist
from weaveshift.audio import AudioAnalysis, TransitionCandidate
from weaveshift.plan import (
    apply_handoff_transition_override,
    apply_manual_transition_override,
    coerce_float_override,
)
from weaveshift.profiles import builtin_profile_path
from weaveshift.setlist import apply_manifest_settings, build_parser, load_setlist
from tests.audio_helpers import write_stereo_wav
from tests.cli_helpers import setlist_args


def analysis(role: str, *, duration: float = 30.0) -> AudioAnalysis:
    times = np.array([0.0, 10.0, 20.0, 30.0], dtype=float)
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
        vocal_method="heuristic",
        analysis_window_start_sec=0.0,
        analysis_window_duration_sec=min(35.0, duration),
        energy_curve=curve,
        loudness_curve=curve,
    )


def base_candidate() -> TransitionCandidate:
    return TransitionCandidate(
        name="vocal_safe",
        score=0.9,
        a_fade_start_sec=20.0,
        a_cut_sec=24.0,
        b_cue_sec=0.0,
        overlap_sec=4.0,
        b_gain_db=0.0,
        trim_a_tail_sec=1.0,
        vocal_collision_score=0.0,
        beat_alignment_score=0.9,
        energy_score=0.9,
        placement_score=0.9,
        loudness_score=0.9,
        perceptual_loudness_score=0.9,
        compatibility_score=0.9,
        notes=["balanced candidate"],
    )


def parsed_args(*extra: str):
    return build_parser().parse_args(["setlist.json", *extra])


# -----------------------------
# Manifest settings allowlist
# -----------------------------

def test_manifest_cannot_override_cli_only_arguments(capsys):
    args = parsed_args()
    apply_manifest_settings(
        args,
        {"output": "elsewhere.wav", "setlist": "other.json", "scoring_config": "/etc/passwd"},
        build_parser(),
    )
    assert args.output == "weaveshift_setlist.wav"
    assert args.setlist == "setlist.json"
    assert args.scoring_config is None
    err = capsys.readouterr().err
    assert "ignoring unsupported manifest setting 'output'" in err
    assert "ignoring unsupported manifest setting 'setlist'" in err
    assert "ignoring unsupported manifest setting 'scoring_config'" in err


def test_manifest_applies_allowlisted_settings_onto_defaults():
    args = parsed_args()
    apply_manifest_settings(
        args,
        {
            "transition_mode": "quick_cut",
            "vocal_method": "none",
            "natural_pause_sec": "1.5",
            "make_snippets": True,
        },
        build_parser(),
    )
    assert args.transition_mode == "quick_cut"
    assert args.vocal_method == "none"
    assert args.natural_pause_sec == 1.5
    assert args.make_snippets is True


def test_explicit_cli_values_win_over_manifest():
    args = parsed_args("--transition-mode", "natural")
    apply_manifest_settings(args, {"transition_mode": "quick_cut"}, build_parser())
    assert args.transition_mode == "natural"


@pytest.mark.parametrize(
    ("settings", "match"),
    [
        ({"search_a_sec": "fast"}, "must be a number"),
        ({"search_a_sec": float("inf")}, "must be a finite number"),
        ({"make_snippets": "yes"}, "must be true or false"),
        ({"transition_mode": 3}, "must be a string"),
        ({"transition_mode": "warp"}, "transition_mode must be one of"),
        ({"vocal_method": "psychic"}, "vocal_method must be one of"),
    ],
)
def test_manifest_setting_validation_errors(settings, match):
    with pytest.raises(ValueError, match=match):
        apply_manifest_settings(parsed_args(), settings, build_parser())


def test_build_continuous_mix_ignores_manifest_output_redirect(tmp_path, monkeypatch, capsys):
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    write_stereo_wav(a, freq=440)
    write_stereo_wav(b, freq=660)
    redirected = tmp_path / "redirected.wav"
    manifest = tmp_path / "setlist.json"
    manifest.write_text(
        json.dumps(
            {
                "tracks": [
                    {"path": "a.wav", "title": "Song A"},
                    {"path": "b.wav", "title": "Song B"},
                ],
                "settings": {"output": str(redirected), "transition_mode": "natural"},
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "mix.wav"
    args = setlist_args(manifest, out, apply_manifest_settings=True)

    weaveshift_setlist.build_continuous_mix(args)

    assert out.exists()
    assert not redirected.exists()
    assert args.transition_mode == "natural"
    assert "ignoring unsupported manifest setting 'output'" in capsys.readouterr().err


def test_load_setlist_rejects_non_dict_settings(tmp_path):
    manifest = tmp_path / "setlist.json"
    manifest.write_text(json.dumps({"tracks": ["a.wav", "b.wav"], "settings": ["nope"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="'settings' must be a JSON object"):
        load_setlist(str(manifest))


@pytest.mark.parametrize(
    ("track", "match"),
    [
        ({"path": "a.wav", "bpm": "fast"}, "bpm must be a number"),
        ({"path": "a.wav", "key": 12}, "key must be a string"),
        ({"path": 42}, "missing a path/file field"),
    ],
)
def test_load_setlist_rejects_invalid_track_fields(tmp_path, track, match):
    manifest = tmp_path / "setlist.json"
    manifest.write_text(json.dumps({"tracks": [track, {"path": "b.wav"}]}), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_setlist(str(manifest))


# -----------------------------
# Transition override bounds
# -----------------------------

@pytest.mark.parametrize("value", ["nan", "inf", float("nan"), float("inf"), True])
def test_coerce_float_override_rejects_non_finite_and_bool(value):
    with pytest.raises(ValueError):
        coerce_float_override({"overlap_sec": value}, "overlap_sec", 1.0)


@pytest.mark.parametrize(
    ("item", "match"),
    [
        ({"a_fade_start_sec": -1.0, "a_cut_sec": 2.0}, "a_fade_start_sec must be >= 0"),
        ({"a_fade_start_sec": 1.0, "a_cut_sec": 2.0, "b_cue_sec": -3.0}, "b_cue_sec must be >= 0"),
        ({"a_fade_start_sec": 1.0, "a_cut_sec": 99.0}, "beyond the end of Track A"),
        ({"a_fade_start_sec": 1.0, "a_cut_sec": 2.0, "b_cue_sec": 45.0}, "beyond the end of Track B"),
    ],
)
def test_manual_override_rejects_out_of_bounds_timing(item, match):
    with pytest.raises(ValueError, match=match):
        apply_manual_transition_override(
            base_candidate(), {"overlap_sec": 1.0, **item}, 1, analysis("a"), analysis("b")
        )


def test_handoff_override_rejects_out_of_bounds_timing():
    item = {
        "a_fade_start_sec": 1.0,
        "a_cut_sec": 99.0,
        "b_cue_sec": 0.5,
        "takeover_overlap_sec": 0.25,
        "b_fade_in_sec": 0.75,
        "b_entry_gain_db": -1.0,
    }
    with pytest.raises(ValueError, match="beyond the end of Track A"):
        apply_handoff_transition_override(base_candidate(), item, 1, analysis("a"), analysis("b"))


def test_manual_override_within_bounds_still_works():
    cand = apply_manual_transition_override(
        base_candidate(),
        {"a_fade_start_sec": 20.0, "a_cut_sec": 24.0, "b_cue_sec": 0.5, "overlap_sec": 4.0, "b_gain_db": -1.0},
        1,
        analysis("a"),
        analysis("b"),
    )
    assert cand.name == "manual"
    assert cand.b_cue_sec == 0.5


# -----------------------------
# Profile name validation
# -----------------------------

@pytest.mark.parametrize("name", ["../evil", "a/b", "..", "edm.toml", "e m"])
def test_builtin_profile_path_rejects_non_identifier_names(name):
    with pytest.raises(ValueError, match="Invalid profile name"):
        builtin_profile_path(name)


def test_builtin_profile_path_accepts_known_names():
    assert builtin_profile_path("vocal_trance").name == "vocal_trance.toml"


# -----------------------------
# Output naming edge cases
# -----------------------------

def test_dotted_output_stem_keeps_full_report_name(tmp_path, monkeypatch):
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    write_stereo_wav(a, freq=440)
    write_stereo_wav(b, freq=660)
    manifest = tmp_path / "setlist.json"
    manifest.write_text(json.dumps({"tracks": ["a.wav", "b.wav"]}), encoding="utf-8")
    out = tmp_path / "mix v1.2.wav"
    args = setlist_args(manifest, out, transition_mode="natural", natural_pause_sec=0.5)

    weaveshift_setlist.build_continuous_mix(args)

    assert out.exists()
    assert (tmp_path / "mix v1.2.weaveshift_setlist_report.json").exists()


def test_load_setlist_rejects_parent_directory_traversal(tmp_path):
    manifest = tmp_path / "setlist.json"
    manifest.write_text(json.dumps({"tracks": ["../outside.wav", "b.wav"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="outside the manifest directory"):
        load_setlist(str(manifest))


def test_load_setlist_rejects_absolute_path_outside_manifest_dir(tmp_path):
    manifest_dir = tmp_path / "set"
    manifest_dir.mkdir()
    manifest = manifest_dir / "setlist.json"
    outside = tmp_path / "elsewhere" / "a.wav"
    manifest.write_text(json.dumps({"tracks": [str(outside), "b.wav"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="outside the manifest directory"):
        load_setlist(str(manifest))


def test_load_setlist_rejects_symlink_escape(tmp_path):
    manifest_dir = tmp_path / "set"
    manifest_dir.mkdir()
    secret_dir = tmp_path / "secret"
    secret_dir.mkdir()
    (manifest_dir / "link").symlink_to(secret_dir, target_is_directory=True)
    manifest = manifest_dir / "setlist.json"
    manifest.write_text(json.dumps({"tracks": ["link/a.wav", "b.wav"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="outside the manifest directory"):
        load_setlist(str(manifest))


def test_load_setlist_allow_outside_opt_in(tmp_path):
    manifest_dir = tmp_path / "set"
    manifest_dir.mkdir()
    manifest = manifest_dir / "setlist.json"
    outside = tmp_path / "elsewhere" / "a.wav"
    manifest.write_text(json.dumps({"tracks": [str(outside), "../b.wav"]}), encoding="utf-8")
    tracks, _ = load_setlist(str(manifest), allow_outside=True)
    assert tracks[0].path == str(outside.resolve())
    assert tracks[1].path == str((tmp_path / "b.wav").resolve())


def test_load_setlist_accepts_nested_relative_and_inside_absolute(tmp_path):
    manifest = tmp_path / "setlist.json"
    inside_abs = tmp_path / "sub" / "a.wav"
    manifest.write_text(json.dumps({"tracks": [str(inside_abs), "sub/../b.wav"]}), encoding="utf-8")
    tracks, _ = load_setlist(str(manifest))
    assert [Path(t.path).name for t in tracks] == ["a.wav", "b.wav"]


def test_cues_resolve_audio_path_rejects_paths_outside_manifest_dir(tmp_path):
    from weaveshift.cues import resolve_audio_path

    manifest_dir = tmp_path / "set"
    manifest_dir.mkdir()
    manifest = manifest_dir / "tracks.csv"
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"x")
    with pytest.raises(ValueError, match="outside the allowed directories"):
        resolve_audio_path(str(outside), manifest, None)
    with pytest.raises(ValueError, match="outside the allowed directories"):
        resolve_audio_path("../outside.wav", manifest, None)
    assert resolve_audio_path(str(outside), manifest, None, allow_outside=True) == outside
    assert resolve_audio_path(str(outside), manifest, tmp_path) == outside  # inside --audio-root


def test_vocal_method_none_skips_vocal_analysis(monkeypatch):
    import weaveshift.audio as audio_mod

    monkeypatch.setattr(
        audio_mod,
        "vocal_segments_heuristic",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("heuristic must not run when method is 'none'")),
    )
    monkeypatch.setattr(
        audio_mod,
        "vocal_segments_demucs",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("demucs must not run when method is 'none'")),
    )
    y = np.zeros((2, 48000), dtype=np.float32)
    segs, method = audio_mod.analyze_vocals("x.wav", y, 48000, 0.0, 1.0, "none", False)
    assert segs == []
    assert method == "none"


def test_vocal_method_none_prints_warning(capsys):
    from weaveshift.audio import warn_if_vocals_skipped

    warn_if_vocals_skipped("heuristic")
    assert capsys.readouterr().err == ""
    warn_if_vocals_skipped("none")
    assert "vocal-collision scores are 0" in capsys.readouterr().err


def test_unknown_home_user_is_value_error(tmp_path):
    manifest = tmp_path / "setlist.json"
    manifest.write_text(json.dumps({"tracks": ["~no_such_user_xyz/a.wav", "b.wav"]}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_setlist(str(manifest))


def test_cues_resolve_tries_remaining_candidates_after_outside_one(tmp_path):
    from weaveshift.cues import resolve_audio_path

    manifest_dir = tmp_path / "set"
    root = tmp_path / "root"
    outside = tmp_path / "elsewhere"
    for d in (manifest_dir, root, outside):
        d.mkdir()
    (outside / "a.wav").write_bytes(b"x")
    (root / "a.wav").symlink_to(outside / "a.wav")  # first candidate escapes --audio-root
    (manifest_dir / "a.wav").write_bytes(b"y")  # later candidate is valid
    assert resolve_audio_path("a.wav", manifest_dir / "t.csv", root) == manifest_dir / "a.wav"


@pytest.mark.parametrize("model", ["-o", "--help", ".", "..", "a/b", ""])
def test_demucs_model_name_rejects_unsafe_values(model):
    import weaveshift.audio as audio_mod

    with pytest.raises(ValueError, match="Invalid Demucs model name"):
        audio_mod._demucs_cli_vocals_path("x.wav", model)


def test_cues_skip_missing_does_not_swallow_containment_errors(tmp_path, capsys):
    from weaveshift.cues import main as cues_main

    manifest_dir = tmp_path / "set"
    manifest_dir.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"x")
    manifest = manifest_dir / "tracks.csv"
    manifest.write_text(f"title,path\nOut,{outside}\n", encoding="utf-8")
    rc = cues_main([str(manifest), "--skip-missing", "--out", str(tmp_path / "o.csv")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "Error:" in err
    assert "outside the allowed directories" in err


def test_cues_does_not_fall_back_to_cwd(tmp_path, monkeypatch):
    from weaveshift.cues import resolve_audio_path

    manifest_dir = tmp_path / "set"
    cwd = tmp_path / "cwd"
    manifest_dir.mkdir()
    cwd.mkdir()
    (cwd / "song.wav").write_bytes(b"x")
    monkeypatch.chdir(cwd)
    with pytest.raises(FileNotFoundError):
        resolve_audio_path("song.wav", manifest_dir / "t.csv", None)


def test_soft_duck_target_override_is_validated():
    from weaveshift.plan import _soft_duck_settings

    base = TransitionCandidate(
        name="x", score=0.0, a_fade_start_sec=0.0, a_cut_sec=1.0, b_cue_sec=0.0, overlap_sec=1.0,
        b_gain_db=0.0, trim_a_tail_sec=0.0, vocal_collision_score=0.0, beat_alignment_score=0.0,
        energy_score=0.0, placement_score=0.0, loudness_score=0.0, perceptual_loudness_score=0.0,
        compatibility_score=0.0, notes=[],
    )
    assert _soft_duck_settings({"soft_duck_target": "b", "soft_duck_db": -3.0}, base, 1) == (-3.0, "b")
    with pytest.raises(ValueError, match="soft_duck_target must be one of"):
        _soft_duck_settings({"soft_duck_target": "typo"}, base, 1)


def test_soft_duck_target_without_negative_db_is_rejected():
    from weaveshift.plan import _soft_duck_settings

    base = TransitionCandidate(
        name="x", score=0.0, a_fade_start_sec=0.0, a_cut_sec=1.0, b_cue_sec=0.0, overlap_sec=1.0,
        b_gain_db=0.0, trim_a_tail_sec=0.0, vocal_collision_score=0.0, beat_alignment_score=0.0,
        energy_score=0.0, placement_score=0.0, loudness_score=0.0, perceptual_loudness_score=0.0,
        compatibility_score=0.0, notes=[],
    )
    with pytest.raises(ValueError, match="needs a negative soft_duck_db"):
        _soft_duck_settings({"soft_duck_target": "b"}, base, 1)
    assert _soft_duck_settings({"soft_duck_target": "b", "soft_duck_db": -3.0}, base, 1) == (-3.0, "b")
    with pytest.raises(ValueError, match="must be <= 0 dB"):
        _soft_duck_settings({"soft_duck_db": 2.0}, base, 1)


def test_soft_duck_db_zero_switches_off_inherited_duck():
    from dataclasses import replace

    from weaveshift.plan import _soft_duck_settings

    base = TransitionCandidate(
        name="vocal_ducked", score=0.0, a_fade_start_sec=0.0, a_cut_sec=1.0, b_cue_sec=0.0, overlap_sec=1.0,
        b_gain_db=0.0, trim_a_tail_sec=0.0, vocal_collision_score=0.0, beat_alignment_score=0.0,
        energy_score=0.0, placement_score=0.0, loudness_score=0.0, perceptual_loudness_score=0.0,
        compatibility_score=0.0, notes=[], soft_duck_db=-2.5, soft_duck_target="a",
    )
    assert _soft_duck_settings({"soft_duck_db": 0}, base, 1) == (0.0, "none")
    assert _soft_duck_settings({}, base, 1) == (-2.5, "a")
    with pytest.raises(ValueError, match="needs a negative soft_duck_db"):
        _soft_duck_settings({"soft_duck_target": "b", "soft_duck_db": 0}, base, 1)


def test_bare_negative_soft_duck_db_defaults_to_outgoing_track():
    from weaveshift.plan import _soft_duck_settings

    base = TransitionCandidate(
        name="x", score=0.0, a_fade_start_sec=0.0, a_cut_sec=1.0, b_cue_sec=0.0, overlap_sec=1.0,
        b_gain_db=0.0, trim_a_tail_sec=0.0, vocal_collision_score=0.0, beat_alignment_score=0.0,
        energy_score=0.0, placement_score=0.0, loudness_score=0.0, perceptual_loudness_score=0.0,
        compatibility_score=0.0, notes=[],
    )
    assert _soft_duck_settings({"soft_duck_db": -4}, base, 1) == (-4.0, "a")
    with pytest.raises(ValueError, match="has no effect"):
        _soft_duck_settings({"soft_duck_db": -4, "soft_duck_target": "none"}, base, 1)


def test_explicit_none_target_switches_off_inherited_duck():
    from weaveshift.plan import _soft_duck_settings

    base = TransitionCandidate(
        name="vocal_ducked", score=0.0, a_fade_start_sec=0.0, a_cut_sec=1.0, b_cue_sec=0.0, overlap_sec=1.0,
        b_gain_db=0.0, trim_a_tail_sec=0.0, vocal_collision_score=0.0, beat_alignment_score=0.0,
        energy_score=0.0, placement_score=0.0, loudness_score=0.0, perceptual_loudness_score=0.0,
        compatibility_score=0.0, notes=[], soft_duck_db=-3.0, soft_duck_target="a",
    )
    assert _soft_duck_settings({"soft_duck_target": "none"}, base, 1) == (0.0, "none")


def test_unknown_home_user_is_value_error_for_scoring_config_and_setlist_path():
    from weaveshift.profiles import load_scoring_profile

    with pytest.raises(ValueError):
        load_scoring_profile(config_path="~no_such_user_xyz/p.toml")
    with pytest.raises(ValueError):
        load_setlist("~no_such_user_xyz/setlist.json")


@pytest.mark.parametrize("name", ["edm\n", "htdemucs\n"])
def test_name_validators_reject_trailing_newline(name):
    import weaveshift.audio as audio_mod
    from weaveshift.profiles import builtin_profile_path

    with pytest.raises(ValueError):
        builtin_profile_path(name)
    with pytest.raises(ValueError, match="Invalid Demucs model name"):
        audio_mod._demucs_cli_vocals_path("x.wav", name)


def test_null_soft_duck_target_is_treated_as_unset():
    from weaveshift.plan import _soft_duck_settings

    base = TransitionCandidate(
        name="x", score=0.0, a_fade_start_sec=0.0, a_cut_sec=1.0, b_cue_sec=0.0, overlap_sec=1.0,
        b_gain_db=0.0, trim_a_tail_sec=0.0, vocal_collision_score=0.0, beat_alignment_score=0.0,
        energy_score=0.0, placement_score=0.0, loudness_score=0.0, perceptual_loudness_score=0.0,
        compatibility_score=0.0, notes=[],
    )
    assert _soft_duck_settings({"soft_duck_target": None, "soft_duck_db": None}, base, 1) == (0.0, "none")


def test_null_soft_duck_db_does_not_count_as_set_with_explicit_none_target():
    from weaveshift.plan import _soft_duck_settings

    base = TransitionCandidate(
        name="vocal_ducked", score=0.0, a_fade_start_sec=0.0, a_cut_sec=1.0, b_cue_sec=0.0, overlap_sec=1.0,
        b_gain_db=0.0, trim_a_tail_sec=0.0, vocal_collision_score=0.0, beat_alignment_score=0.0,
        energy_score=0.0, placement_score=0.0, loudness_score=0.0, perceptual_loudness_score=0.0,
        compatibility_score=0.0, notes=[], soft_duck_db=-3.0, soft_duck_target="a",
    )
    assert _soft_duck_settings({"soft_duck_target": "none", "soft_duck_db": None}, base, 1) == (0.0, "none")
