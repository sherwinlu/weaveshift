"""Security and edge-case hardening tests.

Covers the manifest settings allowlist, transition override bounds checking,
built-in profile name validation, and output naming for dotted stems.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest

import flowmix_plan
import flowmix_setlist
from flowmix_audio import AudioAnalysis, TransitionCandidate
from flowmix_plan import (
    apply_handoff_transition_override,
    apply_manual_transition_override,
    coerce_float_override,
)
from flowmix_profiles import builtin_profile_path
from flowmix_setlist import apply_manifest_settings, build_parser, load_setlist
from tests.audio_helpers import write_stereo_wav


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
    apply_manifest_settings(args, {"output": "elsewhere.wav", "setlist": "other.json"}, build_parser())
    assert args.output == "flowmix_setlist_1_0_0.wav"
    assert args.setlist == "setlist.json"
    err = capsys.readouterr().err
    assert "ignoring unsupported manifest setting 'output'" in err
    assert "ignoring unsupported manifest setting 'setlist'" in err


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
    assert args.vocal_method == "heuristic"  # 'none' maps to heuristic
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
    args = SimpleNamespace(
        setlist=str(manifest),
        output=str(out),
        transition_mode="recommended",
        make_snippets=False,
        profile="edm",
        scoring_config=None,
        vocal_method="heuristic",
        no_mps=True,
        search_a_sec=1.0,
        search_b_sec=1.0,
        max_trim_a_sec=2.0,
        b_cue_max_sec=2.0,
        natural_pause_sec=1.0,
        apply_manifest_settings=True,
    )

    flowmix_setlist.build_continuous_mix(args)

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
    args = SimpleNamespace(
        setlist=str(manifest),
        output=str(out),
        transition_mode="natural",
        make_snippets=False,
        profile="edm",
        scoring_config=None,
        vocal_method="heuristic",
        no_mps=True,
        search_a_sec=1.0,
        search_b_sec=1.0,
        max_trim_a_sec=2.0,
        b_cue_max_sec=2.0,
        natural_pause_sec=0.5,
        apply_manifest_settings=False,
    )

    flowmix_setlist.build_continuous_mix(args)

    assert out.exists()
    assert (tmp_path / "mix v1.2.flowmix_1_0_0_setlist_report.json").exists()
