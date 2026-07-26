from weaveshift_profiles import ScoringProfile, load_scoring_profile
from weaveshift_two_tracks import profile_search_parameters


def test_profile_limits_shape_candidate_search_space():
    profile = ScoringProfile(
        name="test",
        preferences={
            "preferred_overlap_min_sec": 4.0,
            "preferred_overlap_max_sec": 10.0,
        },
        limits={
            "max_tail_trim_sec": 3.0,
            "max_b_cue_sec": 16.0,
            "max_vocal_collision_score": 0.2,
        },
    )
    trim, cue, overlaps, vocal_cutoff = profile_search_parameters(8.0, 10.0, profile)
    assert trim == 3.0
    assert cue == 16.0
    assert overlaps == [4.0, 6.0, 8.0, 10.0]
    assert vocal_cutoff == 0.2


def test_builtin_profiles_have_vocal_collision_cutoffs():
    expected = {
        "edm": 0.35,
        "vocal_trance": 0.25,
        "lounge": 0.30,
        "jazz": 0.25,
        "heart": 0.20,
        "cinematic": 0.30,
    }
    for name, cutoff in expected.items():
        profile = load_scoring_profile(name)
        assert profile.limit("max_vocal_collision_score", -1) == cutoff


def test_pyproject_installs_weaveshift_modules_and_scripts():
    from pathlib import Path
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "[build-system]" in text
    assert "py-modules" in text
    assert "weaveshift_two_tracks" in text
    assert "weaveshift-setlist" in text
    assert "share/weaveshift/configs" in text
