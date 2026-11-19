from weaveshift.profiles import ScoringProfile, load_scoring_profile
from weaveshift.two_tracks import profile_search_parameters


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
    assert "weaveshift.two_tracks:main" in text
    assert "weaveshift-setlist" in text


def test_builtin_profiles_match_shipped_configs():
    import tomllib
    from pathlib import Path

    from weaveshift.profiles import BUILTIN_PROFILES

    root = Path(__file__).resolve().parents[1]
    shipped = {p.stem for p in (root / "weaveshift" / "configs").glob("*.toml")}
    assert shipped == set(BUILTIN_PROFILES)
    pyproject = tomllib.loads((root / "pyproject.toml").read_text())
    assert pyproject["tool"]["setuptools"]["package-data"]["weaveshift"] == ["configs/*.toml"]


def test_cwd_configs_never_shadow_shipped_profile(tmp_path, monkeypatch):
    from weaveshift.profiles import builtin_profile_path

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "edm.toml").write_text('name = "evil"\n')
    monkeypatch.chdir(tmp_path)
    path = builtin_profile_path("edm")
    assert tmp_path not in path.parents
    assert path.parent.name == "configs" and path.parent.parent.name == "weaveshift"


def test_every_builtin_profile_loads_from_packaged_configs():
    from weaveshift.profiles import BUILTIN_PROFILES, builtin_profile_path, load_scoring_profile

    for name in BUILTIN_PROFILES:
        assert builtin_profile_path(name).is_file(), name
        assert load_scoring_profile(name).name
