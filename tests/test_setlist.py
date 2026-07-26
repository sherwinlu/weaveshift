from pathlib import Path
import json
import numpy as np
import soundfile as sf

from weaveshift_setlist import load_setlist, sanitize_filename


def write_wav(path: Path, sr=48000, subtype="PCM_16", channels=2):
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
    y = 0.01 * np.sin(2 * np.pi * 440 * t)
    if channels == 2:
        y = np.column_stack([y, y])
    sf.write(path, y, sr, subtype=subtype)


def test_load_json_setlist_with_metadata(tmp_path):
    a = tmp_path / "A.wav"
    b = tmp_path / "B.wav"
    write_wav(a)
    write_wav(b)
    manifest = tmp_path / "setlist.json"
    manifest.write_text(json.dumps({
        "tracks": [
            {"path": "A.wav", "title": "Track A", "bpm": 128, "key": "E major"},
            {"path": "B.wav", "title": "Track B", "bpm": 130, "key": "A minor"},
        ],
        "settings": {"transition_mode": "recommended"},
    }))
    tracks, settings = load_setlist(str(manifest))
    assert len(tracks) == 2
    assert tracks[0].title == "Track A"
    assert tracks[0].bpm == 128.0
    assert tracks[1].key == "A minor"
    assert settings["transition_mode"] == "recommended"


def test_load_text_setlist_relative_paths(tmp_path):
    (tmp_path / "one.wav").write_bytes(b"RIFFxxxxWAVE")
    (tmp_path / "two.wav").write_bytes(b"RIFFxxxxWAVE")
    txt = tmp_path / "setlist.txt"
    txt.write_text("# comment\none.wav\n\ntwo.wav\n")
    tracks, settings = load_setlist(str(txt))
    assert len(tracks) == 2
    assert tracks[0].path.endswith("one.wav")
    assert settings == {}


def test_sanitize_filename_removes_problem_characters():
    assert sanitize_filename("A/B: Track? *Name") == "AB_Track_Name"


def test_all_builtin_scoring_profiles_load():
    from weaveshift_profiles import load_scoring_profile

    for profile_name in ["edm", "vocal_trance", "vocal_trance_strict", "lounge", "jazz", "heart", "cinematic"]:
        profile = load_scoring_profile(profile_name)
        assert profile.name == profile_name
        assert profile.weights
        assert profile.preferences
        assert profile.limits
