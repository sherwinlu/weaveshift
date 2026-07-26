"""Smoke tests for shipped examples."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from weaveshift_setlist import main as setlist_main

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "examples" / "setlist_example.json"
GENERATE_FIXTURES = REPO / "examples" / "generate_fixtures.py"


def _ensure_example_fixtures() -> None:
    fixtures = EXAMPLE.parent / "fixtures"
    needed = [fixtures / name for name in ("track_01.wav", "track_02.wav", "track_03.wav")]
    if all(p.exists() for p in needed):
        return
    subprocess.run([sys.executable, str(GENERATE_FIXTURES)], check=True, cwd=str(REPO))


@pytest.mark.parametrize("apply_manifest_settings", [True, False])
def test_setlist_example_runs(tmp_path: Path, apply_manifest_settings: bool):
    assert EXAMPLE.exists()
    _ensure_example_fixtures()
    for name in ("track_01.wav", "track_02.wav", "track_03.wav"):
        assert (EXAMPLE.parent / "fixtures" / name).exists()

    out = tmp_path / "demo.wav"
    argv = [
        str(EXAMPLE),
        "-o",
        str(out),
        "--vocal-method",
        "heuristic",
    ]
    if apply_manifest_settings:
        argv.append("--apply-manifest-settings")
    else:
        argv.extend(["--transition-mode", "profile", "--profile", "edm"])
    assert setlist_main(argv) == 0
    assert out.exists()
    assert out.stat().st_size > 0
