"""Unit tests for shared CLI helpers in weaveshift._cli."""
import argparse
from argparse import Namespace
import json

from weaveshift._cli import (
    add_planning_arguments,
    report_widened_windows,
    resolve_output,
    run_cli,
    warn_unused_profile,
    write_report,
)
from weaveshift.plan import SetlistMixPlan
from weaveshift.profiles import BUILTIN_PROFILES


def test_warn_unused_profile_only_warns_outside_profile_mode(capsys):
    warn_unused_profile(Namespace(profile="edm", scoring_config=None), "recommended", "--mode")
    assert capsys.readouterr().err == ""

    warn_unused_profile(Namespace(profile="jazz", scoring_config=None), "profile", "--mode")
    assert capsys.readouterr().err == ""

    warn_unused_profile(Namespace(profile="jazz", scoring_config=None), "recommended", "--mode")
    err = capsys.readouterr().err
    assert "Warning:" in err
    assert "--mode profile" in err

    warn_unused_profile(Namespace(profile="edm", scoring_config="custom.toml"), "smooth", "--transition-mode")
    err = capsys.readouterr().err
    assert "Warning:" in err
    assert "--transition-mode profile" in err


def test_report_widened_windows_prints_only_widened_sides(capsys):
    plan = SetlistMixPlan(tracks=[], effective_search_a_sec=40.0, effective_search_b_sec=35.0)
    report_widened_windows(Namespace(search_a_sec=35, search_b_sec=35.0), plan)
    out = capsys.readouterr().out
    assert "widened Track A analysis window from 35.0s to 40.0s." in out
    assert "Track B" not in out


def test_resolve_output_and_write_report_preserve_dotted_stems(tmp_path):
    out, base, suffix = resolve_output(str(tmp_path / "mix v1.2.wav"))
    assert out == tmp_path / "mix v1.2.wav"
    assert base == tmp_path / "mix v1.2"
    assert suffix == ".wav"

    report_path = write_report(base, ".report.json", {"ok": True})
    assert report_path == tmp_path / "mix v1.2.report.json"
    assert json.loads(report_path.read_text(encoding="utf-8")) == {"ok": True}


def test_run_cli_maps_return_codes(capsys):
    parser = argparse.ArgumentParser()
    assert run_cli(parser, lambda _args: None, []) == 0

    def fail(_args):
        raise ValueError("bad input")

    assert run_cli(parser, fail, []) == 1
    assert "Error: bad input" in capsys.readouterr().err

    def interrupt(_args):
        raise KeyboardInterrupt()

    assert run_cli(parser, interrupt, []) == 130
    assert "Interrupted by user." in capsys.readouterr().err


def test_add_planning_arguments_exposes_profiles_and_defaults():
    parser = argparse.ArgumentParser()
    add_planning_arguments(parser, vocal_default="heuristic")
    help_text = " ".join(parser.format_help().split())
    for name in BUILTIN_PROFILES:
        assert name in help_text
    assert "default: heuristic" in help_text
    assert "default: 35" in help_text
    assert "default: 8" in help_text
    assert "default: 10" in help_text
    args = parser.parse_args([])
    assert args.vocal_method == "heuristic"
    assert args.profile == "edm"
