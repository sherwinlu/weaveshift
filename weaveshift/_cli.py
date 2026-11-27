"""Argument definitions and process plumbing shared by the WeaveShift CLIs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

from weaveshift._paths import atomic_write_text
from weaveshift._version import APP_VERSION
from weaveshift.audio import VOCAL_METHODS, validate_audio_output
from weaveshift.plan import (
    DEFAULT_B_CUE_MAX_SEC,
    DEFAULT_MAX_TRIM_A_SEC,
    DEFAULT_SEARCH_SEC,
    SetlistMixPlan,
    TwoTrackMixPlan,
)
from weaveshift.profiles import BUILTIN_PROFILES

APP_NAME = f"WeaveShift {APP_VERSION}"

CANDIDATE_MODES = (
    "vocal_safe",
    "beat_aligned",
    "quick_cut",
    "smooth",
    "profile",
    "vocal_ducked",
    "long_blend",
)


def add_planning_arguments(parser: argparse.ArgumentParser, *, vocal_default: str) -> None:
    """Add the scoring, vocal and search-window flags both mixers accept."""
    parser.add_argument(
        "--profile",
        default="edm",
        help=f"Built-in scoring profile used with the 'profile' mode (default: edm): {', '.join(BUILTIN_PROFILES)}",
    )
    parser.add_argument("--scoring-config", default=None, help="Path to a TOML scoring profile; overrides --profile")
    parser.add_argument(
        "--vocal-method",
        choices=list(VOCAL_METHODS),
        default=vocal_default,
        help=(
            "Vocal analysis method ('auto' tries Demucs then heuristic; 'none' skips vocal detection; "
            f"default: {vocal_default})"
        ),
    )
    parser.add_argument("--no-mps", action="store_true", help="Do not try Apple Silicon MPS for Demucs")
    parser.add_argument(
        "--search-a-sec",
        type=float,
        default=DEFAULT_SEARCH_SEC,
        help=f"Seconds of the outgoing track's outro to analyze (default: {DEFAULT_SEARCH_SEC:g})",
    )
    parser.add_argument(
        "--search-b-sec",
        type=float,
        default=DEFAULT_SEARCH_SEC,
        help=f"Seconds of the incoming track's intro to analyze (default: {DEFAULT_SEARCH_SEC:g})",
    )
    parser.add_argument(
        "--max-trim-a-sec",
        type=float,
        default=DEFAULT_MAX_TRIM_A_SEC,
        help=f"Maximum seconds allowed to trim from the outgoing tail (default: {DEFAULT_MAX_TRIM_A_SEC:g})",
    )
    parser.add_argument(
        "--b-cue-max-sec",
        type=float,
        default=DEFAULT_B_CUE_MAX_SEC,
        help=f"Latest allowed cue point in the incoming intro (default: {DEFAULT_B_CUE_MAX_SEC:g})",
    )


def resolve_output(raw: str) -> tuple[Path, Path, str]:
    """Validate an output path; return it with its extensionless base and lowercase suffix."""
    out = validate_audio_output(raw)
    return out, out.with_suffix(""), out.suffix.lower()


def write_report(base: Path, suffix: str, report: Mapping[str, object]) -> Path:
    """Write a JSON report beside the output atomically.

    The suffix is appended rather than substituted: a dotted output stem such as
    "mix v1.2.wav" must not lose part of its name to with_suffix().
    """
    path = base.with_name(base.name + suffix)
    payload = json.dumps(report, indent=2, allow_nan=False)
    atomic_write_text(path, payload, encoding="utf-8")
    return path


def warn_unused_profile(args: argparse.Namespace, mode: str, mode_flag: str) -> None:
    """Warn when profile flags were given but the run is not in 'profile' mode."""
    if mode != "profile" and (args.scoring_config is not None or args.profile != "edm"):
        print(
            f"Warning: --profile/--scoring-config are only used with {mode_flag} profile; "
            "ignoring profile settings for this run.",
            file=sys.stderr,
        )


def report_widened_windows(args: argparse.Namespace, plan: TwoTrackMixPlan | SetlistMixPlan) -> None:
    """Tell the user when the planner analyzed more audio than they asked for."""
    for label, requested, effective in (
        ("A", args.search_a_sec, plan.effective_search_a_sec),
        ("B", args.search_b_sec, plan.effective_search_b_sec),
    ):
        req = float(requested)
        if effective > req:
            print(f"Note: widened Track {label} analysis window from {req:.1f}s to {effective:.1f}s.")


def run_cli(
    parser: argparse.ArgumentParser, run: Callable[[argparse.Namespace], object], argv: Optional[Sequence[str]]
) -> int:
    """Parse args, run, and map failures to exit codes."""
    args = parser.parse_args(argv)
    try:
        run(args)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
