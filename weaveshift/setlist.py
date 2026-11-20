#!/usr/bin/env python3
"""WeaveShift Setlist Builder.

Builds a continuous mix from an ordered setlist of full-mix WAV or MP3 masters.
Planning (analysis/scoring/selection) lives in weaveshift.plan; this module handles
manifest loading, render/export, and CLI orchestration.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

from pydub import AudioSegment

from weaveshift._cli import (
    APP_NAME,
    CANDIDATE_MODES,
    add_planning_arguments,
    report_widened_windows,
    resolve_output,
    run_cli,
    warn_unused_profile,
    write_report,
)
from weaveshift._paths import expand_user, resolve_contained
from weaveshift.audio import (
    VOCAL_METHODS,
    NaturalTransition,
    format_timestamp,
    resolve_wav_export_subtype,
    sec_to_ms,
    warn_if_vocals_skipped,
)
from weaveshift.plan import MAX_NATURAL_PAUSE_SEC, TrackSpec, plan_setlist_mix
from weaveshift.rendering import build_transition_audio, export_audio
from weaveshift.reports import build_setlist_report, candidate_verdict

SETLIST_TITLE = f"{APP_NAME} Setlist Builder"
DEFAULT_SETLIST_OUTPUT = "weaveshift_setlist.wav"
SETLIST_REPORT_SUFFIX = ".weaveshift_setlist_report.json"

TRANSITION_MODES = ("recommended", *CANDIDATE_MODES, "natural")

# Manifest settings that --apply-manifest-settings may copy onto CLI defaults,
# mapped to their expected type. This is a deliberate allowlist: anything else
# in the manifest settings object (including CLI-only path arguments like
# 'output' or 'scoring_config') is ignored with a warning so a manifest can
# tune the mix but cannot read arbitrary config paths or redirect output writes.
APPLIABLE_MANIFEST_SETTINGS: Dict[str, type] = {
    "transition_mode": str,
    "profile": str,
    "vocal_method": str,
    "make_snippets": bool,
    "natural_pause_sec": float,
    "search_a_sec": float,
    "search_b_sec": float,
    "max_trim_a_sec": float,
    "b_cue_max_sec": float,
}


@dataclass
class TransitionChoice:
    index: int
    from_title: str
    to_title: str
    selected_candidate: str
    score: Optional[float]
    source_a_fade_start_sec: float
    source_a_cut_sec: float
    source_b_cue_sec: float
    overlap_sec: float
    b_gain_db: float
    takeover_overlap_sec: Optional[float]
    b_fade_in_sec: Optional[float]
    b_entry_gain_db: Optional[float]
    pause_sec: float
    trim_a_tail_sec: float
    mix_transition_start_sec: float
    mix_transition_start_timestamp: str
    next_track_source_zero_in_mix_sec: float
    next_track_source_zero_in_mix_timestamp: str
    ranked_candidates: list[dict[str, object]]
    notes: List[str]


def sanitize_filename(text: str, max_len: int = 50) -> str:
    text = re.sub(r"[^A-Za-z0-9._ -]+", "", text).strip().replace(" ", "_")
    return (text[:max_len] or "track").strip("._-") or "track"


def coerce_manifest_setting(key: str, value: object, kind: type) -> object:
    """Validate and convert one manifest setting to the type its CLI flag expects."""
    if kind is bool:
        if not isinstance(value, bool):
            raise ValueError(f"Manifest setting {key!r} must be true or false. Got: {value!r}")
        return value
    if kind is float:
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ValueError(f"Manifest setting {key!r} must be a number. Got: {value!r}")
        try:
            result = float(value)
        except ValueError as exc:
            raise ValueError(f"Manifest setting {key!r} must be a number. Got: {value!r}") from exc
        if not math.isfinite(result):
            raise ValueError(f"Manifest setting {key!r} must be a finite number. Got: {value!r}")
        return result
    if not isinstance(value, str):
        raise ValueError(f"Manifest setting {key!r} must be a string. Got: {value!r}")
    return value


def apply_manifest_settings(
    args: argparse.Namespace, manifest_settings: dict[str, object], parser: argparse.ArgumentParser
) -> None:
    """Copy allowlisted manifest settings onto CLI defaults.

    Explicit CLI values always win over the manifest. `transition_overrides`
    is skipped here because the planner always honors it directly.
    """
    for key, value in manifest_settings.items():
        if key == "transition_overrides":
            continue
        kind = APPLIABLE_MANIFEST_SETTINGS.get(key)
        if kind is None:
            print(f"Warning: ignoring unsupported manifest setting {key!r}.", file=sys.stderr)
            continue
        if value is None or getattr(args, key) not in (None, parser.get_default(key)):
            continue
        setattr(args, key, coerce_manifest_setting(key, value, kind))

    if args.transition_mode not in TRANSITION_MODES:
        raise ValueError(
            f"Manifest transition_mode must be one of: {', '.join(TRANSITION_MODES)}. Got: {args.transition_mode!r}"
        )
    if args.vocal_method not in VOCAL_METHODS:
        raise ValueError(
            f"Manifest vocal_method must be one of: {', '.join(VOCAL_METHODS)}. Got: {args.vocal_method!r}"
        )


def load_setlist(path: str, allow_outside: bool = False) -> Tuple[List[TrackSpec], dict[str, object]]:
    """Load a JSON manifest or a simple newline-delimited text setlist.

    Track paths must resolve inside the manifest directory unless ``allow_outside``.
    """
    p = expand_user(path)
    if not p.exists():
        raise ValueError(f"Setlist file does not exist: {path}")
    if not p.is_file():
        raise ValueError(f"Setlist path is not a file: {path}")

    settings: dict[str, object] = {}
    tracks_raw = None
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            tracks_raw = data.get("tracks")
            settings_raw = data.get("settings") or {}
            if not isinstance(settings_raw, dict):
                raise ValueError("Setlist 'settings' must be a JSON object.")
            settings = settings_raw
        elif isinstance(data, list):
            tracks_raw = data
        else:
            raise ValueError("Setlist JSON must be an object with a 'tracks' list or a list of tracks.")
    else:
        lines = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            lines.append(line)
        tracks_raw = lines

    if not isinstance(tracks_raw, list) or len(tracks_raw) < 2:
        raise ValueError("Setlist must contain at least two tracks.")

    base_dir = p.parent
    tracks: List[TrackSpec] = []
    for i, item in enumerate(tracks_raw, start=1):
        if isinstance(item, str):
            raw_path = item
            title = Path(item).stem
            bpm = None
            key = None
        elif isinstance(item, dict):
            raw_path = item.get("path") or item.get("file")
            if not raw_path or not isinstance(raw_path, str):
                raise ValueError(f"Track #{i} is missing a path/file field or it is not a string.")
            title = item.get("title") or Path(raw_path).stem
            bpm = item.get("bpm")
            if bpm is not None:
                if isinstance(bpm, bool) or not isinstance(bpm, (int, float)):
                    raise ValueError(f"Track #{i} bpm must be a number. Got: {bpm!r}")
                if not math.isfinite(float(bpm)) or float(bpm) <= 0:
                    raise ValueError(f"Track #{i} bpm must be a positive finite number. Got: {bpm!r}")
            key = item.get("key")
            if key is not None and not isinstance(key, str):
                raise ValueError(f"Track #{i} key must be a string. Got: {key!r}")
        else:
            raise ValueError(f"Track #{i} must be a path string or an object.")

        track_path = resolve_contained(raw_path, base_dir, allow_outside=allow_outside, label=f"Track #{i}")
        tracks.append(
            TrackSpec(
                path=str(track_path),
                title=str(title),
                bpm=float(bpm) if bpm is not None else None,
                key=key,
            )
        )

    return tracks, settings


def build_continuous_mix(args: argparse.Namespace) -> None:
    tracks, manifest_settings = load_setlist(args.setlist, allow_outside=args.allow_outside_manifest_dir)
    if args.apply_manifest_settings:
        apply_manifest_settings(args, manifest_settings, build_parser())
    warn_unused_profile(args, args.transition_mode, "--transition-mode")

    out_path, base, fmt_suffix = resolve_output(args.output)

    print(SETLIST_TITLE)
    print(f"Loaded setlist with {len(tracks)} tracks.")
    print("Chain mode: preserving source timing/pitch; one selected transition per junction.")

    if args.transition_mode != "natural":
        warn_if_vocals_skipped(args.vocal_method)
    mix_plan = plan_setlist_mix(
        tracks,
        manifest_settings=manifest_settings,
        transition_mode=args.transition_mode,
        profile=args.profile,
        scoring_config=args.scoring_config,
        search_a_sec=args.search_a_sec,
        search_b_sec=args.search_b_sec,
        max_trim_a_sec=args.max_trim_a_sec,
        b_cue_max_sec=args.b_cue_max_sec,
        natural_pause_sec=args.natural_pause_sec,
        vocal_method=args.vocal_method,
        prefer_mps=not args.no_mps,
    )

    infos = mix_plan.wav_infos
    segments = mix_plan.segments
    scoring_profile = mix_plan.scoring_profile
    source_subtype = resolve_wav_export_subtype(infos)
    input_formats = ", ".join(
        f"{Path(cast(str, info['path'])).suffix.lower()} ({info['subtype']})" for info in infos[:3]
    )
    if len(infos) > 3:
        input_formats += ", ..."
    print(f"Setlist decode: {infos[0]['samplerate']} Hz, {infos[0]['channels']} ch — inputs: {input_formats}")
    print(f"Output: {fmt_suffix} ({'MP3 320 kbps' if fmt_suffix == '.mp3' else source_subtype or 'PCM_16'})")
    if scoring_profile is not None:
        print(f"Using scoring profile: {scoring_profile.name} — {scoring_profile.description}")
    report_widened_windows(args, mix_plan)

    snippet_requests: List[tuple[int, str, float]] = []
    final = segments[0]
    transitions: List[TransitionChoice] = []

    for junction in mix_plan.junctions:
        i = junction.index - 1
        selected = junction.selected
        print(f"\n--- Transition {junction.index}/{len(tracks)-1}: {junction.from_title} → {junction.to_title} ---")
        if junction.override_mode:
            print(f"Using transition override for #{junction.index}: {junction.from_title} → {junction.to_title} = {junction.override_mode}")

        mix_transition_start_sec = junction.mix_transition_start_sec
        mix_transition_start_ms = sec_to_ms(mix_transition_start_sec)
        tail = build_transition_audio(segments[i], segments[i + 1], selected).body
        final = final[:mix_transition_start_ms] + tail

        if isinstance(selected, NaturalTransition):
            print(
                f"Selected natural: Track A ends at {format_timestamp(mix_transition_start_sec)}; "
                f"pause {selected.pause_sec:.1f}s; Track B starts from 0:00 at "
                f"{format_timestamp(junction.next_track_source_zero_in_mix_sec)}"
            )
        else:
            print(
                f"Selected {selected.name}: score={selected.score:.3f}; new song starts in mix at "
                f"{format_timestamp(junction.next_track_source_zero_in_mix_sec + selected.b_cue_sec)} "
                f"({junction.next_track_source_zero_in_mix_sec + selected.b_cue_sec:.2f}s), "
                f"using Track B cue {format_timestamp(selected.b_cue_sec)} ({selected.b_cue_sec:.2f}s); "
                f"overlap {selected.overlap_sec:.1f}s, B gain {selected.b_gain_db:+.1f} dB"
            )

        transitions.append(
            TransitionChoice(
                index=junction.index,
                from_title=junction.from_title,
                to_title=junction.to_title,
                selected_candidate=selected.name,
                score=None if isinstance(selected, NaturalTransition) else selected.score,
                source_a_fade_start_sec=selected.a_fade_start_sec,
                source_a_cut_sec=selected.a_cut_sec,
                source_b_cue_sec=selected.b_cue_sec,
                overlap_sec=selected.overlap_sec,
                b_gain_db=selected.b_gain_db,
                takeover_overlap_sec=selected.takeover_overlap_sec,
                b_fade_in_sec=selected.b_fade_in_sec,
                b_entry_gain_db=selected.b_entry_gain_db,
                pause_sec=selected.pause_sec if isinstance(selected, NaturalTransition) else 0.0,
                trim_a_tail_sec=selected.trim_a_tail_sec,
                mix_transition_start_sec=junction.mix_transition_start_sec,
                mix_transition_start_timestamp=format_timestamp(mix_transition_start_sec),
                next_track_source_zero_in_mix_sec=junction.next_track_source_zero_in_mix_sec,
                next_track_source_zero_in_mix_timestamp=format_timestamp(junction.next_track_source_zero_in_mix_sec),
                ranked_candidates=junction.ranked,
                notes=([f"transition override: {junction.override_mode}"] if junction.override_mode else [])
                + [candidate_verdict(selected)]
                + selected.notes,
            )
        )
        if args.make_snippets:
            snippet_requests.append((junction.index, f"{junction.from_title}_to_{junction.to_title}", mix_transition_start_sec))

    final_peak_dbfs = final.max_dBFS
    peak_guard_gain_db = min(0.0, -1.0 - final_peak_dbfs)
    if peak_guard_gain_db < 0.0:
        final = final.apply_gain(peak_guard_gain_db)
    final_gain_db = round(peak_guard_gain_db, 3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_audio(final, out_path, source_subtype)

    snippet_outputs = []
    if args.make_snippets:
        snippet_dir = base.parent / f"{base.name}_transition_snippets"
        snippet_dir.mkdir(parents=True, exist_ok=True)
        for idx, label, start_sec in snippet_requests:
            start_ms = sec_to_ms(max(0.0, start_sec - 12.0))
            end_ms = sec_to_ms(min(len(final) / 1000.0, start_sec + 24.0))
            snip_name = f"transition_{idx:02d}_{sanitize_filename(label)}{fmt_suffix}"
            snip_path = snippet_dir / snip_name
            export_audio(
                cast(AudioSegment, final[start_ms:end_ms]),
                snip_path,
                source_subtype,
            )
            snippet_outputs.append(str(snip_path))

    report = build_setlist_report(
        app_name=SETLIST_TITLE,
        transition_mode=args.transition_mode,
        profile=scoring_profile.name if scoring_profile else None,
        scoring_config=args.scoring_config,
        vocal_method=args.vocal_method,
        output_file=str(out_path),
        output_duration_sec=round(len(final) / 1000.0, 3),
        output_duration_timestamp=format_timestamp(len(final) / 1000.0),
        final_gain_db=final_gain_db,
        wav_format_validation={"reference": infos[0], "tracks": infos},
        tracks=[asdict(t) for t in tracks],
        track_source_zero_in_mix_sec=[round(x, 3) for x in mix_plan.source_zero_in_mix_sec],
        transitions=[asdict(t) for t in transitions],
        transition_overrides=manifest_settings.get("transition_overrides"),
        snippet_files=snippet_outputs,
    )
    report_path = write_report(base, SETLIST_REPORT_SUFFIX, report)

    print(f"\n✅ Wrote continuous mix: {out_path}")
    print(f"   Duration: {format_timestamp(len(final) / 1000.0)} ({len(final)/1000.0:.2f}s)")
    if final_gain_db:
        print(f"   Final peak guard applied: {final_gain_db:+.2f} dB")
    print(f"📝 Wrote setlist report: {report_path}")
    if snippet_outputs:
        print(f"🎧 Wrote {len(snippet_outputs)} transition snippets.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=SETLIST_TITLE)
    p.add_argument("setlist", help="JSON manifest or text file with one audio path per line")
    p.add_argument(
        "-o",
        "--output",
        default=DEFAULT_SETLIST_OUTPUT,
        help="Output continuous mix (.wav or .mp3; MP3 exports at 320 kbps)",
    )
    p.add_argument(
        "--transition-mode",
        default="recommended",
        choices=list(TRANSITION_MODES),
        help="Which transition style to select at each junction",
    )
    p.add_argument(
        "--natural-pause-sec",
        type=float,
        default=1.0,
        help=f"Silence between tracks in natural mode, from 0 to {MAX_NATURAL_PAUSE_SEC:g}s (default: 1.0)",
    )
    p.add_argument("--make-snippets", action="store_true", help="Export transition audition snippets from the final continuous mix")
    p.add_argument(
        "--allow-outside-manifest-dir",
        action="store_true",
        help="Allow track paths (absolute, '..', or symlinked) that resolve outside the setlist's directory",
    )
    add_planning_arguments(p, vocal_default="heuristic")
    p.add_argument("--apply-manifest-settings", action="store_true", help="Apply optional settings object from JSON manifest")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    return run_cli(build_parser(), build_continuous_mix, argv)


if __name__ == "__main__":
    raise SystemExit(main())
