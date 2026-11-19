#!/usr/bin/env python3
"""WeaveShift two-track CLI: render and rank transitions between two tracks."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Any, List, Optional

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
from weaveshift.audio import format_timestamp, resolve_wav_export_subtype, warn_if_vocals_skipped
from weaveshift.plan import plan_two_track_mix
from weaveshift.rendering import render_candidate
from weaveshift.reports import build_two_track_report

DEFAULT_TWO_TRACK_OUTPUT = "weaveshift_transition.wav"
TWO_TRACK_REPORT_SUFFIX = ".weaveshift_mix_report.json"

TWO_TRACK_MODES = ("all", *CANDIDATE_MODES)


def run(args) -> None:
    _, base, fmt_suffix = resolve_output(args.output)
    warn_unused_profile(args, args.mode, "--mode")
    warn_if_vocals_skipped(args.vocal_method)
    mix_plan = plan_two_track_mix(
        args.track_a,
        args.track_b,
        mode=args.mode,
        profile=args.profile,
        scoring_config=args.scoring_config,
        search_a_sec=args.search_a_sec,
        search_b_sec=args.search_b_sec,
        max_trim_a_sec=args.max_trim_a_sec,
        b_cue_max_sec=args.b_cue_max_sec,
        a_bpm=args.a_bpm,
        b_bpm=args.b_bpm,
        a_key=args.a_key,
        b_key=args.b_key,
        vocal_method=args.vocal_method,
        prefer_mps=not args.no_mps,
    )

    a_wav_info = mix_plan.track_a_wav_info
    b_wav_info = mix_plan.track_b_wav_info
    seg_a = mix_plan.seg_a
    seg_b = mix_plan.seg_b
    candidates = mix_plan.candidates
    ranked = mix_plan.ranked
    recommended = mix_plan.recommended_name
    scoring_profile = mix_plan.scoring_profile

    print("Preserving source timing/pitch; output format follows the destination suffix (.wav or .mp3).")
    print(f"Track A format: {a_wav_info['samplerate']} Hz, {a_wav_info['channels']} ch, {a_wav_info['subtype']}")
    print(f"Track B format: {b_wav_info['samplerate']} Hz, {b_wav_info['channels']} ch, {b_wav_info['subtype']}")
    if scoring_profile is not None:
        print(f"Using scoring profile: {scoring_profile.name} - {scoring_profile.description}")
    report_widened_windows(args, mix_plan)

    if args.mode != "all" and len(candidates) == 1 and candidates[0].name != args.mode:
        print(
            f"Warning: requested mode '{args.mode}' was not generated as a distinct candidate; "
            f"falling back to '{candidates[0].name}'.",
            file=sys.stderr,
        )

    wav_export_subtype = resolve_wav_export_subtype([a_wav_info, b_wav_info])

    outputs = []
    for cand in candidates:
        # Build names by concatenation, not with_suffix(): a dotted output stem
        # such as "mix v1.2.wav" must not lose part of its name to suffix replacement.
        out = base.with_name(f"{base.name}_{cand.name}{fmt_suffix}")
        snippet = base.with_name(f"{base.name}_{cand.name}_snippet{fmt_suffix}") if args.make_snippets else None
        print(
            f"Rendering {cand.name}: score={cand.score:.3f}, new song starts at {format_timestamp(cand.a_fade_start_sec)} "
            f"({cand.a_fade_start_sec:.2f}s), A fade {cand.a_fade_start_sec:.2f}s -> cut {cand.a_cut_sec:.2f}s, "
            f"B cue {format_timestamp(cand.b_cue_sec)} ({cand.b_cue_sec:.2f}s), overlap {cand.overlap_sec:.1f}s, "
            f"B gain {cand.b_gain_db:+.1f} dB"
        )
        render_candidate(
            seg_a,
            seg_b,
            cand,
            out,
            snippet,
            output_subtype=wav_export_subtype,
        )
        outputs.append(
            {
                "name": cand.name,
                "file": str(out),
                "snippet": str(snippet) if snippet else None,
                "candidate": asdict(cand),
            }
        )

    outputs_by_name = {o["name"]: o for o in outputs}
    print_rankings(ranked, outputs_by_name)
    if recommended:
        print(f"\nRecommended first audition: {recommended}")

    report = build_two_track_report(
        app_name=APP_NAME,
        mode=args.mode,
        recommended=recommended,
        ranked=ranked,
        track_a=mix_plan.track_a_analysis,
        track_b=mix_plan.track_b_analysis,
        outputs=outputs,
        wav_format_validation={"track_a": a_wav_info, "track_b": b_wav_info},
        profile=scoring_profile.name if scoring_profile else None,
        scoring_config=args.scoring_config,
    )
    report_path = write_report(base, TWO_TRACK_REPORT_SUFFIX, report)
    print(f"Wrote ranked mix report: {report_path}")


def print_rankings(ranked: list[dict[str, Any]], outputs_by_name: dict[str, dict[str, Any]]) -> None:
    print("\nRanked transition recommendations, best to worst:")
    for item in ranked:
        out = outputs_by_name.get(item["name"], {})
        file_part = f" -> {out.get('file')}" if out.get("file") else ""
        t = item["transition"]
        start_ts = t.get("new_song_starts_in_mix_timestamp", format_timestamp(t["a_fade_start_sec"]))
        cue_ts = t.get("track_b_source_cue_timestamp", format_timestamp(t["b_cue_sec"]))
        print(
            f"  #{item['rank']} {item['name']}: score={item['score']:.3f}; "
            f"new song starts in mix at {start_ts} ({t['a_fade_start_sec']:.2f}s), "
            f"using Track B cue {cue_ts} ({t['b_cue_sec']:.2f}s); "
            f"A fade {t['a_fade_start_sec']:.2f}s -> cut {t['a_cut_sec']:.2f}s, "
            f"overlap {t['overlap_sec']:.1f}s, B gain {t['b_gain_db']:+.1f} dB{file_part}"
        )
        print(f"     {item['verdict']}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=f"{APP_NAME}: two-track mixing with ranked transition recommendations"
    )
    p.add_argument("track_a", help="Outgoing Track A audio file (.wav or .mp3)")
    p.add_argument("track_b", help="Incoming Track B audio file (.wav or .mp3)")
    p.add_argument(
        "-o",
        "--output",
        default=DEFAULT_TWO_TRACK_OUTPUT,
        help="Output base path (.wav or .mp3; MP3 exports at 320 kbps). Candidate name is appended.",
    )
    p.add_argument(
        "--mode",
        "--transition-mode",
        dest="mode",
        choices=list(TWO_TRACK_MODES),
        default="all",
        help="Which candidate(s) to render",
    )
    p.add_argument("--make-snippets", action="store_true", help="Also export short audition snippets around each transition")
    add_planning_arguments(p, vocal_default="auto")
    p.add_argument("--a-bpm", type=float, default=None, help="Manual Track A BPM override")
    p.add_argument("--b-bpm", type=float, default=None, help="Manual Track B BPM override")
    p.add_argument("--a-key", type=str, default=None, help="Manual Track A key override, e.g. 'E major'")
    p.add_argument("--b-key", type=str, default=None, help="Manual Track B key override, e.g. 'A minor'")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    return run_cli(build_parser(), run, argv)


if __name__ == "__main__":
    raise SystemExit(main())
