#!/usr/bin/env python3
"""FlowMix two-track CLI and backward-compatible public facade."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from pydub import AudioSegment

from flowmix_profiles import load_scoring_profile
from flowmix_audio import (
    AudioAnalysis,
    LOSSLESS_WAV_SUBTYPES,
    MAX_OVERLAP_SEC,
    OVERLAP_LENGTHS,
    SUPPORTED_WAV_SUFFIXES,
    TransitionCandidate,
    VocalSegment,
    analyze_audio,
    analyze_vocals,
    camelot_compat,
    clamp,
    compute_local_energy_curve,
    compute_local_lufs_curve,
    dbfs_from_samples,
    energy_db_from_curve,
    estimate_key,
    format_timestamp,
    get_torch_device,
    librosa,
    lufs_from_curve,
    overlap_vocal_collision,
    parse_manual_key,
    pydub_rms_dbfs,
    safe_float,
    segments_from_activity,
    stereo_dbfs_from_samples,
    validate_loaded_segment_parity,
    validate_audio_input,
    validate_audio_output,
    validate_audio_pair_compatibility,
    validate_wav_input,
    validate_wav_output,
    validate_wav_pair_compatibility,
    output_format_suffix,
    resolve_wav_export_subtype,
    vocal_active_fraction,
    vocal_segments_demucs,
    vocal_segments_heuristic,
    wav_info,
)
from flowmix_scoring import (
    build_candidate_times,
    candidate_too_close,
    choose_candidates,
    nearest_distance,
    profile_candidate_score,
    profile_search_parameters,
    score_candidate,
    smooth_candidate_score,
)
from flowmix_rendering import (
    SUBTYPE_TO_WAV_CODEC,
    apply_gain_ramp,
    apply_soft_duck,
    conservative_sum_peak_db,
    export_audio,
    export_wav_matching_subtype,
    render_candidate,
)
from flowmix_plan import plan_two_track_mix
from flowmix_reports import (
    build_two_track_report,
    candidate_verdict,
    ranked_candidate_summary,
    serialize_analysis,
)

APP_VERSION = "1.0.2"
APP_NAME = f"FlowMix {APP_VERSION}"


def run(args) -> None:
    out_base = validate_audio_output(args.output)
    base = out_base.with_suffix("")
    fmt_suffix = output_format_suffix(out_base)

    if args.mode != "profile" and (args.scoring_config is not None or args.profile != "edm"):
        print(
            "Warning: --profile/--scoring-config are only used with --mode profile; ignoring profile settings for this run.",
            file=sys.stderr,
        )

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
    if mix_plan.effective_search_a_sec > float(args.search_a_sec):
        print(
            f"Note: widened Track A analysis window from {args.search_a_sec:.1f}s to {mix_plan.effective_search_a_sec:.1f}s so local RMS covers max trim + overlap."
        )
    if mix_plan.effective_search_b_sec > float(args.search_b_sec):
        print(
            f"Note: widened Track B analysis window from {args.search_b_sec:.1f}s to {mix_plan.effective_search_b_sec:.1f}s so local RMS covers cue search + overlap."
        )

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
            output_subtype=wav_export_subtype if fmt_suffix == ".wav" else None,
        )
        outputs.append({"name": cand.name, "file": str(out), "snippet": str(snippet) if snippet else None, "candidate": asdict(cand)})

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
        profile=getattr(scoring_profile, "name", None),
        scoring_config=args.scoring_config,
    )
    report_path = base.with_name(base.name + ".1_0_0_wav_mix_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote 1.0.0 WAV ranked report: {report_path}")


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
        description="FlowMix 1.0.2: two-track mixing with ranked transition recommendations"
    )
    p.add_argument("track_a", help="Outgoing Track A audio file (.wav or .mp3)")
    p.add_argument("track_b", help="Incoming Track B audio file (.wav or .mp3)")
    p.add_argument(
        "-o",
        "--output",
        default="flowmix_1_0_0.wav",
        help="Output base path (.wav or .mp3; MP3 exports at 320 kbps). Candidate name is appended.",
    )
    p.add_argument("--mode", choices=["all", "vocal_safe", "beat_aligned", "quick_cut", "smooth", "profile", "vocal_ducked", "long_blend"], default="all", help="Which candidate(s) to render")
    p.add_argument("--profile", default="edm", help="Built-in scoring profile to use when --mode profile is selected")
    p.add_argument("--scoring-config", default=None, help="Path to a TOML scoring profile; overrides --profile")
    p.add_argument("--make-snippets", action="store_true", help="Also export short audition snippets around each transition")
    p.add_argument("--vocal-method", choices=["auto", "demucs", "heuristic", "none"], default="auto", help="Vocal analysis method. 'auto' tries Demucs then fallback.")
    p.add_argument("--no-mps", action="store_true", help="Do not try Apple Silicon MPS for Demucs; use CUDA if available or CPU")
    p.add_argument("--search-a-sec", type=float, default=35.0, help="Analyze this many seconds from the end of Track A")
    p.add_argument("--search-b-sec", type=float, default=35.0, help="Analyze this many seconds from the beginning of Track B")
    p.add_argument("--max-trim-a-sec", type=float, default=8.0, help="Maximum seconds allowed to trim from Track A tail")
    p.add_argument("--b-cue-max-sec", type=float, default=10.0, help="Latest allowed cue point in Track B intro")
    p.add_argument("--a-bpm", type=float, default=None, help="Manual Track A BPM override")
    p.add_argument("--b-bpm", type=float, default=None, help="Manual Track B BPM override")
    p.add_argument("--a-key", type=str, default=None, help="Manual Track A key override, e.g. 'E major'")
    p.add_argument("--b-key", type=str, default=None, help="Manual Track B key override, e.g. 'A minor'")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    if ns.vocal_method == "none":
        ns.vocal_method = "heuristic"
    try:
        run(ns)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
