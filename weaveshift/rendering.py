"""Audio rendering and export helpers for WeaveShift."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple, cast

import numpy as np
from pydub import AudioSegment

from weaveshift.audio import (
    DEFAULT_MP3_BITRATE,
    SUPPORTED_MP3_SUFFIXES,
    NaturalTransition,
    SetlistTransition,
    TransitionCandidate,
    clamp,
)


def sec_to_ms(sec: float) -> int:
    """Convert seconds to milliseconds using round-half-up for splice/render parity."""
    return int(round(float(sec) * 1000.0))


@dataclass(frozen=True)
class TransitionAudio:
    """Rendered transition segments for a candidate.

    ``prefix`` is Track A before the fade start (empty when only the tail is needed).
    ``body`` is the overlap mix plus Track B continuation from the fade start onward.
    """

    prefix: AudioSegment
    body: AudioSegment

    @property
    def full(self) -> AudioSegment:
        return self.prefix + self.body


def conservative_sum_peak_db(a_peak_dbfs: float, b_peak_dbfs: float) -> float:
    """Conservative peak estimate for two overlaid signals."""
    a_amp = 10 ** (float(a_peak_dbfs) / 20.0) if a_peak_dbfs > -120 else 0.0
    b_amp = 10 ** (float(b_peak_dbfs) / 20.0) if b_peak_dbfs > -120 else 0.0
    return 20 * math.log10(max(a_amp + b_amp, 1e-9))


def _apply_envelope(
    seg: AudioSegment,
    build_env: Callable[[int, int], np.ndarray],
    fallback_gain_db: float,
) -> AudioSegment:
    """Apply a 1D per-sample amplitude envelope to an AudioSegment, preserving 24-bit width."""
    if len(seg) <= 0:
        return seg

    # Pydub can raise on get_array_of_samples() for 24-bit audio because Python's
    # array module has no 3-byte integer type. Upcast only for math, then restore.
    original_width = seg.sample_width
    math_seg = seg.set_sample_width(4) if original_width == 3 else seg

    try:
        samples = np.array(math_seg.get_array_of_samples())
    except Exception:
        return seg.apply_gain(fallback_gain_db)

    if samples.size == 0:
        return seg

    channels = max(1, int(math_seg.channels))
    try:
        samples_2d = samples.reshape((-1, channels)).astype(np.float64)
    except ValueError:
        return seg.apply_gain(fallback_gain_db)

    n = samples_2d.shape[0]
    if n <= 1:
        return seg.apply_gain(fallback_gain_db)

    dtype = samples.dtype
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        lo, hi = info.min, info.max
    else:
        lo, hi = -1.0, 1.0

    env = build_env(n, int(math_seg.frame_rate))
    out = np.clip(samples_2d * env[:, None], lo, hi)
    out = np.rint(out).astype(dtype) if np.issubdtype(dtype, np.integer) else out.astype(dtype)

    result = math_seg._spawn(out.reshape(-1).tobytes())
    return result.set_sample_width(original_width) if original_width == 3 else result


def apply_gain_ramp(seg: AudioSegment, start_gain_db: float, end_gain_db: float, step_ms: int = 20) -> AudioSegment:
    """Apply a sample-smooth linear gain ramp with NumPy (`step_ms` retained for signature compatibility)."""
    _ = step_ms
    start_amp = 10.0 ** (float(start_gain_db) / 20.0)
    end_amp = 10.0 ** (float(end_gain_db) / 20.0)
    return _apply_envelope(
        seg,
        lambda n, _sr: np.linspace(start_amp, end_amp, n, dtype=np.float64),
        fallback_gain_db=end_gain_db,
    )


def apply_soft_duck(seg: AudioSegment, duck_db: float, attack_ms: int = 250, release_ms: int = 700) -> AudioSegment:
    """Apply a gentle full-mix duck with a sample-smooth envelope."""
    if len(seg) <= 0 or duck_db >= -0.01:
        return seg

    def _duck_env(n: int, sr: int) -> np.ndarray:
        attack_n = min(n, max(1, int(sr * attack_ms / 1000.0)))
        release_n = min(max(1, int(sr * release_ms / 1000.0)), max(1, n - attack_n))
        hold_n = max(0, n - attack_n - release_n)
        duck_amp = 10.0 ** (float(duck_db) / 20.0)
        env = np.ones(n, dtype=np.float64)
        env[:attack_n] = np.linspace(1.0, duck_amp, attack_n, dtype=np.float64)
        if hold_n > 0:
            env[attack_n:attack_n + hold_n] = duck_amp
        env[attack_n + hold_n:] = np.linspace(duck_amp, 1.0, n - attack_n - hold_n, dtype=np.float64)
        return env

    return _apply_envelope(seg, _duck_env, fallback_gain_db=duck_db)


SUBTYPE_TO_WAV_CODEC = {
    "PCM_16": "pcm_s16le",
    "PCM_24": "pcm_s24le",
    "PCM_32": "pcm_s32le",
    "FLOAT": "pcm_f32le",
    "DOUBLE": "pcm_f64le",
}


def export_wav_matching_subtype(seg: AudioSegment, path: Path, source_subtype: Optional[str]) -> None:
    """Export a WAV file whose on-disk bit depth matches the validated source subtype."""
    path.parent.mkdir(parents=True, exist_ok=True)
    codec = SUBTYPE_TO_WAV_CODEC.get(source_subtype or "")
    if codec:
        seg.export(str(path), format="wav", parameters=["-acodec", codec])
    else:
        seg.export(str(path), format="wav")


def export_mp3(seg: AudioSegment, path: Path, *, bitrate: str = DEFAULT_MP3_BITRATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    seg.export(str(path), format="mp3", bitrate=bitrate)


def export_audio(
    seg: AudioSegment,
    path: Path,
    source_subtype: Optional[str],
    *,
    mp3_bitrate: str = DEFAULT_MP3_BITRATE,
) -> None:
    """Export WAV or MP3 based on the destination suffix."""
    if path.suffix.lower() in SUPPORTED_MP3_SUFFIXES:
        export_mp3(seg, path, bitrate=mp3_bitrate)
        return
    export_wav_matching_subtype(seg, path, source_subtype)


def build_transition_audio(seg_a: AudioSegment, seg_b: AudioSegment, cand: SetlistTransition) -> TransitionAudio:
    """Pure transition render shared by two-track export and setlist tail stitching."""
    if isinstance(cand, NaturalTransition):
        silence = AudioSegment.silent(duration=sec_to_ms(cand.pause_sec), frame_rate=seg_b.frame_rate)
        silence = silence.set_channels(seg_b.channels).set_sample_width(seg_b.sample_width)
        return TransitionAudio(prefix=seg_a, body=silence + seg_b)

    a_fade_start_ms = sec_to_ms(cand.a_fade_start_sec)
    a_cut_ms = sec_to_ms(cand.a_cut_sec)
    b_cue_ms = sec_to_ms(cand.b_cue_sec)
    is_handoff = cand.takeover_overlap_sec is not None
    overlap_sec = cand.takeover_overlap_sec if is_handoff and cand.takeover_overlap_sec is not None else cand.overlap_sec
    overlap_ms = sec_to_ms(overlap_sec)
    b_fade_in_ms = sec_to_ms(cand.b_fade_in_sec if cand.b_fade_in_sec is not None else overlap_sec)

    prefix = cast(AudioSegment, seg_a[:a_fade_start_ms])
    a_slice = cast(AudioSegment, seg_a[a_fade_start_ms:a_cut_ms])
    a_fade_out_ms = max(1, len(a_slice)) if is_handoff else overlap_ms
    raw_a_tail = cast(AudioSegment, a_slice.fade_out(a_fade_out_ms))
    b_after_cue = cast(AudioSegment, seg_b[b_cue_ms:])
    raw_b_after_cue = cast(AudioSegment, b_after_cue.apply_gain(cand.b_gain_db))
    b_start_offset_ms = max(0, a_cut_ms - overlap_ms - a_fade_start_ms) if is_handoff else 0
    b_head_len_ms = max(overlap_ms, b_fade_in_ms)
    b_head_slice = cast(AudioSegment, raw_b_after_cue[:b_head_len_ms])
    raw_b_head = cast(AudioSegment, b_head_slice.fade_in(b_fade_in_ms))

    if cand.soft_duck_db < -0.01:
        if cand.soft_duck_target == "a":
            raw_a_tail = apply_soft_duck(raw_a_tail, cand.soft_duck_db)
        elif cand.soft_duck_target == "b":
            raw_b_head = apply_soft_duck(raw_b_head, cand.soft_duck_db)

    predicted_overlap_peak = conservative_sum_peak_db(raw_a_tail.max_dBFS, raw_b_head.max_dBFS)
    transition_safety_pad_db = clamp(-1.0 - predicted_overlap_peak, -6.0, 0.0)

    a_tail = raw_a_tail.apply_gain(transition_safety_pad_db)
    b_head = raw_b_head.apply_gain(transition_safety_pad_db)
    mixed_base = a_tail
    needed_mixed_len_ms = b_start_offset_ms + len(b_head)
    if needed_mixed_len_ms > len(mixed_base):
        pad = AudioSegment.silent(
            duration=needed_mixed_len_ms - len(mixed_base),
            frame_rate=mixed_base.frame_rate,
        ).set_channels(mixed_base.channels).set_sample_width(mixed_base.sample_width)
        mixed_base += pad
    mixed = mixed_base.overlay(b_head, position=b_start_offset_ms)

    recovery_ms = min(2000, max(0, len(raw_b_after_cue) - b_head_len_ms))
    b_remaining = cast(AudioSegment, raw_b_after_cue[b_head_len_ms:])
    if transition_safety_pad_db < -0.01 and recovery_ms > 0:
        b_recovery = apply_gain_ramp(cast(AudioSegment, b_remaining[:recovery_ms]), transition_safety_pad_db, 0.0)
        b_continuation = b_recovery + cast(AudioSegment, b_remaining[recovery_ms:])
    else:
        b_continuation = b_remaining

    return TransitionAudio(prefix=prefix, body=mixed + b_continuation)


def apply_final_peak_guard(seg: AudioSegment, target_dbfs: float = -1.0) -> AudioSegment:
    """Whole-segment peak guard that avoids splice discontinuities."""
    if seg.max_dBFS > target_dbfs:
        return seg.apply_gain(target_dbfs - seg.max_dBFS)
    return seg


def snippet_window_ms(cand: SetlistTransition, total_len_ms: int) -> Tuple[int, int]:
    """Return audition snippet bounds around a transition."""
    a_fade_start_ms = sec_to_ms(cand.a_fade_start_sec)
    overlap_ms = sec_to_ms(cand.overlap_sec)
    snip_start = max(0, a_fade_start_ms - 12000)
    snip_end = min(total_len_ms, a_fade_start_ms + overlap_ms + 18000)
    return snip_start, snip_end


def render_transition_tail(seg_a: AudioSegment, seg_b: AudioSegment, cand: SetlistTransition) -> AudioSegment:
    """Return only the transition body from Track A fade start onward."""
    return build_transition_audio(seg_a, seg_b, cand).body


def render_candidate(
    seg_a: AudioSegment,
    seg_b: AudioSegment,
    cand: TransitionCandidate,
    output_path: Path,
    snippet_path: Optional[Path] = None,
    output_subtype: Optional[str] = None,
) -> None:
    transition = build_transition_audio(seg_a, seg_b, cand)
    final = apply_final_peak_guard(transition.full)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_audio(final, output_path, output_subtype)

    if snippet_path:
        snip_start, snip_end = snippet_window_ms(cand, len(final))
        export_audio(cast(AudioSegment, final[snip_start:snip_end]), snippet_path, output_subtype)
