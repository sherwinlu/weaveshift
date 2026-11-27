"""Streaming broadcast audio QA: ITU-R BS.1770-4 LUFS, 4x True Peak, and splice-click diagnostics.

Provides bounded-memory post-render verification (~25 MB peak RAM even on multi-hour 48 kHz
mixes) using NumPy, SciPy, and SoundFile, with an FFmpeg float32 decode fallback for compressed
formats when libsndfile cannot open them directly.

Credits & Attribution:
    Clean-room implementation adapted from ``mvdirector`` (``mvdirector.audio.broadcast_qa``),
    originally inspired by ``xenoform`` (``src/xenoform/media_qa.py``).
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
import subprocess
import tempfile
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import lfilter
import soundfile as sf

from weaveshift.audio import AudioAnalysis, NaturalTransition, SetlistTransition

logger = logging.getLogger(__name__)

TRUE_PEAK_WARN_DBTP = -0.5
_TRUE_PEAK_HALF_TAPS = 6
_TRUE_PEAK_OVERLAP = _TRUE_PEAK_HALF_TAPS * 2  # full 12-tap kernel width across chunks
_DEFAULT_CHUNK_SECONDS = 10.0
_MAX_OVERSAMPLE = 16

_BiquadCoeffs = Tuple[float, float, float, float, float]


@dataclass(frozen=True)
class BroadcastAudioReport:
    """Measured integrated loudness, 4x true-peak headroom, and splice-click anomalies."""

    integrated_lufs: float
    true_peak_dbtp: float
    splice_clicks: List[float] = field(default_factory=list)


def _sanitize_audio(samples: np.ndarray, *, copy: bool = True) -> np.ndarray:
    arr = np.asarray(samples, dtype=np.float64)
    return np.nan_to_num(arr, copy=copy, nan=0.0, posinf=0.0, neginf=0.0)


def _as_2d_audio(samples: np.ndarray) -> np.ndarray:
    arr = _sanitize_audio(samples)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Expected 1-D or 2-D audio array, got shape {arr.shape}.")


def _k_weight_coeffs(sr: int) -> Tuple[_BiquadCoeffs, _BiquadCoeffs]:
    """Compute ITU-R BS.1770-4 K-weighting stage-1 high-shelf and stage-2 RLB HPF biquads."""
    # Stage 1: pre-filter high-shelf (+3.9998 dB at 1681.97 Hz, Q=0.7072).
    db = 3.999843853973347
    f0 = 1681.974450955533
    q = 0.7071752369554196
    k = math.tan(math.pi * f0 / sr)
    vh = 10.0 ** (db / 20.0)
    vb = vh ** 0.4996667741545416
    a0 = 1.0 + k / q + k * k
    s1: _BiquadCoeffs = (
        (vh + vb * k / q + k * k) / a0,
        2.0 * (k * k - vh) / a0,
        (vh - vb * k / q + k * k) / a0,
        2.0 * (k * k - 1.0) / a0,
        (1.0 - k / q + k * k) / a0,
    )

    # Stage 2: RLB high-pass filter (f0=38.1355 Hz, Q=0.5003).
    f0_hp = 38.13547087602444
    q_hp = 0.5003270373238773
    k_hp = math.tan(math.pi * f0_hp / sr)
    a0_hp = 1.0 + k_hp / q_hp + k_hp * k_hp
    s2: _BiquadCoeffs = (
        1.0 / a0_hp,
        -2.0 / a0_hp,
        1.0 / a0_hp,
        2.0 * (k_hp * k_hp - 1.0) / a0_hp,
        (1.0 - k_hp / q_hp + k_hp * k_hp) / a0_hp,
    )
    return s1, s2


def _k_weight(samples: np.ndarray, sr: int) -> np.ndarray:
    """Apply BS.1770-4 K-weighting to ``(N, C)`` float64 audio."""
    if sr <= 3400:
        # Stage-1 shelf (1682 Hz) requires Nyquist > 1682 Hz; skip at ultra-low sample rates.
        return np.asarray(samples, dtype=np.float64)
    s1, s2 = _k_weight_coeffs(sr)
    b1, a1 = [s1[0], s1[1], s1[2]], [1.0, s1[3], s1[4]]
    b2, a2 = [s2[0], s2[1], s2[2]], [1.0, s2[3], s2[4]]
    out = np.empty_like(samples, dtype=np.float64)
    for ch in range(samples.shape[1]):
        y = lfilter(b1, a1, samples[:, ch])
        out[:, ch] = lfilter(b2, a2, y)
    return out


def _channel_weights(n_channels: int) -> np.ndarray:
    """ITU-R BS.1770-4 channel weights (L/R/C=1.0, LFE=0.0, Ls/Rs=1.41 for 5.1)."""
    if n_channels >= 6:
        weights = [1.0, 1.0, 1.0, 0.0, 1.41, 1.41] + [1.0] * (n_channels - 6)
    elif n_channels == 5:
        weights = [1.0, 1.0, 1.0, 1.41, 1.41]
    else:
        weights = [1.0] * n_channels
    return np.array(weights, dtype=np.float64)


def _gated_lufs_from_block_powers(block_powers: np.ndarray) -> float:
    """Apply ITU-R BS.1770-4 absolute (-70 LUFS) and relative (-10 LU) gates."""
    if block_powers.size == 0:
        return -70.0
    abs_thresh = 10.0 ** ((-70.0 + 0.691) / 10.0)
    above_abs = block_powers[block_powers > abs_thresh]
    if above_abs.size == 0:
        return -70.0
    ungated_mean = float(np.mean(above_abs))
    rel_thresh = ungated_mean * (10.0 ** (-10.0 / 10.0))
    gated = above_abs[above_abs > rel_thresh]
    if gated.size == 0:
        return -70.0
    return float(-0.691 + 10.0 * math.log10(float(np.mean(gated))))


def integrated_lufs(samples: np.ndarray, sr: int) -> float:
    """Compute ITU-R BS.1770-4 gated integrated loudness in LUFS.

    Returns ``-70.0`` for silent or ultra-short (<400 ms) signals.
    """
    arr = _as_2d_audio(samples)
    if arr.size == 0 or sr <= 0:
        return -70.0

    ch_weights = _channel_weights(arr.shape[1])
    block_len = max(1, int(round(0.400 * sr)))
    hop_len = max(1, int(round(0.100 * sr)))
    if arr.shape[0] < block_len:
        power = float(np.sum(np.mean(arr * arr, axis=0) * ch_weights))
        return -70.0 if power <= 1e-12 else max(-70.0, -0.691 + 10.0 * math.log10(power))

    weighted = _k_weight(arr, sr)
    n_blocks = 1 + (weighted.shape[0] - block_len) // hop_len
    block_powers = np.empty(n_blocks, dtype=np.float64)
    for idx in range(n_blocks):
        seg = weighted[idx * hop_len : idx * hop_len + block_len]
        block_powers[idx] = float(np.sum(np.mean(seg * seg, axis=0) * ch_weights))

    return _gated_lufs_from_block_powers(block_powers)


def _true_peak_kernels(oversample: int) -> List[np.ndarray]:
    factor = min(_MAX_OVERSAMPLE, max(1, int(oversample)))
    if factor == 1:
        return []
    taps = np.arange(-_TRUE_PEAK_HALF_TAPS, _TRUE_PEAK_HALF_TAPS, dtype=np.float64)
    win = 0.5 * (1.0 + np.cos(math.pi * taps / _TRUE_PEAK_HALF_TAPS))
    kernels: List[np.ndarray] = []
    for phase in range(1, factor):
        offset = phase / factor
        k = np.sinc(taps - offset) * win
        k /= np.sum(k)
        kernels.append(k)
    return kernels


def _max_oversampled_peak(arr: np.ndarray, kernels: Sequence[np.ndarray]) -> float:
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if not kernels or arr.shape[0] < 8:
        return peak
    for ch in range(arr.shape[1]):
        col = arr[:, ch]
        for kernel in kernels:
            interp = np.convolve(col, kernel, mode="same")
            peak = max(peak, float(np.max(np.abs(interp))))
    return peak


def true_peak_dbtp(samples: np.ndarray, *, oversample: int = 4) -> float:
    """Compute 4x-oversampled True Peak in dBTP using Hann-windowed sinc interpolation."""
    arr = _as_2d_audio(samples)
    if arr.size == 0:
        return -120.0

    kernels = _true_peak_kernels(oversample)
    peak = _max_oversampled_peak(arr, kernels)
    return -120.0 if peak <= 1e-12 else float(20.0 * math.log10(peak))


def _near_any_onset(t_sec: float, sorted_onsets: Sequence[float], tolerance: float) -> bool:
    """O(log N) proximity check against a sorted onset list."""
    if not sorted_onsets or tolerance < 0.0:
        return False
    pos = bisect_left(sorted_onsets, t_sec)
    if pos < len(sorted_onsets) and abs(sorted_onsets[pos] - t_sec) <= tolerance:
        return True
    if pos > 0 and abs(t_sec - sorted_onsets[pos - 1]) <= tolerance:
        return True
    return False


def scan_splice_clicks(
    samples: np.ndarray,
    sr: int,
    *,
    boundary_times: Optional[Sequence[float]] = None,
    drum_onset_times: Sequence[float] = (),
    spike_ratio: float = 8.0,
    min_hpf_peak: float = 0.02,
    onset_exclude_seconds: float = 0.035,
) -> List[float]:
    """Detect sharp audio splice clicks while excluding legitimate musical drum attacks.

    Applies a second-difference high-frequency discontinuity emphasis, compares impulse
    spikes against a 10 ms local median floor, verifies the spike lacks a sustained 10 ms
    musical tail, and ignores spikes near known beat/drum onsets.
    """
    arr = _as_2d_audio(samples)
    wav = arr.mean(axis=1)
    if wav.size < 64 or sr <= 0:
        return []

    hpf = np.abs(np.diff(wav, n=2, prepend=wav[:2]))
    win_ms = max(8, int(round(0.010 * sr)))
    safe_onsets = sorted(float(t) for t in drum_onset_times if math.isfinite(float(t)))
    clicks: List[float] = []

    if boundary_times is not None:
        sorted_bounds = sorted(float(t) for t in boundary_times if math.isfinite(float(t)) and float(t) > 0.0)
        search_windows = [
            (
                max(win_ms, int(round((t - 0.025) * sr))),
                min(wav.size - win_ms, int(round((t + 0.025) * sr))),
            )
            for t in sorted_bounds
        ]
    else:
        step = max(win_ms, int(round(0.020 * sr)))
        search_windows = [
            (i, min(wav.size - win_ms, i + step)) for i in range(win_ms, wav.size - win_ms, step)
        ]

    for lo, hi in search_windows:
        if hi <= lo:
            continue
        local = hpf[lo:hi]
        rel_idx = int(np.argmax(local))
        peak = float(local[rel_idx])
        if peak < min_hpf_peak:
            continue
        idx = lo + rel_idx
        t_sec = idx / sr
        if _near_any_onset(t_sec, safe_onsets, onset_exclude_seconds):
            continue
        pre = hpf[max(0, idx - win_ms) : max(1, idx - 2)]
        post = hpf[min(wav.size - 1, idx + 3) : min(wav.size, idx + win_ms)]
        if pre.size == 0 or post.size == 0:
            continue
        floor = max(1e-5, float(np.median(np.concatenate([pre, post]))))
        tail_rms = float(np.sqrt(np.mean(post * post)))
        if (
            peak / floor >= spike_ratio
            and tail_rms < 0.35 * peak
            and (not clicks or t_sec - clicks[-1] > 0.05)
        ):
            clicks.append(round(t_sec, 4))
    return clicks


def _decode_to_float32_wav(audio_path: Path, dst_wav: Path) -> Path:
    """Decode *audio_path* to an unclipped 32-bit float WAV for streaming QA."""
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-protocol_whitelist",
        "file",
        "-i",
        str(audio_path.resolve()),
        "-vn",
        "-c:a",
        "pcm_f32le",
        str(dst_wav.resolve()),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=300)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg was not found on PATH; cannot decode compressed audio for broadcast QA."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffmpeg timed out decoding {audio_path.name} for broadcast QA") from exc
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"ffmpeg decode failed for {audio_path.name}")
    return dst_wav


def _analyze_broadcast_soundfile(
    audio_path: Path,
    *,
    boundary_times: Optional[Sequence[float]] = None,
    drum_onset_times: Sequence[float] = (),
    chunk_seconds: float = _DEFAULT_CHUNK_SECONDS,
) -> BroadcastAudioReport:
    safe_boundaries: Optional[List[float]] = (
        sorted(float(t) for t in boundary_times if math.isfinite(float(t)) and float(t) > 0.0)
        if boundary_times is not None
        else None
    )
    safe_onsets = sorted(float(t) for t in drum_onset_times if math.isfinite(float(t)))

    with sf.SoundFile(str(audio_path)) as handle:
        sr = int(handle.samplerate)
        n_channels = int(handle.channels)
        if sr <= 0 or n_channels <= 0:
            return BroadcastAudioReport(integrated_lufs=-70.0, true_peak_dbtp=-120.0)

        ch_weights = _channel_weights(n_channels)
        block_len = max(1, int(round(0.400 * sr)))
        hop_len = max(1, int(round(0.100 * sr)))
        safe_chunk_sec = (
            min(300.0, max(1.0, float(chunk_seconds)))
            if math.isfinite(float(chunk_seconds)) and float(chunk_seconds) > 0.0
            else _DEFAULT_CHUNK_SECONDS
        )
        chunk_frames = max(block_len * 4, int(round(safe_chunk_sec * sr)))
        kernels = _true_peak_kernels(4)
        win_ms = max(8, int(round(0.010 * sr)))
        click_overlap = max(win_ms + 4, int(round(0.050 * sr)))

        use_k = sr > 3400
        if use_k:
            s1, s2 = _k_weight_coeffs(sr)
            b1, a1 = [s1[0], s1[1], s1[2]], [1.0, s1[3], s1[4]]
            b2, a2 = [s2[0], s2[1], s2[2]], [1.0, s2[3], s2[4]]
        else:
            b1 = a1 = b2 = a2 = [1.0]
        zi1 = [np.zeros(2, dtype=np.float64) for _ in range(n_channels)]
        zi2 = [np.zeros(2, dtype=np.float64) for _ in range(n_channels)]

        k_carry = np.empty((0, n_channels), dtype=np.float64)
        tp_carry = np.empty((0, n_channels), dtype=np.float64)
        click_carry = np.empty(0, dtype=np.float64)
        block_powers: List[float] = []
        short_sq_sum = np.zeros(n_channels, dtype=np.float64)
        total_frames = 0
        max_peak = 0.0
        clicks: List[float] = []

        while True:
            raw_chunk = handle.read(chunk_frames, dtype="float64", always_2d=True)
            if raw_chunk.size == 0:
                break
            chunk = _sanitize_audio(raw_chunk, copy=False)
            chunk_start_sample = total_frames
            total_frames += chunk.shape[0]
            short_sq_sum += np.sum(chunk * chunk, axis=0)

            # 1. Streaming 4x True Peak with full kernel-width overlap across chunks.
            tp_buf = np.concatenate([tp_carry, chunk], axis=0) if tp_carry.size else chunk
            max_peak = max(max_peak, _max_oversampled_peak(tp_buf, kernels))
            tp_carry = (
                chunk[-_TRUE_PEAK_OVERLAP:].copy()
                if chunk.shape[0] >= _TRUE_PEAK_OVERLAP
                else tp_buf[-_TRUE_PEAK_OVERLAP:].copy()
            )
            del tp_buf

            # 2. Stateful streaming BS.1770-4 K-weighting + 400 ms / 100 ms blocks.
            if use_k:
                weighted = np.empty_like(chunk, dtype=np.float64)
                for ch in range(n_channels):
                    y, zi1[ch] = lfilter(b1, a1, chunk[:, ch], zi=zi1[ch])
                    weighted[:, ch], zi2[ch] = lfilter(b2, a2, y, zi=zi2[ch])
                    del y
            else:
                weighted = chunk

            k_buf = np.concatenate([k_carry, weighted], axis=0) if k_carry.size else weighted
            del weighted
            if k_buf.shape[0] >= block_len:
                n_blk = 1 + (k_buf.shape[0] - block_len) // hop_len
                for idx in range(n_blk):
                    seg = k_buf[idx * hop_len : idx * hop_len + block_len]
                    block_powers.append(float(np.sum(np.mean(seg * seg, axis=0) * ch_weights)))
                k_carry = k_buf[n_blk * hop_len :].copy()
            else:
                k_carry = k_buf.copy()
            del k_buf

            # 3. Streaming splice-click scan with 50 ms overlap across chunks.
            mono = chunk.mean(axis=1)
            overlap_len = click_carry.size
            c_buf = np.concatenate([click_carry, mono]) if overlap_len else mono
            buf_start_sec = (chunk_start_sample - overlap_len) / sr
            buf_end_sec = (chunk_start_sample + chunk.shape[0]) / sr
            if safe_boundaries is not None:
                local_bounds: Optional[List[float]] = [
                    t - buf_start_sec for t in safe_boundaries if buf_start_sec < t <= buf_end_sec
                ]
            else:
                local_bounds = None
            lo_on = bisect_left(safe_onsets, buf_start_sec - 0.05)
            hi_on = bisect_left(safe_onsets, buf_end_sec + 0.05)
            local_onsets = [t - buf_start_sec for t in safe_onsets[lo_on:hi_on]]
            if local_bounds is None or local_bounds:
                local_clicks = scan_splice_clicks(
                    c_buf,
                    sr,
                    boundary_times=local_bounds,
                    drum_onset_times=local_onsets,
                )
                for rel_t in local_clicks:
                    abs_t = round(buf_start_sec + rel_t, 4)
                    if abs_t > 0.0 and (not clicks or abs_t - clicks[-1] > 0.05):
                        clicks.append(abs_t)
            click_carry = (
                mono[-click_overlap:].copy()
                if mono.size >= click_overlap
                else c_buf[-click_overlap:].copy()
            )
            del mono, c_buf

    if total_frames == 0:
        lufs = -70.0
    elif total_frames < block_len:
        power = float(np.sum((short_sq_sum / total_frames) * ch_weights))
        lufs = -70.0 if power <= 1e-12 else max(-70.0, -0.691 + 10.0 * math.log10(power))
    else:
        lufs = _gated_lufs_from_block_powers(np.array(block_powers, dtype=np.float64))

    tp_db = -120.0 if max_peak <= 1e-12 else float(20.0 * math.log10(max_peak))
    return BroadcastAudioReport(
        integrated_lufs=round(lufs, 2),
        true_peak_dbtp=round(tp_db, 2),
        splice_clicks=clicks,
    )


def analyze_broadcast_audio(
    audio_path: Path,
    *,
    boundary_times: Optional[Sequence[float]] = None,
    drum_onset_times: Sequence[float] = (),
    chunk_seconds: float = _DEFAULT_CHUNK_SECONDS,
) -> BroadcastAudioReport:
    """Measure LUFS, 4x True Peak, and splice clicks on an audio file in bounded-memory chunks."""
    resolved = Path(audio_path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Audio file not found for broadcast QA: {audio_path}")
    try:
        return _analyze_broadcast_soundfile(
            resolved,
            boundary_times=boundary_times,
            drum_onset_times=drum_onset_times,
            chunk_seconds=chunk_seconds,
        )
    except sf.LibsndfileError:
        with tempfile.TemporaryDirectory(prefix="weaveshift_bqa_") as tmp:
            wav_path = _decode_to_float32_wav(resolved, Path(tmp) / "audio.wav")
            return _analyze_broadcast_soundfile(
                wav_path,
                boundary_times=boundary_times,
                drum_onset_times=drum_onset_times,
                chunk_seconds=chunk_seconds,
            )


def format_broadcast_qa_warnings(
    report: BroadcastAudioReport,
    *,
    true_peak_warn_dbtp: float = TRUE_PEAK_WARN_DBTP,
) -> List[str]:
    """Return human-readable warnings when headroom or splice checks fail."""
    warnings: List[str] = []
    if math.isfinite(report.true_peak_dbtp) and report.true_peak_dbtp > true_peak_warn_dbtp:
        warnings.append(
            f"Broadcast QA: True Peak {report.true_peak_dbtp:.2f} dBTP exceeds "
            f"{true_peak_warn_dbtp:.1f} dBTP headroom ceiling (inter-sample clipping risk)."
        )
    finite_clicks = [t for t in report.splice_clicks if math.isfinite(t)]
    if finite_clicks:
        stamps = ", ".join(f"{t:.3f}s" for t in finite_clicks[:8])
        suffix = f" (+{len(finite_clicks) - 8} more)" if len(finite_clicks) > 8 else ""
        warnings.append(
            f"Broadcast QA: detected {len(finite_clicks)} potential splice click(s) at {stamps}{suffix}."
        )
    return warnings


def transition_qa_targets(
    selected: SetlistTransition,
    *,
    a_zero_sec: float,
    b_zero_sec: float,
    track_a_analysis: Optional[AudioAnalysis] = None,
    track_b_analysis: Optional[AudioAnalysis] = None,
) -> Tuple[List[float], List[float]]:
    """Return ``(splice_boundaries_sec, mapped_onsets_sec)`` in the rendered mix timeline."""
    boundaries: List[float] = []
    if isinstance(selected, NaturalTransition):
        boundary = a_zero_sec + selected.boundary_sec
        boundaries.append(boundary)
        if selected.pause_sec > 0.0:
            boundaries.append(boundary + selected.pause_sec)
    else:
        fade_start = a_zero_sec + selected.a_fade_start_sec
        a_cut = a_zero_sec + selected.a_cut_sec
        boundaries.extend([fade_start, a_cut, fade_start + selected.overlap_sec])
        takeover = selected.takeover_overlap_sec
        if takeover is not None:
            b_entry = a_cut - takeover
            b_fade_in = selected.b_fade_in_sec if selected.b_fade_in_sec is not None else takeover
            boundaries.extend([b_entry, b_entry + max(takeover, b_fade_in)])

    clean_boundaries = sorted({round(t, 4) for t in boundaries if math.isfinite(t) and t > 0.0})

    onsets: List[float] = []
    for analysis, zero in ((track_a_analysis, a_zero_sec), (track_b_analysis, b_zero_sec)):
        if analysis is None:
            continue
        for t in (*analysis.beats_sec, *analysis.onsets_sec):
            mapped = zero + float(t)
            if math.isfinite(mapped) and mapped >= 0.0:
                onsets.append(mapped)
    return clean_boundaries, onsets


def run_broadcast_qa(
    audio_path: Path,
    *,
    boundary_times: Optional[Sequence[float]] = None,
    drum_onset_times: Sequence[float] = (),
) -> Optional[BroadcastAudioReport]:
    """Diagnostic-only wrapper: never raises on QA failures so completed renders stay intact."""
    try:
        return analyze_broadcast_audio(
            audio_path,
            boundary_times=boundary_times,
            drum_onset_times=drum_onset_times,
        )
    except Exception as exc:
        logger.warning("Skipped broadcast audio QA on %s: %s", audio_path.name, exc)
        return None
