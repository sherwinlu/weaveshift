#!/usr/bin/env python3
"""Mix-polish cleanup for AI-generated songs and stems.

A simplified offline port of the idea behind Soundkey's CLEANROOM
(https://soundkey.io): analyze a whole track once, then tighten the low
end, cut mud, tame harsh top-end "fizz", restore a little air, correct
loudness drift, and run a lookahead true-peak-safe ceiling before export,
all tuned from a Song Type preset that reuses WeaveShift's existing
scoring-profile names (edm, vocal_trance, lounge, jazz, heart, cinematic).

What this module deliberately does NOT attempt, since it is a one-shot
offline pass over a whole file rather than a real-time interactive tool:

- No manual per-band "PRO" control surface. Only --cleanup-song-type,
  --cleanup-intensity, and --cleanup-loud are exposed.
- De-Fizz, Mud Cut, and the stereo width adjustment are static corrections
  sized once from a whole-track analysis, not dynamic/level-following
  processors the way a real-time engine's would be.
- "Mono below" uses a plain Butterworth low-pass split on the Side signal,
  not a phase-matched Linkwitz-Riley crossover.
- The true-peak ceiling is a best-effort 4x-oversampled lookahead limiter
  with block-smoothed release, not a sample-accurate ITU-R BS.1770 limiter.

This can't fix audio that was never generated, codec-like texture, or
smeared vocals -- it's mix polish, not source repair.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from pydub import AudioSegment
from scipy.ndimage import minimum_filter1d
from scipy.signal import butter, lfilter, resample_poly, sosfilt, welch

try:
    import pyloudnorm as pyln
except Exception:  # pragma: no cover
    pyln = None

from weaveshift_audio import SUPPORTED_MP3_SUFFIXES, audio_info, clamp, read_audio_window

APP_VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# Song Type presets. Names are shared with weaveshift_profiles' built-in
# scoring profiles so --cleanup-song-type edm lines up with --profile edm,
# but the values here tune mastering-style cleanup, not transition scoring.
# All caps and scale factors are reasonable starting points (TUNE), not a
# precise match to CLEANROOM's own per-genre tuning.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SongType:
    hp_freq: float  # low-tighten high-pass, Hz
    mono_freq: float  # collapse stereo width below this, Hz
    mono_amt: float  # 0..1, how much of the low side band to remove
    mud_cap_db: float  # max mud cut
    fizz_cap_db: float  # max de-fizz cut
    air_cap_db: float  # max air boost
    width: float  # side-channel multiplier above mono_freq
    level_cap_db: float  # max leveler turn-down
    line: str  # one-line description, echoed in "what changed" output


SONG_TYPES: Dict[str, SongType] = {
    "custom": SongType(30, 120, 0.80, 3.0, 4.0, 2.5, 1.00, 3.0, "No genre bias. AUTO decides from the audio alone."),
    "edm": SongType(28, 150, 1.00, 3.0, 3.0, 2.0, 1.15, 2.0, "Mono low end, wide top, never over-levels the drops."),
    "vocal_trance": SongType(30, 130, 0.85, 3.0, 4.0, 2.5, 1.10, 2.5, "Keeps space around the lead vocal, gentle air."),
    "vocal_trance_strict": SongType(30, 130, 0.90, 2.0, 3.0, 2.0, 1.05, 2.0, "Tighter mud/fizz caps for a cleaner vocal-safe mix."),
    "lounge": SongType(35, 110, 0.60, 2.5, 3.0, 2.5, 1.00, 2.0, "Light touch; preserves a relaxed, open top end."),
    "jazz": SongType(35, 100, 0.60, 2.5, 3.0, 1.5, 0.95, 2.0, "Warm and natural, narrower image, dynamics protected."),
    "heart": SongType(30, 110, 0.70, 2.5, 3.5, 2.5, 1.00, 2.0, "Protects dynamics; keeps the vocal forward."),
    "cinematic": SongType(28, 100, 0.50, 2.5, 3.0, 1.5, 1.00, 2.0, "Protects dynamics and space; only mud/harshness control."),
}

INTENSITY: Dict[str, float] = {"light": 0.5, "balanced": 1.0, "strong": 1.5}
LOUD_MODES: Tuple[str, ...] = ("louder", "match", "streaming")
LOUDER_BOOST_DB = 1.5
STREAMING_TARGET_LUFS = -14.0
CEILING_DBTP = -1.0


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


@dataclass
class TrackMetrics:
    sr: int
    duration_sec: float
    channels: int
    integrated_lufs: float
    crest_db: float
    correlation: float
    low_width_db: float
    mud_peak_hz: float
    mud_excess_db: float
    fizz_peak_hz: float
    fizz_excess_db: float
    air_deficit_db: float
    loudness_drift_lu: float
    true_peak_db: float


def _band_db(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return -99.0
    power = float(np.mean(psd[mask]))
    return 10.0 * math.log10(max(power, 1e-20))


def _tilt_fit(freqs: np.ndarray, psd: np.ndarray, lo: float = 200.0, hi: float = 6000.0) -> Tuple[float, float]:
    """Least-squares reference tilt (dB per octave) fit over [lo, hi] Hz."""
    mask = (freqs >= lo) & (freqs <= hi) & (psd > 0)
    if np.count_nonzero(mask) < 4:
        return 0.0, -60.0
    x = np.log2(freqs[mask])
    y = 10.0 * np.log10(psd[mask])
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def _expected_db(slope: float, intercept: float, freq: float) -> float:
    return intercept + slope * math.log2(max(freq, 1.0))


def _frame_rms_db(x: np.ndarray, sr: int, frame_sec: float = 0.02) -> np.ndarray:
    n = max(1, int(sr * frame_sec))
    usable = (len(x) // n) * n
    if usable == 0:
        return np.array([-99.0])
    frames = x[:usable].reshape(-1, n)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-9))


def _mono_for_loudness(y: np.ndarray) -> np.ndarray:
    if y.ndim == 2 and y.shape[1] > 1:
        return y
    return y[:, 0] if y.ndim == 2 else y


def _integrated_lufs(y: np.ndarray, sr: int) -> float:
    if pyln is not None:
        try:
            meter = pyln.Meter(sr)
            val = meter.integrated_loudness(_mono_for_loudness(y).astype(np.float64))
            if math.isfinite(val):
                return float(val)
        except Exception:
            pass
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2) + 1e-20))
    return -0.691 + 20.0 * math.log10(max(rms, 1e-9))


def _loudness_drift_lu(y: np.ndarray, sr: int, duration_sec: float) -> float:
    """Linear-fit loudness drift (LU) over the song, using 3 s blocks on a 1 s hop."""
    if duration_sec < 20.0 or pyln is None:
        return 0.0
    mono = np.mean(y, axis=1) if y.ndim == 2 and y.shape[1] > 1 else (y[:, 0] if y.ndim == 2 else y)
    block_n = int(3.0 * sr)
    hop_n = int(1.0 * sr)
    meter = pyln.Meter(sr)
    vals: List[float] = []
    for start in range(0, max(1, len(mono) - block_n), hop_n):
        block = mono[start:start + block_n]
        try:
            lu = meter.integrated_loudness(block.astype(np.float64))
        except Exception:
            continue
        if math.isfinite(lu) and lu > -60.0:
            vals.append(lu)
    if len(vals) < 5:
        return 0.0
    x = np.arange(len(vals), dtype=np.float64)
    slope, _ = np.polyfit(x, np.array(vals), 1)
    return float(slope * len(vals))


def _true_peak_db(y: np.ndarray) -> float:
    try:
        os_y = resample_poly(y, up=4, down=1, axis=0)
    except Exception:
        os_y = y
    peak = float(np.max(np.abs(os_y))) if os_y.size else 0.0
    return 20.0 * math.log10(max(peak, 1e-9))


def analyze_track(y: np.ndarray, sr: int) -> TrackMetrics:
    """Whole-track metrics driving the AUTO cleanup settings."""
    if y.ndim == 1:
        y = y[:, None]
    y = np.asarray(y, dtype=np.float64)
    channels = y.shape[1]
    duration_sec = y.shape[0] / float(sr) if sr else 0.0
    mid = np.mean(y, axis=1) if channels > 1 else y[:, 0]
    side = (y[:, 0] - y[:, 1]) * 0.5 if channels > 1 else np.zeros_like(mid)

    nperseg = int(min(8192, max(256, len(mid))))
    freqs, psd = welch(mid, sr, nperseg=nperseg) if mid.size else (np.array([0.0]), np.array([0.0]))
    slope, intercept = _tilt_fit(freqs, psd)

    mud_lo, mud_hi = 120.0, 600.0
    mud_mask = (freqs >= mud_lo) & (freqs <= mud_hi)
    if np.any(mud_mask):
        mud_peak_hz = float(freqs[mud_mask][int(np.argmax(psd[mud_mask]))])
    else:
        mud_peak_hz = 300.0
    mud_actual = _band_db(freqs, psd, mud_peak_hz / 1.2, mud_peak_hz * 1.2)
    mud_excess_db = max(0.0, mud_actual - _expected_db(slope, intercept, mud_peak_hz))

    fizz_lo, fizz_hi = 3000.0, 12000.0
    fizz_mask = (freqs >= fizz_lo) & (freqs <= fizz_hi)
    if np.any(fizz_mask):
        band_db_vals = 10.0 * np.log10(np.maximum(psd[fizz_mask], 1e-20))
        expected_vals = np.array([_expected_db(slope, intercept, f) for f in freqs[fizz_mask]])
        excess_vals = band_db_vals - expected_vals
        idx = int(np.argmax(excess_vals))
        fizz_peak_hz = float(freqs[fizz_mask][idx])
        fizz_excess_db = float(max(0.0, excess_vals[idx]))
    else:
        fizz_peak_hz = 6000.0
        fizz_excess_db = 0.0

    nyquist = sr / 2.0 if sr else 0.0
    air_hi = min(16000.0, max(10001.0, nyquist - 1.0))
    air_actual = _band_db(freqs, psd, 10000.0, air_hi)
    air_deficit_db = max(0.0, _expected_db(slope, intercept, 12000.0) - air_actual)

    frame_db = _frame_rms_db(mid, sr)
    usable = frame_db[frame_db > -70]
    crest_db = float(np.percentile(usable, 95) - np.percentile(usable, 50)) if usable.size >= 10 else 10.0

    if channels > 1:
        l, r = y[:, 0], y[:, 1]
        denom = math.sqrt(float(np.sum(l * l)) * float(np.sum(r * r))) + 1e-12
        correlation = float(np.sum(l * r) / denom)
        low_sos = butter(2, min(150.0, max(10.0, nyquist - 10.0)), btype="low", fs=sr, output="sos")
        low_mid = np.asarray(sosfilt(low_sos, mid))
        low_side = np.asarray(sosfilt(low_sos, side))
        low_width_db = 10.0 * math.log10((float(np.mean(low_side ** 2)) + 1e-20) / (float(np.mean(low_mid ** 2)) + 1e-20))
    else:
        correlation = 1.0
        low_width_db = -60.0

    return TrackMetrics(
        sr=sr,
        duration_sec=duration_sec,
        channels=channels,
        integrated_lufs=_integrated_lufs(y, sr),
        crest_db=crest_db,
        correlation=correlation,
        low_width_db=low_width_db,
        mud_peak_hz=mud_peak_hz,
        mud_excess_db=mud_excess_db,
        fizz_peak_hz=fizz_peak_hz,
        fizz_excess_db=fizz_excess_db,
        air_deficit_db=air_deficit_db,
        loudness_drift_lu=_loudness_drift_lu(y, sr, duration_sec),
        true_peak_db=_true_peak_db(y),
    )


# ---------------------------------------------------------------------------
# AUTO: metrics + Song Type -> cleanup parameters
# ---------------------------------------------------------------------------


@dataclass
class CleanupParams:
    hp_freq: float
    mono_freq: float
    mono_amt: float
    mud_freq: float
    mud_depth_db: float
    fizz_freq: float
    fizz_depth_db: float
    air_freq: float
    air_gain_db: float
    width: float
    level_amt: float
    level_allowance_lu: float
    level_range_db: float
    ceiling_db: float = CEILING_DBTP


def auto_params(metrics: TrackMetrics, song_type: str = "custom", intensity: str = "balanced") -> CleanupParams:
    st = SONG_TYPES.get(song_type, SONG_TYPES["custom"])
    k = INTENSITY.get(intensity, 1.0)

    mono_amt = st.mono_amt if metrics.low_width_db > -18.0 else 0.0
    mud_depth_db = clamp(metrics.mud_excess_db * 0.75 * k, 0.0, st.mud_cap_db)
    fizz_depth_db = clamp(metrics.fizz_excess_db * k, 0.0, st.fizz_cap_db)
    air_gain_db = clamp(metrics.air_deficit_db * 0.5 * k, 0.0, st.air_cap_db)
    width = st.width if (metrics.channels > 1 and metrics.correlation > 0.15) else 1.0
    engage_leveler = metrics.duration_sec >= 20.0 and abs(metrics.loudness_drift_lu) > 1.0

    return CleanupParams(
        hp_freq=st.hp_freq,
        mono_freq=st.mono_freq,
        mono_amt=mono_amt,
        mud_freq=metrics.mud_peak_hz,
        mud_depth_db=mud_depth_db,
        fizz_freq=metrics.fizz_peak_hz,
        fizz_depth_db=fizz_depth_db,
        air_freq=12000.0,
        air_gain_db=air_gain_db,
        width=width,
        level_amt=1.0 if engage_leveler else 0.0,
        level_allowance_lu=2.0,
        level_range_db=st.level_cap_db,
    )


# ---------------------------------------------------------------------------
# DSP building blocks (RBJ cookbook biquads via scipy)
# ---------------------------------------------------------------------------


def _highpass(x: np.ndarray, freq: float, sr: int, order: int = 2) -> np.ndarray:
    freq = clamp(freq, 1.0, max(2.0, sr / 2.0 - 10.0))
    sos = butter(order, freq, btype="high", fs=sr, output="sos")
    return np.asarray(sosfilt(sos, x))


def _lowpass(x: np.ndarray, freq: float, sr: int, order: int = 2) -> np.ndarray:
    freq = clamp(freq, 1.0, max(2.0, sr / 2.0 - 10.0))
    sos = butter(order, freq, btype="low", fs=sr, output="sos")
    return np.asarray(sosfilt(sos, x))


def _peaking_coeffs(freq: float, sr: int, q: float, gain_db: float) -> Tuple[np.ndarray, np.ndarray]:
    freq = clamp(freq, 20.0, max(21.0, sr / 2.0 - 100.0))
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * freq / sr
    alpha = math.sin(w0) / (2.0 * q)
    cos_w0 = math.cos(w0)
    b0, b1, b2 = 1 + alpha * a, -2 * cos_w0, 1 - alpha * a
    a0, a1, a2 = 1 + alpha / a, -2 * cos_w0, 1 - alpha / a
    return np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0])


def _highshelf_coeffs(freq: float, sr: int, gain_db: float, slope: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    freq = clamp(freq, 20.0, max(21.0, sr / 2.0 - 100.0))
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * freq / sr
    cos_w0, sin_w0 = math.cos(w0), math.sin(w0)
    alpha = sin_w0 / 2.0 * math.sqrt((a + 1.0 / a) * (1.0 / slope - 1.0) + 2.0)
    sqrt_a = math.sqrt(a)
    b0 = a * ((a + 1) + (a - 1) * cos_w0 + 2 * sqrt_a * alpha)
    b1 = -2 * a * ((a - 1) + (a + 1) * cos_w0)
    b2 = a * ((a + 1) + (a - 1) * cos_w0 - 2 * sqrt_a * alpha)
    a0 = (a + 1) - (a - 1) * cos_w0 + 2 * sqrt_a * alpha
    a1 = 2 * ((a - 1) - (a + 1) * cos_w0)
    a2 = (a + 1) - (a - 1) * cos_w0 - 2 * sqrt_a * alpha
    return np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0])


def _apply_biquad(x: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    return np.asarray(lfilter(b, a, x))


def _apply_leveler(y: np.ndarray, sr: int, params: CleanupParams) -> np.ndarray:
    """Turn down sections that run louder than the song's own usual level."""
    mono = np.mean(y, axis=1) if y.shape[1] > 1 else y[:, 0]
    block_n = max(1, int(0.4 * sr))
    n_blocks = len(mono) // block_n
    if n_blocks < 4:
        return y
    usable = n_blocks * block_n
    blocks = mono[:usable].reshape(n_blocks, block_n)
    rms_db = 20.0 * np.log10(np.maximum(np.sqrt(np.mean(blocks.astype(np.float64) ** 2, axis=1)), 1e-9))
    loud_blocks = rms_db[rms_db > -60]
    if loud_blocks.size < 4:
        return y
    reference_db = float(np.median(loud_blocks))
    threshold_db = reference_db + params.level_allowance_lu
    over = np.maximum(0.0, rms_db - threshold_db)
    reduction_db = np.minimum(over * params.level_amt, params.level_range_db)

    # Block-rate smoothing (not per-sample): fast-ish turn-down, slower recovery.
    attack, release = 0.5, 0.15
    smoothed = np.empty_like(reduction_db)
    prev = 0.0
    for i, target in enumerate(reduction_db):
        coeff = attack if target > prev else release
        prev += (target - prev) * coeff
        smoothed[i] = prev

    gain = 10.0 ** (-smoothed / 20.0)
    gain_full = np.repeat(gain, block_n)
    if len(gain_full) < len(mono):
        gain_full = np.concatenate([gain_full, np.full(len(mono) - len(gain_full), gain_full[-1])])
    else:
        gain_full = gain_full[: len(mono)]
    return y * gain_full[:, None]


def _true_peak_ceiling(y: np.ndarray, sr: int, ceiling_db: float, lookahead_ms: float = 5.0) -> np.ndarray:
    """Best-effort 4x-oversampled lookahead true-peak-safe ceiling."""
    if y.size == 0:
        return y
    ceil_amp = 10.0 ** (ceiling_db / 20.0)
    os_y = resample_poly(y, up=4, down=1, axis=0)
    os_sr = sr * 4
    peak_per_sample = np.max(np.abs(os_y), axis=1)
    required = np.minimum(1.0, ceil_amp / np.maximum(peak_per_sample, 1e-9))

    look_n = max(1, int(os_sr * (lookahead_ms / 1000.0)))
    looked_ahead = minimum_filter1d(required, size=look_n, mode="nearest")

    # Smooth the release in ~1 ms blocks (not per oversampled sample) so a
    # multi-minute track stays fast. Attack is immediate: the lookahead
    # minimum above already guarantees no overshoot inside the window.
    block_n = max(1, int(os_sr * 0.001))
    n_blocks = max(1, len(looked_ahead) // block_n)
    usable = n_blocks * block_n
    block_min = looked_ahead[:usable].reshape(n_blocks, block_n).min(axis=1)

    release_coeff = 1.0 - math.exp(-0.001 / 0.05)  # ~50 ms release, 1 ms step
    smoothed_blocks = np.empty_like(block_min)
    g = 1.0
    for i, req in enumerate(block_min):
        g = req if req < g else g + (req - g) * release_coeff
        smoothed_blocks[i] = g

    gain_os = np.repeat(smoothed_blocks, block_n)
    if usable < len(looked_ahead):
        gain_os = np.concatenate([gain_os, np.full(len(looked_ahead) - usable, smoothed_blocks[-1])])

    gain_orig = gain_os[::4]
    if len(gain_orig) < len(y):
        tail = gain_orig[-1] if len(gain_orig) else 1.0
        gain_orig = np.concatenate([gain_orig, np.full(len(y) - len(gain_orig), tail)])
    else:
        gain_orig = gain_orig[: len(y)]

    out = y * gain_orig[:, None]
    return np.clip(out, -ceil_amp, ceil_amp)


def apply_cleanup(y: np.ndarray, sr: int, params: CleanupParams) -> np.ndarray:
    if y.ndim == 1:
        y = y[:, None]
    y = y.astype(np.float64)
    channels = y.shape[1]

    if channels > 1:
        mid = (y[:, 0] + y[:, 1]) * 0.5
        side = (y[:, 0] - y[:, 1]) * 0.5
    else:
        mid = y[:, 0].copy()
        side = np.zeros_like(mid)

    # 1. Low tighten: remove sub rumble, then collapse stereo width below mono_freq.
    mid = _highpass(mid, params.hp_freq, sr)
    if channels > 1:
        side = _highpass(side, params.hp_freq, sr)
        side_low = _lowpass(side, params.mono_freq, sr)
        side_high = side - side_low
        side = side_high + side_low * (1.0 - params.mono_amt)
        l, r = mid + side, mid - side
    else:
        l = r = mid

    # 2-4. Mud cut, De-Fizz, Air: static stereo-linked EQ.
    if params.mud_depth_db > 0.05:
        b, a = _peaking_coeffs(params.mud_freq, sr, 1.0, -params.mud_depth_db)
        l = _apply_biquad(l, b, a)
        r = _apply_biquad(r, b, a) if channels > 1 else l
    if params.fizz_depth_db > 0.05:
        b, a = _peaking_coeffs(params.fizz_freq, sr, 1.4, -params.fizz_depth_db)
        l = _apply_biquad(l, b, a)
        r = _apply_biquad(r, b, a) if channels > 1 else l
    if params.air_gain_db > 0.05:
        b, a = _highshelf_coeffs(params.air_freq, sr, params.air_gain_db)
        l = _apply_biquad(l, b, a)
        r = _apply_biquad(r, b, a) if channels > 1 else l

    # 5. Stereo width.
    if channels > 1 and abs(params.width - 1.0) > 0.01:
        mid2 = (l + r) * 0.5
        side2 = (l - r) * 0.5 * params.width
        l, r = mid2 + side2, mid2 - side2

    out = np.stack([l, r], axis=1) if channels > 1 else l[:, None]

    # 6. Leveler.
    if params.level_amt > 0.01:
        out = _apply_leveler(out, sr, params)

    return out


def export_target_gain(original_lufs: float, processed_lufs: float, loud: str) -> float:
    if loud == "match":
        target = original_lufs
    elif loud == "streaming":
        target = STREAMING_TARGET_LUFS
    else:
        target = original_lufs + LOUDER_BOOST_DB
    if not math.isfinite(processed_lufs) or not math.isfinite(target):
        return 0.0
    return clamp(target - processed_lufs, -12.0, 12.0)


def summarize_changes(metrics: TrackMetrics, params: CleanupParams, song_type: str) -> List[str]:
    st = SONG_TYPES.get(song_type, SONG_TYPES["custom"])
    lines = [f"Song Type: {song_type}. {st.line}"]
    if params.mono_amt > 0.01:
        lines.append(f"Low end was wide below {params.mono_freq:.0f} Hz. Mono {params.mono_amt * 100:.0f} %.")
    else:
        lines.append(f"Sub rumble cleaned below {params.hp_freq:.0f} Hz. Lows already mono.")
    if params.mud_depth_db > 0.05:
        lines.append(f"Mud build-up of +{metrics.mud_excess_db:.1f} dB near {params.mud_freq:.0f} Hz. Cutting {params.mud_depth_db:.1f} dB.")
    else:
        lines.append("No mud problem found.")
    if params.fizz_depth_db > 0.05:
        lines.append(f"Harsh spike near {params.fizz_freq:.0f} Hz. De-fizzing {params.fizz_depth_db:.1f} dB.")
    else:
        lines.append("Top end is clean.")
    if params.air_gain_db > 0.05:
        lines.append(f"Top end was dull above 10 kHz. Adding {params.air_gain_db:.1f} dB of air.")
    if params.level_amt > 0.01:
        lines.append(f"Level drifted {metrics.loudness_drift_lu:+.1f} LU over the song. Leveler engaged, up to {params.level_range_db:.1f} dB.")
    if abs(params.width - 1.0) > 0.01:
        lines.append(f"Stereo width set to {params.width * 100:.0f} %.")
    return lines


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


@dataclass
class CleanupResult:
    samples: np.ndarray
    sr: int
    metrics_before: TrackMetrics
    params: CleanupParams
    gain_db: float
    output_lufs: float
    output_true_peak_db: float
    changed: List[str]


def clean_audio(
    y: np.ndarray,
    sr: int,
    *,
    song_type: str = "custom",
    intensity: str = "balanced",
    loud: str = "louder",
) -> CleanupResult:
    if song_type not in SONG_TYPES:
        raise ValueError(f"Unknown cleanup song type: {song_type!r}. Choose one of: {', '.join(SONG_TYPES)}")
    if intensity not in INTENSITY:
        raise ValueError(f"Unknown cleanup intensity: {intensity!r}. Choose one of: {', '.join(INTENSITY)}")
    if loud not in LOUD_MODES:
        raise ValueError(f"Unknown cleanup loudness target: {loud!r}. Choose one of: {', '.join(LOUD_MODES)}")

    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]

    metrics = analyze_track(y, sr)
    params = auto_params(metrics, song_type, intensity)
    processed = apply_cleanup(y, sr, params)

    processed_lufs = _integrated_lufs(processed, sr)
    gain_db = export_target_gain(metrics.integrated_lufs, processed_lufs, loud)
    processed = processed * (10.0 ** (gain_db / 20.0))
    processed = _true_peak_ceiling(processed, sr, params.ceiling_db)

    return CleanupResult(
        samples=processed,
        sr=sr,
        metrics_before=metrics,
        params=params,
        gain_db=gain_db,
        output_lufs=_integrated_lufs(processed, sr),
        output_true_peak_db=_true_peak_db(processed),
        changed=summarize_changes(metrics, params, song_type),
    )


def load_track(path: Path) -> Tuple[np.ndarray, int]:
    info = audio_info(path)
    duration_sec = float(info["duration_sec"])  # pyright: ignore[reportArgumentType]
    y, sr = read_audio_window(str(path), 0.0, duration_sec)
    return np.asarray(y, dtype=np.float64), int(sr)


def write_wav(y: np.ndarray, sr: int, path: Path, subtype: str = "PCM_24") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = y[:, 0] if y.ndim == 2 and y.shape[1] == 1 else y
    sf.write(str(path), np.clip(data, -1.0, 1.0), sr, subtype=subtype)


def write_output(y: np.ndarray, sr: int, path: Path, *, subtype: str = "PCM_24", mp3_bitrate: str = "320k") -> None:
    if path.suffix.lower() in SUPPORTED_MP3_SUFFIXES:
        fd, tmp_name = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            write_wav(y, sr, tmp_path, subtype="PCM_24")
            path.parent.mkdir(parents=True, exist_ok=True)
            AudioSegment.from_file(str(tmp_path), format="wav").export(str(path), format="mp3", bitrate=mp3_bitrate)
        finally:
            tmp_path.unlink(missing_ok=True)
    else:
        write_wav(y, sr, path, subtype=subtype)


def clean_file(
    input_path: Path,
    output_path: Path,
    *,
    song_type: str = "custom",
    intensity: str = "balanced",
    loud: str = "louder",
    subtype: str = "PCM_24",
) -> CleanupResult:
    y, sr = load_track(input_path)
    result = clean_audio(y, sr, song_type=song_type, intensity=intensity, loud=loud)
    write_output(result.samples, sr, output_path, subtype=subtype)
    return result


def report_dict(src: Path, result: CleanupResult) -> dict:
    return {
        "input": str(src),
        "cleanup_params": asdict(result.params),
        "metrics_before": asdict(result.metrics_before),
        "gain_db": round(result.gain_db, 2),
        "output_lufs": round(result.output_lufs, 2),
        "output_true_peak_dbtp": round(result.output_true_peak_db, 2),
        "what_changed": result.changed,
    }


# ---------------------------------------------------------------------------
# Pipeline integration: lets weaveshift_two_tracks.py / weaveshift_setlist.py
# run every input track through cleanup before transition analysis/mixing.
# ---------------------------------------------------------------------------


@dataclass
class CleanupOptions:
    song_type: str = "custom"
    intensity: str = "balanced"
    loud: str = "louder"
    report_dir: Optional[Path] = None


def cleaned_temp_copy(src: Path, workdir: Path, opts: CleanupOptions, label: str) -> Path:
    """Clean src into a new WAV under workdir, same sample rate/channels as src.

    Always writes WAV (never MP3) so the rest of the pipeline's existing
    format-parity checks keep working, and so a cleaned MP3 input isn't
    re-compressed before WeaveShift's own transition analysis runs on it.
    """
    out_path = workdir / f"{label}_cleaned.wav"
    result = clean_file(src, out_path, song_type=opts.song_type, intensity=opts.intensity, loud=opts.loud)
    if opts.report_dir is not None:
        opts.report_dir.mkdir(parents=True, exist_ok=True)
        report_path = opts.report_dir / f"{label}_cleanup_report.json"
        report_path.write_text(json.dumps(report_dict(src, result), indent=2), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="WeaveShift cleanup: mix-polish for AI-generated songs and stems."
    )
    p.add_argument("input", help="Input audio file (.wav or .mp3)")
    p.add_argument("-o", "--output", default="weaveshift_cleanup.wav", help="Output file (.wav or .mp3)")
    p.add_argument("--song-type", choices=list(SONG_TYPES), default="custom", help="Genre-tuned preset (shares names with WeaveShift's --profile)")
    p.add_argument("--intensity", choices=list(INTENSITY), default="balanced", help="How hard to apply the fixes")
    p.add_argument(
        "--loud",
        choices=list(LOUD_MODES),
        default="louder",
        help="Export loudness target: louder (+1.5 dB over the input), match (same as input), or streaming (-14 LUFS)",
    )
    p.add_argument("--report", default=None, help="Write a JSON 'what changed' report to this path")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    try:
        input_path = Path(ns.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file does not exist: {input_path}")
        output_path = Path(ns.output)
        result = clean_file(input_path, output_path, song_type=ns.song_type, intensity=ns.intensity, loud=ns.loud)

        print(f"Song Type: {ns.song_type} | Intensity: {ns.intensity} | Loudness target: {ns.loud}")
        for line in result.changed:
            print(f"  - {line}")
        print(f"Output: {output_path} ({result.output_lufs:.1f} LUFS, {result.output_true_peak_db:.1f} dBTP)")

        if ns.report:
            report_path = Path(ns.report)
            report_path.write_text(json.dumps(report_dict(input_path, result), indent=2), encoding="utf-8")
            print(f"Wrote report: {report_path}")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
