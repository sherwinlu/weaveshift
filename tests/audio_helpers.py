"""Shared synthetic audio helpers for tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def write_stereo_wav(
    path: Path,
    *,
    duration: float = 4.0,
    sr: int = 48000,
    freq: float = 440.0,
    amp: float = 0.05,
    subtype: str = "PCM_16",
) -> Path:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = amp * np.column_stack([np.sin(2 * np.pi * freq * t), np.sin(2 * np.pi * freq * t)])
    sf.write(path, y, sr, subtype=subtype)
    return path


def write_stereo_mp3(
    path: Path,
    *,
    duration: float = 4.0,
    sr: int = 48000,
    freq: float = 440.0,
    amp: float = 0.05,
    bitrate: str = "320k",
) -> Path:
    wav_path = path.with_suffix(".wav")
    write_stereo_wav(wav_path, duration=duration, sr=sr, freq=freq, amp=amp)
    from pydub import AudioSegment

    AudioSegment.from_file(str(wav_path), format="wav").export(str(path), format="mp3", bitrate=bitrate)
    wav_path.unlink(missing_ok=True)
    return path


def write_click_track(path: Path, *, duration: float, sr: int = 48000, bpm: float = 128.0) -> Path:
    """Stereo WAV with periodic clicks for beat detection tests."""
    n = int(sr * duration)
    y = np.zeros((n, 2), dtype=np.float32)
    interval = int(sr * 60.0 / bpm)
    for i in range(0, n, max(1, interval)):
        y[i:i + min(400, n - i), :] = 0.4
    sf.write(path, y, sr, subtype="PCM_16")
    return path


def write_mono_wav(
    path: Path,
    *,
    duration: float = 4.0,
    sr: int = 22050,
    envelope=None,
    freq: float = 440.0,
    amp: float = 0.08,
    subtype: str = "PCM_16",
) -> Path:
    """Write mono WAV with optional per-sample amplitude envelope."""
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    if envelope is None:
        env = np.ones(n, dtype=np.float32)
    else:
        env = np.asarray(envelope(t), dtype=np.float32)
    y = (amp * env * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(path, y, sr, subtype=subtype)
    return path


def cue_fixture_track(path: Path, *, duration: float = 180.0, sr: int = 22050) -> Path:
    """Synthetic EDM-ish track with predictable cue regions for golden tests."""
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)

    env = np.full(n, 0.08, dtype=np.float32)
    # Quiet intro
    env[t < 12.0] = 0.02
    # First sustained lift around 18s
    env[(t >= 18.0) & (t < 28.0)] = 0.35
    # Main peak/drop region around 48s with rhythmic bursts
    peak_mask = (t >= 44.0) & (t < 58.0)
    env[peak_mask] = 0.95
    env[peak_mask] *= 0.85 + 0.15 * np.sin(2 * np.pi * 2.0 * t[peak_mask])
    # Breakdown dip around 120s
    env[(t >= 116.0) & (t < 128.0)] = 0.04
    # Fade out tail
    env[t >= duration * 0.92] = np.linspace(env[t >= duration * 0.92][0], 0.01, np.sum(t >= duration * 0.92))

    carrier = np.sin(2 * np.pi * 130.0 * t) + 0.35 * np.sin(2 * np.pi * 260.0 * t)
    y = (env * carrier).astype(np.float32)
    sf.write(path, y, sr, subtype="PCM_16")
    return path
