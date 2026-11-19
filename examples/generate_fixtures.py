#!/usr/bin/env python3
"""Generate synthetic demo WAV fixtures for examples/setlist_example.json.

Output is procedurally generated (clicks + sine tone). Do not replace with
copyrighted masters; fixtures are not committed to the repository.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SR = 22050
DURATION = 60.0


def write_click_track(path: Path, *, bpm: float) -> None:
    n = int(SR * DURATION)
    y = np.zeros(n, dtype=np.float32)
    interval = int(SR * 60.0 / bpm)
    for i in range(0, n, max(1, interval)):
        y[i : i + min(400, n - i)] = 0.35
    tone = np.sin(2 * np.pi * 220.0 * np.linspace(0, DURATION, n, endpoint=False))
    y += 0.04 * tone
    sf.write(path, y, SR, subtype="PCM_16")


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    tracks = [
        ("track_01.wav", 110.0),
        ("track_02.wav", 126.0),
        ("track_03.wav", 126.0),
    ]
    for name, bpm in tracks:
        out = FIXTURES / name
        write_click_track(out, bpm=bpm)
        print(f"Wrote {out} ({bpm} BPM, {DURATION:.0f}s)")


if __name__ == "__main__":
    main()
