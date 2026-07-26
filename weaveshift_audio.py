"""Audio validation and analysis primitives for WeaveShift."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import soundfile as sf
from pydub import AudioSegment

try:
    import pyloudnorm as pyln
except Exception:  # pragma: no cover
    pyln = None

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import librosa
except Exception as exc:  # pragma: no cover
    print("ERROR: librosa is required. Install with: python3 -m pip install librosa soundfile", file=sys.stderr)
    raise exc

try:
    from scipy.ndimage import maximum_filter1d
except Exception:  # pragma: no cover
    maximum_filter1d = None


logger = logging.getLogger(__name__)


# -----------------------------
# Audio format validation
# -----------------------------

SUPPORTED_WAV_SUFFIXES = {".wav", ".wave"}
SUPPORTED_MP3_SUFFIXES = {".mp3"}
SUPPORTED_INPUT_SUFFIXES = SUPPORTED_WAV_SUFFIXES | SUPPORTED_MP3_SUFFIXES
SUPPORTED_OUTPUT_SUFFIXES = SUPPORTED_WAV_SUFFIXES | SUPPORTED_MP3_SUFFIXES

LOSSLESS_WAV_SUBTYPES = {"PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"}
LOSSY_MP3_SUBTYPE = "MP3"
DEFAULT_MP3_BITRATE = "320k"

# Shared transition search constants. Keep candidate generation and analysis-window
# coverage tied to the same source of truth so future overlap additions cannot
# silently create stale local-energy curves.
OVERLAP_LENGTHS = [2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0]
MAX_OVERLAP_SEC = max(OVERLAP_LENGTHS)

def _is_lossless_wav(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_WAV_SUFFIXES


def _is_mp3(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_MP3_SUFFIXES


@dataclass(frozen=True)
class ProbedAudioMetadata:
    samplerate: int
    channels: int
    duration_sec: float


def _ffprobe_audio_metadata(path: Path) -> ProbedAudioMetadata:
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=sample_rate,channels",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffprobe was not found on PATH. Install ffmpeg (which provides ffprobe) to use MP3 inputs."
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"ffprobe failed for {path.name}")
    payload = json.loads(proc.stdout or "{}")
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError(f"No audio stream found in {path.name}")
    stream = streams[0]
    fmt = payload.get("format") or {}
    return ProbedAudioMetadata(
        samplerate=int(stream.get("sample_rate") or 0),
        channels=int(stream.get("channels") or 0),
        duration_sec=float(fmt.get("duration") or 0.0),
    )


def _validate_lossless_wav_input(p: Path, label: str) -> None:
    with p.open("rb") as f:
        header = f.read(12)
    if len(header) < 12 or header[0:4] not in (b"RIFF", b"RF64") or header[8:12] != b"WAVE":
        raise ValueError(f"{label} has a .wav extension but is not a RIFF/RF64 WAVE file: {p.name}")

    info = sf.info(str(p))
    if info.format != "WAV":
        raise ValueError(f"{label} is not reported as WAV by soundfile: {p.name} ({info.format})")
    if info.subtype not in LOSSLESS_WAV_SUBTYPES:
        raise ValueError(
            f"{label} WAV subtype is {info.subtype}, not one of {sorted(LOSSLESS_WAV_SUBTYPES)}.\n"
            "Export a standard PCM or float WAV master before mixing."
        )


def validate_audio_input(path: str, label: str) -> Path:
    """Validate that an input is a supported WAV or MP3 file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{label} file does not exist: {p}")
    suffix = p.suffix.lower()
    if suffix not in SUPPORTED_INPUT_SUFFIXES:
        raise ValueError(
            f"{label} must be a WAV or MP3 file. Got: {p.name}\n"
            "Supported inputs: .wav, .wave, .mp3"
        )

    try:
        if _is_lossless_wav(p):
            _validate_lossless_wav_input(p, label)
        else:
            meta = _ffprobe_audio_metadata(p)
            if meta.duration_sec <= 0.0:
                raise ValueError(f"{label} MP3 has zero duration: {p.name}")
            if meta.samplerate <= 0 or meta.channels <= 0:
                raise ValueError(f"{label} MP3 stream metadata is invalid: {p.name}")
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"{label} could not be read as a valid audio file: {p.name}. Details: {exc}") from exc
    return p


def validate_wav_input(path: str, label: str) -> Path:
    """Validate that an input is a real, supported WAV file, not just a .wav suffix."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{label} file does not exist: {p}")
    if p.suffix.lower() not in SUPPORTED_WAV_SUFFIXES:
        raise ValueError(
            f"{label} must be a WAV file. Got: {p.name}\n"
            "Use validate_audio_input() for MP3 sources, or export a WAV master before mixing."
        )
    try:
        _validate_lossless_wav_input(p, label)
    except RuntimeError as exc:
        raise ValueError(f"{label} could not be read as a valid WAV file: {p.name}. Details: {exc}") from exc
    return p


def wav_info(path: Path) -> Dict[str, object]:
    info = sf.info(str(path))
    return {
        "path": str(path),
        "samplerate": int(info.samplerate),
        "channels": int(info.channels),
        "subtype": str(info.subtype),
        "format": str(info.format),
        "duration_sec": float(info.duration),
        "lossless": True,
    }


def audio_info(path: Path) -> Dict[str, object]:
    if _is_lossless_wav(path):
        return wav_info(path)
    meta = _ffprobe_audio_metadata(path)
    return {
        "path": str(path),
        "samplerate": meta.samplerate,
        "channels": meta.channels,
        "subtype": LOSSY_MP3_SUBTYPE,
        "format": "MP3",
        "duration_sec": meta.duration_sec,
        "lossless": False,
    }


def compatibility_fields(a_info: Dict[str, object], b_info: Dict[str, object]) -> Tuple[str, ...]:
    fields: List[str] = ["samplerate", "channels"]
    if a_info.get("lossless") and b_info.get("lossless"):
        fields.append("subtype")
    return tuple(fields)


def validate_audio_pair_compatibility(track_a: Path, track_b: Path) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Require A/B format parity so pydub does not silently resample/reconcile masters."""
    a_info = audio_info(track_a)
    b_info = audio_info(track_b)
    fields = compatibility_fields(a_info, b_info)
    mismatches = [(f, a_info[f], b_info[f]) for f in fields if a_info[f] != b_info[f]]
    if mismatches:
        details = "\n".join(f"  - {field}: Track A={a_val}, Track B={b_val}" for field, a_val, b_val in mismatches)
        raise ValueError(
            "Track A and Track B audio formats do not match. Refusing to mix rather than letting "
            "pydub/ffmpeg silently resample or change bit depth.\n"
            f"{details}\n"
            "Export/resample both masters to the same sample rate and channel count before mixing. "
            "Mixed WAV and MP3 inputs are allowed when those match. "
            "For all-WAV chains, subtype must also match. Recommended: 48 kHz, stereo, PCM_24."
        )
    return a_info, b_info


def validate_wav_pair_compatibility(track_a: Path, track_b: Path) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Require A/B WAV format parity (lossless subtype included)."""
    a_info = wav_info(track_a)
    b_info = wav_info(track_b)
    fields = ("samplerate", "channels", "subtype")
    mismatches = [(f, a_info[f], b_info[f]) for f in fields if a_info[f] != b_info[f]]
    if mismatches:
        details = "\n".join(f"  - {field}: Track A={a_val}, Track B={b_val}" for field, a_val, b_val in mismatches)
        raise ValueError(
            "Track A and Track B WAV formats do not match. Refusing to mix rather than letting "
            "pydub/ffmpeg silently resample or change bit depth.\n"
            f"{details}\n"
            "Export/resample both masters to the same sample rate, channel count, and WAV subtype "
            "before running AutoMix. Recommended: 48 kHz, stereo, PCM_24 or 44.1 kHz, stereo, PCM_24."
        )
    return a_info, b_info


def load_audio_segment(path: Path) -> AudioSegment:
    if _is_mp3(path):
        return AudioSegment.from_file(str(path), format="mp3")
    return AudioSegment.from_file(str(path), format="wav")


def lossy_input_note(path: Path) -> Optional[str]:
    if _is_mp3(path):
        return (
            f"Note: {path.name} is a lossy MP3 source; transitions may inherit encoder "
            "delay/padding or re-encoding artifacts."
        )
    return None


def _audio_segment_to_stereo_float32(seg: AudioSegment) -> Tuple[np.ndarray, int]:
    sr = int(seg.frame_rate)
    channels = int(seg.channels)
    if len(seg) == 0:
        return np.zeros((0, max(1, channels)), dtype=np.float32), sr
    samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
    if channels > 1:
        samples = samples.reshape((-1, channels))
    else:
        samples = samples.reshape((-1, 1))
    peak = float(2 ** (8 * seg.sample_width - 1))
    return (samples / peak).astype(np.float32), sr


def read_audio_window(path: str, start_sec: float, duration_sec: float) -> Tuple[np.ndarray, int]:
    """Read a stereo float32 window without loading the full file when possible."""
    p = Path(path)
    if duration_sec <= 0:
        return np.zeros((0, 2), dtype=np.float32), 48000
    if _is_lossless_wav(p):
        info = sf.info(str(p))
        sr = int(info.samplerate)
        start_frame = max(0, int(round(start_sec * sr)))
        frames = int(round(duration_sec * sr))
        frames = min(frames, max(0, int(info.frames) - start_frame))
        y, sr_read = sf.read(str(p), start=start_frame, frames=frames, dtype="float32", always_2d=True)
        return y, int(sr_read)
    seg = AudioSegment.from_file(
        str(p),
        format="mp3",
        start_second=max(0.0, start_sec),
        duration=max(0.0, duration_sec),
    )
    return _audio_segment_to_stereo_float32(seg)


def load_mono_analysis_audio(path: str | Path) -> Tuple[np.ndarray, int, float]:
    """Load a full track as mono float32 for librosa analysis (WAV or MP3)."""
    p = Path(path)
    meta = audio_info(p)
    duration_sec = float(meta["duration_sec"])  # pyright: ignore[reportArgumentType]
    y_stereo, sr_native = read_audio_window(str(p), 0.0, duration_sec)
    y_mono = (
        np.mean(y_stereo, axis=1).astype(np.float32)
        if y_stereo.size
        else np.zeros(0, dtype=np.float32)
    )
    return y_mono, int(sr_native), duration_sec


def validate_loaded_segment_parity(seg_a: AudioSegment, seg_b: AudioSegment) -> None:
    """Secondary guard after pydub load; catches decoded sample-rate/channel mismatches.

    Decoded sample width is intentionally not compared: MP3 and WAV masters that share
    sample rate and channels can still decode to different internal bit depths.
    """
    mismatches = []
    if seg_a.frame_rate != seg_b.frame_rate:
        mismatches.append(("frame_rate", seg_a.frame_rate, seg_b.frame_rate))
    if seg_a.channels != seg_b.channels:
        mismatches.append(("channels", seg_a.channels, seg_b.channels))
    if mismatches:
        details = "\n".join(f"  - {field}: Track A={a_val}, Track B={b_val}" for field, a_val, b_val in mismatches)
        raise ValueError(
            "Decoded AudioSegment formats do not match after loading. Refusing to render.\n" + details
        )


def resolve_wav_export_subtype(infos: Sequence[Dict[str, object]]) -> Optional[str]:
    """Choose a WAV PCM subtype for export when inputs may mix WAV and MP3."""
    for info in infos:
        if info.get("lossless") and info.get("subtype") in LOSSLESS_WAV_SUBTYPES:
            return str(info["subtype"])
    return "PCM_16"

def validate_audio_output(path: str) -> Path:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix and suffix not in SUPPORTED_OUTPUT_SUFFIXES:
        raise ValueError(f"Output must be .wav or .mp3. Got: {p.name}")
    if not p.suffix:
        p = p.with_suffix(".wav")
    return p


def validate_wav_output(path: str) -> Path:
    p = Path(path)
    if p.suffix and p.suffix.lower() not in SUPPORTED_WAV_SUFFIXES:
        raise ValueError(f"Output must be .wav. Got: {p.name}")
    if not p.suffix:
        p = p.with_suffix(".wav")
    return p


def output_format_suffix(path: Path) -> str:
    return path.suffix.lower() if path.suffix else ".wav"

# -----------------------------
# Utility types
# -----------------------------

@dataclass
class VocalSegment:
    start_sec: float
    end_sec: float
    mean_db: float = 0.0


@dataclass
class AudioAnalysis:
    path: str
    duration_sec: float
    bpm: Optional[float]
    key: Optional[str]
    camelot: Optional[str]
    rms_dbfs: float
    peak_dbfs: float
    beats_sec: List[float]
    onsets_sec: List[float]
    vocal_segments: List[VocalSegment]
    vocal_method: str
    analysis_window_start_sec: float
    analysis_window_duration_sec: float
    energy_curve: Optional[Dict[str, np.ndarray]] = None
    loudness_curve: Optional[Dict[str, np.ndarray]] = None


@dataclass
class TransitionCandidate:
    name: str
    score: float
    a_fade_start_sec: float
    a_cut_sec: float
    b_cue_sec: float
    overlap_sec: float
    b_gain_db: float
    trim_a_tail_sec: float
    vocal_collision_score: float
    beat_alignment_score: float
    energy_score: float
    placement_score: float
    loudness_score: float
    perceptual_loudness_score: float
    compatibility_score: float
    notes: List[str]
    soft_duck_db: float = 0.0
    soft_duck_target: str = "none"
    takeover_overlap_sec: Optional[float] = None
    b_fade_in_sec: Optional[float] = None
    b_entry_gain_db: Optional[float] = None


@dataclass
class NaturalTransition:
    """Unscored track boundary that preserves both source tracks."""

    boundary_sec: float
    pause_sec: float
    name: str = "natural"
    notes: List[str] = field(default_factory=list)

    @property
    def a_fade_start_sec(self) -> float:
        return self.boundary_sec

    @property
    def a_cut_sec(self) -> float:
        return self.boundary_sec

    @property
    def b_cue_sec(self) -> float:
        return 0.0

    @property
    def overlap_sec(self) -> float:
        return 0.0

    @property
    def b_gain_db(self) -> float:
        return 0.0

    @property
    def trim_a_tail_sec(self) -> float:
        return 0.0

    @property
    def takeover_overlap_sec(self) -> None:
        return None

    @property
    def b_fade_in_sec(self) -> None:
        return None

    @property
    def b_entry_gain_db(self) -> None:
        return None


SetlistTransition = Union[TransitionCandidate, NaturalTransition]


# -----------------------------
# Audio helpers
# -----------------------------

def dbfs_from_samples(y: np.ndarray) -> Tuple[float, float]:
    if y.size == 0:
        return -120.0, -120.0
    y = np.asarray(y, dtype=np.float32)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    rms = float(np.sqrt(np.mean(y ** 2))) if y.size else 0.0
    rms_db = 20 * math.log10(max(rms, 1e-9))
    peak_db = 20 * math.log10(max(peak, 1e-9))
    return rms_db, peak_db


def pydub_rms_dbfs(seg: AudioSegment) -> float:
    if len(seg) <= 0 or seg.rms == 0:
        return -120.0
    return float(seg.dBFS)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def format_timestamp(seconds: float) -> str:
    """Format seconds as M:SS for human-readable mix timeline positions."""
    if seconds is None:
        return "unknown"
    total = max(0, int(round(float(seconds))))
    minutes = total // 60
    secs = total % 60
    return f"{minutes}:{secs:02d}"


def safe_float(x, default=None):
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


# In-memory local energy curves are attached directly to AudioAnalysis.
# Scoring reads curves from the analysis object, not from module-level caches.

def stereo_dbfs_from_samples(y: np.ndarray) -> Tuple[float, float]:
    """Stereo-aware RMS/peak in dBFS for float samples in [-1, 1].

    This computes RMS across all samples and channels without summing L/R to mono,
    avoiding phase-cancellation errors from wide EDM masters.
    """
    arr = np.asarray(y, dtype=np.float32)
    if arr.size == 0:
        return -120.0, -120.0
    peak = float(np.max(np.abs(arr)))
    rms = float(np.sqrt(np.mean(np.square(arr))))
    return 20 * math.log10(max(rms, 1e-9)), 20 * math.log10(max(peak, 1e-9))

def compute_local_energy_curve(y: np.ndarray, sr: int, offset_sec: float, *, win_sec: float = 0.25, hop_sec: float = 0.10) -> Dict[str, np.ndarray]:
    """Precompute a local stereo RMS curve for cheap candidate scoring.

    y is expected as (samples, channels) or (samples,). Times are absolute
    full-track seconds and dbfs is stereo-aware.
    """
    arr = np.asarray(y, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    win = max(1, int(sr * win_sec))
    hop = max(1, int(sr * hop_sec))
    if len(arr) == 0:
        return {"times": np.array([], dtype=np.float32), "dbfs": np.array([], dtype=np.float32)}
    vals = []
    times = []
    for start_i in range(0, max(1, len(arr) - win + 1), hop):
        chunk = arr[start_i:start_i + win]
        rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
        vals.append(20 * math.log10(max(rms, 1e-9)))
        times.append(offset_sec + (start_i + len(chunk) / 2.0) / sr)
    if not vals:
        rms = float(np.sqrt(np.mean(np.square(arr)))) if arr.size else 0.0
        vals = [20 * math.log10(max(rms, 1e-9))]
        times = [offset_sec + len(arr) / (2.0 * sr)]
    return {"times": np.asarray(times, dtype=np.float32), "dbfs": np.asarray(vals, dtype=np.float32)}


def energy_db_from_curve(curve: Optional[Dict[str, np.ndarray]], start: float, duration: float) -> float:
    """Return mean local stereo RMS dBFS from a precomputed curve."""
    if not curve:
        return -40.0
    times = curve.get("times", np.array([]))
    dbfs = curve.get("dbfs", np.array([]))
    if len(times) == 0 or len(dbfs) == 0:
        return -40.0
    end = start + max(0.05, duration)
    mask = (times >= start) & (times <= end)
    if not np.any(mask):
        # Use nearest local energy sample as a graceful fallback.
        idx = int(np.argmin(np.abs(times - (start + duration / 2.0))))
        return float(dbfs[idx])
    # Convert dB to power, average, then return dB.
    power = np.power(10.0, dbfs[mask] / 10.0)
    return float(10.0 * math.log10(max(float(np.mean(power)), 1e-12)))


def compute_local_lufs_curve(y: np.ndarray, sr: int, offset_sec: float, *, win_sec: float = 3.0, hop_sec: float = 0.25) -> Dict[str, np.ndarray]:
    """Precompute short-term-ish LUFS windows for perceptual loudness scoring.

    This uses pyloudnorm when available. LUFS windows are intentionally longer
    than RMS windows because perceived loudness needs a little time context.
    If pyloudnorm is unavailable or a window is too short/silent, callers fall
    back to the RMS curve.
    """
    if pyln is None:
        return {"times": np.array([], dtype=np.float32), "lufs": np.array([], dtype=np.float32)}
    arr = np.asarray(y, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if len(arr) == 0:
        return {"times": np.array([], dtype=np.float32), "lufs": np.array([], dtype=np.float32)}
    win = max(int(sr * 0.5), int(sr * win_sec))
    hop = max(1, int(sr * hop_sec))
    meter = pyln.Meter(sr)
    vals = []
    times = []
    if len(arr) < win:
        starts = [0]
    else:
        starts = list(range(0, len(arr) - win + 1, hop))
        if starts[-1] != len(arr) - win:
            starts.append(len(arr) - win)
    for start_i in starts:
        chunk = arr[start_i:min(len(arr), start_i + win)]
        if chunk.size == 0:
            continue
        try:
            val = float(meter.integrated_loudness(chunk))
        except Exception:
            continue
        if math.isfinite(val):
            vals.append(val)
            times.append(offset_sec + (start_i + len(chunk) / 2.0) / sr)
    return {"times": np.asarray(times, dtype=np.float32), "lufs": np.asarray(vals, dtype=np.float32)}

def lufs_from_curve(curve: Optional[Dict[str, np.ndarray]], start: float, duration: float) -> Optional[float]:
    """Return local perceived loudness in LUFS from a precomputed curve.

    Returns None if no LUFS curve exists, allowing callers to fall back to RMS.
    """
    if not curve:
        return None
    times = curve.get("times", np.array([]))
    vals = curve.get("lufs", np.array([]))
    if len(times) == 0 or len(vals) == 0:
        return None
    end = start + max(0.05, duration)
    mask = (times >= start) & (times <= end)
    if not np.any(mask):
        idx = int(np.argmin(np.abs(times - (start + duration / 2.0))))
        return float(vals[idx])
    # LUFS is logarithmic; average by energy then convert back.
    power = np.power(10.0, vals[mask] / 10.0)
    return float(10.0 * math.log10(max(float(np.mean(power)), 1e-12)))


# -----------------------------
# Key / Camelot rough estimate
# -----------------------------

MAJOR_PROFILES = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILES = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ENHARMONIC_DISPLAY = {"C#": "D-flat", "D#": "E-flat", "F#": "F-sharp", "G#": "A-flat", "A#": "B-flat"}
CAMELOT_MAJOR = {
    "B": "1B", "F#": "2B", "C#": "3B", "G#": "4B", "D#": "5B", "A#": "6B",
    "F": "7B", "C": "8B", "G": "9B", "D": "10B", "A": "11B", "E": "12B",
}
CAMELOT_MINOR = {
    "G#": "1A", "D#": "2A", "A#": "3A", "F": "4A", "C": "5A", "G": "6A",
    "D": "7A", "A": "8A", "E": "9A", "B": "10A", "F#": "11A", "C#": "12A",
}


def estimate_key(y: np.ndarray, sr: int) -> Tuple[Optional[str], Optional[str]]:
    if y.size < sr:
        return None, None
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        if np.max(chroma_mean) > 0:
            chroma_mean = chroma_mean / np.max(chroma_mean)
        best = (-1e9, None, None)
        for root in range(12):
            maj = np.roll(MAJOR_PROFILES, root)
            minor = np.roll(MINOR_PROFILES, root)
            maj_score = np.corrcoef(chroma_mean, maj)[0, 1]
            min_score = np.corrcoef(chroma_mean, minor)[0, 1]
            if maj_score > best[0]:
                best = (float(maj_score), KEY_NAMES[root], "major")
            if min_score > best[0]:
                best = (float(min_score), KEY_NAMES[root], "minor")
        root, mode = best[1], best[2]
        if not root or not mode:
            return None, None
        display_root = ENHARMONIC_DISPLAY.get(root, root)
        key = f"{display_root} {mode}"
        camelot = (CAMELOT_MAJOR if mode == "major" else CAMELOT_MINOR).get(root)
        return key, camelot
    except Exception:
        return None, None


def parse_manual_key(key: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not key:
        return None, None
    k = key.strip().lower().replace("♭", "-flat").replace("#", "-sharp")
    mode = "minor" if "minor" in k or k.endswith(" min") else "major" if "major" in k or k.endswith(" maj") else None
    root = k.replace("major", "").replace("minor", "").replace("maj", "").replace("min", "").strip()
    canonical = {
        "c": "C", "c-sharp": "C#", "d-flat": "C#", "db": "C#", "c#": "C#",
        "d": "D", "d-sharp": "D#", "e-flat": "D#", "eb": "D#", "d#": "D#",
        "e": "E", "f": "F", "f-sharp": "F#", "g-flat": "F#", "gb": "F#", "f#": "F#",
        "g": "G", "g-sharp": "G#", "a-flat": "G#", "ab": "G#", "g#": "G#",
        "a": "A", "a-sharp": "A#", "b-flat": "A#", "bb": "A#", "a#": "A#", "b": "B",
    }.get(root)
    if not canonical or not mode:
        return key, None
    display_root = ENHARMONIC_DISPLAY.get(canonical, canonical)
    camelot = (CAMELOT_MAJOR if mode == "major" else CAMELOT_MINOR).get(canonical)
    return f"{display_root} {mode}", camelot


def camelot_compat(a: Optional[str], b: Optional[str]) -> float:
    if not a or not b or len(a) < 2 or len(b) < 2:
        return 0.55
    try:
        an, at = int(a[:-1]), a[-1]
        bn, bt = int(b[:-1]), b[-1]
    except Exception:
        return 0.55
    if a == b:
        return 1.0
    if an == bn and at != bt:  # relative major/minor
        return 0.88
    diff = min((an - bn) % 12, (bn - an) % 12)
    if diff == 1 and at == bt:
        return 0.82
    if diff == 2 and at == bt:
        return 0.68
    return 0.42


# -----------------------------
# Vocal activity analysis
# -----------------------------

def get_torch_device(prefer_mps: bool = True) -> str:
    try:
        import torch  # pyright: ignore[reportMissingImports]
        if prefer_mps and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def segments_from_activity(times: np.ndarray, active: np.ndarray, scores_db: np.ndarray, *, offset: float, merge_gap: float, min_len: float) -> List[VocalSegment]:
    segs: List[VocalSegment] = []
    if len(times) == 0 or len(active) == 0:
        return segs
    starts: List[int] = []
    ends: List[int] = []
    in_seg = False
    start_i = 0
    for i, is_active in enumerate(active):
        if is_active and not in_seg:
            start_i = i
            in_seg = True
        elif not is_active and in_seg:
            starts.append(start_i)
            ends.append(i - 1)
            in_seg = False
    if in_seg:
        starts.append(start_i)
        ends.append(len(active) - 1)

    raw: List[VocalSegment] = []
    hop = float(np.median(np.diff(times))) if len(times) > 1 else 0.1
    for s, e in zip(starts, ends):
        start_t = float(times[s])
        end_t = float(times[e] + hop)
        if end_t - start_t >= min_len:
            mean_db = float(np.mean(scores_db[s:e + 1])) if e >= s else 0.0
            raw.append(VocalSegment(offset + start_t, offset + end_t, mean_db))

    # Merge close segments.
    for seg in raw:
        if not segs or seg.start_sec - segs[-1].end_sec > merge_gap:
            segs.append(seg)
        else:
            prev = segs[-1]
            total_len = max(1e-6, (prev.end_sec - prev.start_sec) + (seg.end_sec - seg.start_sec))
            prev.mean_db = ((prev.mean_db * (prev.end_sec - prev.start_sec)) + (seg.mean_db * (seg.end_sec - seg.start_sec))) / total_len
            prev.end_sec = max(prev.end_sec, seg.end_sec)
    return segs


def vocal_segments_heuristic(y: np.ndarray, sr: int, offset_sec: float, threshold_db: float = -36.0) -> List[VocalSegment]:
    """Fast fallback: estimate vocal-like harmonic/midrange activity. This is not stem separation."""
    sample_count = y.shape[-1] if y.ndim > 1 else y.shape[0]
    if sample_count < sr // 2:
        return []
    y_mono = librosa.to_mono(y) if y.ndim > 1 else y
    try:
        harmonic, percussive = librosa.effects.hpss(y_mono)
        # Mel bands around typical vocal midrange. AI vocals can be wider, so this is only a proxy.
        S = np.abs(librosa.stft(harmonic, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        mid = (freqs >= 250) & (freqs <= 3500)
        full = (freqs >= 80) & (freqs <= 8000)
        mid_energy = np.mean(S[mid, :], axis=0) if np.any(mid) else np.mean(S, axis=0)
        full_energy = np.mean(S[full, :], axis=0) if np.any(full) else np.mean(S, axis=0)
        ratio = mid_energy / (full_energy + 1e-9)
        rms = librosa.feature.rms(y=harmonic, hop_length=512)[0]
        db = 20 * np.log10(rms + 1e-9)
        # Adaptive threshold plus ratio gate.
        thr = max(threshold_db, float(np.percentile(db, 70) - 8.0))
        active = (db > thr) & (ratio > np.percentile(ratio, 45))
        # Smooth to avoid stuttering.
        if maximum_filter1d is not None:
            active = maximum_filter1d(active.astype(float), size=5) > 0
        times = librosa.frames_to_time(np.arange(len(active)), sr=sr, hop_length=512)
        return segments_from_activity(times, active, db, offset=offset_sec, merge_gap=1.2, min_len=0.4)
    except Exception:
        return []


def _audio_cache_key(audio_path: str) -> str:
    p = Path(audio_path).resolve()
    try:
        st = p.stat()
        payload = f"{p}|{st.st_mtime_ns}|{st.st_size}"
    except OSError:
        payload = str(p)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _segments_from_vocals_wav(vocals_path: Path, start_sec: float, duration_sec: float, threshold_db: float) -> List[VocalSegment]:
    info = sf.info(str(vocals_path))
    sr = int(info.samplerate)
    frame_start = max(0, int(start_sec * sr))
    frame_stop = min(int(info.frames), frame_start + max(1, int(duration_sec * sr)))
    if frame_stop <= frame_start:
        return []
    data, sr_read = sf.read(str(vocals_path), start=frame_start, stop=frame_stop, always_2d=True, dtype="float32")
    if data.size == 0:
        return []
    vocals_mono = np.mean(data, axis=1).astype(np.float32)
    win = max(1, int(sr_read * 0.10))
    hop = win
    rms = librosa.feature.rms(y=vocals_mono, frame_length=win, hop_length=hop)[0]
    db = 20 * np.log10(rms + 1e-9)
    if len(db):
        adaptive = float(np.percentile(db, 72) - 10.0)
        thr = max(threshold_db, adaptive)
    else:
        thr = threshold_db
    active = db > thr
    if maximum_filter1d is not None:
        active = maximum_filter1d(active.astype(float), size=3) > 0
    times = librosa.frames_to_time(np.arange(len(active)), sr=sr_read, hop_length=hop)
    return segments_from_activity(times, active, db, offset=start_sec, merge_gap=1.5, min_len=0.35)


def _demucs_cli_vocals_path(audio_path: str, model: str = "htdemucs") -> Path:
    """Run/cache Demucs CLI separation and return the generated vocals.wav path.

    Some Demucs installs expose only the CLI module (demucs.separate) and not
    demucs.api. This fallback keeps --vocal-method demucs usable in those envs,
    including Python 3.14 + TorchCodec setups.
    """
    src = Path(audio_path)
    cache_dir = Path.cwd() / ".weaveshift_demucs_cache" / _audio_cache_key(audio_path)
    model_dir = cache_dir / model
    existing = sorted(model_dir.glob("*/vocals.wav")) if model_dir.exists() else []
    if existing:
        return existing[0]

    cache_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "demucs.separate",
        "-n",
        model,
        "--two-stems",
        "vocals",
        "-o",
        str(cache_dir),
        str(src),
    ]
    logger.info("Demucs API unavailable; running CLI fallback: %s ...", " ".join(cmd[:6]))
    subprocess.run(cmd, check=True)
    generated = sorted(model_dir.glob("*/vocals.wav"))
    if not generated:
        raise RuntimeError(f"Demucs CLI completed but no vocals.wav was found under {model_dir}")
    return generated[0]


def vocal_segments_demucs(audio_path: str, start_sec: float, duration_sec: float, threshold_db: float = -38.0, prefer_mps: bool = True) -> List[VocalSegment]:
    """Neural source separation with Demucs. Returns full-song absolute vocal segments for the requested chunk.

    Preferred path: in-process demucs.api Separator when available.
    Fallback path: demucs.separate CLI + cached vocals.wav when demucs.api is absent.
    """
    try:
        from demucs.api import Separator  # pyright: ignore[reportMissingImports]
    except ModuleNotFoundError as exc:
        if exc.name not in {"demucs.api", "demucs"}:
            raise
        vocals_path = _demucs_cli_vocals_path(audio_path)
        return _segments_from_vocals_wav(vocals_path, start_sec, duration_sec, threshold_db)

    import torchaudio  # pyright: ignore[reportMissingImports]

    info = torchaudio.info(audio_path)  # pyright: ignore[reportAttributeAccessIssue]
    sr_native = int(info.sample_rate)
    frame_offset = int(start_sec * sr_native)
    requested_frames = int(duration_sec * sr_native)
    available_frames = max(0, int(info.num_frames) - frame_offset)
    num_frames = min(requested_frames, available_frames)
    if num_frames <= 0:
        return []
    wav, sr = torchaudio.load(audio_path, frame_offset=frame_offset, num_frames=num_frames)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2, :]

    device = get_torch_device(prefer_mps=prefer_mps)
    try:
        separator = Separator(model="htdemucs", device=device)
        origin, separated = separator.separate_tensor(wav.to(device), sr)
    except Exception as exc:
        if device != "cpu":
            logger.warning("Demucs failed on %s; retrying on CPU. Reason: %s", device, exc)
            device = "cpu"
            separator = Separator(model="htdemucs", device=device)
            origin, separated = separator.separate_tensor(wav.to(device), sr)
        else:
            raise

    vocals = separated["vocals"]
    if vocals.ndim == 3:  # batch, channel, time
        vocals = vocals[0]
    vocals_mono = vocals.mean(dim=0).detach().cpu().numpy().astype(np.float32)

    win = int(sr * 0.10)
    hop = win
    rms = librosa.feature.rms(y=vocals_mono, frame_length=win, hop_length=hop)[0]
    db = 20 * np.log10(rms + 1e-9)
    if len(db):
        adaptive = float(np.percentile(db, 72) - 10.0)
        thr = max(threshold_db, adaptive)
    else:
        thr = threshold_db
    active = db > thr
    if maximum_filter1d is not None:
        active = maximum_filter1d(active.astype(float), size=3) > 0
    times = librosa.frames_to_time(np.arange(len(active)), sr=sr, hop_length=hop)
    return segments_from_activity(times, active, db, offset=start_sec, merge_gap=1.5, min_len=0.35)

def analyze_vocals(audio_path: str, y_chunk: np.ndarray, sr: int, start_sec: float, duration_sec: float, method: str, prefer_mps: bool) -> Tuple[List[VocalSegment], str]:
    method = method.lower()
    if method in {"demucs", "auto"}:
        try:
            logger.info("Vocal analysis: Demucs chunk %.1fs-%.1fs", start_sec, start_sec + duration_sec)
            segs = vocal_segments_demucs(audio_path, start_sec, duration_sec, prefer_mps=prefer_mps)
            return segs, "demucs"
        except Exception as exc:
            if method == "demucs":
                raise
            logger.warning("Demucs unavailable/failed; using heuristic vocal analysis. Reason: %s", exc)
    segs = vocal_segments_heuristic(y_chunk, sr, start_sec)
    return segs, "heuristic"


def vocal_active_fraction(segments: Sequence[VocalSegment], start: float, end: float, step: float = 0.1) -> float:
    if end <= start:
        return 0.0
    t = np.arange(start, end, step)
    if len(t) == 0:
        return 0.0
    active = np.zeros(len(t), dtype=bool)
    for seg in segments:
        active |= (t >= seg.start_sec) & (t <= seg.end_sec)
    return float(np.mean(active))


def overlap_vocal_collision(a_segments: Sequence[VocalSegment], b_segments: Sequence[VocalSegment], a_start: float, b_start: float, overlap: float, step: float = 0.1) -> float:
    """Fraction of overlap frames where both A and B have active vocals."""
    if overlap <= 0:
        return 0.0
    t = np.arange(0, overlap, step)
    if len(t) == 0:
        return 0.0
    a_active = np.zeros(len(t), dtype=bool)
    b_active = np.zeros(len(t), dtype=bool)
    for seg in a_segments:
        a_active |= ((a_start + t) >= seg.start_sec) & ((a_start + t) <= seg.end_sec)
    for seg in b_segments:
        b_active |= ((b_start + t) >= seg.start_sec) & ((b_start + t) <= seg.end_sec)
    return float(np.mean(a_active & b_active))


# -----------------------------
# Analysis
# -----------------------------

def analyze_audio(path: str, *, role: str, window_sec: float, manual_bpm: Optional[float], manual_key: Optional[str], vocal_method: str, prefer_mps: bool) -> AudioAnalysis:
    audio_meta = audio_info(Path(path))
    duration_sec = float(audio_meta["duration_sec"])  # pyright: ignore[reportArgumentType]
    if role == "a":
        start = max(0.0, duration_sec - window_sec)
        dur = duration_sec - start
    else:
        start = 0.0
        dur = min(window_sec, duration_sec)

    logger.info("Analyzing %s: %s", "Track A outro" if role == "a" else "Track B intro", Path(path).name)
    y_stereo, sr = read_audio_window(path, start, dur)
    rms_db, peak_db = stereo_dbfs_from_samples(y_stereo)
    energy_curve = compute_local_energy_curve(y_stereo, sr, start)
    loudness_curve = compute_local_lufs_curve(y_stereo, sr, start)
    # For beat/key/vocal heuristics, mono analysis is fine; gain/loudness decisions stay stereo-aware.
    y_mono = np.mean(y_stereo, axis=1).astype(np.float32) if y_stereo.size else np.array([], dtype=np.float32)
    y_chunk = y_stereo.T  # channels-first for librosa.to_mono in heuristic vocal analysis

    # Tempo / beat analysis: Track A uses the outro window; Track B uses the intro.
    beat_load_offset = start if role == "a" else 0.0
    beat_load_dur = min(dur if role == "a" else duration_sec, 180.0)
    try:
        # Same rationale as weaveshift_cues.py: avoid librosa.load(), which unconditionally
        # touches audioread.available_backends() (and therefore the stdlib aifc/sunau
        # modules removed in Python 3.13) even for plain WAV input. Read the windowed
        # chunk with soundfile/pydub directly and resample with librosa instead.
        y_tempo_native, sr_tempo_native = read_audio_window(path, beat_load_offset, beat_load_dur)
        y_tempo_native = (
            np.mean(y_tempo_native, axis=1) if y_tempo_native.size else np.zeros(0, dtype=np.float32)
        )
        sr_tempo = 22050
        y_tempo = (
            librosa.resample(y_tempo_native, orig_sr=sr_tempo_native, target_sr=sr_tempo)
            if sr_tempo_native != sr_tempo
            else y_tempo_native
        )
        tempo, beats = librosa.beat.beat_track(y=y_tempo, sr=sr_tempo, units="time")
        tempo = safe_float(np.asarray(tempo).flatten()[0] if np.asarray(tempo).size else tempo, None)
    except Exception as exc:
        logger.warning(
            "Tempo/beat analysis failed for %s; continuing without beat grid. stage=tempo path=%s reason=%s",
            Path(path).name,
            path,
            type(exc).__name__,
        )
        tempo, beats = None, np.array([])

    try:
        onset_env = librosa.onset.onset_strength(y=y_mono, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units="frames", backtrack=True)
        onsets = librosa.frames_to_time(onset_frames, sr=sr).tolist()
        onsets_abs = [round(start + float(x), 3) for x in onsets]
    except Exception as exc:
        logger.warning(
            "Onset analysis failed for %s; continuing without onsets. stage=onset path=%s reason=%s",
            Path(path).name,
            path,
            type(exc).__name__,
        )
        onsets_abs = []

    beats_abs = []
    try:
        for bt in np.asarray(beats).flatten().tolist():
            bt_abs = round(beat_load_offset + float(bt), 3)
            if start <= bt_abs <= start + dur:
                beats_abs.append(bt_abs)
    except Exception as exc:
        logger.warning(
            "Beat window filtering failed for %s; continuing without beats. stage=beats path=%s reason=%s",
            Path(path).name,
            path,
            type(exc).__name__,
        )

    est_key, est_camelot = estimate_key(y_mono, sr)
    manual_key_norm, manual_cam = parse_manual_key(manual_key)
    key = manual_key_norm or est_key
    camelot = manual_cam or est_camelot
    bpm = manual_bpm or tempo

    vocals, used_method = analyze_vocals(path, y_chunk, sr, start, dur, vocal_method, prefer_mps)

    logger.info(
        "duration=%.2fs bpm=%s key=%s camelot=%s vocals=%d segments method=%s",
        duration_sec,
        bpm if bpm else "unknown",
        key or "unknown",
        camelot or "unknown",
        len(vocals),
        used_method,
    )
    return AudioAnalysis(
        path=path,
        duration_sec=duration_sec,
        bpm=round(float(bpm), 2) if bpm else None,
        key=key,
        camelot=camelot,
        rms_dbfs=round(rms_db, 2),
        peak_dbfs=round(peak_db, 2),
        beats_sec=beats_abs,
        onsets_sec=onsets_abs,
        vocal_segments=vocals,
        vocal_method=used_method,
        analysis_window_start_sec=round(start, 3),
        analysis_window_duration_sec=round(dur, 3),
        energy_curve=energy_curve,
        loudness_curve=loudness_curve,
    )
