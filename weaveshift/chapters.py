"""Chapter and tracklist exporters for WeaveShift continuous mixes.

Generates YouTube chapter lists (``.chapters.txt``), FFmpeg chapter metadata
(``.ffmetadata``), standard ``.cue`` sheets, and structured tracklist JSON
(``.tracklist.json``) directly from a rendered setlist mix timeline.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from weaveshift._paths import atomic_write_text

CHAPTER_FILE_SUFFIXES: Dict[str, str] = {
    "youtube": ".chapters.txt",
    "ffmetadata": ".ffmetadata",
    "cue": ".cue",
    "tracklist": ".tracklist.json",
}


def _safe_sec(value: float) -> float:
    """Clamp non-finite or negative timestamps to 0.0 for safe formatting and JSON export."""
    if isinstance(value, bool):
        return 0.0
    try:
        num = float(value)
    except (TypeError, ValueError):
        return 0.0
    return num if math.isfinite(num) and num > 0.0 else 0.0


@dataclass(frozen=True)
class MixChapter:
    """One track chapter inside a rendered continuous mix."""

    index: int
    start_sec: float
    end_sec: float
    title: str
    artist: Optional[str] = None
    bpm: Optional[float] = None
    key: Optional[str] = None

    @property
    def duration_sec(self) -> float:
        return max(0.0, _safe_sec(self.end_sec) - _safe_sec(self.start_sec))


def _clean_line(text: str) -> str:
    """Collapse whitespace and strip control/newline characters."""
    cleaned = "".join(ch if ch.isprintable() else " " for ch in text)
    return " ".join(cleaned.split())


def format_chapter_label(title: str, artist: Optional[str] = None) -> str:
    """Return a clean single-line chapter label."""
    clean_title = _clean_line(title) or "Untitled"
    clean_artist = _clean_line(artist) if artist else ""
    if clean_artist and not clean_title.casefold().startswith(clean_artist.casefold()):
        return f"{clean_artist} - {clean_title}"
    return clean_title


def _finite_float(value: object) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        num = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


def chapters_from_setlist_report(report: Mapping[str, Any]) -> List[MixChapter]:
    """Derive ordered, gap-free chapters from a WeaveShift setlist report dict."""
    raw_tracks = report.get("tracks")
    if not isinstance(raw_tracks, Sequence) or isinstance(raw_tracks, (str, bytes)) or not raw_tracks:
        return []

    raw_transitions = report.get("transitions")
    transitions: Sequence[Any] = (
        raw_transitions
        if isinstance(raw_transitions, Sequence) and not isinstance(raw_transitions, (str, bytes))
        else ()
    )
    raw_zeros = report.get("track_source_zero_in_mix_sec")
    zeros: Sequence[Any] = (
        raw_zeros
        if isinstance(raw_zeros, Sequence) and not isinstance(raw_zeros, (str, bytes))
        else ()
    )

    total_duration = _finite_float(report.get("output_duration_sec")) or 0.0
    total_duration = max(0.0, total_duration)

    starts: List[float] = []
    for idx in range(len(raw_tracks)):
        if idx == 0:
            start = 0.0
        else:
            start_opt: Optional[float] = None
            if idx - 1 < len(transitions) and isinstance(transitions[idx - 1], Mapping):
                tr = transitions[idx - 1]
                next_zero = _finite_float(tr.get("next_track_source_zero_in_mix_sec"))
                b_cue = _finite_float(tr.get("source_b_cue_sec"))
                if next_zero is not None and b_cue is not None:
                    start_opt = next_zero + b_cue
                else:
                    start_opt = _finite_float(tr.get("mix_transition_start_sec"))
            if start_opt is None and idx < len(zeros):
                start_opt = _finite_float(zeros[idx])
            start = start_opt if start_opt is not None else (starts[-1] if starts else 0.0)
            start = max(0.0, start)
            if total_duration > 0.0:
                start = min(start, total_duration)
            if starts:
                start = max(starts[-1], start)
        starts.append(round(start, 3))

    if total_duration <= 0.0 and starts:
        total_duration = starts[-1]

    chapters: List[MixChapter] = []
    for idx, raw_track in enumerate(raw_tracks):
        if isinstance(raw_track, Mapping):
            title_val = raw_track.get("title") or Path(str(raw_track.get("path") or "")).stem or f"Track {idx + 1:02d}"
            artist_raw = raw_track.get("artist")
            artist = _clean_line(str(artist_raw)) if artist_raw else None
            bpm_raw = _finite_float(raw_track.get("bpm"))
            bpm = round(bpm_raw, 2) if bpm_raw is not None and bpm_raw > 0 else None
            key_raw = raw_track.get("key")
            key = _clean_line(str(key_raw)) if key_raw else None
        else:
            title_val = f"Track {idx + 1:02d}"
            artist = None
            bpm = None
            key = None

        start_sec = starts[idx]
        end_sec = round(starts[idx + 1] if idx + 1 < len(starts) else max(start_sec, total_duration), 3)
        chapters.append(
            MixChapter(
                index=idx,
                start_sec=start_sec,
                end_sec=end_sec,
                title=_clean_line(str(title_val)) or f"Track {idx + 1:02d}",
                artist=artist or None,
                bpm=bpm,
                key=key or None,
            )
        )
    return chapters


def _format_yt_timestamp(seconds: float, *, use_hours: bool) -> str:
    total = max(0, int(round(seconds)))
    if use_hours:
        hours, rem = divmod(total, 3600)
        minutes, secs = divmod(rem, 60)
        return f"{hours}:{minutes:02d}:{secs:02d}"
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"


def format_youtube_chapters(
    chapters: Sequence[MixChapter],
    total_duration_sec: float,
) -> Tuple[str, List[str]]:
    """Render a YouTube-compatible ``00:00 Track Name`` chapter list and validation warnings."""
    if not chapters:
        return "", ["No chapters available to export."]

    warnings: List[str] = []
    if len(chapters) < 3:
        warnings.append(
            f"YouTube requires at least 3 chapters to activate timeline scrubbing (got {len(chapters)})."
        )
    short = [f"#{ch.index + 1} ({ch.title})" for ch in chapters if ch.duration_sec < 10.0 - 1e-6]
    if short:
        warnings.append(
            f"YouTube requires every chapter to be >= 10s; short chapter(s): {', '.join(short)}."
        )

    max_stamp = max(
        [
            _safe_sec(total_duration_sec),
            *(_safe_sec(ch.start_sec) for ch in chapters),
            *(_safe_sec(ch.end_sec) for ch in chapters),
        ]
    )
    use_hours = max_stamp >= 3600.0
    lines: List[str] = []
    prev_sec = -1
    for idx, ch in enumerate(chapters):
        sec = 0 if idx == 0 else max(prev_sec + 1, int(round(_safe_sec(ch.start_sec))))
        prev_sec = sec
        stamp = _format_yt_timestamp(float(sec), use_hours=use_hours)
        label = format_chapter_label(ch.title, ch.artist)
        lines.append(f"{stamp} {label}")
    return "\n".join(lines) + "\n", warnings


def _escape_ffmetadata(value: str) -> str:
    cleaned = _clean_line(value)
    for char in ("\\", "=", ";", "#"):
        cleaned = cleaned.replace(char, f"\\{char}")
    return cleaned


def format_ffmetadata_chapters(
    chapters: Sequence[MixChapter],
    title: str = "WeaveShift Mix",
) -> str:
    """Render an FFmpeg ``;FFMETADATA1`` document with millisecond chapter markers."""
    lines = [";FFMETADATA1", f"title={_escape_ffmetadata(title or 'WeaveShift Mix')}"]
    for ch in chapters:
        start_ms = max(0, int(round(_safe_sec(ch.start_sec) * 1000.0)))
        end_ms = max(start_ms + 1, int(round(_safe_sec(ch.end_sec) * 1000.0)))
        label = format_chapter_label(ch.title, ch.artist)
        lines.extend(
            [
                "",
                "[CHAPTER]",
                "TIMEBASE=1/1000",
                f"START={start_ms}",
                f"END={end_ms}",
                f"title={_escape_ffmetadata(label)}",
            ]
        )
    return "\n".join(lines) + "\n"


def _escape_cue_string(value: str) -> str:
    return _clean_line(value).replace('"', "'")


def _format_cue_index(seconds: float) -> str:
    """Format seconds as CUE ``MM:SS:FF`` (75 CD frames/sec) without frame overflow."""
    total_frames = max(0, int(round(_safe_sec(seconds) * 75.0)))
    ff = total_frames % 75
    total_secs = total_frames // 75
    ss = total_secs % 60
    mm = total_secs // 60
    return f"{mm:02d}:{ss:02d}:{ff:02d}"


def format_cue_sheet(
    chapters: Sequence[MixChapter],
    audio_filename: str,
    *,
    title: str = "WeaveShift Mix",
) -> str:
    """Render a standard CUE sheet (compatible with DJ players and mvdirector)."""
    file_name = _escape_cue_string(Path(audio_filename).name) or "mix.wav"
    file_type = "MP3" if Path(file_name).suffix.lower() == ".mp3" else "WAVE"
    sheet_title = _escape_cue_string(title) or "WeaveShift Mix"
    lines = [
        f'TITLE "{sheet_title}"',
        f'FILE "{file_name}" {file_type}',
    ]
    for idx, ch in enumerate(chapters, start=1):
        ch_title = _escape_cue_string(ch.title) or f"Track {idx:02d}"
        lines.append(f"  TRACK {idx:02d} AUDIO")
        lines.append(f'    TITLE "{ch_title}"')
        artist = _escape_cue_string(ch.artist) if ch.artist else ""
        if artist:
            lines.append(f'    PERFORMER "{artist}"')
        if ch.bpm is not None and math.isfinite(ch.bpm) and ch.bpm > 0:
            lines.append(f"    REM BPM {ch.bpm:.2f}")
        key = _escape_cue_string(ch.key) if ch.key else ""
        if key:
            lines.append(f'    REM KEY "{key}"')
        lines.append(f"    INDEX 01 {_format_cue_index(ch.start_sec)}")
    return "\n".join(lines) + "\n"


def build_tracklist_document(
    chapters: Sequence[MixChapter],
    *,
    title: str = "WeaveShift Mix",
) -> Dict[str, Any]:
    """Build a standard tracklist JSON dict (directly ingestible by mvdirector mix mode)."""
    tracks: List[Dict[str, Any]] = []
    for idx, ch in enumerate(chapters, start=1):
        start_val = round(_safe_sec(ch.start_sec), 3)
        end_val = round(_safe_sec(ch.end_sec), 3)
        ch_title = _clean_line(ch.title) or f"Track {idx:02d}"
        entry: Dict[str, Any] = {
            "start": start_val,
            "title": ch_title,
        }
        if end_val > start_val:
            entry["end"] = end_val
        artist = _clean_line(ch.artist) if ch.artist else ""
        if artist:
            entry["artist"] = artist
        if ch.bpm is not None and math.isfinite(ch.bpm) and 20.0 < ch.bpm < 300.0:
            entry["bpm"] = round(ch.bpm, 2)
        key = _clean_line(ch.key) if ch.key else ""
        if key:
            entry["key"] = key
        tracks.append(entry)
    return {
        "title": _clean_line(title) or "WeaveShift Mix",
        "source": "weaveshift",
        "tracks": tracks,
    }


def export_chapters(
    chapters: Sequence[MixChapter],
    base: Path,
    audio_filename: str,
    total_duration_sec: float,
    *,
    title: str = "WeaveShift Mix",
) -> Tuple[Dict[str, Path], List[str]]:
    """Write YouTube, FFmpeg metadata, CUE sheet, and tracklist JSON chapter sidecars atomically."""
    yt_text, yt_warnings = format_youtube_chapters(chapters, total_duration_sec)
    ffmeta_text = format_ffmetadata_chapters(chapters, title=title)
    cue_text = format_cue_sheet(chapters, audio_filename, title=title)
    tracklist_text = json.dumps(
        build_tracklist_document(chapters, title=title),
        indent=2,
        allow_nan=False,
    ) + "\n"

    payloads = {
        "youtube": yt_text,
        "ffmetadata": ffmeta_text,
        "cue": cue_text,
        "tracklist": tracklist_text,
    }
    written: Dict[str, Path] = {}
    for kind, suffix in CHAPTER_FILE_SUFFIXES.items():
        path = base.with_name(base.name + suffix)
        atomic_write_text(path, payloads[kind], encoding="utf-8")
        written[kind] = path
    return written, yt_warnings
