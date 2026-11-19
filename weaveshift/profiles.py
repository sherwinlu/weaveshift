"""Scoring profile support for WeaveShift.

Profiles are declarative TOML files that let the same transition engine behave
differently for EDM/workout mixes, smooth blends, vocal trance sets, lounge/background
journeys, jazz/blues, emotional ballad sets, or cinematic continuous-listening mixes
without hard-coding style preferences into the DSP code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import re
from importlib import resources

import tomllib  # Python 3.11+

from weaveshift._paths import expand_user

# Built-in profile names are plain identifiers. Anything else (path separators,
# "..", etc.) must go through an explicit CLI --scoring-config path instead, so a
# profile name from a manifest can never traverse outside the packaged configs directory.
_PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")

BUILTIN_PROFILES = (
    "edm",
    "smooth",
    "vocal_trance",
    "vocal_trance_strict",
    "lounge",
    "jazz",
    "heart",
    "cinematic",
)


@dataclass
class ScoringProfile:
    name: str = "edm"
    description: str = "Beat-aware continuous mix profile."
    weights: Dict[str, float] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    limits: Dict[str, float] = field(default_factory=dict)

    def weight(self, key: str, default: float) -> float:
        try:
            return float(self.weights.get(key, default))
        except (TypeError, ValueError):
            return default

    def pref(self, key: str, default: Any) -> Any:
        return self.preferences.get(key, default)

    def limit(self, key: str, default: float) -> float:
        try:
            return float(self.limits.get(key, default))
        except (TypeError, ValueError):
            return default


def builtin_profile_path(name: str) -> Path:
    """Return the path of a built-in profile, shipped as package data in weaveshift/configs/."""
    if not _PROFILE_NAME_RE.fullmatch(name):
        raise ValueError(
            f"Invalid profile name: {name!r}. Built-in profile names may only contain letters, digits, "
            "hyphens, and underscores; use --scoring-config to load a profile from a TOML path."
        )
    return Path(str(resources.files("weaveshift") / "configs" / f"{name}.toml"))


def load_scoring_profile(profile: str = "edm", config_path: Optional[str] = None) -> ScoringProfile:
    path = expand_user(config_path) if config_path else builtin_profile_path(profile)
    if not path.exists():
        raise FileNotFoundError(
            f"Scoring profile not found: {path}. Built-in profiles: {', '.join(BUILTIN_PROFILES)}. "
            "Use --scoring-config to load a custom TOML profile from any path."
        )
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return ScoringProfile(
        name=str(data.get("name") or profile),
        description=str(data.get("description") or ""),
        weights=dict(data.get("weights") or {}),
        preferences=dict(data.get("preferences") or {}),
        limits=dict(data.get("limits") or {}),
    )
