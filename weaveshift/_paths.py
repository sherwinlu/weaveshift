"""Shared manifest path resolution for WeaveShift CLIs."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable


def expand_user(raw: str) -> Path:
    """``Path(raw).expanduser()`` that reports unknown ``~user`` as a ValueError."""
    try:
        return Path(raw).expanduser()
    except RuntimeError as exc:
        raise ValueError(f"Cannot expand home directory in path {raw!r}: {exc}") from exc


def is_within(path: Path, roots: Iterable[Path]) -> bool:
    """True if the resolved *path* is inside any of the resolved *roots*."""
    try:
        resolved = path.resolve()
        return any(resolved.is_relative_to(root.resolve()) for root in roots)
    except (OSError, RuntimeError) as exc:  # e.g. symlink loops
        raise ValueError(f"Cannot resolve path {path}: {exc}") from exc


def resolve_contained(
    raw: str,
    base_dir: Path,
    *,
    allow_outside: bool = False,
    label: str = "Track",
) -> Path:
    """Resolve a manifest path against *base_dir*, keeping it inside *base_dir*.

    Absolute paths, ``..`` segments, ``~`` and symlinks are all judged by where
    the path finally resolves, so none of them can escape the manifest directory
    unless the caller opts in with ``allow_outside``.
    """
    p = expand_user(raw)
    try:
        resolved = p.resolve() if p.is_absolute() else (base_dir / p).resolve()
    except (OSError, RuntimeError) as exc:  # e.g. symlink loops
        raise ValueError(f"Cannot resolve {label} path {raw!r}: {exc}") from exc
    if not allow_outside and not is_within(resolved, [base_dir]):
        raise ValueError(
            f"{label} path {raw!r} resolves outside the manifest directory ({base_dir.resolve()}). "
            "Symlinks are followed, so a link to a file elsewhere also counts as outside. "
            "Move the file next to the manifest or pass --allow-outside-manifest-dir."
        )
    return resolved
