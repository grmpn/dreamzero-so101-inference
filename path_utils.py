"""Path helpers for the standalone SO-101 inference scripts."""

from __future__ import annotations

import os
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parent


def find_project_root(code_root: Path = CODE_ROOT) -> Path:
    """Return the directory that owns checkpoints, data, and DreamZero sources.

    The SO-101 files may be checked out as either:

    - ``/workspace/dreamzero-so101/*.py`` with assets inside the same directory;
    - ``/workspace/dreamzero-so101/*.py`` with ``dreamzero/``, ``checkpoints/``,
      and ``data/`` mounted beside it.

    ``DREAMZERO_SO101_ROOT`` can override both layouts for Docker runs.
    """

    override = os.environ.get("DREAMZERO_SO101_ROOT")
    if override:
        return Path(override).expanduser().resolve()

    candidates = (code_root, code_root.parent)
    markers = ("dreamzero", "checkpoints", "data")
    for candidate in candidates:
        if any((candidate / marker).exists() for marker in markers):
            return candidate.resolve()
    return code_root.resolve()


PROJECT_ROOT = find_project_root()
DREAMZERO_ROOT = PROJECT_ROOT / "dreamzero"


def project_path(path: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    """Resolve project asset paths against ``base``.

    Existing absolute paths are preserved. Missing absolute paths that contain a
    known project directory, such as ``/home/user/run/checkpoints/...``, are
    rebased onto ``base`` so copied checkpoint configs still work in containers.
    """

    resolved = Path(path).expanduser()
    if resolved.is_absolute() and resolved.exists():
        return resolved.resolve()

    normalized = str(path).replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    for marker in ("checkpoints", "data"):
        if marker in parts:
            return (base / Path(*parts[parts.index(marker) :])).resolve()

    if resolved.is_absolute():
        return resolved.resolve()
    return (base / resolved).resolve()


def default_config_path() -> Path:
    """Prefer the editable project config, then fall back to checkpoint config."""

    local_config = CODE_ROOT / "config.json"
    if local_config.is_file():
        return local_config
    return PROJECT_ROOT / "checkpoints" / "dreamzero-so101-lora" / "config.json"
