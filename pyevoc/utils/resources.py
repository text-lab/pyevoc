"""Utilities for locating bundled and user-level PyEvoc resources."""
from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Iterable

PACKAGE = "pyevoc.models"
USER_MODEL_DIR = Path.home() / ".pyevoc" / "models"


def model_path(name: str) -> Path:
    """Return a path to a bundled resource in :mod:`pyevoc.models`.

    Parameters
    ----------
    name:
        File name or relative path inside ``pyevoc/models``.
    """
    return Path(resources.files(PACKAGE).joinpath(name))


def user_model_path(name: str) -> Path:
    """Return the standard user-level model path, creating the directory."""
    USER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return USER_MODEL_DIR / name


def list_bundled_models() -> list[str]:
    """List bundled resource files."""
    base = Path(resources.files(PACKAGE))
    return [str(p.relative_to(base)) for p in base.rglob("*") if p.is_file()]
