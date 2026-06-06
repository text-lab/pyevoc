"""General utility functions used across PyEvoc."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PUBLIC_OBJECTS: dict[str, str] = {'read_table': 'io', 'ensure_dir': 'io', 'model_path': 'resources', 'user_model_path': 'resources', 'list_bundled_models': 'resources', 'validate_columns': 'validation', 'validate_column_map': 'validation'}

__all__ = sorted(_PUBLIC_OBJECTS)

def __getattr__(name: str) -> Any:
    """Load public objects lazily from their implementation module."""
    try:
        module_name = _PUBLIC_OBJECTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value

def __dir__() -> list[str]:
    """Return the public API exposed by this subpackage."""
    return sorted(set(globals()) | set(__all__))
