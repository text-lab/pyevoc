"""PyEvoc: computational Hierarchical Evocation Analysis for digital corpora.

The top-level package intentionally exposes only the dataset helpers and version
metadata. Subpackage-level APIs are available from ``pyevoc.preprocessing``,
``pyevoc.features``, ``pyevoc.analysis`` and ``pyevoc.visualisation``.

Imports are lazy to make API documentation generation robust in environments
where optional analytical dependencies are not installed.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.2.1"

_PUBLIC_OBJECTS: dict[str, str] = {
    "DatasetConfig": "data.dataset",
    "load_dataset": "data.dataset",
    "standardise_dataset": "data.dataset",
    "corpus_summary": "data.dataset",
}

__all__ = ["__version__", *sorted(_PUBLIC_OBJECTS)]

def __getattr__(name: str) -> Any:
    """Load top-level public objects lazily."""
    try:
        module_name = _PUBLIC_OBJECTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value

def __dir__() -> list[str]:
    """Return the top-level public API."""
    return sorted(set(globals()) | set(__all__))
