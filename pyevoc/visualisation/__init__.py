"""PyEvoc visualisation layer.

This subpackage exposes plotting functions for EVOC concentric maps, semantic
trees, emoji EVOC maps and temporal Sankey diagrams. Plotting dependencies are
kept local to the plotting module; this wrapper loads public names lazily.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PUBLIC_OBJECTS: dict[str, str] = {'build_evoc_target_plot': 'plots', 'build_evoc_collocation_tree_for_upos': 'plots', 'build_emoji_evoc_plot': 'plots', 'build_temporal_sankey_ordered': 'plots'}

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
