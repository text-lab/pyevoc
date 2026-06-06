"""PyEvoc analysis layer.

This subpackage contains EVOC quadrant assignment, compact HTML reports,
dependency-based collocations, named-entity n-grams, temporal stability and
quadrant-mobility analysis. Public objects are loaded lazily for documentation
and faster package import.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PUBLIC_OBJECTS: dict[str, str] = {'QuadrantConfig': 'quadrants', 'assign_evoc_quadrants': 'quadrants', 'assign_quadrants': 'quadrants', 'quadrant_summary_by_pos': 'quadrants', 'generate_evoc_html_reports': 'quadrants', 'build_evoc_html_for_upos': 'quadrants', 'CollocationEntityConfig': 'collocations_entities', 'extract_collocations_and_entities': 'collocations_entities', 'extract_dependency_collocations': 'collocations_entities', 'extract_named_entity_ngrams': 'collocations_entities', 'extract_collocations': 'collocations_entities', 'aggregate_named_entities': 'collocations_entities', 'TemporalStabilityConfig': 'temporal_stability', 'build_time_periods': 'temporal_stability', 'compute_period_quadrants': 'temporal_stability', 'compute_stability_metrics': 'temporal_stability', 'compute_temporal_engagement': 'temporal_stability', 'write_temporal_report_html': 'temporal_stability', 'run_temporal_stability_analysis': 'temporal_stability', 'quadrant_trajectories': 'temporal_stability'}

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
