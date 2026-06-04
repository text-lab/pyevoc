"""PyEvoc analysis layer.

This subpackage contains the core analytical routines used after token-level
preprocessing and feature construction:

- EVOC quadrant assignment and compact HTML EVOC reports;
- dependency-based collocations and PROPN named-entity n-grams;
- temporal stability and quadrant mobility analysis.

Only stable public functions are exported here. Internal helper functions remain
available from their own modules but are intentionally not re-exported at the
subpackage level.
"""

# -----------------------------------------------------
# EVOC quadrants and compact EVOC HTML reports
# -----------------------------------------------------

from .quadrants import (
    QuadrantConfig,
    assign_evoc_quadrants,
    assign_quadrants,
    quadrant_summary_by_pos,
    generate_evoc_html_reports,
    build_evoc_html_for_upos,
)

# -----------------------------------------------------
# Collocations and named entities
# -----------------------------------------------------

from .collocations_entities import (
    CollocationEntityConfig,
    extract_collocations_and_entities,
    extract_dependency_collocations,
    extract_named_entity_ngrams,
    extract_collocations,
    aggregate_named_entities,
)

# -----------------------------------------------------
# Temporal stability
# -----------------------------------------------------

from .temporal_stability import (
    TemporalStabilityConfig,
    build_time_periods,
    compute_period_quadrants,
    compute_stability_metrics,
    compute_temporal_engagement,
    write_temporal_report_html,
    run_temporal_stability_analysis,
    quadrant_trajectories,
)

__all__ = [
    # EVOC quadrants
    "QuadrantConfig",
    "assign_evoc_quadrants",
    "assign_quadrants",
    "quadrant_summary_by_pos",
    "generate_evoc_html_reports",
    "build_evoc_html_for_upos",

    # Collocations and named entities
    "CollocationEntityConfig",
    "extract_collocations_and_entities",
    "extract_dependency_collocations",
    "extract_named_entity_ngrams",
    "extract_collocations",
    "aggregate_named_entities",

    # Temporal stability
    "TemporalStabilityConfig",
    "build_time_periods",
    "compute_period_quadrants",
    "compute_stability_metrics",
    "compute_temporal_engagement",
    "write_temporal_report_html",
    "run_temporal_stability_analysis",
    "quadrant_trajectories",
]
