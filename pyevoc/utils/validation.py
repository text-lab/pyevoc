"""Validation helpers used across PyEvoc."""
from __future__ import annotations

from collections.abc import Mapping

REQUIRED_COLUMNS = ("user_id", "doc_id", "time", "text")


def validate_columns(
    columns: list[str] | tuple[str, ...],
    required: tuple[str, ...] = REQUIRED_COLUMNS,
) -> None:
    """Validate that all required columns are present."""
    missing = [c for c in required if c not in columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_column_map(column_map: Mapping[str, str]) -> None:
    """Validate mapping toward the canonical PyEvoc schema."""
    targets = set(column_map.values())

    missing_targets = [c for c in REQUIRED_COLUMNS if c not in targets]

    if missing_targets:
        raise ValueError(
            "column_map must map source columns to the canonical schema "
            f"{REQUIRED_COLUMNS}; missing targets: {missing_targets}"
        )
