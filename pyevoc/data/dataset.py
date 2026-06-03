"""Dataset ingestion and canonical corpus preparation.

Canonical PyEvoc schema

Required columns
----------------
user_id
doc_id
time
text

Optional metadata
-----------------
source
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import pandas as pd

from pyevoc.utils.io import read_table
from pyevoc.utils.validation import (
    REQUIRED_COLUMNS,
    CANONICAL_COLUMNS,
    validate_column_map,
    validate_columns,
)


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for converting a table into a canonical PyEvoc corpus."""

    column_map: Mapping[str, str]

    start_date: str | None = None
    end_date: str | None = None

    timezone: str = "UTC"

    drop_missing_text: bool = True

    keep_extra_columns: bool = False

    read_kwargs: dict = field(default_factory=dict)


def _parse_bound(
    value: str | None,
    *,
    end: bool,
    timezone: str,
) -> pd.Timestamp | None:
    if value is None:
        return None

    ts = pd.Timestamp(value)

    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone)
    else:
        ts = ts.tz_convert(timezone)

    if end and len(value.strip()) <= 10:
        ts = ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)

    return ts


def standardise_dataset(
    df: pd.DataFrame,
    config: DatasetConfig,
) -> pd.DataFrame:
    """Return a canonical PyEvoc corpus from an existing DataFrame."""

    validate_column_map(config.column_map)

    missing_input = [
        c for c in config.column_map
        if c not in df.columns
    ]

    if missing_input:
        raise ValueError(
            f"Input data are missing columns: {missing_input}"
        )

    selected = (
        df.loc[:, list(config.column_map.keys())]
        .rename(columns=dict(config.column_map))
        .copy()
    )

    if config.keep_extra_columns:
        extra = df.drop(
            columns=list(config.column_map.keys()),
            errors="ignore",
        )
        selected = pd.concat(
            [selected, extra],
            axis=1,
        )

    # ------------------------------------------------------------------
    # Optional metadata
    # ------------------------------------------------------------------

    if "source" not in selected.columns:
        selected["source"] = "corpus"

    validate_columns(
        list(selected.columns),
        required=REQUIRED_COLUMNS,
    )

    for col in ("user_id", "doc_id", "source", "text"):
        selected[col] = selected[col].astype("string")

    selected["time"] = pd.to_datetime(
        selected["time"],
        utc=True,
        errors="coerce",
    )

    selected = selected.dropna(subset=["time"])

    if config.drop_missing_text:
        selected = selected.dropna(subset=["text"])

        selected = selected[
            selected["text"]
            .str.strip()
            .fillna("")
            .ne("")
        ]

    start = _parse_bound(
        config.start_date,
        end=False,
        timezone=config.timezone,
    )

    end = _parse_bound(
        config.end_date,
        end=True,
        timezone=config.timezone,
    )

    if start is not None:
        selected = selected[
            selected["time"] >= start
        ]

    if end is not None:
        selected = selected[
            selected["time"] <= end
        ]

    ordered_cols = [
        "user_id",
        "doc_id",
        "time",
        "source",
        "text",
    ]

    remaining_cols = [
        c for c in selected.columns
        if c not in ordered_cols
    ]

    return (
        selected.loc[:, ordered_cols + remaining_cols]
        .reset_index(drop=True)
    )


def load_dataset(
    path: str | Path,
    config: DatasetConfig,
) -> pd.DataFrame:
    """Load a table from disk and convert it to the canonical PyEvoc schema."""

    df = read_table(
        path,
        **config.read_kwargs,
    )

    return standardise_dataset(
        df,
        config,
    )


def corpus_summary(
    df: pd.DataFrame,
) -> dict:
    """Return basic corpus-level counts for a canonical PyEvoc corpus."""

    validate_columns(
        list(df.columns),
        required=REQUIRED_COLUMNS,
    )

    return {
        "documents": int(len(df)),
        "users": int(df["user_id"].nunique(dropna=True)),
        "sources": int(df["source"].nunique(dropna=True))
        if "source" in df.columns
        else 1,
        "start_time": df["time"].min(),
        "end_time": df["time"].max(),
    }
