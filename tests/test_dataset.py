import pandas as pd
import pytest

from pyevoc import DatasetConfig, corpus_summary, load_dataset, standardise_dataset


def _raw_dataset():
    return pd.DataFrame(
        {
            "u": ["a", "b", "c", "d"],
            "d": ["1", "2", "3", "4"],
            "t": [
                "2024-01-01",
                "2024-01-31 23:00:00",
                "2024-02-01",
                "2024-01-15",
            ],
            "s": ["reddit", "reddit", "reddit", "reddit"],
            "txt": ["one", "two", "three", ""],
            "extra": [10, 20, 30, 40],
        }
    )


def _config(**kwargs):
    params = {
        "column_map": {
            "u": "user_id",
            "d": "doc_id",
            "t": "time",
            "s": "source",
            "txt": "text",
        },
        "show_progress": False,
    }
    params.update(kwargs)
    return DatasetConfig(**params)


def test_standardise_dataset_returns_canonical_schema():
    out = standardise_dataset(_raw_dataset(), _config())

    assert list(out.columns) == ["user_id", "doc_id", "time", "text"]
    assert len(out) == 3
    assert pd.api.types.is_datetime64_any_dtype(out["time"])


def test_standardise_dataset_date_bounds_are_inclusive_for_calendar_dates():
    cfg = _config(start_date="2024-01-01", end_date="2024-01-31")
    out = standardise_dataset(_raw_dataset(), cfg)

    assert len(out) == 2
    assert set(out["doc_id"].astype(str)) == {"1", "2"}


def test_standardise_dataset_drops_missing_or_empty_text_by_default():
    out = standardise_dataset(_raw_dataset(), _config())

    assert "4" not in set(out["doc_id"].astype(str))


def test_standardise_dataset_can_keep_extra_columns():
    cfg = _config(keep_extra_columns=True)
    out = standardise_dataset(_raw_dataset(), cfg)

    assert "extra" not in out.columns
