import pandas as pd
from pyevoc.data import DatasetConfig, standardise_dataset


def test_standardise_dataset_date_bounds():
    raw = pd.DataFrame({
        "u": ["a", "b", "c"],
        "d": ["1", "2", "3"],
        "t": ["2024-01-01", "2024-01-31 23:00:00", "2024-02-01"],
        "s": ["x", "x", "x"],
        "txt": ["one", "two", "three"],
    })
    cfg = DatasetConfig(column_map={"u":"user_id", "d":"doc_id", "t":"time", "s":"source", "txt":"text"}, start_date="2024-01-01", end_date="2024-01-31")
    out = standardise_dataset(raw, cfg)
    assert len(out) == 2
