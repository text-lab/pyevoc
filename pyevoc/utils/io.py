"""Input/output helpers."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def read_table(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False, **kwargs)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t", low_memory=False, **kwargs)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, **kwargs)
    if suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    if suffix == ".json":
        return pd.read_json(path, **kwargs)
    raise ValueError(f"Unsupported file type: {suffix}")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
