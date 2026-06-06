"""Input/output helpers."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def read_table(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read a tabular data file into a :class:`pandas.DataFrame`.

    Dispatches to the appropriate pandas reader based on the file extension.
    Supported formats: ``.csv``, ``.tsv`` / ``.tab``, ``.xlsx`` / ``.xls``,
    ``.parquet``, and ``.json``.

    Args:
        path: Path to the data file.
        **kwargs: Extra keyword arguments forwarded to the underlying
            pandas reader.

    Returns:
        A :class:`~pandas.DataFrame` containing the file contents.

    Raises:
        ValueError: If the file extension is not supported.
    """
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
    """Create *path* and all missing parent directories, then return it.

    Equivalent to ``mkdir -p``; safe to call on a directory that already
    exists.

    Args:
        path: Directory path to create.

    Returns:
        The resolved :class:`~pathlib.Path` object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
