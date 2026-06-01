"""Named-entity aggregation utilities."""
from __future__ import annotations
import pandas as pd


def aggregate_named_entities(entities: pd.DataFrame, text_col: str = "text", type_col: str = "type") -> pd.DataFrame:
    if entities.empty:
        return pd.DataFrame(columns=["entity", "type", "freq"])
    return entities.groupby([text_col, type_col]).size().reset_index(name="freq").rename(columns={text_col:"entity", type_col:"type"}).sort_values("freq", ascending=False)
