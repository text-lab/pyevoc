"""Sankey plot for temporal quadrant transitions."""
from __future__ import annotations
from pathlib import Path
import pandas as pd


def plot_quadrant_sankey(transitions: pd.DataFrame, source_col="source", target_col="target", value_col="value", output_html: str | Path | None = None):
    import plotly.graph_objects as go
    labels = list(pd.unique(pd.concat([transitions[source_col], transitions[target_col]], ignore_index=True)))
    idx = {label:i for i,label in enumerate(labels)}
    fig = go.Figure(data=[go.Sankey(
        node={"label": labels},
        link={"source": transitions[source_col].map(idx), "target": transitions[target_col].map(idx), "value": transitions[value_col]},
    )])
    if output_html:
        fig.write_html(str(output_html))
    return fig
