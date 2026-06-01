"""Plotly emoji EVOC map."""
from __future__ import annotations
from pathlib import Path
import pandas as pd


def plot_emoji_evoc_map(terms: pd.DataFrame, output_html: str | Path | None = None):
    import plotly.express as px
    fig = px.scatter(
        terms,
        x="relative_diffusion",
        y="AOE",
        color="quadrant",
        text="term",
        hover_data=[c for c in ["emoji_description", "abs_freq"] if c in terms.columns],
    )
    fig.update_yaxes(autorange="reversed")
    if output_html:
        fig.write_html(str(output_html))
    return fig
