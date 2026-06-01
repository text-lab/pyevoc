"""EVOC map plotting."""
from __future__ import annotations
from pathlib import Path
import pandas as pd


def plot_evoc_scatter(terms: pd.DataFrame, output: str | Path | None = None, diffusion_col: str = "relative_diffusion", rank_col: str = "AOE"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    for q, grp in terms.groupby("quadrant"):
        ax.scatter(grp[diffusion_col], grp[rank_col], label=q, alpha=0.7)
    ax.invert_yaxis()
    ax.set_xlabel("Relative diffusion")
    ax.set_ylabel("AOE")
    ax.legend()
    if output:
        fig.savefig(output, bbox_inches="tight", dpi=300)
    return fig, ax
