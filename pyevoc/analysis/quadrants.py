"""EVOC thresholds and quadrant assignment."""
from __future__ import annotations
import pandas as pd


def compute_thresholds(
    terms: pd.DataFrame,
    *,
    upos_col: str = "upos",
    diffusion_col: str = "relative_diffusion",
    rank_col: str = "AOE",
) -> pd.DataFrame:
    return terms.groupby(upos_col).agg(
        theta_F=(diffusion_col, "mean"),
        theta_R=(rank_col, "mean"),
    ).reset_index().rename(columns={upos_col: "upos"})


def assign_quadrants(
    terms: pd.DataFrame,
    thresholds: pd.DataFrame | None = None,
    *,
    upos_col: str = "upos",
    diffusion_col: str = "relative_diffusion",
    rank_col: str = "AOE",
) -> pd.DataFrame:
    th = thresholds if thresholds is not None else compute_thresholds(terms, upos_col=upos_col, diffusion_col=diffusion_col, rank_col=rank_col)
    out = terms.merge(th, left_on=upos_col, right_on="upos", how="left", suffixes=("", "_threshold"))
    high_diff = out[diffusion_col] >= out["theta_F"]
    high_sal = out[rank_col] <= out["theta_R"]
    out["quadrant"] = "Peripheral system"
    out.loc[high_diff & high_sal, "quadrant"] = "Central nucleus"
    out.loc[high_diff & ~high_sal, "quadrant"] = "First periphery"
    out.loc[~high_diff & high_sal, "quadrant"] = "Contrast zone"
    return out
