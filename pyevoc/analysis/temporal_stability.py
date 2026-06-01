"""Temporal stability and quadrant mobility."""
from __future__ import annotations
import pandas as pd


def quadrant_trajectories(period_terms: pd.DataFrame, term_col: str = "term", period_col: str = "period", quadrant_col: str = "quadrant") -> pd.DataFrame:
    wide = period_terms.pivot_table(index=term_col, columns=period_col, values=quadrant_col, aggfunc="first")
    out = wide.reset_index()
    periods = list(wide.columns)
    out["quadrant_trajectory"] = wide.apply(lambda r: " → ".join([str(x) if pd.notna(x) else "Absent" for x in r]), axis=1).values
    out["quadrant_changes"] = wide.apply(lambda r: sum(1 for a,b in zip(r.tolist(), r.tolist()[1:]) if a != b), axis=1).values
    return out


def core_jaccard(period_terms: pd.DataFrame, period_a, period_b, term_col: str = "term", period_col: str = "period", quadrant_col: str = "quadrant") -> float:
    a = set(period_terms[(period_terms[period_col]==period_a) & (period_terms[quadrant_col]=="Central nucleus")][term_col])
    b = set(period_terms[(period_terms[period_col]==period_b) & (period_terms[quadrant_col]=="Central nucleus")][term_col])
    return len(a & b) / len(a | b) if (a | b) else 0.0
