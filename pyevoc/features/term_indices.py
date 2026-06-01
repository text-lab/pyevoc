"""Term-level AFE/AOE-style indices."""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class SalienceConfig:
    alpha: float = 0.50
    r_max: float = 5.0


def compute_term_indices(
    tokens: pd.DataFrame,
    *,
    term_col: str = "term",
    upos_col: str = "upos",
    user_col: str = "user_id",
    doc_col: str = "doc_id",
    r_pos_col: str = "r_pos",
    r_str_col: str = "r_str",
    config: SalienceConfig = SalienceConfig(),
) -> pd.DataFrame:
    """Compute diffusion, composite salience and AOE-like rank for each term."""
    required = {term_col, upos_col, user_col, doc_col, r_pos_col, r_str_col}
    missing = required.difference(tokens.columns)
    if missing:
        raise ValueError(f"Token table missing columns: {sorted(missing)}")
    n_users = tokens[user_col].nunique(dropna=True)
    grouped = tokens.groupby([term_col, upos_col], dropna=False)
    out = grouped.agg(
        abs_freq=(term_col, "size"),
        n_docs=(doc_col, "nunique"),
        n_users=(user_col, "nunique"),
        r_pos_mean=(r_pos_col, "mean"),
        r_str_mean=(r_str_col, "mean"),
    ).reset_index()
    out["relative_diffusion"] = 100 * out["n_users"] / n_users if n_users else 0.0
    out["S"] = config.alpha * out["r_pos_mean"] + (1 - config.alpha) * out["r_str_mean"]
    out["AOE"] = 1 + (1 - out["S"])*(config.r_max - 1)
    out = out.rename(columns={term_col: "term", upos_col: "upos"})
    return out.sort_values(["upos", "relative_diffusion", "AOE"], ascending=[True, False, True]).reset_index(drop=True)
