"""Term-level AFE/AOE-style indices and EVOC quadrant assignment.

This module computes the term-level indicators used by PyEvoc to reconstruct
computational Hierarchical Evocation Analysis (HEM) structures from token-level
tables.

The module implements three analytical layers:

1. term-level statistics;
2. POS-specific AFE/AOE thresholds;
3. EVOC quadrant assignment.

It supports both user-level diffusion and document-level diffusion, preserves
temporal information when available, and remains backward compatible with the
earlier ``compute_term_indices`` interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import gc

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


DEFAULT_FOCAL_UPOS = {"NOUN", "ADJ", "EMOJI"}


@dataclass(frozen=True)
class SalienceConfig:
    """Configuration for term-level salience and diffusion indicators."""

    alpha: float = 0.50
    max_rank_value: float = 5.0
    use_users_for_frequency: bool = True
    focal_upos: set[str] = field(default_factory=lambda: set(DEFAULT_FOCAL_UPOS))


def _validate_config(config: SalienceConfig) -> None:
    """Validate salience configuration."""

    if not 0 <= config.alpha <= 1:
        raise ValueError("alpha must be in [0, 1].")

    if config.max_rank_value <= 1:
        raise ValueError("max_rank_value must be greater than 1.")

    if not config.focal_upos:
        raise ValueError("focal_upos cannot be empty.")


def _require_columns(df: pd.DataFrame, columns: set[str]) -> None:
    """Validate that required columns are present."""

    missing = columns.difference(df.columns)

    if missing:
        raise ValueError(f"Token table missing columns: {sorted(missing)}")


def _normalise_quadrant_labels(value: str) -> str:
    """Return human-readable EVOC quadrant labels."""

    labels = {
        "central_nucleus": "Central nucleus",
        "first_periphery": "First periphery",
        "contrast_zone": "Contrast zone",
        "peripheral_system": "Peripheral system",
    }

    return labels.get(value, value)


def compute_term_statistics(
    tokens: pd.DataFrame,
    *,
    term_col: str = "term",
    upos_col: str = "upos",
    user_col: str = "user_id",
    doc_col: str = "doc_id",
    time_col: str = "time",
    r_pos_col: str = "r_pos",
    r_str_col: str = "r_str",
    term_type_col: str = "term_type",
    config: SalienceConfig = SalienceConfig(),
    human_readable_quadrants: bool = False,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute term-level statistics, thresholds and quadrant summaries."""

    _validate_config(config)

    required = {
        term_col,
        upos_col,
        user_col,
        doc_col,
        r_pos_col,
        r_str_col,
    }

    _require_columns(tokens, required)

    df = tokens.loc[
        tokens[upos_col].isin(config.focal_upos)
    ].copy()

    if df.empty:
        raise ValueError(
            "No tokens remain after POS filtering. "
            "Check UPOS labels and focal_upos."
        )

    df[term_col] = df[term_col].astype("string")
    df[upos_col] = df[upos_col].astype("string")
    df[doc_col] = df[doc_col].astype("string")
    df[user_col] = df[user_col].astype("string")

    df[r_pos_col] = pd.to_numeric(df[r_pos_col], errors="coerce")
    df[r_str_col] = pd.to_numeric(df[r_str_col], errors="coerce")

    if df[r_pos_col].isna().any():
        raise ValueError(f"Some tokens have missing {r_pos_col} values.")

    if df[r_str_col].isna().any():
        raise ValueError(f"Some tokens have missing {r_str_col} values.")

    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    else:
        df[time_col] = pd.NaT

    if term_type_col not in df.columns:
        df[term_type_col] = "unigram"

    n_unique_users = int(df[user_col].nunique(dropna=True))
    n_unique_docs = int(df[doc_col].nunique(dropna=True))

    steps = [
        "Aggregating term statistics",
        "Computing salience and rank",
        "Computing diffusion",
        "Computing thresholds",
        "Assigning quadrants",
    ]

    with tqdm(
        total=len(steps),
        desc="Computing term statistics",
        disable=not show_progress,
    ) as pbar:

        term_stats_df = (
            df
            .groupby([term_col, upos_col], observed=True, sort=False)
            .agg(
                n_tokens=(term_col, "size"),
                n_docs=(doc_col, "nunique"),
                n_users=(user_col, "nunique"),
                R_pos=(r_pos_col, "mean"),
                R_struct=(r_str_col, "mean"),
                term_type=(term_type_col, "first"),
                first_mention=(time_col, "min"),
                last_mention=(time_col, "max"),
            )
            .reset_index()
        )

        pbar.set_description(steps[0])
        pbar.update(1)

        if term_stats_df["R_pos"].isna().any():
            raise ValueError("Some terms have missing R_pos values.")

        if term_stats_df["R_struct"].isna().any():
            raise ValueError("Some terms have missing R_struct values.")

        term_stats_df["S"] = (
            config.alpha * term_stats_df["R_pos"]
            + (1 - config.alpha) * term_stats_df["R_struct"]
        )

        term_stats_df["Rank"] = (
            1
            + (1 - term_stats_df["S"])
            * (config.max_rank_value - 1)
        )

        pbar.set_description(steps[1])
        pbar.update(1)

        if config.use_users_for_frequency:
            term_stats_df["freq_for_quadrant"] = term_stats_df["n_users"]
            diffusion_denominator = n_unique_users
            diffusion_metric = "user-level diffusion"
        else:
            term_stats_df["freq_for_quadrant"] = term_stats_df["n_docs"]
            diffusion_denominator = n_unique_docs
            diffusion_metric = "document-level diffusion"

        term_stats_df["diffusion_penetration"] = np.where(
            diffusion_denominator > 0,
            (
                term_stats_df["freq_for_quadrant"]
                / diffusion_denominator
                * 100
            ).round(1),
            np.nan,
        )

        term_stats_df["user_penetration"] = np.where(
            n_unique_users > 0,
            (term_stats_df["n_users"] / n_unique_users * 100).round(1),
            np.nan,
        )

        term_stats_df["document_penetration"] = np.where(
            n_unique_docs > 0,
            (term_stats_df["n_docs"] / n_unique_docs * 100).round(1),
            np.nan,
        )

        term_stats_df["posts_per_user"] = np.where(
            term_stats_df["n_users"] > 0,
            (term_stats_df["n_docs"] / term_stats_df["n_users"]).round(2),
            np.nan,
        )

        pbar.set_description(steps[2])
        pbar.update(1)

        pos_thresholds_df = (
            term_stats_df
            .groupby(upos_col, observed=True, sort=False)
            .agg(
                AFE=("freq_for_quadrant", "mean"),
                AOE=("Rank", "mean"),
                n_terms=(term_col, "size"),
                n_tokens=("n_tokens", "sum"),
            )
            .reset_index()
            .rename(columns={upos_col: "upos"})
        )

        if upos_col != "upos":
            term_stats_df = term_stats_df.rename(columns={upos_col: "upos"})
        if term_col != "term":
            term_stats_df = term_stats_df.rename(columns={term_col: "term"})

        term_stats_df = term_stats_df.merge(
            pos_thresholds_df[["upos", "AFE", "AOE"]],
            on="upos",
            how="left",
        )

        pbar.set_description(steps[3])
        pbar.update(1)

        term_stats_df["diffusion_class"] = np.where(
            term_stats_df["freq_for_quadrant"] >= term_stats_df["AFE"],
            "high_diffusion",
            "low_diffusion",
        )

        term_stats_df["salience_class"] = np.where(
            term_stats_df["Rank"] <= term_stats_df["AOE"],
            "high_salience",
            "low_salience",
        )

        term_stats_df["quadrant"] = np.select(
            [
                (
                    term_stats_df["diffusion_class"].eq("high_diffusion")
                    & term_stats_df["salience_class"].eq("high_salience")
                ),
                (
                    term_stats_df["diffusion_class"].eq("high_diffusion")
                    & term_stats_df["salience_class"].eq("low_salience")
                ),
                (
                    term_stats_df["diffusion_class"].eq("low_diffusion")
                    & term_stats_df["salience_class"].eq("high_salience")
                ),
                (
                    term_stats_df["diffusion_class"].eq("low_diffusion")
                    & term_stats_df["salience_class"].eq("low_salience")
                ),
            ],
            [
                "central_nucleus",
                "first_periphery",
                "contrast_zone",
                "peripheral_system",
            ],
            default=pd.NA,
        )

        if human_readable_quadrants:
            term_stats_df["quadrant"] = term_stats_df["quadrant"].map(
                _normalise_quadrant_labels
            )

        quadrant_summary_df = (
            term_stats_df
            .groupby(["upos", "quadrant"], observed=True)
            .size()
            .reset_index(name="n_terms")
            .sort_values(["upos", "quadrant"])
            .reset_index(drop=True)
        )

        pbar.set_description(steps[4])
        pbar.update(1)

    ordered_cols = [
        "term",
        "upos",
        "term_type",
        "n_tokens",
        "n_docs",
        "n_users",
        "R_pos",
        "R_struct",
        "S",
        "Rank",
        "freq_for_quadrant",
        "diffusion_penetration",
        "user_penetration",
        "document_penetration",
        "posts_per_user",
        "first_mention",
        "last_mention",
        "AFE",
        "AOE",
        "diffusion_class",
        "salience_class",
        "quadrant",
    ]

    remaining_cols = [
        col for col in term_stats_df.columns
        if col not in ordered_cols
    ]

    term_stats_df = term_stats_df.loc[:, ordered_cols + remaining_cols]

    term_stats_df = term_stats_df.sort_values(
        ["upos", "diffusion_penetration", "Rank"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    print(f"Term statistics computed using {diffusion_metric}.")
    print(f"Retained focal tokens: {len(df):,}")
    print(f"Retained term-UPOS pairs: {len(term_stats_df):,}")

    gc.collect()

    return term_stats_df, pos_thresholds_df, quadrant_summary_df


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
    return_thresholds: bool = False,
    return_quadrants: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Backward-compatible wrapper for term-level index computation."""

    term_stats_df, pos_thresholds_df, quadrant_summary_df = compute_term_statistics(
        tokens,
        term_col=term_col,
        upos_col=upos_col,
        user_col=user_col,
        doc_col=doc_col,
        r_pos_col=r_pos_col,
        r_str_col=r_str_col,
        config=config,
    )

    if return_thresholds or return_quadrants:
        return term_stats_df, pos_thresholds_df, quadrant_summary_df

    return term_stats_df


def term_statistics_summary(
    term_stats: pd.DataFrame,
    *,
    upos_col: str = "upos",
    quadrant_col: str = "quadrant",
) -> pd.DataFrame:
    """Return a compact summary of term-level statistics by UPOS and quadrant."""

    required = {
        upos_col,
        quadrant_col,
        "n_tokens",
        "n_docs",
        "n_users",
        "Rank",
        "diffusion_penetration",
    }

    _require_columns(term_stats, required)

    return (
        term_stats
        .groupby([upos_col, quadrant_col], observed=True)
        .agg(
            n_terms=("term", "size"),
            n_tokens=("n_tokens", "sum"),
            median_rank=("Rank", "median"),
            mean_diffusion=("diffusion_penetration", "mean"),
            median_docs=("n_docs", "median"),
            median_users=("n_users", "median"),
        )
        .reset_index()
        .sort_values([upos_col, quadrant_col])
        .reset_index(drop=True)
    )
