"""Temporal stability and quadrant mobility for PyEvoc.

This module reconstructs temporal EVOC/HEM dynamics from token-level data and
a reference EVOC quadrant table.

It supports three main periodisation strategies:

- ``custom``: user-supplied cuts or breaks;
- ``equal_days`` / ``equal_width``: periods with approximately equal calendar duration;
- ``equal_posts`` / ``equal_count``: periods with approximately equal numbers of posts/documents.

The module computes:

- period-specific term quadrants;
- stable nucleus terms;
- volatile core terms;
- quadrant mobility and stability indices;
- transition diagnostics;
- temporal engagement diagnostics;
- optional HTML report.

The implementation is compatible with the rebuilt PyEvoc package schema
(``doc_id``) and with the original notebook schema (``post_id``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import gc
import html
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


QUADRANT_ORDER = [
    "Central nucleus",
    "First periphery",
    "Contrast zone",
    "Peripheral system",
]

QUADRANT_ORDER_WITH_ABSENT = QUADRANT_ORDER + ["Absent"]

QUADRANT_COLOURS_HEX = {
    "Central nucleus": "#4f7fa6",
    "First periphery": "#4f9a55",
    "Contrast zone": "#b84848",
    "Peripheral system": "#8250a0",
    "Absent": "#a0a0a0",
}

DEFAULT_FOCAL_UPOS = ["NOUN", "ADJ", "EMOJI"]


@dataclass(frozen=True)
class TemporalStabilityConfig:
    """Configuration for temporal stability analysis."""

    n_periods: int = 4
    time_period_mode: str = "equal_posts"

    custom_time_breaks: list[str] | None = None
    custom_time_cuts: list[str] | None = None
    period_labels: list[str] | None = None
    right: bool = True

    alpha: float = 0.5
    max_rank_value: float = 5.0
    use_users_for_frequency: bool = True
    focal_upos: list[str] = field(default_factory=lambda: list(DEFAULT_FOCAL_UPOS))

    round_digits: int = 2
    diffusion_multiplier: float = 100.0

    time_col: str = "time"
    doc_col: str = "doc_id"
    user_col: str = "user_id"
    term_col: str = "term"
    upos_col: str = "upos"
    r_pos_col: str = "r_pos"
    r_str_col: str = "r_str"

    expected_min_timestamp: str | None = None
    expected_max_timestamp: str | None = None
    warn_on_time_range_mismatch: bool = False

    output_dir: str | Path = "evoc_outputs"
    write_html_report: bool = True
    show_tables: bool = False

    verbose: bool = True


def _display(df: pd.DataFrame, name: str | None = None) -> None:
    """Display a dataframe when running in notebooks, otherwise print it."""

    try:
        display(df)  # type: ignore[name-defined]
    except NameError:
        if name:
            print(f"\n{name}")
        print(df)


def _ensure_post_id_alias(df: pd.DataFrame, doc_col: str = "doc_id") -> pd.DataFrame:
    """Ensure both ``post_id`` and package ``doc_id`` conventions can be used."""

    out = df.copy()

    if "post_id" not in out.columns and doc_col in out.columns:
        out["post_id"] = out[doc_col]

    if doc_col not in out.columns and "post_id" in out.columns:
        out[doc_col] = out["post_id"]

    return out


def _ensure_salience_aliases(
    df: pd.DataFrame,
    *,
    r_pos_col: str = "r_pos",
    r_str_col: str = "r_str",
) -> pd.DataFrame:
    """Create notebook-compatible salience aliases when needed."""

    out = df.copy()

    if "norm_rank" not in out.columns and r_pos_col in out.columns:
        out["norm_rank"] = out[r_pos_col]

    if "structural_score" not in out.columns and r_str_col in out.columns:
        out["structural_score"] = out[r_str_col]

    if r_pos_col not in out.columns and "norm_rank" in out.columns:
        out[r_pos_col] = out["norm_rank"]

    if r_str_col not in out.columns and "structural_score" in out.columns:
        out[r_str_col] = out["structural_score"]

    return out


def format_float(x: object, digits: int = 3) -> str:
    """Format a float for reporting."""

    if x is None or pd.isna(x):
        return "-"

    return f"{float(x):.{digits}f}"


def format_int(x: object) -> str:
    """Format an integer for reporting."""

    if x is None or pd.isna(x):
        return "-"

    return f"{int(round(float(x))):,}"


def quadrant_entropy(x: pd.Series | list[object]) -> float:
    """Compute entropy of quadrant memberships."""

    s = pd.Series(x).dropna()

    if len(s) == 0:
        return np.nan

    p = s.value_counts(normalize=True)

    return float(-(p * np.log(p)).sum())


def gini_coefficient(x: np.ndarray | list[float]) -> float:
    """Compute the Gini coefficient."""

    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) < 2:
        return np.nan

    if np.all(arr == 0):
        return 0.0

    arr = np.sort(arr)
    n = len(arr)

    return float(
        (2 * np.sum(arr * np.arange(1, n + 1)) / (n * np.sum(arr)))
        - ((n + 1) / n)
    )


def safe_std(x: pd.Series | list[float], ddof: int = 1) -> float:
    """Return standard deviation with safe handling of degenerate inputs."""

    s = pd.to_numeric(pd.Series(x), errors="coerce")

    if np.sum(np.isfinite(s)) <= ddof:
        return 0.0

    return float(np.nanstd(s, ddof=ddof))


def safe_spearman(x: pd.Series | list[float], y: pd.Series | list[float]) -> float:
    """Compute Spearman correlation with safe handling of constants."""

    xs = pd.Series(x).fillna(0)
    ys = pd.Series(y).fillna(0)

    if xs.nunique(dropna=True) <= 1 or ys.nunique(dropna=True) <= 1:
        return np.nan

    rho, _ = spearmanr(xs, ys)

    return float(rho) if np.isfinite(rho) else np.nan


def quarter_label_from_timestamp(ts: object) -> str:
    """Return a quarterly label from a timestamp."""

    t = pd.Timestamp(ts)

    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)

    return str(t.to_period("Q"))


def quarter_start_utc(ts: object) -> pd.Timestamp:
    """Return the UTC start of the quarter containing ``ts``."""

    t = pd.Timestamp(ts)

    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)

    return pd.Timestamp(t.to_period("Q").start_time, tz="UTC")


def next_quarter_start_utc(ts: object) -> pd.Timestamp:
    """Return the UTC start of the next quarter after ``ts``."""

    t = pd.Timestamp(ts)

    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)

    return pd.Timestamp((t.to_period("Q") + 1).start_time, tz="UTC")


def validate_time_range(
    observed_min: object,
    observed_max: object,
    expected_min: object | None = None,
    expected_max: object | None = None,
) -> dict[str, object]:
    """Compare observed and expected time ranges."""

    out = {
        "observed_min": observed_min,
        "observed_max": observed_max,
        "expected_min": expected_min,
        "expected_max": expected_max,
        "min_matches": None,
        "max_matches": None,
    }

    if expected_min is not None:
        exp_min = pd.to_datetime(expected_min, utc=True)
        out["expected_min"] = exp_min
        out["min_matches"] = pd.Timestamp(observed_min) == exp_min

    if expected_max is not None:
        exp_max = pd.to_datetime(expected_max, utc=True)
        out["expected_max"] = exp_max
        out["max_matches"] = pd.Timestamp(observed_max) == exp_max

    return out


def normalise_threshold_table(
    pos_thresholds_round_df: pd.DataFrame | None = None,
    evoc_quadrants_df: pd.DataFrame | None = None,
    *,
    round_digits: int = 2,
) -> pd.DataFrame:
    """Return internal POS-specific thresholds with exact and rounded columns."""

    if pos_thresholds_round_df is not None:
        thr = pos_thresholds_round_df.copy()

        rename_map = {}

        if "AFE_thr_round" in thr.columns and "AFE_thr" not in thr.columns:
            rename_map["AFE_thr_round"] = "AFE_thr"

        if "AOE_thr_round" in thr.columns and "AOE_thr" not in thr.columns:
            rename_map["AOE_thr_round"] = "AOE_thr"

        thr = thr.rename(columns=rename_map)

        required = {"upos", "AFE_thr", "AOE_thr"}
        missing = required.difference(thr.columns)

        if missing:
            raise ValueError(
                "pos_thresholds_round_df must contain ['upos','AFE_thr','AOE_thr'] "
                "or ['upos','AFE_thr_round','AOE_thr_round']. "
                f"Missing: {missing}"
            )

        thr = thr[["upos", "AFE_thr", "AOE_thr"]].copy()

    else:
        if evoc_quadrants_df is None:
            raise ValueError(
                "Either pos_thresholds_round_df or evoc_quadrants_df is required."
            )

        eq = evoc_quadrants_df.copy()

        if {"upos", "AFE_thr", "AOE_thr"}.issubset(eq.columns):
            thr = eq[["upos", "AFE_thr", "AOE_thr"]].drop_duplicates("upos").copy()
        else:
            required = {"upos", "relative_diffusion", "Rank"}
            missing = required.difference(eq.columns)

            if missing:
                raise ValueError(
                    f"Cannot infer POS-specific thresholds. Missing columns: {missing}"
                )

            eq["relative_diffusion"] = pd.to_numeric(
                eq["relative_diffusion"],
                errors="coerce",
            )
            eq["Rank"] = pd.to_numeric(eq["Rank"], errors="coerce")

            thr = (
                eq.dropna(subset=["relative_diffusion", "Rank"])
                .groupby("upos", observed=True, sort=False)
                .agg(
                    AFE_thr=("relative_diffusion", "mean"),
                    AOE_thr=("Rank", "mean"),
                )
                .reset_index()
            )

    thr["upos"] = thr["upos"].astype(str)
    thr["AFE_thr"] = pd.to_numeric(thr["AFE_thr"], errors="coerce")
    thr["AOE_thr"] = pd.to_numeric(thr["AOE_thr"], errors="coerce")
    thr = thr.dropna(subset=["upos", "AFE_thr", "AOE_thr"]).copy()

    if thr.empty:
        raise ValueError("The POS-specific threshold table is empty after cleaning.")

    thr["AFE_thr_r"] = thr["AFE_thr"].round(round_digits)
    thr["AOE_thr_r"] = thr["AOE_thr"].round(round_digits)

    return thr.reset_index(drop=True)


def build_threshold_display_table(thresholds: pd.DataFrame) -> pd.DataFrame:
    """Return a display-ready threshold table."""

    out = thresholds[["upos", "AFE_thr_r", "AOE_thr_r"]].copy()

    return out.rename(
        columns={
            "upos": "UPOS",
            "AFE_thr_r": "AFE threshold",
            "AOE_thr_r": "AOE threshold",
        }
    )


def build_time_periods(
    df: pd.DataFrame,
    *,
    time_col: str = "time",
    doc_col: str = "doc_id",
    user_col: str = "user_id",
    n_periods: int = 4,
    mode: str = "equal_posts",
    custom_time_breaks: list[str] | None = None,
    custom_time_cuts: list[str] | None = None,
    period_labels: list[str] | None = None,
    right: bool = True,
    time_range_reference_df: pd.DataFrame | None = None,
    expected_min_timestamp: object | None = None,
    expected_max_timestamp: object | None = None,
    warn_on_time_range_mismatch: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.Timestamp], dict[str, object]]:
    """Assign observations to temporal periods.

    Supported modes
    ---------------
    ``custom``
        Use user-defined breaks or internal cuts.

    ``equal_days`` or ``equal_width``
        Generate periods with approximately equal calendar duration.

    ``equal_posts`` or ``equal_count``
        Generate periods with approximately equal numbers of unique posts.

    ``quarters``
        Generate calendar-quarter periods.
    """

    data = _ensure_post_id_alias(df, doc_col=doc_col)

    required = {time_col, "post_id", user_col}
    missing = required.difference(data.columns)

    if missing:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    data[time_col] = pd.to_datetime(data[time_col], errors="coerce", utc=True)
    data = data.dropna(subset=[time_col]).copy()

    if data.empty:
        raise ValueError("No valid time values after datetime conversion.")

    if time_range_reference_df is not None:
        ref = _ensure_post_id_alias(time_range_reference_df, doc_col=doc_col)

        if time_col not in ref.columns:
            raise ValueError(f"time_range_reference_df is missing '{time_col}'.")

        ref[time_col] = pd.to_datetime(ref[time_col], errors="coerce", utc=True)
        ref = ref.dropna(subset=[time_col])

        if ref.empty:
            raise ValueError("time_range_reference_df contains no valid timestamps.")

        tmin = ref[time_col].min()
        tmax = ref[time_col].max()

    else:
        tmin = data[time_col].min()
        tmax = data[time_col].max()

    range_check = validate_time_range(
        tmin,
        tmax,
        expected_min_timestamp,
        expected_max_timestamp,
    )

    if warn_on_time_range_mismatch:
        messages = []

        if range_check["min_matches"] is False:
            messages.append(
                f"Minimum timestamp differs: observed={tmin}, "
                f"expected={range_check['expected_min']}."
            )

        if range_check["max_matches"] is False:
            messages.append(
                f"Maximum timestamp differs: observed={tmax}, "
                f"expected={range_check['expected_max']}."
            )

        if messages:
            warnings.warn(" ".join(messages))

    mode = mode.lower()

    if mode in {"equal_width", "equal_days", "same_days"}:
        breaks = pd.date_range(
            start=tmin,
            end=tmax,
            periods=n_periods + 1,
            tz="UTC",
        ).to_list()

    elif mode in {"equal_count", "equal_posts", "same_posts"}:
        posts = (
            data[["post_id", time_col]]
            .dropna(subset=["post_id", time_col])
            .drop_duplicates("post_id")
            .sort_values(time_col)
            .reset_index(drop=True)
        )

        if posts.empty:
            raise ValueError("No valid post identifiers for equal_posts periodisation.")

        probs = np.linspace(0, 1, n_periods + 1)
        quantile_positions = np.ceil(probs * (len(posts) - 1)).astype(int)
        breaks = posts.loc[quantile_positions, time_col].to_list()
        breaks[0] = tmin
        breaks[-1] = tmax

        for i in range(1, len(breaks)):
            if breaks[i] <= breaks[i - 1]:
                breaks[i] = breaks[i - 1] + pd.Timedelta(seconds=1)

    elif mode == "custom":
        if custom_time_breaks is not None and custom_time_cuts is not None:
            raise ValueError("Use either custom_time_breaks or custom_time_cuts, not both.")

        if custom_time_breaks is not None:
            breaks = (
                pd.Series(pd.to_datetime(custom_time_breaks, errors="coerce", utc=True))
                .dropna()
                .sort_values()
                .drop_duplicates()
                .to_list()
            )

            if len(breaks) < 2:
                raise ValueError(
                    "custom_time_breaks must contain at least two valid date-time values."
                )

        elif custom_time_cuts is not None:
            cuts = (
                pd.Series(pd.to_datetime(custom_time_cuts, errors="coerce", utc=True))
                .dropna()
                .sort_values()
                .drop_duplicates()
                .to_list()
            )
            cuts = [cut for cut in cuts if tmin < cut < tmax]
            breaks = [tmin] + cuts + [tmax]

        else:
            raise ValueError(
                "For mode='custom', provide custom_time_breaks or custom_time_cuts."
            )

        n_periods = len(breaks) - 1

    elif mode == "quarters":
        start_q = quarter_start_utc(tmin)
        end_q = next_quarter_start_utc(tmax)
        breaks = pd.date_range(start=start_q, end=end_q, freq="QS", tz="UTC").to_list()
        n_periods = len(breaks) - 1

    else:
        raise ValueError(
            "mode must be one of: 'custom', 'equal_days', 'equal_width', "
            "'equal_posts', 'equal_count', 'same_days', 'same_posts', 'quarters'."
        )

    if period_labels is None:
        labels = (
            [quarter_label_from_timestamp(b) for b in breaks[:-1]]
            if mode == "quarters"
            else [f"P{i}" for i in range(1, n_periods + 1)]
        )
    else:
        if len(period_labels) != n_periods:
            raise ValueError(
                f"period_labels length must equal generated periods. "
                f"Expected {n_periods}, got {len(period_labels)}."
            )

        labels = list(period_labels)

    data["time_period"] = pd.cut(
        data[time_col],
        bins=breaks,
        labels=labels,
        include_lowest=True,
        right=right,
    )

    period_ranges = pd.DataFrame(
        {
            "period": labels,
            "start": pd.to_datetime(breaks[:-1], utc=True),
            "end": pd.to_datetime(breaks[1:], utc=True),
        }
    )

    period_ranges["start_fmt"] = period_ranges["start"].dt.strftime("%Y-%m-%d")
    period_ranges["end_fmt"] = period_ranges["end"].dt.strftime("%Y-%m-%d")
    period_ranges["n_days"] = (
        period_ranges["end"].dt.date
        - period_ranges["start"].dt.date
    ).map(lambda x: x.days + 1)

    start_naive = period_ranges["start"].dt.tz_localize(None)
    end_naive = (period_ranges["end"] - pd.Timedelta(seconds=1)).dt.tz_localize(None)

    period_ranges["quarter_start"] = start_naive.dt.to_period("Q").astype(str)
    period_ranges["quarter_end"] = end_naive.dt.to_period("Q").astype(str)

    valid = data.dropna(subset=["time_period"])

    posts_per_period = (
        valid.drop_duplicates(["time_period", "post_id"])
        .groupby("time_period", observed=True)["post_id"]
        .nunique()
        .reset_index(name="n_posts")
        .rename(columns={"time_period": "period"})
    )

    users_per_period = (
        valid.drop_duplicates(["time_period", user_col])
        .groupby("time_period", observed=True)[user_col]
        .nunique()
        .reset_index(name="n_users")
        .rename(columns={"time_period": "period"})
    )

    posts_per_period["period"] = posts_per_period["period"].astype(str)
    users_per_period["period"] = users_per_period["period"].astype(str)
    period_ranges["period"] = period_ranges["period"].astype(str)

    period_ranges = period_ranges.merge(posts_per_period, on="period", how="left")
    period_ranges = period_ranges.merge(users_per_period, on="period", how="left")
    period_ranges["n_posts"] = period_ranges["n_posts"].fillna(0).astype(int)
    period_ranges["n_users"] = period_ranges["n_users"].fillna(0).astype(int)

    return data, period_ranges, breaks, range_check


def _compute_period_denominator(
    period_tokens: pd.DataFrame,
    *,
    use_users_for_frequency: bool = True,
) -> int:
    """Return the denominator for period-level diffusion."""

    col = "user_id" if use_users_for_frequency else "post_id"

    return int(period_tokens[col].astype("string").nunique(dropna=True))


def compute_period_quadrants(
    tokens_df: pd.DataFrame,
    evoc_quadrants_df: pd.DataFrame,
    pos_thresholds_round_df: pd.DataFrame | None = None,
    *,
    alpha: float = 0.5,
    max_rank_value: float = 5.0,
    use_users_for_frequency: bool = True,
    focal_upos: list[str] | None = None,
    round_digits: int = 2,
    diffusion_multiplier: float = 100.0,
) -> pd.DataFrame:
    """Reconstruct period-specific term quadrants."""

    focal_upos = focal_upos or list(DEFAULT_FOCAL_UPOS)

    data = _ensure_post_id_alias(tokens_df)
    data = _ensure_salience_aliases(data)

    required = {
        "post_id",
        "user_id",
        "time_period",
        "term",
        "upos",
        "norm_rank",
        "structural_score",
    }

    missing = required.difference(data.columns)

    if missing:
        raise ValueError(f"tokens_df is missing required columns: {missing}")

    if not isinstance(data["time_period"].dtype, CategoricalDtype):
        raise ValueError(
            "tokens_df['time_period'] must be a pandas categorical created by "
            "build_time_periods()."
        )

    thr = normalise_threshold_table(
        pos_thresholds_round_df=pos_thresholds_round_df,
        evoc_quadrants_df=evoc_quadrants_df,
        round_digits=round_digits,
    )

    all_terms_ref = evoc_quadrants_df[["term", "upos"]].drop_duplicates().copy()
    all_terms_ref["upos"] = all_terms_ref["upos"].astype(str)
    all_terms_ref = all_terms_ref.loc[
        all_terms_ref["upos"].isin(focal_upos)
    ].copy()

    periods = list(data["time_period"].cat.categories)
    period_tables = []

    df = data.loc[data["upos"].astype(str).isin(focal_upos)].copy()
    df["upos"] = df["upos"].astype(str)
    df["norm_rank"] = pd.to_numeric(df["norm_rank"], errors="coerce")
    df["structural_score"] = pd.to_numeric(df["structural_score"], errors="coerce")
    df["S"] = alpha * df["norm_rank"] + (1 - alpha) * df["structural_score"]

    for period in tqdm(periods, desc="Computing period quadrants", leave=False):
        tok_p = df.loc[df["time_period"].astype(str).eq(str(period))].copy()

        if tok_p.empty:
            period_stats = all_terms_ref.copy()
            period_stats["frequency"] = 0
            period_stats["relative_diffusion"] = 0.0
            period_stats["Rank"] = np.nan

        else:
            freq_col = "user_id" if use_users_for_frequency else "post_id"
            denominator = _compute_period_denominator(
                tok_p,
                use_users_for_frequency=use_users_for_frequency,
            )

            period_stats = (
                tok_p
                .groupby(["term", "upos"], observed=True)
                .agg(
                    frequency=(freq_col, "nunique"),
                    Sbar=("S", "mean"),
                )
                .reset_index()
            )

            period_stats["Rank"] = (
                1
                + (1 - period_stats["Sbar"])
                * (max_rank_value - 1)
            )

            period_stats = period_stats.drop(columns="Sbar")

            period_stats = all_terms_ref.merge(
                period_stats,
                on=["term", "upos"],
                how="left",
            )

            period_stats["frequency"] = period_stats["frequency"].fillna(0)
            period_stats["relative_diffusion"] = np.where(
                denominator > 0,
                diffusion_multiplier * period_stats["frequency"] / denominator,
                np.nan,
            )

        period_stats = period_stats.merge(thr, on="upos", how="left")

        if period_stats[["AFE_thr", "AOE_thr"]].isna().any().any():
            missing_upos = sorted(
                period_stats.loc[
                    period_stats["AFE_thr"].isna()
                    | period_stats["AOE_thr"].isna(),
                    "upos",
                ].unique()
            )
            raise ValueError(f"Missing POS-specific thresholds for UPOS: {missing_upos}")

        period_stats["relative_diffusion_r"] = period_stats["relative_diffusion"].round(
            round_digits
        )
        period_stats["Rank_r"] = period_stats["Rank"].round(round_digits)

        period_stats["diffusion_class"] = np.where(
            period_stats["relative_diffusion_r"] >= period_stats["AFE_thr_r"],
            "high_diffusion",
            "low_diffusion",
        )

        period_stats["salience_class"] = np.where(
            period_stats["Rank_r"] <= period_stats["AOE_thr_r"],
            "high_salience",
            "low_salience",
        )

        period_stats["quadrant"] = np.select(
            [
                (
                    period_stats["diffusion_class"].eq("high_diffusion")
                    & period_stats["salience_class"].eq("high_salience")
                ),
                (
                    period_stats["diffusion_class"].eq("high_diffusion")
                    & period_stats["salience_class"].eq("low_salience")
                ),
                (
                    period_stats["diffusion_class"].eq("low_diffusion")
                    & period_stats["salience_class"].eq("high_salience")
                ),
                (
                    period_stats["diffusion_class"].eq("low_diffusion")
                    & period_stats["salience_class"].eq("low_salience")
                ),
            ],
            QUADRANT_ORDER,
            default="Peripheral system",
        )

        period_stats.loc[period_stats["frequency"].eq(0), "quadrant"] = "Absent"
        period_stats["period"] = str(period)

        period_tables.append(
            period_stats[
                [
                    "term",
                    "upos",
                    "period",
                    "quadrant",
                    "frequency",
                    "relative_diffusion",
                    "Rank",
                    "diffusion_class",
                    "salience_class",
                    "AFE_thr",
                    "AOE_thr",
                ]
            ]
        )

    return pd.concat(period_tables, ignore_index=True)


def add_term_diffusion_summary(
    term_table: pd.DataFrame,
    all_period_quadrants: pd.DataFrame,
) -> pd.DataFrame:
    """Add mean and maximum relative diffusion summaries."""

    if term_table.empty:
        return term_table.copy()

    summary = (
        all_period_quadrants
        .groupby(["term", "upos"], observed=True)
        .agg(
            mean_relative_diffusion=("relative_diffusion", "mean"),
            max_relative_diffusion=("relative_diffusion", "max"),
        )
        .reset_index()
    )

    out = term_table.merge(summary, on=["term", "upos"], how="left")

    return (
        out
        .sort_values(
            ["mean_relative_diffusion", "max_relative_diffusion", "term"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )


def compute_stability_metrics(
    all_period_quadrants: pd.DataFrame,
    n_periods: int,
) -> dict[str, object]:
    """Compute term-level and transition-level temporal stability metrics."""

    nucleus_terms = (
        all_period_quadrants.loc[
            all_period_quadrants["quadrant"].eq("Central nucleus")
        ]
        .groupby(["term", "upos"], observed=True)
        .agg(
            n_periods=("period", "nunique"),
            periods=("period", lambda x: ", ".join(map(str, x))),
        )
        .reset_index()
    )

    stable_nucleus = nucleus_terms.loc[
        nucleus_terms["n_periods"].eq(n_periods)
    ].copy()

    quadrant_stability = (
        all_period_quadrants
        .groupby(["term", "upos"], observed=True)
        .agg(
            quadrants=("quadrant", lambda x: " → ".join(pd.unique(x.astype(str)))),
            n_quadrant_changes=("quadrant", lambda x: x.nunique(dropna=True) - 1),
        )
        .reset_index()
    )

    modal_quadrant = (
        all_period_quadrants.loc[
            ~all_period_quadrants["quadrant"].eq("Absent")
        ]
        .groupby(["term", "upos"], observed=True)["quadrant"]
        .agg(lambda x: x.value_counts().idxmax())
        .reset_index(name="modal_quadrant")
    )

    quadrant_stability = quadrant_stability.merge(
        modal_quadrant,
        on=["term", "upos"],
        how="left",
    )
    quadrant_stability["modal_quadrant"] = (
        quadrant_stability["modal_quadrant"].fillna("Absent")
    )

    quadrant_counts = (
        quadrant_stability
        .groupby("modal_quadrant", observed=True)
        .agg(
            n_stable=("n_quadrant_changes", lambda x: int((x == 0).sum())),
            n_volatile=("n_quadrant_changes", lambda x: int((x > 0).sum())),
        )
        .reset_index()
        .rename(columns={"modal_quadrant": "quadrant"})
    )

    quadrant_counts["quadrant"] = pd.Categorical(
        quadrant_counts["quadrant"],
        categories=QUADRANT_ORDER_WITH_ABSENT,
        ordered=True,
    )
    quadrant_counts = quadrant_counts.sort_values("quadrant").reset_index(drop=True)

    core_by_period = (
        all_period_quadrants.loc[
            all_period_quadrants["quadrant"].eq("Central nucleus")
        ]
        .groupby(["term", "upos"], observed=True)
        .agg(
            n_core_periods=("period", "nunique"),
            core_periods=("period", lambda x: ", ".join(map(str, x))),
        )
        .reset_index()
    )

    volatile_core_terms = core_by_period.merge(
        quadrant_stability,
        on=["term", "upos"],
        how="left",
    )

    volatile_core_terms = (
        volatile_core_terms.loc[volatile_core_terms["n_core_periods"] < n_periods]
        .sort_values(
            ["n_quadrant_changes", "n_core_periods", "term"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )

    term_stability = (
        all_period_quadrants
        .groupby(["term", "upos"], observed=True)
        .agg(
            quadrant_entropy=("quadrant", quadrant_entropy),
            diffusion_sd=("relative_diffusion", safe_std),
            freq_sd=("frequency", safe_std),
            rank_sd=("Rank", safe_std),
            core_persistence=(
                "quadrant",
                lambda x: float(np.mean(x.eq("Central nucleus"))),
            ),
        )
        .reset_index()
    )

    components = pd.DataFrame(
        {
            "inv_entropy": 1 / (1 + term_stability["quadrant_entropy"].fillna(0)),
            "inv_diffusion_sd": 1 / (1 + term_stability["diffusion_sd"].fillna(0)),
            "inv_rank_sd": 1 / (1 + term_stability["rank_sd"].fillna(0)),
            "core_persistence": term_stability["core_persistence"].fillna(0),
        }
    )

    z = StandardScaler().fit_transform(components)
    raw = np.nanmean(z, axis=1)

    if np.nanmax(raw) == np.nanmin(raw):
        stability_index = np.repeat(0.5, len(raw))
    else:
        stability_index = (raw - np.nanmin(raw)) / (np.nanmax(raw) - np.nanmin(raw))

    term_stability["stability_index"] = stability_index
    term_stability["stability_class"] = pd.cut(
        term_stability["stability_index"],
        bins=[-np.inf, 0.33, 0.66, np.inf],
        labels=["unstable", "intermediate", "highly stable"],
    )

    term_stability = term_stability.merge(
        modal_quadrant,
        on=["term", "upos"],
        how="left",
    )
    term_stability["modal_quadrant"] = term_stability["modal_quadrant"].fillna(
        "Absent"
    )

    stability_summary = (
        term_stability.loc[
            term_stability["modal_quadrant"].ne("Absent")
            & term_stability["stability_index"].notna()
        ]
        .groupby("modal_quadrant", observed=True)
        .agg(
            n_terms=("term", "size"),
            median=("stability_index", "median"),
            q25=("stability_index", lambda x: x.quantile(0.25)),
            q75=("stability_index", lambda x: x.quantile(0.75)),
            mean_cp=("core_persistence", "mean"),
        )
        .reset_index()
        .rename(columns={"modal_quadrant": "quadrant"})
    )

    stability_summary["quadrant"] = pd.Categorical(
        stability_summary["quadrant"],
        categories=QUADRANT_ORDER,
        ordered=True,
    )
    stability_summary = stability_summary.sort_values("quadrant").reset_index(drop=True)

    stable_nucleus = stable_nucleus.rename(columns={"n_periods": "n_core_periods"})

    if "periods" in stable_nucleus.columns and "core_periods" not in stable_nucleus.columns:
        stable_nucleus["core_periods"] = stable_nucleus["periods"]

    stable_nucleus = stable_nucleus.drop(columns=["periods"], errors="ignore")

    stable_nucleus = stable_nucleus.merge(
        quadrant_stability[
            [
                "term",
                "upos",
                "n_quadrant_changes",
                "quadrants",
                "modal_quadrant",
            ]
        ],
        on=["term", "upos"],
        how="left",
    )

    stable_nucleus = add_term_diffusion_summary(
        stable_nucleus,
        all_period_quadrants,
    )

    stable_nucleus = stable_nucleus.sort_values(
        [
            "n_quadrant_changes",
            "n_core_periods",
            "mean_relative_diffusion",
            "max_relative_diffusion",
            "term",
        ],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)

    volatile_core_terms = add_term_diffusion_summary(
        volatile_core_terms,
        all_period_quadrants,
    )

    volatile_core_terms = volatile_core_terms.sort_values(
        [
            "n_quadrant_changes",
            "mean_relative_diffusion",
            "max_relative_diffusion",
            "term",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    periods = list(pd.unique(all_period_quadrants["period"]))
    transition_rows = []
    core_overlap = {}
    diffusion_correlation = {}

    for i in range(len(periods) - 1):
        p0, p1 = periods[i], periods[i + 1]

        a = set(
            all_period_quadrants.loc[
                all_period_quadrants["period"].eq(p0)
                & all_period_quadrants["quadrant"].eq("Central nucleus"),
                "term",
            ]
        )
        b = set(
            all_period_quadrants.loc[
                all_period_quadrants["period"].eq(p1)
                & all_period_quadrants["quadrant"].eq("Central nucleus"),
                "term",
            ]
        )

        jaccard = np.nan if len(a) == 0 and len(b) == 0 else len(a & b) / len(a | b)

        core_overlap[f"{p0}→{p1}"] = jaccard

        f0 = all_period_quadrants.loc[
            all_period_quadrants["period"].eq(p0),
            ["term", "upos", "relative_diffusion"],
        ].rename(columns={"relative_diffusion": "diff_t"})

        f1 = all_period_quadrants.loc[
            all_period_quadrants["period"].eq(p1),
            ["term", "upos", "relative_diffusion"],
        ].rename(columns={"relative_diffusion": "diff_t1"})

        merged = f0.merge(f1, on=["term", "upos"], how="outer").fillna(0)
        rho_d = safe_spearman(merged["diff_t"], merged["diff_t1"])

        diffusion_correlation[f"{p0}→{p1}"] = rho_d

        transition_rows.append(
            {
                "transition": f"{p0}→{p1}",
                "core_jaccard": jaccard,
                "diffusion_spearman": rho_d,
            }
        )

    transition_diagnostics = pd.DataFrame(transition_rows)

    return {
        "stable_nucleus": stable_nucleus,
        "quadrant_stability": quadrant_stability,
        "quadrant_counts": quadrant_counts,
        "volatile_core_terms": volatile_core_terms,
        "term_stability": term_stability,
        "stability_summary": stability_summary,
        "core_overlap": core_overlap,
        "diffusion_correlation": diffusion_correlation,
        "transition_diagnostics": transition_diagnostics,
    }


def compute_temporal_engagement(
    periodised_df: pd.DataFrame,
    *,
    time_col: str = "time",
    doc_col: str = "doc_id",
    user_col: str = "user_id",
) -> pd.DataFrame:
    """Compute engagement diagnostics at post/user level."""

    data = _ensure_post_id_alias(periodised_df, doc_col=doc_col)

    required = {"post_id", user_col, time_col, "time_period"}
    missing = required.difference(data.columns)

    if missing:
        raise ValueError(f"periodised_df is missing required columns: {missing}")

    base = (
        data[["post_id", user_col, time_col, "time_period"]]
        .dropna(subset=["post_id", user_col, "time_period"])
        .drop_duplicates(["post_id", user_col, "time_period"])
        .copy()
    )

    if base.empty:
        return pd.DataFrame()

    base[time_col] = pd.to_datetime(base[time_col], errors="coerce", utc=True)
    base = base.dropna(subset=[time_col]).copy()

    first_seen = base.groupby(user_col, observed=True)[time_col].min()

    rows = []

    for period in base["time_period"].cat.categories:
        sub = base.loc[base["time_period"].astype(str).eq(str(period))].copy()

        if sub.empty:
            rows.append(
                {
                    "period": str(period),
                    "n_posts": 0,
                    "n_users": 0,
                    "posts_per_user": np.nan,
                    "gini_posts": np.nan,
                    "top10_share": np.nan,
                    "new_users": 0,
                    "returning_users": 0,
                    "new_user_share": np.nan,
                }
            )
            continue

        n_posts = sub["post_id"].nunique()
        n_users = sub[user_col].nunique()

        user_counts = sub.groupby(user_col, observed=True)["post_id"].nunique().values

        if len(user_counts) > 0:
            sorted_counts = np.sort(user_counts)[::-1]
            k = max(1, int(np.ceil(0.10 * len(sorted_counts))))
            top10_share = sorted_counts[:k].sum() / sorted_counts.sum()
        else:
            top10_share = np.nan

        users_period = sub[user_col].dropna().unique()
        first_dates = first_seen.loc[users_period]
        p_start = sub[time_col].min()
        p_end = sub[time_col].max()

        new_users = int(((first_dates >= p_start) & (first_dates <= p_end)).sum())

        rows.append(
            {
                "period": str(period),
                "n_posts": n_posts,
                "n_users": n_users,
                "posts_per_user": n_posts / n_users if n_users > 0 else np.nan,
                "gini_posts": gini_coefficient(user_counts),
                "top10_share": top10_share,
                "new_users": new_users,
                "returning_users": max(0, n_users - new_users),
                "new_user_share": new_users / n_users if n_users > 0 else np.nan,
            }
        )

    return pd.DataFrame(rows)


def _format_html_value(value: object) -> str:
    """Format values for HTML output."""

    if value is None or pd.isna(value):
        return "-"

    if isinstance(value, (float, np.floating)):
        return f"{value:.3f}"

    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"

    return html.escape(str(value))


def dataframe_to_html_table(
    df: pd.DataFrame,
    title: str,
    *,
    max_rows: int = 100,
    cols: list[str] | None = None,
) -> str:
    """Convert a dataframe to a compact HTML table section."""

    if cols is not None:
        existing = [col for col in cols if col in df.columns]
        show = df[existing].head(max_rows).copy()
    else:
        show = df.head(max_rows).copy()

    if show.empty:
        show = pd.DataFrame({"message": ["No retained records"]})

    columns = list(show.columns)
    head = "".join(f"<th>{html.escape(str(col))}</th>" for col in columns)

    body_rows = []

    for _, row in show.iterrows():
        cells = []

        for col in columns:
            value = row[col]
            is_num = pd.api.types.is_numeric_dtype(show[col])
            cls = "num" if is_num else "txt"
            cells.append(f'<td class="{cls}">{_format_html_value(value)}</td>')

        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    return f"""
    <section class="card">
      <h2>{html.escape(title)}</h2>
      <table>
        <thead><tr>{head}</tr></thead>
        <tbody>{''.join(body_rows)}</tbody>
      </table>
    </section>
    """


def write_temporal_report_html(
    results: dict[str, object],
    output_file: str | Path,
    *,
    max_rows: int = 80,
) -> str:
    """Write a compact HTML report for temporal stability analysis."""

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    term_cols = [
        "term",
        "upos",
        "n_core_periods",
        "n_quadrant_changes",
        "periods",
        "core_periods",
        "quadrants",
        "mean_relative_diffusion",
        "max_relative_diffusion",
    ]

    sections = [
        dataframe_to_html_table(
            results.get("period_ranges", pd.DataFrame()),  # type: ignore[arg-type]
            "Temporal periods",
            max_rows=max_rows,
            cols=[
                "period",
                "start_fmt",
                "end_fmt",
                "quarter_start",
                "quarter_end",
                "n_days",
                "n_posts",
                "n_users",
            ],
        ),
        dataframe_to_html_table(
            results.get("pos_thresholds_display", pd.DataFrame()),  # type: ignore[arg-type]
            "POS-specific thresholds",
            max_rows=max_rows,
        ),
        dataframe_to_html_table(
            results.get("transition_diagnostics", pd.DataFrame()),  # type: ignore[arg-type]
            "Transition diagnostics",
            max_rows=max_rows,
            cols=["transition", "core_jaccard", "diffusion_spearman"],
        ),
        dataframe_to_html_table(
            results.get("stability_summary", pd.DataFrame()),  # type: ignore[arg-type]
            "Stability summary by modal quadrant",
            max_rows=max_rows,
        ),
        dataframe_to_html_table(
            results.get("quadrant_counts", pd.DataFrame()),  # type: ignore[arg-type]
            "Stable and volatile terms by modal quadrant",
            max_rows=max_rows,
        ),
        dataframe_to_html_table(
            results.get("engagement_time", pd.DataFrame()),  # type: ignore[arg-type]
            "Temporal engagement",
            max_rows=max_rows,
            cols=[
                "period",
                "n_posts",
                "n_users",
                "posts_per_user",
                "gini_posts",
                "top10_share",
                "new_users",
                "returning_users",
                "new_user_share",
            ],
        ),
        dataframe_to_html_table(
            results.get("stable_nucleus_top20", pd.DataFrame()),  # type: ignore[arg-type]
            "Stable nucleus terms: top 20 most stable, ordered by diffusion",
            max_rows=20,
            cols=term_cols,
        ),
        dataframe_to_html_table(
            results.get("volatile_core_terms_top20", pd.DataFrame()),  # type: ignore[arg-type]
            "Volatile core terms: top 20 most volatile, ordered by diffusion",
            max_rows=20,
            cols=term_cols,
        ),
    ]

    doc = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PyEvoc Temporal Stability Report</title>
<style>
body {{
    font-family: Arial, Helvetica, sans-serif;
    margin: 12px auto;
    padding: 0 6px;
    background: #ffffff;
    color: #222222;
    max-width: 950px;
}}

h1 {{
    font-size: 18px;
    margin: 0 0 4px 0;
    font-weight: 700;
}}

.card {{
    margin: 0 0 8px 0;
    break-inside: avoid;
}}

h2 {{
    font-size: 15px;
    font-weight: 700;
    padding: 5px 7px;
    background: #ebebeb;
    border: 1px solid #d8d8d8;
    border-bottom: 0;
    margin: 0;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    table-layout: fixed;
    border: 1px solid #d8d8d8;
}}

th {{
    font-size: 11px;
    padding: 4px 4px;
    background: #f4f4f4;
    border-bottom: 1px solid #dddddd;
    text-align: center;
    font-weight: 700;
}}

td {{
    font-size: 11.5px;
    padding: 3px 4px;
    border-bottom: 1px solid #eeeeee;
    vertical-align: top;
    line-height: 1.18;
    overflow-wrap: anywhere;
}}

td.num {{
    text-align: center;
    white-space: nowrap;
}}

td.txt {{
    text-align: center;
}}
</style>
</head>
<body>
<h1>PyEvoc temporal stability report</h1>
{''.join(sections)}
</body>
</html>
"""

    output_path.write_text(doc, encoding="utf-8")

    print(f"Saved: {output_path}")

    return str(output_path)


def run_temporal_stability_analysis(
    df: pd.DataFrame,
    evoc_quadrants_df: pd.DataFrame,
    pos_thresholds_round_df: pd.DataFrame | None = None,
    *,
    n_periods: int = 4,
    time_period_mode: str = "equal_posts",
    custom_time_breaks: list[str] | None = None,
    custom_time_cuts: list[str] | None = None,
    period_labels: list[str] | None = None,
    alpha: float = 0.5,
    max_rank_value: float = 5.0,
    use_users_for_frequency: bool = True,
    focal_upos: list[str] | None = None,
    round_digits: int = 2,
    diffusion_multiplier: float = 100.0,
    time_range_reference_df: pd.DataFrame | None = None,
    count_reference_df: pd.DataFrame | None = None,
    expected_min_timestamp: object | None = None,
    expected_max_timestamp: object | None = None,
    warn_on_time_range_mismatch: bool = False,
    output_dir: str | Path = "evoc_outputs",
    write_html_report: bool = True,
    show_tables: bool = True,
    config: TemporalStabilityConfig | None = None,
) -> dict[str, object]:
    """Run the complete temporal stability analysis."""

    if config is not None:
        n_periods = config.n_periods
        time_period_mode = config.time_period_mode
        custom_time_breaks = config.custom_time_breaks
        custom_time_cuts = config.custom_time_cuts
        period_labels = config.period_labels
        alpha = config.alpha
        max_rank_value = config.max_rank_value
        use_users_for_frequency = config.use_users_for_frequency
        focal_upos = config.focal_upos
        round_digits = config.round_digits
        diffusion_multiplier = config.diffusion_multiplier
        expected_min_timestamp = config.expected_min_timestamp
        expected_max_timestamp = config.expected_max_timestamp
        warn_on_time_range_mismatch = config.warn_on_time_range_mismatch
        output_dir = config.output_dir
        write_html_report = config.write_html_report
        show_tables = config.show_tables

    focal_upos = focal_upos or list(DEFAULT_FOCAL_UPOS)

    print("Temporal stability analysis started...")

    df_periodised, period_ranges, period_breaks, range_check = build_time_periods(
        df=df,
        time_col="time",
        n_periods=n_periods,
        mode=time_period_mode,
        custom_time_breaks=custom_time_breaks,
        custom_time_cuts=custom_time_cuts,
        period_labels=period_labels,
        time_range_reference_df=time_range_reference_df,
        expected_min_timestamp=expected_min_timestamp,
        expected_max_timestamp=expected_max_timestamp,
        warn_on_time_range_mismatch=warn_on_time_range_mismatch,
    )

    if count_reference_df is not None:
        ref_periodised, ref_period_ranges, _, _ = build_time_periods(
            df=count_reference_df,
            time_col="time",
            n_periods=n_periods,
            mode=time_period_mode,
            custom_time_breaks=custom_time_breaks,
            custom_time_cuts=custom_time_cuts,
            period_labels=period_labels,
            time_range_reference_df=time_range_reference_df,
            expected_min_timestamp=expected_min_timestamp,
            expected_max_timestamp=expected_max_timestamp,
            warn_on_time_range_mismatch=False,
        )

        period_ranges = period_ranges.drop(columns=["n_posts", "n_users"], errors="ignore")
        period_ranges = period_ranges.merge(
            ref_period_ranges[["period", "n_posts", "n_users"]],
            on="period",
            how="left",
        )
        period_ranges["n_posts"] = period_ranges["n_posts"].fillna(0).astype(int)
        period_ranges["n_users"] = period_ranges["n_users"].fillna(0).astype(int)
        engagement_reference = ref_periodised

    else:
        engagement_reference = df_periodised

    pos_thresholds_used = normalise_threshold_table(
        pos_thresholds_round_df=pos_thresholds_round_df,
        evoc_quadrants_df=evoc_quadrants_df,
        round_digits=round_digits,
    )

    pos_thresholds_display = build_threshold_display_table(pos_thresholds_used)

    all_period_quadrants = compute_period_quadrants(
        tokens_df=df_periodised,
        evoc_quadrants_df=evoc_quadrants_df,
        pos_thresholds_round_df=pos_thresholds_used,
        alpha=alpha,
        max_rank_value=max_rank_value,
        use_users_for_frequency=use_users_for_frequency,
        focal_upos=focal_upos,
        round_digits=round_digits,
        diffusion_multiplier=diffusion_multiplier,
    )

    stability_results = compute_stability_metrics(
        all_period_quadrants,
        n_periods=len(period_ranges),
    )

    engagement_df = compute_temporal_engagement(engagement_reference)

    if not engagement_df.empty:
        coherence = period_ranges[["period", "n_posts", "n_users"]].merge(
            engagement_df[["period", "n_posts", "n_users"]],
            on="period",
            how="left",
            suffixes=("_periods", "_engagement"),
        )
        coherence["posts_match"] = (
            coherence["n_posts_periods"] == coherence["n_posts_engagement"]
        )
        coherence["users_match"] = (
            coherence["n_users_periods"] == coherence["n_users_engagement"]
        )
    else:
        coherence = pd.DataFrame()

    term_cols = [
        "term",
        "upos",
        "n_core_periods",
        "n_quadrant_changes",
        "periods",
        "core_periods",
        "quadrants",
        "mean_relative_diffusion",
        "max_relative_diffusion",
    ]

    stable_top20_full = stability_results["stable_nucleus"].head(20).copy()
    volatile_top20_full = stability_results["volatile_core_terms"].head(20).copy()

    stable_top20 = stable_top20_full[
        [col for col in term_cols if col in stable_top20_full.columns]
    ].copy()

    volatile_top20 = volatile_top20_full[
        [col for col in term_cols if col in volatile_top20_full.columns]
    ].copy()

    results: dict[str, object] = {
        "period_quadrants": all_period_quadrants,
        "period_ranges": period_ranges,
        "period_breaks": period_breaks,
        "time_range_check": range_check,
        "pos_thresholds_used": pos_thresholds_used,
        "pos_thresholds_display": pos_thresholds_display,
        "engagement_time": engagement_df,
        "period_engagement_coherence": coherence,
        "stable_nucleus_top20": stable_top20,
        "volatile_core_terms_top20": volatile_top20,
        "stable_nucleus_top20_full": stable_top20_full,
        "volatile_core_terms_top20_full": volatile_top20_full,
        **stability_results,
    }

    if write_html_report:
        report_html = Path(output_dir) / "evoc_temporal_stability.html"
        results["report_html"] = write_temporal_report_html(results, report_html)

    if show_tables:
        _display(period_ranges, "Temporal periods")
        _display(pos_thresholds_display, "POS-specific thresholds")
        _display(stability_results["transition_diagnostics"], "Transition diagnostics")
        _display(stability_results["stability_summary"], "Stability summary")
        _display(stability_results["quadrant_counts"], "Quadrant stability counts")
        _display(engagement_df, "Temporal engagement")
        _display(coherence, "Period/engagement count coherence")
        _display(stable_top20, "Stable nucleus top 20")
        _display(volatile_top20, "Volatile core terms top 20")

    print("Temporal stability analysis complete.")

    gc.collect()

    return results


def quadrant_trajectories(
    period_terms: pd.DataFrame,
    term_col: str = "term",
    period_col: str = "period",
    quadrant_col: str = "quadrant",
) -> pd.DataFrame:
    """Backward-compatible utility returning quadrant trajectories."""

    with tqdm(total=3, desc="Computing quadrant trajectories") as pbar:
        wide = period_terms.pivot_table(
            index=term_col,
            columns=period_col,
            values=quadrant_col,
            aggfunc="first",
        )
        pbar.update(1)

        out = wide.reset_index()
        out["quadrant_trajectory"] = wide.apply(
            lambda row: " → ".join(
                [str(value) if pd.notna(value) else "Absent" for value in row]
            ),
            axis=1,
        ).values
        pbar.update(1)

        out["quadrant_changes"] = wide.apply(
            lambda row: sum(
                1
                for a, b in zip(row.tolist(), row.tolist()[1:])
                if a != b
            ),
            axis=1,
        ).values
        pbar.update(1)

    return out
