"""Token-level structural foregrounding and positional salience.

This module computes the token-level indicators used by PyEvoc to reconstruct
the AOE-like salience component of computational Hierarchical Evocation
Analysis.

The module is metadata-preserving: every input column is retained and new
salience-related columns are appended. Identifiers such as ``user_id``,
``doc_id``, ``time`` and ``source`` are therefore not removed or modified.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd
from tqdm.auto import tqdm


EMPHASIS_RE = re.compile(
    r"(?:\*\*|__|<b>|</b>|<strong>|</strong>|_{2,}|\*{2,})",
    flags=re.IGNORECASE,
)

LIST_OR_QUOTE_RE = re.compile(
    r"^\s*(?:-|\*|>|•|–|—|\d+[\.\)]|[a-zA-Z][\.\)])"
)

INTENSIFICATION_RE = re.compile(
    r"(?:[A-Z]{3,}|!{2,}|\?{2,}|[!?]{3,})"
)


@dataclass(frozen=True)
class ForegroundingWeights:
    """Weights used to compute structural foregrounding salience."""

    opening_sentence: float = 0.35
    emphasis: float = 0.25
    list_or_quote: float = 0.20
    intensification: float = 0.20

    def normalised(self) -> "ForegroundingWeights":
        """Return a copy of the weights scaled so that they sum to 1.

        Raises:
            ValueError: If the weights sum to zero or a negative value.
        """
        total = (
            self.opening_sentence
            + self.emphasis
            + self.list_or_quote
            + self.intensification
        )

        if total <= 0:
            raise ValueError(
                "Foregrounding weights must sum to a positive value."
            )

        return ForegroundingWeights(
            opening_sentence=self.opening_sentence / total,
            emphasis=self.emphasis / total,
            list_or_quote=self.list_or_quote / total,
            intensification=self.intensification / total,
        )

    def as_dict(self) -> dict[str, float]:
        """Return the weights as a plain :class:`dict` keyed by indicator name."""
        return {
            "opening_sentence": self.opening_sentence,
            "emphasis": self.emphasis,
            "list_or_quote": self.list_or_quote,
            "intensification": self.intensification,
        }


@dataclass(frozen=True)
class ForegroundingConfig:
    """Configuration for token-level foregrounding indicators."""

    doc_col: str = "doc_id"
    user_col: str = "user_id"
    time_col: str = "time"
    source_col: str = "source"

    token_col: str = "text"
    sentence_col: str = "sentence_id"
    position_col: str = "position"

    opening_sentence_value: int = 1

    emphasis_pattern: re.Pattern = EMPHASIS_RE
    list_or_quote_pattern: re.Pattern = LIST_OR_QUOTE_RE
    intensification_pattern: re.Pattern = INTENSIFICATION_RE

    weights: ForegroundingWeights = ForegroundingWeights()

    keep_component_scores: bool = True
    show_progress: bool = True


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise :exc:`ValueError` if any of *columns* are absent from *df*.

    Args:
        df: DataFrame to inspect.
        columns: Column names that must be present.

    Raises:
        ValueError: If one or more columns are missing.
    """
    missing = [
        col for col in columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _safe_text_series(df: pd.DataFrame, token_col: str) -> pd.Series:
    """Return *token_col* from *df* as a string Series with NaN filled as empty strings.

    Args:
        df: Token-level DataFrame.
        token_col: Name of the column containing token strings.

    Returns:
        A :class:`~pandas.Series` of strings with no null values.

    Raises:
        ValueError: If *token_col* is not present in *df*.
    """
    if token_col not in df.columns:
        raise ValueError(f"Column '{token_col}' not found in token table.")

    return df[token_col].fillna("").astype(str)


def validate_token_metadata(
    tokens: pd.DataFrame,
    *,
    doc_col: str = "doc_id",
    user_col: str = "user_id",
    time_col: str = "time",
    require_user: bool = False,
    require_time: bool = False,
) -> None:
    """Validate that the token table preserves key metadata."""

    required = [doc_col]

    if require_user:
        required.append(user_col)

    if require_time:
        required.append(time_col)

    _require_columns(tokens, required)


def add_token_position(
    tokens: pd.DataFrame,
    *,
    doc_col: str = "doc_id",
    position_col: str = "position",
    overwrite: bool = False,
) -> pd.DataFrame:
    """Add or recompute one-based token positions within documents."""

    validate_token_metadata(tokens, doc_col=doc_col)

    out = tokens.copy()

    if overwrite or position_col not in out.columns:
        out[position_col] = (
            out.groupby(doc_col, sort=False)
            .cumcount()
            .add(1)
        )

    return out


def add_foregrounding_indicators(
    tokens: pd.DataFrame,
    *,
    sentence_col: str = "sentence_id",
    token_col: str = "text",
    opening_sentence_value: int = 1,
    emphasis_pattern: re.Pattern = EMPHASIS_RE,
    list_or_quote_pattern: re.Pattern = LIST_OR_QUOTE_RE,
    intensification_pattern: re.Pattern = INTENSIFICATION_RE,
) -> pd.DataFrame:
    """Add binary foregrounding indicators to a token table."""

    _require_columns(tokens, [sentence_col, token_col])

    out = tokens.copy()

    text = _safe_text_series(out, token_col)

    out["fg_opening_sentence"] = (
        out[sentence_col]
        .fillna(-1)
        .eq(opening_sentence_value)
        .astype(int)
    )

    # Use pattern strings rather than compiled patterns to avoid pandas
    # group-extraction warnings in older pandas versions.
    out["fg_emphasis"] = (
        text
        .str.contains(emphasis_pattern.pattern, regex=True, na=False)
        .astype(int)
    )

    out["fg_list_or_quote"] = (
        text
        .str.match(list_or_quote_pattern.pattern, na=False)
        .astype(int)
    )

    out["fg_intensification"] = (
        text
        .str.contains(intensification_pattern.pattern, regex=True, na=False)
        .astype(int)
    )

    return out


def compute_structural_salience(
    tokens: pd.DataFrame,
    weights: ForegroundingWeights = ForegroundingWeights(),
    *,
    keep_component_scores: bool = True,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Compute weighted structural salience ``r_str``."""

    required_indicators = [
        "fg_opening_sentence",
        "fg_emphasis",
        "fg_list_or_quote",
        "fg_intensification",
    ]

    if any(col not in tokens.columns for col in required_indicators):
        out = add_foregrounding_indicators(tokens)
    else:
        out = tokens.copy()

    w = weights.normalised()

    with tqdm(
        total=1,
        desc="Computing structural salience",
        leave=False,
        disable=not show_progress,
    ) as pbar:
        opening_component = (
            w.opening_sentence * out["fg_opening_sentence"]
        )
        emphasis_component = (
            w.emphasis * out["fg_emphasis"]
        )
        list_component = (
            w.list_or_quote * out["fg_list_or_quote"]
        )
        intensification_component = (
            w.intensification * out["fg_intensification"]
        )

        if keep_component_scores:
            out["r_str_opening"] = opening_component
            out["r_str_emphasis"] = emphasis_component
            out["r_str_list_or_quote"] = list_component
            out["r_str_intensification"] = intensification_component

        out["r_str"] = (
            opening_component
            + emphasis_component
            + list_component
            + intensification_component
        )

        pbar.update(1)

    return out


def add_positional_salience(
    tokens: pd.DataFrame,
    doc_col: str = "doc_id",
    position_col: str = "position",
    *,
    output_col: str = "r_pos",
    show_progress: bool = True,
) -> pd.DataFrame:
    """Compute positional salience ``r_pos`` within documents."""

    validate_token_metadata(tokens, doc_col=doc_col)

    out = add_token_position(
        tokens,
        doc_col=doc_col,
        position_col=position_col,
        overwrite=False,
    )

    with tqdm(
        total=1,
        desc="Computing positional salience",
        leave=False,
        disable=not show_progress,
    ) as pbar:
        lengths = (
            out.groupby(doc_col, sort=False)[position_col]
            .transform("max")
        )

        out[output_col] = 1.0

        mask = lengths > 1

        out.loc[mask, output_col] = 1 - (
            (out.loc[mask, position_col] - 1)
            / (lengths.loc[mask] - 1)
        )

        pbar.update(1)

    return out


def add_salience_indicators(
    tokens: pd.DataFrame,
    *,
    config: ForegroundingConfig | None = None,
) -> pd.DataFrame:
    """Add foregrounding, structural salience and positional salience."""

    cfg = config or ForegroundingConfig()

    validate_token_metadata(
        tokens,
        doc_col=cfg.doc_col,
        user_col=cfg.user_col,
        time_col=cfg.time_col,
        require_user=False,
        require_time=False,
    )

    out = add_token_position(
        tokens,
        doc_col=cfg.doc_col,
        position_col=cfg.position_col,
        overwrite=False,
    )

    out = add_foregrounding_indicators(
        out,
        sentence_col=cfg.sentence_col,
        token_col=cfg.token_col,
        opening_sentence_value=cfg.opening_sentence_value,
        emphasis_pattern=cfg.emphasis_pattern,
        list_or_quote_pattern=cfg.list_or_quote_pattern,
        intensification_pattern=cfg.intensification_pattern,
    )

    out = compute_structural_salience(
        out,
        weights=cfg.weights,
        keep_component_scores=cfg.keep_component_scores,
        show_progress=cfg.show_progress,
    )

    out = add_positional_salience(
        out,
        doc_col=cfg.doc_col,
        position_col=cfg.position_col,
        output_col="r_pos",
        show_progress=cfg.show_progress,
    )

    return out


def foregrounding_diagnostics(
    tokens: pd.DataFrame,
    *,
    doc_col: str = "doc_id",
    user_col: str = "user_id",
    time_col: str = "time",
) -> pd.DataFrame:
    """Return diagnostics for foregrounding and salience indicators."""

    n_tokens = int(len(tokens))

    rows: list[tuple[str, object]] = [
        ("Tokens", n_tokens),
        (
            "Documents",
            int(tokens[doc_col].nunique(dropna=True))
            if doc_col in tokens.columns
            else None,
        ),
        (
            "Users",
            int(tokens[user_col].nunique(dropna=True))
            if user_col in tokens.columns
            else None,
        ),
        (
            "Minimum timestamp",
            pd.to_datetime(tokens[time_col], errors="coerce", utc=True).min()
            if time_col in tokens.columns
            else None,
        ),
        (
            "Maximum timestamp",
            pd.to_datetime(tokens[time_col], errors="coerce", utc=True).max()
            if time_col in tokens.columns
            else None,
        ),
    ]

    indicator_cols = [
        "fg_opening_sentence",
        "fg_emphasis",
        "fg_list_or_quote",
        "fg_intensification",
    ]

    for col in indicator_cols:
        if col in tokens.columns:
            count = int(tokens[col].fillna(0).astype(int).sum())
            share = round(count / n_tokens * 100, 3) if n_tokens > 0 else 0.0
            rows.append((f"{col} tokens", count))
            rows.append((f"{col} share (%)", share))

    if "r_str" in tokens.columns:
        rows.extend(
            [
                ("Mean r_str", round(float(tokens["r_str"].mean()), 6)),
                ("Median r_str", round(float(tokens["r_str"].median()), 6)),
                ("Max r_str", round(float(tokens["r_str"].max()), 6)),
            ]
        )

    if "r_pos" in tokens.columns:
        rows.extend(
            [
                ("Mean r_pos", round(float(tokens["r_pos"].mean()), 6)),
                ("Median r_pos", round(float(tokens["r_pos"].median()), 6)),
                ("Min r_pos", round(float(tokens["r_pos"].min()), 6)),
                ("Max r_pos", round(float(tokens["r_pos"].max()), 6)),
            ]
        )

    return (
        pd.DataFrame(rows, columns=["Statistic", "Value"])
        .dropna(subset=["Value"])
        .reset_index(drop=True)
    )


def metadata_coverage(
    tokens: pd.DataFrame,
    *,
    required_metadata: list[str] | None = None,
) -> pd.DataFrame:
    """Return a diagnostic table describing metadata availability."""

    required_metadata = required_metadata or [
        "user_id",
        "doc_id",
        "time",
        "source",
    ]

    rows = []

    for col in required_metadata:
        present = col in tokens.columns

        non_missing = (
            int(tokens[col].notna().sum())
            if present
            else 0
        )

        coverage = (
            round(non_missing / len(tokens) * 100, 3)
            if present and len(tokens) > 0
            else 0.0
        )

        rows.append(
            {
                "column": col,
                "present": present,
                "non_missing": non_missing,
                "coverage_percent": coverage,
            }
        )

    return pd.DataFrame(rows)


def assert_metadata_preserved(
    before: pd.DataFrame,
    after: pd.DataFrame,
    *,
    metadata_cols: list[str] | None = None,
) -> None:
    """Raise an error if metadata columns were lost during transformation."""

    metadata_cols = metadata_cols or [
        "user_id",
        "doc_id",
        "time",
        "source",
    ]

    missing = [
        col for col in metadata_cols
        if col in before.columns and col not in after.columns
    ]

    if missing:
        raise RuntimeError(
            "Metadata columns were lost during transformation: "
            f"{missing}"
        )
