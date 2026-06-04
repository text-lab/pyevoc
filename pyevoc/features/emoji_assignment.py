"""Emoji detection, token-level emoji assignment, and diagnostics.

This module provides utilities for identifying emoji tokens in an annotated
token-level table and assigning them to the dedicated ``EMOJI`` UPOS category.

The module is intentionally conservative: a token is treated as an emoji if at
least one Unicode character or grapheme cluster contained in the token is found
in the emoji package metadata.

Main functions
--------------
is_emoji_token
    Detect whether a token contains emoji content.

assign_emoji_upos
    Relabel emoji tokens as UPOS = "EMOJI" and set their lemma equal to the
    original emoji token.

emoji_summary
    Produce a frequency/coverage table for emoji tokens.

emoji_assignment_diagnostics
    Produce a compact one-row diagnostic table for the emoji assignment step.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from tqdm.auto import tqdm

try:
    import emoji as emoji_lib  # type: ignore
except Exception:  # pragma: no cover
    emoji_lib = None


@dataclass(frozen=True)
class EmojiAssignmentConfig:
    """Configuration for emoji UPOS assignment and diagnostics."""

    token_col: str = "text"
    lemma_col: str = "lemma"
    upos_col: str = "upos"
    doc_col: str = "doc_id"
    user_col: str = "user_id"

    emoji_upos: str = "EMOJI"

    show_progress: bool = True
    verbose: bool = True
    top_n: int = 10


def is_emoji_token(value: object) -> bool:
    """Return True if a token contains emoji content.

    Parameters
    ----------
    value:
        Token value.

    Returns
    -------
    bool
        True if the token contains at least one emoji recognised by the
        ``emoji`` package.
    """

    if emoji_lib is None:
        return False

    if value is None or pd.isna(value):
        return False

    token = str(value)

    if token == "":
        return False

    return any(ch in emoji_lib.EMOJI_DATA for ch in token)


def _emoji_mask(
    tokens: pd.DataFrame,
    *,
    token_col: str,
    show_progress: bool = True,
) -> pd.Series:
    """Return a Boolean mask identifying emoji tokens."""

    if token_col not in tokens.columns:
        raise ValueError(f"Column '{token_col}' not found in token table.")

    values = tokens[token_col].fillna("").astype(str)

    if show_progress:
        mask_values = [
            is_emoji_token(value)
            for value in tqdm(
                values,
                desc="Detecting emojis",
                unit="token",
                leave=False,
            )
        ]
    else:
        mask_values = [
            is_emoji_token(value)
            for value in values
        ]

    return pd.Series(mask_values, index=tokens.index)


def emoji_summary(
    tokens: pd.DataFrame,
    *,
    token_col: str = "text",
    upos_col: str = "upos",
    doc_col: str = "doc_id",
    user_col: str = "user_id",
    emoji_upos: str = "EMOJI",
    top_n: int | None = None,
    only_assigned: bool = True,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Return a summary table of emoji usage.

    Parameters
    ----------
    tokens:
        Token-level dataframe.
    token_col:
        Column containing the original token text.
    upos_col:
        Column containing the UPOS tag.
    doc_col:
        Column containing document identifiers. If absent, document counts are
        omitted.
    user_col:
        Column containing user identifiers. If absent, user counts are omitted.
    emoji_upos:
        UPOS value used for emoji tokens.
    top_n:
        If provided, return only the top ``n`` emojis by frequency.
    only_assigned:
        If True, emojis are selected using ``upos_col == emoji_upos``. If False,
        emoji detection is recomputed from ``token_col``.
    show_progress:
        Whether to display a progress bar if emoji detection is recomputed.

    Returns
    -------
    pandas.DataFrame
        Emoji summary table with frequency and, when available, document and
        user coverage.
    """

    if token_col not in tokens.columns:
        raise ValueError(f"Column '{token_col}' not found in token table.")

    if only_assigned and upos_col in tokens.columns:
        mask = tokens[upos_col].eq(emoji_upos)
    else:
        mask = _emoji_mask(
            tokens,
            token_col=token_col,
            show_progress=show_progress,
        )

    emoji_tokens = tokens.loc[mask].copy()

    if emoji_tokens.empty:
        columns = ["emoji", "frequency"]
        if doc_col in tokens.columns:
            columns.append("documents")
        if user_col in tokens.columns:
            columns.append("users")
        columns.append("relative_frequency")
        return pd.DataFrame(columns=columns)

    total_emoji_tokens = len(emoji_tokens)

    summary = (
        emoji_tokens[token_col]
        .fillna("")
        .astype(str)
        .value_counts()
        .rename_axis("emoji")
        .reset_index(name="frequency")
    )

    if doc_col in emoji_tokens.columns:
        doc_counts = (
            emoji_tokens
            .groupby(token_col, dropna=False)[doc_col]
            .nunique(dropna=True)
            .rename("documents")
            .reset_index()
            .rename(columns={token_col: "emoji"})
        )
        summary = summary.merge(doc_counts, on="emoji", how="left")

    if user_col in emoji_tokens.columns:
        user_counts = (
            emoji_tokens
            .groupby(token_col, dropna=False)[user_col]
            .nunique(dropna=True)
            .rename("users")
            .reset_index()
            .rename(columns={token_col: "emoji"})
        )
        summary = summary.merge(user_counts, on="emoji", how="left")

    summary["relative_frequency"] = (
        summary["frequency"] / total_emoji_tokens
    )

    if top_n is not None:
        summary = summary.head(top_n)

    return summary.reset_index(drop=True)


def emoji_assignment_diagnostics(
    tokens: pd.DataFrame,
    *,
    token_col: str = "text",
    upos_col: str = "upos",
    doc_col: str = "doc_id",
    user_col: str = "user_id",
    emoji_upos: str = "EMOJI",
) -> pd.DataFrame:
    """Return one-row diagnostics for emoji assignment.

    Parameters
    ----------
    tokens:
        Token-level dataframe after emoji assignment.
    token_col:
        Column containing token text.
    upos_col:
        Column containing UPOS values.
    doc_col:
        Column containing document identifiers.
    user_col:
        Column containing user identifiers.
    emoji_upos:
        UPOS value assigned to emoji tokens.

    Returns
    -------
    pandas.DataFrame
        One-row diagnostic table.
    """

    n_tokens = int(len(tokens))

    if upos_col in tokens.columns:
        mask = tokens[upos_col].eq(emoji_upos)
    else:
        mask = _emoji_mask(
            tokens,
            token_col=token_col,
            show_progress=False,
        )

    n_emoji_tokens = int(mask.sum())

    emoji_share = (
        (n_emoji_tokens / n_tokens) * 100
        if n_tokens > 0
        else 0.0
    )

    unique_emojis = (
        int(tokens.loc[mask, token_col].nunique(dropna=True))
        if token_col in tokens.columns
        else 0
    )

    documents_with_emoji = (
        int(tokens.loc[mask, doc_col].nunique(dropna=True))
        if doc_col in tokens.columns
        else None
    )

    users_with_emoji = (
        int(tokens.loc[mask, user_col].nunique(dropna=True))
        if user_col in tokens.columns
        else None
    )

    diagnostics = {
        "tokens": n_tokens,
        "emoji_tokens": n_emoji_tokens,
        "emoji_share_percent": round(emoji_share, 3),
        "unique_emojis": unique_emojis,
        "documents_with_emoji": documents_with_emoji,
        "users_with_emoji": users_with_emoji,
    }

    return pd.DataFrame(
        {
            "Statistic": [
                "Tokens inspected",
                "Emoji tokens",
                "Emoji share (%)",
                "Unique emojis",
                "Documents with emoji",
                "Users with emoji",
            ],
            "Value": [
                diagnostics["tokens"],
                diagnostics["emoji_tokens"],
                diagnostics["emoji_share_percent"],
                diagnostics["unique_emojis"],
                diagnostics["documents_with_emoji"],
                diagnostics["users_with_emoji"],
            ],
        }
    ).dropna(subset=["Value"]).reset_index(drop=True)


def assign_emoji_upos(
    tokens: pd.DataFrame,
    token_col: str = "text",
    upos_col: str = "upos",
    lemma_col: str = "lemma",
    *,
    config: EmojiAssignmentConfig | None = None,
    return_diagnostics: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assign ``UPOS='EMOJI'`` to emoji tokens.

    Parameters
    ----------
    tokens:
        Token-level dataframe.
    token_col:
        Column containing token text. Ignored when ``config`` is supplied.
    upos_col:
        Column containing UPOS tags. Ignored when ``config`` is supplied.
    lemma_col:
        Column containing lemma values. Ignored when ``config`` is supplied.
    config:
        Optional :class:`EmojiAssignmentConfig`.
    return_diagnostics:
        If True, return ``(tokens, diagnostics, summary)``. If False, return
        only the modified token dataframe.

    Returns
    -------
    pandas.DataFrame or tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        Modified token table, and optionally diagnostic and summary tables.
    """

    cfg = config or EmojiAssignmentConfig(
        token_col=token_col,
        upos_col=upos_col,
        lemma_col=lemma_col,
    )

    for col in (cfg.token_col, cfg.upos_col):
        if col not in tokens.columns:
            raise ValueError(f"Column '{col}' not found in token table.")

    out = tokens.copy()

    if cfg.lemma_col not in out.columns:
        out[cfg.lemma_col] = pd.NA

    mask = _emoji_mask(
        out,
        token_col=cfg.token_col,
        show_progress=cfg.show_progress,
    )

    out.loc[mask, cfg.upos_col] = cfg.emoji_upos
    out.loc[mask, cfg.lemma_col] = out.loc[mask, cfg.token_col]

    diagnostics = emoji_assignment_diagnostics(
        out,
        token_col=cfg.token_col,
        upos_col=cfg.upos_col,
        doc_col=cfg.doc_col,
        user_col=cfg.user_col,
        emoji_upos=cfg.emoji_upos,
    )

    summary = emoji_summary(
        out,
        token_col=cfg.token_col,
        upos_col=cfg.upos_col,
        doc_col=cfg.doc_col,
        user_col=cfg.user_col,
        emoji_upos=cfg.emoji_upos,
        top_n=cfg.top_n,
        only_assigned=True,
        show_progress=False,
    )

    if cfg.verbose:
        print("\nEmoji assignment summary")
        print("-" * 40)
        print(diagnostics.to_string(index=False))

        if not summary.empty:
            print("\nTop emojis")
            print("-" * 40)
            print(summary.to_string(index=False))
        else:
            print("\nNo emoji tokens detected.")

    if return_diagnostics:
        return out, diagnostics, summary

    return out
