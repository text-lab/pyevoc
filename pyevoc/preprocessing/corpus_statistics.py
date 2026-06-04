"""Corpus-level descriptive statistics.

This module computes descriptive statistics on raw textual data before
linguistic preprocessing. The tokenisation is Unicode-aware and preserves
words, contractions, numbers, hashtags, mentions, and emojis.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import regex as re
from tqdm.auto import tqdm


TOKEN_PATTERN = re.compile(
    r"""
    \p{L}+(?:['’]\p{L}+)?      # words and contractions
    |
    \p{N}+                     # numbers
    |
    [#@]?\p{L}[\p{L}\p{N}_]*   # hashtags and mentions
    |
    \X                         # Unicode grapheme clusters, including emojis
    """,
    flags=re.VERBOSE,
)


@dataclass(frozen=True)
class CorpusStatisticsConfig:
    """Configuration for corpus-level descriptive statistics."""

    text_col: str = "text"
    doc_col: str = "doc_id"
    user_col: str = "user_id"
    source_col: str = "source"
    time_col: str = "time"

    lowercase: bool = True
    show_progress: bool = True
    as_percent: bool = True


def simple_tokenise(text: object) -> list[str]:
    """Tokenise raw text using a Unicode-aware regular expression.

    The function preserves:
    - words;
    - contractions;
    - numbers;
    - hashtags;
    - mentions;
    - emojis and other Unicode grapheme clusters.
    """

    if pd.isna(text):
        return []

    return TOKEN_PATTERN.findall(str(text))


def is_valid_stat_token(token: object) -> bool:
    """Return True if a token should be retained for corpus statistics.

    Empty strings, whitespace-only tokens, and punctuation-only tokens are
    removed. Emojis and symbolic Unicode grapheme clusters are retained.
    """

    tok = str(token).strip()

    if tok == "":
        return False

    if re.fullmatch(r"\s+", tok):
        return False

    if re.fullmatch(r"\p{P}+", tok):
        return False

    return True


def extract_tokens(
    texts: Iterable[object],
    *,
    lowercase: bool = True,
    show_progress: bool = True,
) -> list[str]:
    """Extract valid tokens from a sequence of raw texts."""

    iterator = texts

    if show_progress:
        iterator = tqdm(
            texts,
            desc="Computing corpus statistics",
            unit="doc",
        )

    tokens: list[str] = []

    for text in iterator:
        toks = simple_tokenise(text)

        if lowercase:
            toks = [
                t.lower()
                for t in toks
                if is_valid_stat_token(t)
            ]
        else:
            toks = [
                t
                for t in toks
                if is_valid_stat_token(t)
            ]

        tokens.extend(toks)

    return tokens


def _format_value(value: object) -> object:
    """Return values in a clean display format without forcing strings."""

    if isinstance(value, float):
        return round(value, 3)

    return value


def corpus_statistics(
    df: pd.DataFrame,
    text_col: str = "text",
    *,
    doc_col: str = "doc_id",
    user_col: str = "user_id",
    time_col: str = "time",
    lowercase: bool = True,
    show_progress: bool = True,
    as_dataframe: bool = True,
) -> pd.DataFrame | dict:
    """Compute corpus-level descriptive statistics.

    Parameters
    ----------
    df:
        Input dataframe.
    text_col:
        Name of the raw text column.
    doc_col:
        Name of the document identifier column.
    user_col:
        Name of the user identifier column.
    source_col:
        Name of the source/community/platform column.
    time_col:
        Name of the timestamp column.
    lowercase:
        If True, tokens are lowercased before computing lexical statistics.
    show_progress:
        If True, display a progress bar during token extraction.
    as_dataframe:
        If True, return a display-ready pandas DataFrame. If False, return
        a dictionary.

    Returns
    -------
    pandas.DataFrame or dict
        Corpus-level descriptive statistics.
    """

    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in dataframe.")

    texts = df[text_col].fillna("").astype(str)

    tokens = extract_tokens(
        texts,
        lowercase=lowercase,
        show_progress=show_progress,
    )

    token_counts = Counter(tokens)

    n_documents = int(len(df))
    n_tokens = int(sum(token_counts.values()))
    n_types = int(len(token_counts))

    n_unique_docs = (
        int(df[doc_col].nunique(dropna=True))
        if doc_col in df.columns
        else None
    )

    n_users = (
        int(df[user_col].nunique(dropna=True))
        if user_col in df.columns
        else None
    )

    if time_col in df.columns:
        time_values = pd.to_datetime(
            df[time_col],
            errors="coerce",
            utc=True,
        )

        min_time = time_values.min()
        max_time = time_values.max()

        n_days = (
            int((max_time.normalize() - min_time.normalize()).days + 1)
            if pd.notna(min_time) and pd.notna(max_time)
            else None
        )
    else:
        min_time = None
        max_time = None
        n_days = None

    n_hapax = int(
        sum(1 for freq in token_counts.values() if freq == 1)
    )

    type_token_ratio = (
        (n_types / n_tokens) * 100
        if n_tokens > 0
        else 0.0
    )

    hapax_share = (
        (n_hapax / n_types) * 100
        if n_types > 0
        else 0.0
    )

    stats = {
        "documents": n_documents,
        "unique_documents": n_unique_docs,
        "unique_users": n_users,
        "minimum_timestamp": min_time,
        "maximum_timestamp": max_time,
        "number_of_days": n_days,
        "tokens": n_tokens,
        "types": n_types,
        "type_token_ratio_percent": round(type_token_ratio, 3),
        "hapax": n_hapax,
        "hapax_share_percent": round(hapax_share, 3),
    }

    if not as_dataframe:
        return stats

    display_names = {
        "documents": "Posts/documents",
        "unique_documents": "Unique document IDs",
        "unique_users": "Unique user IDs",
        "minimum_timestamp": "Minimum timestamp",
        "maximum_timestamp": "Maximum timestamp",
        "number_of_days": "Number of days",
        "tokens": "Tokens",
        "types": "Types",
        "type_token_ratio_percent": "Type/token ratio (%)",
        "hapax": "Hapax",
        "hapax_share_percent": "Hapax share (%)",
    }

    out = pd.DataFrame(
        {
            "Statistic": [
                display_names[k]
                for k in stats.keys()
                if stats[k] is not None
            ],
            "Value": [
                _format_value(v)
                for v in stats.values()
                if v is not None
            ],
        }
    )

    return out