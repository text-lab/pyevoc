"""Unigram cleaning and selection."""
from __future__ import annotations

import pandas as pd

DEFAULT_UPOS = {"NOUN", "ADJ", "PROPN", "EMOJI"}


def select_unigrams(
    tokens: pd.DataFrame,
    *,
    lemma_col: str = "lemma",
    upos_col: str = "upos",
    keep_upos: set[str] | None = None,
    min_chars: int = 2,
    lowercase: bool = True,
) -> pd.DataFrame:
    keep_upos = keep_upos or DEFAULT_UPOS
    out = tokens.copy()
    out["term"] = out[lemma_col].fillna(out.get("text", "")).astype(str)
    if lowercase:
        mask_non_emoji = out[upos_col] != "EMOJI"
        out.loc[mask_non_emoji, "term"] = out.loc[mask_non_emoji, "term"].str.lower()
    out = out[out[upos_col].isin(keep_upos)]
    out = out[out["term"].str.len() >= min_chars]
    out = out[out["term"].str.strip().ne("")]
    return out.reset_index(drop=True)
