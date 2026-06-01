"""Emoji detection and token-level emoji assignment."""
from __future__ import annotations

import pandas as pd

try:
    import emoji as emoji_lib  # type: ignore
except Exception:  # pragma: no cover
    emoji_lib = None


def is_emoji_token(value: str) -> bool:
    if emoji_lib is None:
        return False
    return bool(value) and any(ch in emoji_lib.EMOJI_DATA for ch in str(value))


def assign_emoji_upos(tokens: pd.DataFrame, token_col: str = "text", upos_col: str = "upos") -> pd.DataFrame:
    out = tokens.copy()
    mask = out[token_col].fillna("").astype(str).apply(is_emoji_token)
    out.loc[mask, upos_col] = "EMOJI"
    out.loc[mask, "lemma"] = out.loc[mask, token_col]
    return out
