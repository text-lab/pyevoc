"""Corpus-level descriptive statistics."""
from __future__ import annotations

import regex as re
import pandas as pd

TOKEN_RE = re.compile(r"\X", flags=re.UNICODE)
WORD_RE = re.compile(r"[\p{L}\p{N}_\-]+|\p{Emoji_Presentation}", flags=re.UNICODE)


def tokenize_simple(text: str) -> list[str]:
    return WORD_RE.findall(str(text).lower())


def corpus_statistics(df: pd.DataFrame, text_col: str = "text") -> dict:
    tokens = []
    for text in df[text_col].fillna("").astype(str):
        tokens.extend(tokenize_simple(text))
    n_tokens = len(tokens)
    types = set(tokens)
    hapax = sum(1 for t in set(tokens) if tokens.count(t) == 1) if tokens else 0
    return {
        "documents": int(len(df)),
        "tokens": int(n_tokens),
        "types": int(len(types)),
        "type_token_ratio": float(len(types) / n_tokens) if n_tokens else 0.0,
        "hapax": int(hapax),
        "hapax_share": float(hapax / len(types)) if types else 0.0,
    }
