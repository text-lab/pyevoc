"""Basic text cleaning for social-media corpora."""
from __future__ import annotations

import re
import pandas as pd

URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.I)
WS_RE = re.compile(r"\s+")


def clean_text(text: str, *, lowercase: bool = False, remove_urls: bool = True) -> str:
    value = "" if text is None else str(text)
    if remove_urls:
        value = URL_RE.sub(" ", value)
    value = WS_RE.sub(" ", value).strip()
    return value.lower() if lowercase else value


def clean_corpus(df: pd.DataFrame, text_col: str = "text", output_col: str = "clean_text", **kwargs) -> pd.DataFrame:
    out = df.copy()
    out[output_col] = out[text_col].apply(lambda x: clean_text(x, **kwargs))
    return out
