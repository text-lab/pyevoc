"""Emoji description labelling with Unicode codepoint fallback."""
from __future__ import annotations
from pathlib import Path
import unicodedata
import pandas as pd
from pyevoc.utils.resources import model_path


def codepoints(value: str) -> str:
    return " ".join(f"U+{ord(ch):04X}" for ch in str(value))


def unicode_description(value: str) -> str:
    names = []
    for ch in str(value):
        try:
            names.append(unicodedata.name(ch).lower())
        except ValueError:
            names.append(f"unknown {ord(ch):04X}")
    return "; ".join(names)


def load_emoji_lookup(path: str | Path | None = None) -> pd.DataFrame:
    p = Path(path) if path else model_path("emoji_lookup.csv")
    return pd.read_csv(p)


def label_emojis(terms: pd.DataFrame, lookup: pd.DataFrame | None = None, term_col: str = "term") -> pd.DataFrame:
    look = lookup.copy() if lookup is not None else load_emoji_lookup()
    emoji_col = "emoji" if "emoji" in look.columns else look.columns[0]
    desc_col = "description" if "description" in look.columns else look.columns[-1]
    look = look[[emoji_col, desc_col]].rename(columns={emoji_col: term_col, desc_col: "emoji_description"}).drop_duplicates()
    out = terms.merge(look, on=term_col, how="left")
    mask = out["emoji_description"].isna() & (out.get("upos", "") == "EMOJI")
    out.loc[mask, "emoji_description"] = out.loc[mask, term_col].apply(unicode_description)
    out["codepoints"] = out[term_col].apply(codepoints)
    return out
