"""Concreteness labelling."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from pyevoc.utils.resources import model_path


def load_concreteness_lexicon(path: str | Path | None = None) -> pd.DataFrame:
    p = Path(path) if path else model_path("concreteness_lexicon.csv")
    return pd.read_csv(p)


def label_concreteness(
    terms: pd.DataFrame,
    lexicon: pd.DataFrame | None = None,
    *,
    term_col: str = "term",
    lexicon_term_col: str | None = None,
    label_col: str = "concreteness",
) -> pd.DataFrame:
    lex = lexicon.copy() if lexicon is not None else load_concreteness_lexicon()
    if lexicon_term_col is None:
        candidates = [c for c in lex.columns if c.lower() in {"term", "word", "lemma"}]
        lexicon_term_col = candidates[0] if candidates else lex.columns[0]
    if label_col not in lex.columns:
        # use a plausible existing label column or leave numeric information if available
        candidates = [c for c in lex.columns if "conc" in c.lower() or "class" in c.lower()]
        label_col = candidates[0] if candidates else lex.columns[-1]
    lex = lex[[lexicon_term_col, label_col]].drop_duplicates().rename(columns={lexicon_term_col: term_col, label_col: "concreteness"})
    lex[term_col] = lex[term_col].astype(str).str.lower()
    out = terms.copy()
    out[term_col] = out[term_col].astype(str)
    return out.merge(lex, on=term_col, how="left")
