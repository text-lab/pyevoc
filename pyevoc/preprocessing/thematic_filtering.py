"""Thematic corpus filtering using external anchor lists."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class ThematicFilterConfig:
    text_col: str = "text"
    min_anchor_hits: int = 1
    similarity_threshold: float = 0.10
    min_expansion_hits: int = 2
    max_features: int | None = 50000
    lowercase: bool = True
    ngram_range: tuple[int, int] = (1, 2)


def load_anchor_terms(path: str | Path, column: str | None = None) -> list[str]:
    """Load anchor terms from TXT, CSV/TSV, or JSON.

    TXT files should contain one term per line. CSV/TSV files use ``column`` if
    provided; otherwise the first column is used. JSON files may contain a list or a
    dictionary with an ``anchors`` key.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".txt", ".lst"}:
        terms = [line.strip() for line in p.read_text(encoding="utf-8").splitlines()]
    elif suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        frame = pd.read_csv(p, sep=sep)
        col = column or frame.columns[0]
        terms = frame[col].dropna().astype(str).tolist()
    elif suffix == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
        terms = obj.get("anchors", []) if isinstance(obj, dict) else obj
    else:
        raise ValueError(f"Unsupported anchor-list format: {suffix}")
    return sorted({t.strip().lower() for t in terms if str(t).strip()})


def _anchor_pattern(terms: list[str]) -> re.Pattern:
    escaped = sorted([re.escape(t) for t in terms], key=len, reverse=True)
    return re.compile(r"(?<!\w)(?:" + "|".join(escaped) + r")(?!\w)", re.IGNORECASE)


def direct_anchor_filter(df: pd.DataFrame, anchors: list[str], config: ThematicFilterConfig) -> pd.Series:
    pattern = _anchor_pattern(anchors)
    return df[config.text_col].fillna("").astype(str).apply(lambda x: len(pattern.findall(x)) >= config.min_anchor_hits)


def derive_expansion_terms(
    df: pd.DataFrame,
    anchors: list[str],
    config: ThematicFilterConfig = ThematicFilterConfig(),
) -> list[str]:
    """Derive distributional expansion terms using sparse count vectors.

    This is a transparent baseline expansion strategy rather than an embedding model.
    """
    texts = df[config.text_col].fillna("").astype(str)
    vectorizer = CountVectorizer(
        lowercase=config.lowercase,
        ngram_range=config.ngram_range,
        max_features=config.max_features,
        token_pattern=r"(?u)\b\w[\w\-]+\b",
    )
    X = vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names_out()
    anchor_idx = [i for i, v in enumerate(vocab) if v.lower() in set(anchors)]
    if not anchor_idx:
        return []
    anchor_profile = X[:, anchor_idx].sum(axis=1)
    sims = cosine_similarity(X.T, anchor_profile.reshape(1, -1)).ravel()
    terms = [vocab[i] for i, s in enumerate(sims) if s >= config.similarity_threshold and vocab[i] not in anchors]
    return sorted(set(terms))


def build_thematic_subset(
    df: pd.DataFrame,
    anchor_file: str | Path,
    config: ThematicFilterConfig = ThematicFilterConfig(),
    expansion_terms: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Return a thematic subcorpus and metadata.

    Rows are retained if they contain anchor terms directly or, when expansion terms are
    available, at least ``min_expansion_hits`` expansion terms.
    """
    anchors = load_anchor_terms(anchor_file)
    direct = direct_anchor_filter(df, anchors, config)
    if expansion_terms is None:
        expansion_terms = derive_expansion_terms(df.loc[direct].copy(), anchors, config) if direct.any() else []
    exp_pattern = _anchor_pattern(expansion_terms) if expansion_terms else None
    if exp_pattern:
        indirect = df[config.text_col].fillna("").astype(str).apply(
            lambda x: len(exp_pattern.findall(x)) >= config.min_expansion_hits
        )
    else:
        indirect = pd.Series(False, index=df.index)
    keep = direct | indirect
    meta = {
        "n_input": int(len(df)),
        "n_retained": int(keep.sum()),
        "n_anchors": int(len(anchors)),
        "n_expansion_terms": int(len(expansion_terms)),
        "anchors": anchors,
        "expansion_terms": expansion_terms,
    }
    return df.loc[keep].reset_index(drop=True), meta
