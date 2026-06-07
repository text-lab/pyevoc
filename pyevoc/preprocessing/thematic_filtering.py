"""Generic thematic corpus filtering using anchor terms and distributional expansion.

This module implements the notebook-style thematic-subsetting procedure used in
PyEvoc:

1. normalise an external anchor inventory;
2. identify direct anchor matches with a compiled regular expression;
3. build a document-term matrix with ``CountVectorizer``;
4. identify vectorizer terms containing at least one anchor string;
5. compute cosine similarity between all terms and anchor-related terms;
6. select the most similar expansion terms;
7. retain documents with either direct anchor hits or enough expansion hits.

The implementation is deliberately generic. It does not assume a renewable-energy
domain and therefore uses neutral column names such as ``theme_subset_flag`` and
``filter_type``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm


@dataclass(frozen=True)
class ThematicFilterConfig:
    """Configuration for anchor-based thematic filtering."""

    text_col: str = "text"

    min_anchor_hits: int = 1
    similarity_threshold: float = 0.10
    min_expansion_hits: int = 2

    top_n_expansion_terms: int = 150

    min_df: int | float = 10
    max_df: int | float = 0.5
    max_features: int | None = 50000
    stop_words: str | list[str] | None = "english"
    lowercase: bool = True
    ngram_range: tuple[int, int] = (1, 2)

    anchor_col: str = "anchor_hit"
    anchor_hits_col: str = "anchor_hits"
    expansion_hits_col: str = "expansion_hits"
    expansion_col: str = "expansion_hit"
    subset_flag_col: str = "theme_subset_flag"
    filter_type_col: str = "filter_type"

    retain_columns: tuple[str, ...] | None = None
    verbose: bool = True
    show_progress: bool = True


# ---------------------------------------------------------------------------
# Anchor handling
# ---------------------------------------------------------------------------

def normalise_anchor_terms(anchors: Iterable[object]) -> list[str]:
    """Lowercase, strip, deduplicate and length-sort anchor terms."""

    return sorted(
        {
            str(anchor).strip().lower()
            for anchor in anchors
            if isinstance(anchor, str) and str(anchor).strip()
        },
        key=len,
        reverse=True,
    )


def load_anchor_terms(path: str | Path, column: str | None = None) -> list[str]:
    """Load anchor terms from TXT, CSV/TSV, or JSON.

    TXT files should contain one term per line. CSV/TSV files use ``column`` if
    provided; otherwise the first column is used. JSON files may contain a list
    or a dictionary with an ``anchors`` key.
    """

    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Anchor file not found: {p}")

    suffix = p.suffix.lower()

    if suffix in {".txt", ".lst"}:
        terms = [
            line.strip()
            for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    elif suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        frame = pd.read_csv(p, sep=sep)
        col = column or frame.columns[0]
        terms = frame[col].dropna().astype(str).tolist()

    elif suffix == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))

        if isinstance(obj, dict):
            terms = obj.get("anchors", [])
        else:
            terms = obj

    else:
        raise ValueError(f"Unsupported anchor-list format: {suffix}")

    return normalise_anchor_terms(terms)


def compile_anchor_pattern(anchors: list[str]) -> re.Pattern:
    """Compile the notebook-style anchor regex pattern."""

    if not anchors:
        raise ValueError("The anchor inventory is empty.")

    escaped = "|".join(map(re.escape, anchors))

    return re.compile(
        r"\b(?:" + escaped + r")\b",
        flags=re.IGNORECASE,
    )


# ---------------------------------------------------------------------------
# Direct anchor matching
# ---------------------------------------------------------------------------

def direct_anchor_filter(
    df: pd.DataFrame,
    anchors: list[str],
    config: ThematicFilterConfig,
) -> pd.DataFrame:
    """Add anchor hit indicators to a dataframe."""

    if config.text_col not in df.columns:
        raise ValueError(f"Input dataframe is missing text column {config.text_col!r}.")

    out = df.copy()
    anchors = normalise_anchor_terms(anchors)
    pattern = compile_anchor_pattern(anchors)

    texts = out[config.text_col].fillna("").astype(str)

    # ``str.count`` provides the hit count; ``str.contains`` reproduces the
    # original notebook's direct boolean filter.
    out[config.anchor_hits_col] = texts.str.count(pattern)
    out[config.anchor_col] = out[config.anchor_hits_col].ge(config.min_anchor_hits)

    return out


# ---------------------------------------------------------------------------
# Distributional expansion
# ---------------------------------------------------------------------------

def build_count_vectorizer(config: ThematicFilterConfig) -> CountVectorizer:
    """Construct the CountVectorizer used for expansion-term discovery."""

    return CountVectorizer(
        stop_words=config.stop_words,
        min_df=config.min_df,
        max_df=config.max_df,
        max_features=config.max_features,
        lowercase=config.lowercase,
        ngram_range=config.ngram_range,
    )


def derive_expansion_terms(
    df: pd.DataFrame,
    anchors: list[str],
    config: ThematicFilterConfig,
) -> tuple[set[str], pd.DataFrame, CountVectorizer, object]:
    """Derive expansion terms using notebook-style term-term cosine similarity.

    The function returns ``expansion_terms``, a diagnostic dataframe,
    the fitted vectorizer, and the document-term matrix.
    """

    if config.text_col not in df.columns:
        raise ValueError(f"Input dataframe is missing text column {config.text_col!r}.")

    anchors = normalise_anchor_terms(anchors)

    vectorizer = build_count_vectorizer(config)
    texts = df[config.text_col].fillna("").astype(str)

    X = vectorizer.fit_transform(texts)
    terms = np.array(vectorizer.get_feature_names_out())

    if len(terms) == 0:
        empty = pd.DataFrame(
            columns=[
                "term",
                "similarity",
                "is_anchor_related",
                "selected_expansion",
            ]
        )
        return set(), empty, vectorizer, X

    # This reproduces the original notebook logic:
    # a vectorizer feature is anchor-related if any anchor string is contained
    # in that feature.
    term_iterator = terms

    if config.show_progress:
        term_iterator = tqdm(
            terms,
            desc="Detecting anchor-related terms",
            unit="term",
        )

    anchor_mask = np.array(
        [
            any(anchor in term for anchor in anchors)
            for term in term_iterator
        ],
        dtype=bool,
    )

    anchor_indices = np.where(anchor_mask)[0]

    if len(anchor_indices) == 0:
        diagnostics = pd.DataFrame(
            {
                "term": terms,
                "similarity": np.zeros(len(terms), dtype=float),
                "is_anchor_related": anchor_mask,
                "selected_expansion": np.zeros(len(terms), dtype=bool),
            }
        )
        return set(), diagnostics, vectorizer, X

    term_matrix = X.T

    sim = cosine_similarity(
        term_matrix,
        term_matrix[anchor_indices],
    )

    sim_scores = np.asarray(sim.mean(axis=1)).reshape(-1)

    expansion_indices = np.where(sim_scores > config.similarity_threshold)[0]

    if len(expansion_indices) > 0:
        order = np.argsort(sim_scores[expansion_indices])[::-1]
        expansion_indices = expansion_indices[order[: config.top_n_expansion_terms]]

    expansion_terms = set(terms[expansion_indices])

    diagnostics = pd.DataFrame(
        {
            "term": terms,
            "similarity": sim_scores,
            "is_anchor_related": anchor_mask,
            "selected_expansion": np.isin(np.arange(len(terms)), expansion_indices),
        }
    ).sort_values(
        ["selected_expansion", "similarity", "term"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    return expansion_terms, diagnostics, vectorizer, X


def add_expansion_hits(
    df: pd.DataFrame,
    expansion_terms: set[str],
    vectorizer: "CountVectorizer",
    X: Any,
    config: "ThematicFilterConfig",
) -> pd.DataFrame:
    """Add expansion hit counts and boolean expansion-hit flags."""

    out = df.copy()

    if not expansion_terms:
        out[config.expansion_hits_col] = 0
        out[config.expansion_col] = False
        return out

    terms = np.array(vectorizer.get_feature_names_out())
    term_to_idx = {term: idx for idx, term in enumerate(terms)}

    exp_idx = [
        term_to_idx[term]
        for term in expansion_terms
        if term in term_to_idx
    ]

    if not exp_idx:
        out[config.expansion_hits_col] = 0
        out[config.expansion_col] = False
        return out

    expansion_hits = X[:, exp_idx].sum(axis=1)
    expansion_hits = np.asarray(expansion_hits).reshape(-1)

    out[config.expansion_hits_col] = expansion_hits
    out[config.expansion_col] = out[config.expansion_hits_col].ge(
        config.min_expansion_hits,
    )

    return out


# ---------------------------------------------------------------------------
# Main thematic-subset builder
# ---------------------------------------------------------------------------

def build_thematic_subset(
    df: pd.DataFrame,
    anchor_file: str | Path | None = None,
    config: ThematicFilterConfig = ThematicFilterConfig(),
    expansion_terms: set[str] | list[str] | None = None,
    anchors: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Build a thematic subcorpus using anchor and expansion filters.

    Parameters
    ----------
    df:
        Input document-level dataframe.
    anchor_file:
        Optional external anchor file. Required when ``anchors`` is not given.
    config:
        Filtering configuration.
    expansion_terms:
        Optional pre-computed expansion terms. If not supplied, expansion terms
        are derived from the input corpus.
    anchors:
        Optional in-memory anchor inventory. If supplied, ``anchor_file`` is not
        required.
    """

    if config.text_col not in df.columns:
        raise ValueError(f"Input dataframe is missing text column {config.text_col!r}.")

    progress = (
        tqdm(total=5, desc="Building thematic subset", unit="step")
        if config.show_progress
        else None
    )

    try:
        if anchors is None:
            if anchor_file is None:
                raise ValueError("Either anchor_file or anchors must be supplied.")
            anchors = load_anchor_terms(anchor_file)
        else:
            anchors = normalise_anchor_terms(anchors)

        if progress is not None:
            progress.update(1)

        out = direct_anchor_filter(df, anchors, config=config)

        if progress is not None:
            progress.update(1)

        vectorizer = None
        X = None
        expansion_diagnostics = pd.DataFrame()

        if expansion_terms is None:
            expansion_terms, expansion_diagnostics, vectorizer, X = derive_expansion_terms(
                out,
                anchors,
                config=config,
            )
        else:
            expansion_terms = set(normalise_anchor_terms(expansion_terms))

            vectorizer = build_count_vectorizer(config)
            X = vectorizer.fit_transform(out[config.text_col].fillna("").astype(str))

            expansion_diagnostics = pd.DataFrame(
                {
                    "term": vectorizer.get_feature_names_out(),
                    "similarity": np.nan,
                    "is_anchor_related": np.nan,
                    "selected_expansion": [
                        term in expansion_terms
                        for term in vectorizer.get_feature_names_out()
                    ],
                }
            )

        if progress is not None:
            progress.update(1)

        out = add_expansion_hits(
            out,
            set(expansion_terms),
            vectorizer,
            X,
            config=config,
        )

        if progress is not None:
            progress.update(1)

    except Exception:
        if progress is not None:
            progress.close()
        raise

    out[config.subset_flag_col] = (
        out[config.anchor_col] | out[config.expansion_col]
    )

    out[config.filter_type_col] = np.select(
        [
            out[config.anchor_col] & out[config.expansion_col],
            out[config.anchor_col] & ~out[config.expansion_col],
            ~out[config.anchor_col] & out[config.expansion_col],
        ],
        [
            "anchor+expansion",
            "anchor_only",
            "expansion_only",
        ],
        default="excluded",
    )

    subset = out.loc[out[config.subset_flag_col]].copy()

    if config.retain_columns is not None:
        existing = [col for col in config.retain_columns if col in subset.columns]
        subset = subset.loc[:, existing].copy()

    if progress is not None:
        progress.update(1)
        progress.close()

    metadata = {
        "n_input": int(len(df)),
        "n_retained": int(len(subset)),
        "retention_rate": float(len(subset) / len(df)) if len(df) else 0.0,
        "n_anchor_hits": int(out[config.anchor_col].sum()),
        "n_expansion_hits": int(out[config.expansion_col].sum()),
        "n_anchor_plus_expansion": int(
            (out[config.anchor_col] & out[config.expansion_col]).sum()
        ),
        "n_anchor_only": int(
            (out[config.anchor_col] & ~out[config.expansion_col]).sum()
        ),
        "n_expansion_only": int(
            (~out[config.anchor_col] & out[config.expansion_col]).sum()
        ),
        "n_anchors": int(len(anchors)),
        "n_expansion_terms": int(len(expansion_terms)),
        "anchors": list(anchors),
        "expansion_terms": sorted(expansion_terms),
        "expansion_diagnostics": expansion_diagnostics,
    }

    if config.verbose:
        print(f"Number of unique anchors: {len(anchors)}")
        print(out[config.anchor_col].value_counts(dropna=False))
        print(f"Anchor-hit documents: {int(out[config.anchor_col].sum()):,} / {len(out):,}")
        print(f"Expansion-hit documents: {int(out[config.expansion_col].sum()):,} / {len(out):,}")
        print(f"Thematic subset documents: {len(subset):,} / {len(out):,}")

    return subset, metadata


__all__ = [
    "ThematicFilterConfig",
    "normalise_anchor_terms",
    "load_anchor_terms",
    "compile_anchor_pattern",
    "direct_anchor_filter",
    "build_count_vectorizer",
    "derive_expansion_terms",
    "add_expansion_hits",
    "build_thematic_subset",
]
