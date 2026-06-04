"""Dependency collocations and PROPN named-entity n-grams.

This module implements the PyEvoc collocation and named-entity extraction
workflow used after token-level annotation.

It provides one integrated interface that can extract:

- dependency-based lexical collocations;
- proper-noun named-entity n-grams;
- both tables with optional overlap adjudication;
- compact HTML reports.

The implementation follows the original PyEvoc notebook logic while adapting
it to a package-ready API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import gc
import html

import numpy as np
import pandas as pd
import regex
from scipy.stats import chi2


DEFAULT_COLLOC_RELATIONS = {
    "amod",
    "compound",
    "nmod",
    "flat",
    "name",
    "appos",
}

DEFAULT_FALSE_ENTITY_TERMS = {
    "reddit",
    "edit",
    "update",
    "source",
    "thanks",
}

DEFAULT_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "that", "the", "to", "with",
    "lol", "lmao", "haha", "yeah", "ok", "okay", "edit", "reddit",
}

PUNCT_ONLY_RE = regex.compile(r"^\p{P}+$")
LATIN_TERM_RE = regex.compile(r"^[\p{Latin}' -]+$")


@dataclass(frozen=True)
class CollocationEntityConfig:
    """Configuration for collocation and named-entity extraction."""

    output_dir: str | Path = "evoc_outputs"

    entity_n: int = 2

    min_freq: int = 3
    min_docs: int = 3
    min_users: int = 3
    g2_alpha: float = 0.001

    caps_thr: float = 0.60
    min_caps_obs: int = 20

    include_collocations: bool = True
    include_entities: bool = True
    resolve_overlap: bool = True
    prefer_overlap: str = "evidence"

    write_html: bool = True
    html_max_colloc_rows: int = 60
    html_max_entity_rows: int = 60
    html_max_overlap_rows: int = 50

    doc_col: str = "doc_id"
    user_col: str = "user_id"
    sentence_col: str = "sentence_id"
    token_id_col: str = "token_id"
    token_col: str = "token"
    lemma_col: str = "lemma"
    upos_col: str = "upos"
    surface_col: str = "token"
    head_col: str = "head_token_id"
    dep_rel_col: str = "dep_rel"

    colloc_relations: set[str] = field(
        default_factory=lambda: set(DEFAULT_COLLOC_RELATIONS)
    )
    false_entity_terms: set[str] = field(
        default_factory=lambda: set(DEFAULT_FALSE_ENTITY_TERMS)
    )
    stop_words: set[str] = field(default_factory=lambda: set(DEFAULT_STOP_WORDS))

    verbose: bool = True


def _empty_candidate_table() -> pd.DataFrame:
    """Return an empty candidate table with the canonical columns."""

    return pd.DataFrame(
        columns=[
            "term",
            "candidate_type",
            "freq",
            "n_docs",
            "n_posts",
            "n_users",
            "g2",
            "p_value",
            "example_form",
            "source",
        ]
    )


def _require_columns(df: pd.DataFrame, columns: set[str], label: str = "dataframe") -> None:
    """Validate required columns."""

    missing = columns.difference(df.columns)

    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")


def normalise_annotation_columns(
    tokens: pd.DataFrame,
    *,
    config: CollocationEntityConfig | None = None,
) -> pd.DataFrame:
    """Normalise token-table column names for compatibility.

    The canonical annotation schema uses ``doc_id`` and ``token``. This
    function also accepts older aliases such as ``text`` and creates the
    internal ``post_id`` alias required by this module.
    """

    cfg = config or CollocationEntityConfig()

    out = tokens.copy()

    if cfg.doc_col in out.columns and "post_id" not in out.columns:
        out = out.rename(columns={cfg.doc_col: "post_id"})

    if "token" not in out.columns:
        if cfg.token_col in out.columns:
            out = out.rename(columns={cfg.token_col: "token"})
        elif "text" in out.columns:
            out = out.rename(columns={"text": "token"})

    if cfg.surface_col not in out.columns:
        if "token" in out.columns:
            out[cfg.surface_col] = out["token"]
        elif "token_display" in out.columns:
            out[cfg.surface_col] = out["token_display"]
        elif cfg.lemma_col in out.columns:
            out[cfg.surface_col] = out[cfg.lemma_col]
        else:
            out[cfg.surface_col] = pd.NA

    return out


def is_valid_term(value: object, stop_words: set[str] | None = None) -> bool:
    """Return True if a lexical item is eligible for collocation/NER extraction."""

    if value is None or pd.isna(value):
        return False

    x = str(value).strip().lower()

    if not x:
        return False

    if stop_words is not None and x in stop_words:
        return False

    if x == "url":
        return False

    if len(x) < 2:
        return False

    if PUNCT_ONLY_RE.fullmatch(x):
        return False

    if not LATIN_TERM_RE.fullmatch(x):
        return False

    return True


def prepare_tokens(
    tokens: pd.DataFrame,
    *,
    config: CollocationEntityConfig | None = None,
) -> pd.DataFrame:
    """Prepare and validate a token-level annotation table."""

    cfg = config or CollocationEntityConfig()

    df = normalise_annotation_columns(tokens, config=cfg)

    required_cols = {
        "post_id",
        cfg.sentence_col,
        cfg.token_id_col,
        cfg.upos_col,
        cfg.lemma_col,
        cfg.surface_col,
    }

    _require_columns(df, required_cols, label="tokens_df")

    if cfg.user_col not in df.columns:
        df[cfg.user_col] = pd.NA

    df["post_id"] = df["post_id"].astype("string")
    df[cfg.user_col] = df[cfg.user_col].astype("string")
    df[cfg.sentence_col] = pd.to_numeric(df[cfg.sentence_col], errors="coerce")
    df[cfg.token_id_col] = pd.to_numeric(df[cfg.token_id_col], errors="coerce")
    df[cfg.upos_col] = df[cfg.upos_col].astype("string")
    df["lemma_lc"] = (
        df[cfg.lemma_col]
        .astype("string")
        .str.lower()
        .str.strip()
    )
    df["surface"] = df[cfg.surface_col].astype("string")

    df = (
        df
        .dropna(
            subset=[
                "post_id",
                cfg.sentence_col,
                cfg.token_id_col,
                "lemma_lc",
            ]
        )
        .drop_duplicates(
            ["post_id", cfg.sentence_col, cfg.token_id_col],
            keep="first",
        )
        .sort_values(["post_id", cfg.sentence_col, cfg.token_id_col])
        .reset_index(drop=True)
    )

    return df


def score_phrase_table(
    phrase_df: pd.DataFrame,
    component_cols: list[str],
    *,
    term_col: str = "term",
    min_freq: int = 3,
    min_docs: int = 3,
    min_users: int = 3,
    alpha: float = 0.001,
    source_label: str = "candidate",
) -> pd.DataFrame:
    """Score phrase candidates using the log-likelihood G² statistic."""

    if phrase_df.empty:
        return pd.DataFrame()

    df = phrase_df.copy()

    group_cols = [term_col] + component_cols

    optional_first_cols = [
        col for col in ["candidate_type", "example_form"]
        if col in df.columns and col not in group_cols
    ]

    agg_dict = {
        "freq": (term_col, "size"),
        "n_docs": ("post_id", "nunique"),
        "n_posts": ("post_id", "nunique"),
        "n_users": ("user_id", "nunique"),
    }

    for col in optional_first_cols:
        agg_dict[col] = (col, "first")

    grouped = (
        df
        .groupby(group_cols, observed=True)
        .agg(**agg_dict)
        .reset_index()
    )

    grouped = grouped.loc[
        (grouped["freq"] >= min_freq)
        & (grouped["n_docs"] >= min_docs)
        & (grouped["n_users"] >= min_users)
    ].copy()

    if grouped.empty:
        return pd.DataFrame()

    total = max(len(df), 1)

    if len(component_cols) == 2:
        left_col, right_col = component_cols

        left_marg = (
            grouped
            .groupby(left_col, observed=True)["freq"]
            .sum()
            .rename("n_x")
            .reset_index()
        )

        right_marg = (
            grouped
            .groupby(right_col, observed=True)["freq"]
            .sum()
            .rename("n_y")
            .reset_index()
        )

        grouped = grouped.merge(left_marg, on=left_col, how="left")
        grouped = grouped.merge(right_marg, on=right_col, how="left")

        n11 = grouped["freq"].astype(float)
        n1_ = grouped["n_x"].astype(float)
        n_1 = grouped["n_y"].astype(float)
        n = float(total)

        n12 = n1_ - n11
        n21 = n_1 - n11
        n22 = n - n11 - n12 - n21

        observed = np.vstack([n11, n12, n21, n22]).T

        row1 = n11 + n12
        row2 = n21 + n22
        col1 = n11 + n21
        col2 = n12 + n22

        expected = np.vstack(
            [
                row1 * col1 / n,
                row1 * col2 / n,
                row2 * col1 / n,
                row2 * col2 / n,
            ]
        ).T

        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(
                observed > 0,
                observed * np.log(observed / expected),
                0.0,
            )

        grouped["g2"] = 2.0 * np.nansum(terms, axis=1)
        grouped["p_value"] = chi2.sf(grouped["g2"], df=1)

        grouped = grouped.drop(columns=["n_x", "n_y"], errors="ignore")

        g2_thr = chi2.ppf(1 - alpha, df=1)

        grouped = grouped.loc[grouped["g2"] >= g2_thr].copy()

    else:
        grouped["g2"] = 2.0 * grouped["freq"] * np.log(
            grouped["freq"] / max(grouped["freq"].mean(), 1e-9)
        )
        grouped["p_value"] = np.nan

    grouped["source"] = source_label

    return (
        grouped
        .sort_values(
            ["g2", "freq", "n_docs", "n_posts", "n_users"],
            ascending=[False, False, False, False, False],
        )
        .reset_index(drop=True)
    )


def extract_dependency_collocations(
    tokens_df: pd.DataFrame,
    *,
    config: CollocationEntityConfig | None = None,
    min_freq: int | None = None,
    min_docs: int | None = None,
    min_users: int | None = None,
    alpha: float | None = None,
) -> pd.DataFrame:
    """Extract dependency-based collocations."""

    cfg = config or CollocationEntityConfig()

    min_freq = cfg.min_freq if min_freq is None else min_freq
    min_docs = cfg.min_docs if min_docs is None else min_docs
    min_users = cfg.min_users if min_users is None else min_users
    alpha = cfg.g2_alpha if alpha is None else alpha

    if cfg.verbose:
        print("Extracting dependency-based collocations...")

    df = prepare_tokens(tokens_df, config=cfg)

    required_cols = {cfg.head_col, cfg.dep_rel_col}
    _require_columns(df, required_cols, label="tokens_df")

    df[cfg.head_col] = pd.to_numeric(df[cfg.head_col], errors="coerce")
    df[cfg.dep_rel_col] = df[cfg.dep_rel_col].astype("string")

    stop_words = set(cfg.stop_words)

    heads = (
        df.loc[
            df[cfg.upos_col].isin(["NOUN", "PROPN"])
            & df["lemma_lc"].map(lambda x: is_valid_term(x, stop_words)),
            [
                "post_id",
                cfg.sentence_col,
                cfg.token_id_col,
                "lemma_lc",
            ],
        ]
        .rename(
            columns={
                cfg.token_id_col: cfg.head_col,
                "lemma_lc": "head",
            }
        )
    )

    deps = (
        df.loc[
            df[cfg.upos_col].isin(["NOUN", "PROPN", "ADJ"])
            & df[cfg.head_col].notna()
            & df[cfg.dep_rel_col].isin(cfg.colloc_relations)
            & df["lemma_lc"].map(lambda x: is_valid_term(x, stop_words)),
            [
                "post_id",
                cfg.user_col,
                cfg.sentence_col,
                cfg.token_id_col,
                cfg.head_col,
                "lemma_lc",
                cfg.upos_col,
                cfg.dep_rel_col,
                "surface",
            ],
        ]
        .rename(columns={"lemma_lc": "dep"})
    )

    cand = deps.merge(
        heads,
        on=["post_id", cfg.sentence_col, cfg.head_col],
        how="inner",
    )

    cand = cand.loc[cand["dep"].ne(cand["head"])].copy()

    if cand.empty:
        return _empty_candidate_table()

    cand["term"] = cand["dep"] + " " + cand["head"]
    cand["example_form"] = cand["term"]
    cand["candidate_type"] = np.where(
        cand[cfg.upos_col].eq("ADJ"),
        "ADJ_NOUN",
        "NOUN_NOUN",
    )

    scored = score_phrase_table(
        cand,
        component_cols=["dep", "head"],
        min_freq=min_freq,
        min_docs=min_docs,
        min_users=min_users,
        alpha=alpha,
        source_label="collocation",
    )

    if scored.empty:
        return _empty_candidate_table()

    out = scored[
        [
            "term",
            "candidate_type",
            "freq",
            "n_docs",
            "n_posts",
            "n_users",
            "g2",
            "p_value",
            "example_form",
            "source",
        ]
    ].copy()

    if cfg.verbose:
        print(f"Collocations retained: {len(out):,}")

    return out


def extract_named_entity_ngrams(
    tokens_df: pd.DataFrame,
    *,
    config: CollocationEntityConfig | None = None,
    entity_n: int | None = None,
    min_freq: int | None = None,
    min_docs: int | None = None,
    min_users: int | None = None,
    alpha: float | None = None,
) -> pd.DataFrame:
    """Extract contiguous PROPN named-entity n-grams."""

    cfg = config or CollocationEntityConfig()

    entity_n = cfg.entity_n if entity_n is None else entity_n
    min_freq = cfg.min_freq if min_freq is None else min_freq
    min_docs = cfg.min_docs if min_docs is None else min_docs
    min_users = cfg.min_users if min_users is None else min_users
    alpha = cfg.g2_alpha if alpha is None else alpha

    if entity_n < 2:
        raise ValueError("entity_n must be >= 2. Unigram NEs are intentionally skipped.")

    if cfg.verbose:
        print(f"Extracting PROPN named-entity {entity_n}-grams...")

    df = prepare_tokens(tokens_df, config=cfg)

    stop_words = set(cfg.stop_words)

    propn = df.loc[
        df[cfg.upos_col].eq("PROPN")
        & df["lemma_lc"].map(lambda x: is_valid_term(x, stop_words))
        & ~df["lemma_lc"].isin(cfg.false_entity_terms),
        [
            "post_id",
            cfg.user_col,
            cfg.sentence_col,
            cfg.token_id_col,
            "lemma_lc",
            "surface",
        ],
    ].copy()

    if propn.empty:
        return _empty_candidate_table()

    propn = propn.sort_values(["post_id", cfg.sentence_col, cfg.token_id_col]).copy()

    for i in range(entity_n):
        propn[f"lemma_{i}"] = (
            propn
            .groupby(["post_id", cfg.sentence_col], observed=True)["lemma_lc"]
            .shift(-i)
        )

        propn[f"surface_{i}"] = (
            propn
            .groupby(["post_id", cfg.sentence_col], observed=True)["surface"]
            .shift(-i)
        )

        propn[f"token_{i}"] = (
            propn
            .groupby(["post_id", cfg.sentence_col], observed=True)[cfg.token_id_col]
            .shift(-i)
        )

    mask = pd.Series(True, index=propn.index)

    for i in range(entity_n):
        mask &= propn[f"lemma_{i}"].notna()

    for i in range(1, entity_n):
        mask &= propn[f"token_{i}"].eq(propn["token_0"] + i)

    cand = propn.loc[mask].copy()

    if cand.empty:
        return _empty_candidate_table()

    lemma_cols = [f"lemma_{i}" for i in range(entity_n)]
    surface_cols = [f"surface_{i}" for i in range(entity_n)]

    cand["term"] = cand[lemma_cols].agg(" ".join, axis=1)
    cand["example_form"] = cand[surface_cols].astype(str).agg(" ".join, axis=1)
    cand["candidate_type"] = f"PROPN_{entity_n}GRAM"

    component_cols = lemma_cols if entity_n > 2 else ["lemma_0", "lemma_1"]

    scored = score_phrase_table(
        cand,
        component_cols=component_cols,
        min_freq=min_freq,
        min_docs=min_docs,
        min_users=min_users,
        alpha=alpha,
        source_label="named_entity",
    )

    if scored.empty:
        return _empty_candidate_table()

    out = scored[
        [
            "term",
            "candidate_type",
            "freq",
            "n_docs",
            "n_posts",
            "n_users",
            "g2",
            "p_value",
            "example_form",
            "source",
        ]
    ].copy()

    if cfg.verbose:
        print(f"Named entities retained: {len(out):,}")

    return out


def compute_caps_evidence(
    tokens_df: pd.DataFrame,
    terms: list[str] | set[str],
    *,
    config: CollocationEntityConfig | None = None,
) -> pd.DataFrame:
    """Compute capitalisation evidence for overlapping collocation/NE terms."""

    cfg = config or CollocationEntityConfig()

    df = prepare_tokens(tokens_df, config=cfg)

    terms = sorted(set(str(term).strip().lower() for term in terms if pd.notna(term)))

    if not terms:
        return pd.DataFrame(columns=["term", "caps_obs", "caps_ratio"])

    records = []

    for term in terms:
        parts = term.split()
        n = len(parts)

        tmp = df.copy()

        for i in range(n):
            tmp[f"lemma_{i}"] = (
                tmp
                .groupby(["post_id", cfg.sentence_col], observed=True)["lemma_lc"]
                .shift(-i)
            )

            tmp[f"surface_{i}"] = (
                tmp
                .groupby(["post_id", cfg.sentence_col], observed=True)["surface"]
                .shift(-i)
            )

            tmp[f"token_{i}"] = (
                tmp
                .groupby(["post_id", cfg.sentence_col], observed=True)[cfg.token_id_col]
                .shift(-i)
            )

        mask = pd.Series(True, index=tmp.index)

        for i, part in enumerate(parts):
            mask &= tmp[f"lemma_{i}"].eq(part)

        for i in range(1, n):
            mask &= tmp[f"token_{i}"].eq(tmp["token_0"] + i)

        matched = tmp.loc[mask]

        if matched.empty:
            records.append({"term": term, "caps_obs": 0, "caps_ratio": np.nan})
            continue

        cap_flags = []

        for _, row in matched.iterrows():
            surfaces = [str(row[f"surface_{i}"]) for i in range(n)]

            flags = [
                surface[:1].isupper()
                for surface in surfaces
                if any(ch.isalpha() for ch in surface)
            ]

            cap_flags.append(bool(flags) and sum(flags) / len(flags) >= 0.75)

        records.append(
            {
                "term": term,
                "caps_obs": len(cap_flags),
                "caps_ratio": float(np.mean(cap_flags)) if cap_flags else np.nan,
            }
        )

    return pd.DataFrame(records)


def resolve_ne_colloc_overlap(
    collocations_df: pd.DataFrame,
    entities_df: pd.DataFrame,
    tokens_df: pd.DataFrame,
    *,
    config: CollocationEntityConfig | None = None,
    prefer: str | None = None,
    caps_thr: float | None = None,
    min_caps_obs: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Resolve overlaps between collocations and named entities."""

    cfg = config or CollocationEntityConfig()

    prefer = cfg.prefer_overlap if prefer is None else prefer
    caps_thr = cfg.caps_thr if caps_thr is None else caps_thr
    min_caps_obs = cfg.min_caps_obs if min_caps_obs is None else min_caps_obs

    if prefer not in {"evidence", "entity", "collocation"}:
        raise ValueError("prefer must be one of: 'evidence', 'entity', 'collocation'.")

    colloc = collocations_df.copy()
    ents = entities_df.copy()

    if colloc.empty or ents.empty:
        return colloc, ents, pd.DataFrame()

    colloc["term_norm"] = colloc["term"].astype(str).str.lower().str.strip()
    ents["term_norm"] = ents["term"].astype(str).str.lower().str.strip()

    overlap_terms = sorted(set(colloc["term_norm"]).intersection(set(ents["term_norm"])))

    if not overlap_terms:
        return (
            colloc.drop(columns=["term_norm"], errors="ignore"),
            ents.drop(columns=["term_norm"], errors="ignore"),
            pd.DataFrame(),
        )

    caps_df = compute_caps_evidence(tokens_df, overlap_terms, config=cfg)
    caps_map = caps_df.set_index("term").to_dict(orient="index")

    drop_colloc: set[str] = set()
    drop_entity: set[str] = set()
    logs = []

    for term in overlap_terms:
        c = colloc.loc[colloc["term_norm"].eq(term)].iloc[0]
        e = ents.loc[ents["term_norm"].eq(term)].iloc[0]

        caps = caps_map.get(term, {})
        caps_obs = caps.get("caps_obs", 0)
        caps_ratio = caps.get("caps_ratio", np.nan)

        if prefer == "entity":
            decision = "keep_entity"
            reason = "manual_preference_entity"

        elif prefer == "collocation":
            decision = "keep_collocation"
            reason = "manual_preference_collocation"

        else:
            if (
                pd.notna(caps_ratio)
                and caps_obs >= min_caps_obs
                and caps_ratio >= caps_thr
            ):
                decision = "keep_entity"
                reason = "capitalisation_evidence"

            elif pd.notna(e.get("g2", np.nan)) and pd.notna(c.get("g2", np.nan)):
                if e["g2"] > c["g2"]:
                    decision = "keep_entity"
                    reason = "higher_g2_entity"
                else:
                    decision = "keep_collocation"
                    reason = "higher_g2_collocation"

            else:
                decision = "keep_collocation"
                reason = "insufficient_entity_evidence"

        if decision == "keep_entity":
            drop_colloc.add(term)
        else:
            drop_entity.add(term)

        logs.append(
            {
                "term": term,
                "colloc_freq": c.get("freq", np.nan),
                "entity_freq": e.get("freq", np.nan),
                "colloc_g2": c.get("g2", np.nan),
                "entity_g2": e.get("g2", np.nan),
                "caps_obs": caps_obs,
                "caps_ratio": caps_ratio,
                "decision": decision,
                "reason": reason,
            }
        )

    colloc = (
        colloc
        .loc[~colloc["term_norm"].isin(drop_colloc)]
        .drop(columns=["term_norm"], errors="ignore")
        .reset_index(drop=True)
    )

    ents = (
        ents
        .loc[~ents["term_norm"].isin(drop_entity)]
        .drop(columns=["term_norm"], errors="ignore")
        .reset_index(drop=True)
    )

    return colloc, ents, pd.DataFrame(logs)


def _format_p(value: object) -> str:
    """Format p-values for HTML tables."""

    if value is None or pd.isna(value):
        return "-"

    x = float(value)

    if x < 0.001:
        return f"{x:.2e}"

    return f"{x:.4f}"


def _format_num(value: object, digits: int = 2) -> str:
    """Format numeric values for HTML tables."""

    if value is None or pd.isna(value):
        return "-"

    return f"{float(value):.{digits}f}"


def write_html_table(
    df: pd.DataFrame,
    title: str,
    output_file: str | Path,
    max_rows: int,
) -> str:
    """Write a compact HTML table and return the output path."""

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    show = df.head(max_rows).copy()

    if show.empty:
        show = pd.DataFrame({"message": ["No retained candidates"]})

    cols = list(show.columns)
    rows = []

    num_cols = {
        "freq", "n_docs", "n_posts", "n_users", "g2", "p_value", "caps_obs",
        "caps_ratio", "colloc_freq", "entity_freq", "colloc_g2", "entity_g2",
    }

    for _, row in show.iterrows():
        cells = []

        for col in cols:
            value = row[col]

            if col == "p_value":
                txt = _format_p(value)
            elif col in {"g2", "caps_ratio", "colloc_g2", "entity_g2"}:
                txt = _format_num(value, 2)
            elif col in {
                "freq", "n_docs", "n_posts", "n_users", "caps_obs",
                "colloc_freq", "entity_freq",
            }:
                txt = "-" if pd.isna(value) else f"{int(value):,}"
            else:
                txt = html.escape(str(value))

            cls = "num" if col in num_cols else "txt"

            cells.append(f'<td class="{cls}">{txt}</td>')

        rows.append("<tr>" + "".join(cells) + "</tr>")

    header = "".join(f"<th>{html.escape(col)}</th>" for col in cols)

    html_doc = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
    body {{
        font-family: Arial, Helvetica, sans-serif;
        margin: 14px auto;
        padding: 0 8px;
        max-width: 950px;
        color: #222222;
        background: #ffffff;
    }}
    h1 {{
        font-size: 18px;
        margin: 0 0 4px 0;
        font-weight: 700;
    }}
    .subtitle {{
        font-size: 11px;
        color: #555555;
        margin: 0 0 8px 0;
        line-height: 1.25;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        table-layout: fixed;
        border: 1px solid #d8d8d8;
    }}
    th {{
        font-size: 11px;
        padding: 5px 5px;
        background: #f1f1f1;
        border-bottom: 1px solid #d8d8d8;
        text-align: left;
        font-weight: 700;
    }}
    td {{
        font-size: 11.5px;
        padding: 4px 5px;
        border-bottom: 1px solid #eeeeee;
        vertical-align: top;
        line-height: 1.18;
        overflow-wrap: anywhere;
    }}
    td.num {{
        text-align: center;
        white-space: nowrap;
    }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<div class="subtitle">Showing {min(len(df), max_rows):,} of {len(df):,} retained candidates.</div>
<table>
<thead>
<tr>{header}</tr>
</thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
</body>
</html>
"""

    output_file.write_text(html_doc, encoding="utf-8")

    print(f"Saved: {output_file}")

    return str(output_file)


def extract_collocations_and_entities(
    tokens_df: pd.DataFrame,
    *,
    config: CollocationEntityConfig | None = None,
    output_dir: str | Path | None = None,
    include_collocations: bool | None = None,
    include_entities: bool | None = None,
    entity_n: int | None = None,
    min_freq: int | None = None,
    min_docs: int | None = None,
    min_users: int | None = None,
    g2_alpha: float | None = None,
    resolve_overlap: bool | None = None,
    prefer_overlap: str | None = None,
    write_html: bool | None = None,
) -> dict[str, object]:
    """Extract collocations and/or named entities through one wrapper.

    This is the recommended public API. The user can run only collocations,
    only named entities, or both.
    """

    base_cfg = config or CollocationEntityConfig()

    cfg = CollocationEntityConfig(
        output_dir=output_dir if output_dir is not None else base_cfg.output_dir,
        entity_n=entity_n if entity_n is not None else base_cfg.entity_n,
        min_freq=min_freq if min_freq is not None else base_cfg.min_freq,
        min_docs=min_docs if min_docs is not None else base_cfg.min_docs,
        min_users=min_users if min_users is not None else base_cfg.min_users,
        g2_alpha=g2_alpha if g2_alpha is not None else base_cfg.g2_alpha,
        caps_thr=base_cfg.caps_thr,
        min_caps_obs=base_cfg.min_caps_obs,
        include_collocations=(
            include_collocations
            if include_collocations is not None
            else base_cfg.include_collocations
        ),
        include_entities=(
            include_entities
            if include_entities is not None
            else base_cfg.include_entities
        ),
        resolve_overlap=(
            resolve_overlap
            if resolve_overlap is not None
            else base_cfg.resolve_overlap
        ),
        prefer_overlap=(
            prefer_overlap
            if prefer_overlap is not None
            else base_cfg.prefer_overlap
        ),
        write_html=write_html if write_html is not None else base_cfg.write_html,
        html_max_colloc_rows=base_cfg.html_max_colloc_rows,
        html_max_entity_rows=base_cfg.html_max_entity_rows,
        html_max_overlap_rows=base_cfg.html_max_overlap_rows,
        doc_col=base_cfg.doc_col,
        user_col=base_cfg.user_col,
        sentence_col=base_cfg.sentence_col,
        token_id_col=base_cfg.token_id_col,
        token_col=base_cfg.token_col,
        lemma_col=base_cfg.lemma_col,
        upos_col=base_cfg.upos_col,
        surface_col=base_cfg.surface_col,
        head_col=base_cfg.head_col,
        dep_rel_col=base_cfg.dep_rel_col,
        colloc_relations=set(base_cfg.colloc_relations),
        false_entity_terms=set(base_cfg.false_entity_terms),
        stop_words=set(base_cfg.stop_words),
        verbose=base_cfg.verbose,
    )

    collocations_df = pd.DataFrame()
    entities_df = pd.DataFrame()

    if cfg.include_collocations:
        collocations_df = extract_dependency_collocations(
            tokens_df=tokens_df,
            config=cfg,
        )

    if cfg.include_entities:
        entities_df = extract_named_entity_ngrams(
            tokens_df=tokens_df,
            config=cfg,
        )

    overlap_log_df = pd.DataFrame()

    if (
        cfg.resolve_overlap
        and cfg.include_collocations
        and cfg.include_entities
        and not collocations_df.empty
        and not entities_df.empty
    ):
        collocations_df, entities_df, overlap_log_df = resolve_ne_colloc_overlap(
            collocations_df=collocations_df,
            entities_df=entities_df,
            tokens_df=tokens_df,
            config=cfg,
        )

    colloc_html = None
    entity_html = None
    overlap_html = None

    if cfg.write_html:
        outdir = Path(cfg.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        if cfg.include_collocations:
            colloc_html = write_html_table(
                collocations_df,
                title=f"Dependency-based collocations: top {cfg.html_max_colloc_rows}",
                output_file=outdir / "evoc_collocations.html",
                max_rows=cfg.html_max_colloc_rows,
            )

        if cfg.include_entities:
            entity_html = write_html_table(
                entities_df,
                title=f"Named entities ({cfg.entity_n}-grams): top {cfg.html_max_entity_rows}",
                output_file=outdir / f"evoc_named_entities_{cfg.entity_n}grams.html",
                max_rows=cfg.html_max_entity_rows,
            )

        if not overlap_log_df.empty:
            overlap_html = write_html_table(
                overlap_log_df,
                title=f"NE-collocation overlap adjudication: top {cfg.html_max_overlap_rows}",
                output_file=outdir / "evoc_ne_colloc_overlap.html",
                max_rows=cfg.html_max_overlap_rows,
            )

    diagnostics_df = pd.DataFrame(
        {
            "table": [
                "collocations",
                "named_entities",
                "overlap_adjudications",
            ],
            "n_rows_total": [
                len(collocations_df),
                len(entities_df),
                len(overlap_log_df),
            ],
            "n_rows_shown_html": [
                min(len(collocations_df), cfg.html_max_colloc_rows),
                min(len(entities_df), cfg.html_max_entity_rows),
                min(len(overlap_log_df), cfg.html_max_overlap_rows),
            ],
        }
    )

    gc.collect()

    return {
        "collocations_df": collocations_df,
        "entities_df": entities_df,
        "overlap_log_df": overlap_log_df,
        "collocations_html": colloc_html,
        "entities_html": entity_html,
        "overlap_html": overlap_html,
        "diagnostics_df": diagnostics_df,
    }


def extract_collocations(
    tokens: pd.DataFrame,
    *,
    config: CollocationEntityConfig | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Backward-compatible wrapper returning only dependency collocations."""

    return extract_dependency_collocations(tokens, config=config, **kwargs)


def aggregate_named_entities(
    entities: pd.DataFrame,
    text_col: str = "text",
    type_col: str = "type",
) -> pd.DataFrame:
    """Backward-compatible utility for aggregating pre-extracted NER tables."""

    if entities.empty:
        return pd.DataFrame(columns=["entity", "type", "freq"])

    return (
        entities
        .groupby([text_col, type_col])
        .size()
        .reset_index(name="freq")
        .rename(columns={text_col: "entity", type_col: "type"})
        .sort_values("freq", ascending=False)
        .reset_index(drop=True)
    )
