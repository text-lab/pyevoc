"""Concreteness labelling for PyEvoc term-level tables.

This module enriches term-level EVOC outputs with lexical concreteness
information. It is designed to operate after term-level statistics have been
computed.

The expected input is a dataframe containing at least:

- ``term``
- ``upos``

The module adds:

- ``concreteness_score``
- ``concreteness_label``
- ``concreteness_in_lexicon``

and returns a coverage diagnostic table.

Lexicon lookup
--------------
The concreteness lexicon is searched in the following order:

1. explicit ``path`` argument;
2. explicit ``lexicon_file`` if it is already a valid path;
3. ``models_dir / lexicon_file`` when ``models_dir`` is supplied;
4. package-bundled resource locations:
   - ``pyevoc.models``
   - ``pyevoc.resources``
   - ``pyevoc.data``
5. local package-adjacent folders:
   - ``pyevoc/models``
   - ``models``
   - ``mdl``
6. user model directory, usually ``~/.pyevoc/models``.

Emoji terms are explicitly retained but excluded from lexical concreteness
labelling, because emojis are not standard lexical items in concreteness norms.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from importlib import resources
import gc

import numpy as np
import pandas as pd

try:
    from pyevoc.utils.resources import user_model_path
except Exception:  # pragma: no cover
    user_model_path = None


TOKEN_COLUMN_CANDIDATES = (
    "word",
    "term",
    "lemma",
    "token",
    "surface",
    "form",
    "base_term",
    "base",
)

SCORE_COLUMN_CANDIDATES = (
    "conc.m",
    "conc_m",
    "conc.mean",
    "conc_mean",
    "concreteness",
    "concreteness_score",
    "conc",
    "score",
    "rating",
    "mean",
    "value",
)

PREVIOUS_CONCRETENESS_COLUMNS = (
    "conc_score",
    "concreteness_score",
    "concreteness_label",
    "concreteness_in_lexicon",
    "word_lc",
    "base_term_lc",
)


@dataclass(frozen=True)
class ConcretenessConfig:
    """Configuration for concreteness labelling."""

    lexicon_file: str = "concreteness_lexicon.csv"
    models_dir: str | Path | None = None

    term_col: str = "term"
    upos_col: str = "upos"

    absent_label: str = "-"
    cut_abstract: float = 2.5
    cut_concrete: float = 3.5

    eligible_upos: tuple[str, ...] = ("NOUN", "ADJ")
    emoji_upos: str = "EMOJI"

    token_column_candidates: tuple[str, ...] = TOKEN_COLUMN_CANDIDATES
    score_column_candidates: tuple[str, ...] = SCORE_COLUMN_CANDIDATES

    verbose: bool = True


def _require_columns(df: pd.DataFrame, columns: set[str]) -> None:
    """Validate that all required columns are present."""

    missing = columns.difference(df.columns)

    if missing:
        raise ValueError(
            f"Dataframe is missing required columns: {sorted(missing)}"
        )


def _user_lexicon_path(filename: str) -> Path:
    """Return the user-level model path for a lexicon file."""

    if user_model_path is not None:
        return Path(user_model_path(filename))

    return Path.home() / ".pyevoc" / "models" / filename


def _package_resource_candidates(filename: str) -> list[Path]:
    """Return possible package-bundled lexicon paths.

    The function uses ``importlib.resources`` when possible and falls back to
    local package-relative paths. It returns only paths that currently exist.
    """

    candidates: list[Path] = []

    for package in (
        "pyevoc.models",
        "pyevoc.resources",
        "pyevoc.data",
    ):
        try:
            resource = resources.files(package).joinpath(filename)
            if resource.is_file():
                with resources.as_file(resource) as resource_path:
                    candidates.append(Path(resource_path))
        except Exception:
            pass

    module_dir = Path(__file__).resolve().parent
    package_root = module_dir.parent

    local_candidates = [
        package_root / "models" / filename,
        package_root / "resources" / filename,
        package_root / "data" / filename,
        package_root.parent / "models" / filename,
        package_root.parent / "mdl" / filename,
        Path.cwd() / "models" / filename,
        Path.cwd() / "mdl" / filename,
    ]

    candidates.extend(
        path for path in local_candidates
        if path.exists()
    )

    # Preserve order while removing duplicates.
    unique: list[Path] = []
    seen: set[str] = set()

    for path in candidates:
        key = str(path.resolve())
        if key not in seen:
            unique.append(path)
            seen.add(key)

    return unique


def resolve_concreteness_lexicon_path(
    lexicon_file: str | Path = "concreteness_lexicon.csv",
    *,
    models_dir: str | Path | None = None,
) -> Path:
    """Resolve the concreteness lexicon path.

    Resolution order
    ----------------
    1. ``lexicon_file`` if it already points to an existing file;
    2. ``models_dir / lexicon_file`` if ``models_dir`` is supplied;
    3. package-bundled resources and package-adjacent model folders;
    4. user model path, usually ``~/.pyevoc/models/lexicon_file``.
    """

    lexicon_path = Path(lexicon_file)

    if lexicon_path.exists():
        return lexicon_path

    if models_dir is not None:
        candidate = Path(models_dir) / lexicon_path.name
        if candidate.exists():
            return candidate
        return candidate

    package_candidates = _package_resource_candidates(lexicon_path.name)

    if package_candidates:
        return package_candidates[0]

    return _user_lexicon_path(lexicon_path.name)


def available_concreteness_lexicon_paths(
    lexicon_file: str = "concreteness_lexicon.csv",
) -> pd.DataFrame:
    """Return a diagnostic table of detected lexicon locations."""

    rows = []

    explicit = Path(lexicon_file)
    rows.append(
        {
            "location": "explicit_or_cwd",
            "path": str(explicit.resolve()),
            "exists": explicit.exists(),
        }
    )

    for path in _package_resource_candidates(lexicon_file):
        rows.append(
            {
                "location": "package_or_local_resource",
                "path": str(path.resolve()),
                "exists": path.exists(),
            }
        )

    user_path = _user_lexicon_path(lexicon_file)
    rows.append(
        {
            "location": "user_model_path",
            "path": str(user_path),
            "exists": user_path.exists(),
        }
    )

    return pd.DataFrame(rows)


def load_concreteness_lexicon(
    path: str | Path | None = None,
    *,
    models_dir: str | Path | None = None,
    lexicon_file: str = "concreteness_lexicon.csv",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """Load a concreteness lexicon.

    Parameters
    ----------
    path:
        Explicit lexicon path. If omitted, PyEvoc searches package resources,
        local model folders, and the user model directory.
    models_dir:
        Optional model directory.
    lexicon_file:
        Lexicon filename used when ``path`` is omitted.
    encoding:
        File encoding.

    Returns
    -------
    pandas.DataFrame
        Raw concreteness lexicon.
    """

    lexicon_path = (
        Path(path)
        if path is not None
        else resolve_concreteness_lexicon_path(
            lexicon_file,
            models_dir=models_dir,
        )
    )

    if not lexicon_path.exists():
        diagnostics = available_concreteness_lexicon_paths(lexicon_file)
        raise FileNotFoundError(
            "Concreteness lexicon not found. Last attempted path: "
            f"{lexicon_path}\n\nSearched locations:\n"
            f"{diagnostics.to_string(index=False)}"
        )

    return pd.read_csv(
        lexicon_path,
        encoding=encoding,
        low_memory=False,
    )


def _normalise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalised column names."""

    out = df.copy()
    out.columns = out.columns.astype(str).str.lower().str.strip()

    return out


def infer_lexicon_columns(
    lexicon: pd.DataFrame,
    *,
    token_candidates: tuple[str, ...] = TOKEN_COLUMN_CANDIDATES,
    score_candidates: tuple[str, ...] = SCORE_COLUMN_CANDIDATES,
) -> tuple[str, str]:
    """Infer lexical-item and score columns from a concreteness lexicon."""

    conc_lex = _normalise_column_names(lexicon)

    token_col = next(
        (col for col in token_candidates if col in conc_lex.columns),
        None,
    )

    if token_col is None:
        object_cols = [
            col for col in conc_lex.columns
            if conc_lex[col].dtype == "object"
            or pd.api.types.is_string_dtype(conc_lex[col])
        ]

        if object_cols:
            token_col = object_cols[0]

    if token_col is None:
        raise ValueError(
            "Could not identify a lexical item column in the concreteness lexicon."
        )

    score_col = next(
        (col for col in score_candidates if col in conc_lex.columns),
        None,
    )

    if score_col is None:
        numeric_cols = [
            col for col in conc_lex.columns
            if pd.api.types.is_numeric_dtype(conc_lex[col])
        ]

        if numeric_cols:
            score_col = numeric_cols[0]

    if score_col is None:
        raise ValueError(
            "Could not identify a numeric concreteness score column."
        )

    return token_col, score_col


def prepare_concreteness_map(
    lexicon: pd.DataFrame,
    *,
    config: ConcretenessConfig | None = None,
) -> tuple[pd.DataFrame, str, str]:
    """Prepare a cleaned concreteness lookup table."""

    cfg = config or ConcretenessConfig()

    conc_lex = _normalise_column_names(lexicon)

    token_col, score_col = infer_lexicon_columns(
        conc_lex,
        token_candidates=cfg.token_column_candidates,
        score_candidates=cfg.score_column_candidates,
    )

    conc_lex["word_lc"] = (
        conc_lex[token_col]
        .astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )

    conc_lex["concreteness_score"] = pd.to_numeric(
        conc_lex[score_col],
        errors="coerce",
    )

    conc_lex = conc_lex.loc[
        conc_lex["word_lc"].notna()
        & conc_lex["word_lc"].ne("")
        & conc_lex["concreteness_score"].notna()
    ].copy()

    conc_map = (
        conc_lex
        .groupby("word_lc", observed=True, sort=False)
        .agg(concreteness_score=("concreteness_score", "mean"))
        .reset_index()
    )

    conc_map["concreteness_label"] = np.select(
        [
            conc_map["concreteness_score"] < cfg.cut_abstract,
            conc_map["concreteness_score"] > cfg.cut_concrete,
        ],
        [
            "abstract",
            "concrete",
        ],
        default="mixed",
    )

    return conc_map, token_col, score_col


def add_concreteness_labels(
    term_stats_df: pd.DataFrame,
    *,
    lexicon: pd.DataFrame | None = None,
    config: ConcretenessConfig | None = None,
    models_dir: str | Path | None = None,
    lexicon_file: str | None = None,
    absent_label: str | None = None,
    cut_abstract: float | None = None,
    cut_concrete: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add concreteness labels to a term-level dataframe."""

    base_cfg = config or ConcretenessConfig()

    cfg = ConcretenessConfig(
        lexicon_file=lexicon_file or base_cfg.lexicon_file,
        models_dir=models_dir if models_dir is not None else base_cfg.models_dir,
        term_col=base_cfg.term_col,
        upos_col=base_cfg.upos_col,
        absent_label=absent_label if absent_label is not None else base_cfg.absent_label,
        cut_abstract=cut_abstract if cut_abstract is not None else base_cfg.cut_abstract,
        cut_concrete=cut_concrete if cut_concrete is not None else base_cfg.cut_concrete,
        eligible_upos=base_cfg.eligible_upos,
        emoji_upos=base_cfg.emoji_upos,
        token_column_candidates=base_cfg.token_column_candidates,
        score_column_candidates=base_cfg.score_column_candidates,
        verbose=base_cfg.verbose,
    )

    _require_columns(term_stats_df, {cfg.term_col, cfg.upos_col})

    if cfg.verbose:
        print(f"Concreteness labelling started: {cfg.lexicon_file}")

    if lexicon is None:
        lexicon = load_concreteness_lexicon(
            models_dir=cfg.models_dir,
            lexicon_file=cfg.lexicon_file,
        )

    out = term_stats_df.copy()

    out = out.drop(
        columns=list(PREVIOUS_CONCRETENESS_COLUMNS),
        errors="ignore",
    )

    conc_map, token_col, score_col = prepare_concreteness_map(
        lexicon,
        config=cfg,
    )

    out["base_term_lc"] = (
        out[cfg.term_col]
        .astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )

    out = out.merge(
        conc_map,
        left_on="base_term_lc",
        right_on="word_lc",
        how="left",
        suffixes=("", "_lex"),
    )

    out["concreteness_in_lexicon"] = (
        out["concreteness_score"].notna()
    ).astype("int8")

    emoji_mask = out[cfg.upos_col].eq(cfg.emoji_upos)

    out.loc[emoji_mask, "concreteness_score"] = np.nan
    out.loc[emoji_mask, "concreteness_label"] = pd.NA
    out.loc[emoji_mask, "concreteness_in_lexicon"] = 0

    out["concreteness_label"] = (
        out["concreteness_label"]
        .fillna(cfg.absent_label)
        .astype("string")
    )

    eligible = out[cfg.upos_col].isin(cfg.eligible_upos)

    n_eligible = int(eligible.sum())

    n_hit = int(
        (eligible & out["concreteness_in_lexicon"].eq(1)).sum()
    )

    pct_hit = (
        round(100 * n_hit / n_eligible, 1)
        if n_eligible > 0
        else np.nan
    )

    coverage_df = pd.DataFrame(
        {
            "lexicon_file": [cfg.lexicon_file],
            "token_column": [token_col],
            "score_column": [score_col],
            "eligible_terms": [n_eligible],
            "in_lexicon": [n_hit],
            "pct_in_lexicon": [pct_hit],
            "cut_abstract": [cfg.cut_abstract],
            "cut_concrete": [cfg.cut_concrete],
        }
    )

    out = out.drop(
        columns=["word_lc", "base_term_lc"],
        errors="ignore",
    )

    if cfg.verbose:
        print(
            "Concreteness labelling complete: "
            f"{n_hit}/{n_eligible} eligible terms covered ({pct_hit}%)."
        )

    gc.collect()

    return out, coverage_df


def label_concreteness(
    terms: pd.DataFrame,
    *,
    lexicon: pd.DataFrame | None = None,
    term_col: str = "term",
    upos_col: str = "upos",
    label_col: str = "concreteness_label",
    score_col: str = "concreteness_score",
    in_lexicon_col: str = "concreteness_in_lexicon",
    absent_label: str = "-",
    cut_abstract: float = 2.5,
    cut_concrete: float = 3.5,
    return_coverage: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Backward-compatible wrapper for concreteness labelling."""

    config = ConcretenessConfig(
        term_col=term_col,
        upos_col=upos_col,
        absent_label=absent_label,
        cut_abstract=cut_abstract,
        cut_concrete=cut_concrete,
        verbose=False,
    )

    out, coverage = add_concreteness_labels(
        terms,
        lexicon=lexicon,
        config=config,
    )

    rename_map = {}

    if label_col != "concreteness_label":
        rename_map["concreteness_label"] = label_col

    if score_col != "concreteness_score":
        rename_map["concreteness_score"] = score_col

    if in_lexicon_col != "concreteness_in_lexicon":
        rename_map["concreteness_in_lexicon"] = in_lexicon_col

    if rename_map:
        out = out.rename(columns=rename_map)

    if return_coverage:
        return out, coverage

    return out
