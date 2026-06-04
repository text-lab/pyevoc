"""Emoji description labelling with codepoint fallback.

This module enriches term-level EVOC outputs with emoji descriptions. It is
intended to run after term-level statistics and, typically, after concreteness
labelling.

The expected input is a dataframe containing at least:

- ``term``
- ``upos``

The module adds:

- ``emoji_description``
- ``emoji_in_lookup``
- ``emoji_match_method``
- ``emoji_match_similarity``

Matching strategy
-----------------
Emoji matching is performed through several increasingly permissive passes:

1. exact glyph;
2. exact codepoints;
3. normalised glyph;
4. normalised codepoints;
5. conservative codepoint-similarity fallback.

The normalisation removes variation selectors and dangling zero-width joiners,
while preserving the visible base emoji sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from importlib import resources
import gc
import unicodedata

import pandas as pd

try:
    from pyevoc.utils.resources import user_model_path
except Exception:  # pragma: no cover
    user_model_path = None


VARIATION_SELECTORS = {
    "\ufe0e",  # text presentation
    "\ufe0f",  # emoji presentation
}

ZERO_WIDTH_JOINER = "\u200d"

PREVIOUS_EMOJI_COLUMNS = (
    "emoji_description",
    "emoji_in_lookup",
    "emoji_match_method",
    "emoji_match_similarity",
    "emoji_norm",
    "codepoints_hex",
    "codepoints_norm",
)


@dataclass(frozen=True)
class EmojiLabellingConfig:
    """Configuration for emoji description labelling."""

    lookup_file: str = "emoji_lookup.csv"
    models_dir: str | Path | None = None

    term_col: str = "term"
    upos_col: str = "upos"
    emoji_upos: str = "EMOJI"

    min_similarity: float = 0.80

    verbose: bool = True


def _require_columns(df: pd.DataFrame, columns: set[str]) -> None:
    """Validate required columns."""

    missing = columns.difference(df.columns)

    if missing:
        raise ValueError(f"Dataframe is missing required columns: {sorted(missing)}")


def _user_lookup_path(filename: str) -> Path:
    """Return the user-level model path for an emoji lookup file."""

    if user_model_path is not None:
        return Path(user_model_path(filename))

    return Path.home() / ".pyevoc" / "models" / filename


def _package_resource_candidates(filename: str) -> list[Path]:
    """Return possible package-bundled lookup paths that currently exist."""

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

    candidates.extend(path for path in local_candidates if path.exists())

    unique: list[Path] = []
    seen: set[str] = set()

    for path in candidates:
        key = str(path.resolve())
        if key not in seen:
            unique.append(path)
            seen.add(key)

    return unique


def resolve_emoji_lookup_path(
    lookup_file: str | Path = "emoji_lookup.csv",
    *,
    models_dir: str | Path | None = None,
) -> Path:
    """Resolve the emoji lookup file path."""

    lookup_path = Path(lookup_file)

    if lookup_path.exists():
        return lookup_path

    if models_dir is not None:
        candidate = Path(models_dir) / lookup_path.name
        if candidate.exists():
            return candidate
        return candidate

    package_candidates = _package_resource_candidates(lookup_path.name)

    if package_candidates:
        return package_candidates[0]

    return _user_lookup_path(lookup_path.name)


def available_emoji_lookup_paths(
    lookup_file: str = "emoji_lookup.csv",
) -> pd.DataFrame:
    """Return a diagnostic table of detected emoji lookup locations."""

    rows = []

    explicit = Path(lookup_file)

    rows.append(
        {
            "location": "explicit_or_cwd",
            "path": str(explicit.resolve()),
            "exists": explicit.exists(),
        }
    )

    for path in _package_resource_candidates(lookup_file):
        rows.append(
            {
                "location": "package_or_local_resource",
                "path": str(path.resolve()),
                "exists": path.exists(),
            }
        )

    user_path = _user_lookup_path(lookup_file)

    rows.append(
        {
            "location": "user_model_path",
            "path": str(user_path),
            "exists": user_path.exists(),
        }
    )

    return pd.DataFrame(rows)


def load_emoji_lookup(
    path: str | Path | None = None,
    *,
    models_dir: str | Path | None = None,
    lookup_file: str = "emoji_lookup.csv",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """Load an emoji lookup table."""

    lookup_path = (
        Path(path)
        if path is not None
        else resolve_emoji_lookup_path(
            lookup_file,
            models_dir=models_dir,
        )
    )

    if not lookup_path.exists():
        diagnostics = available_emoji_lookup_paths(lookup_file)
        raise FileNotFoundError(
            "Emoji lookup file not found. Last attempted path: "
            f"{lookup_path}\n\nSearched locations:\n"
            f"{diagnostics.to_string(index=False)}"
        )

    return pd.read_csv(
        lookup_path,
        encoding=encoding,
        low_memory=False,
    )


def emoji_codepoints_hex(value: object) -> str:
    """Return Unicode codepoints for an emoji/string as uppercase hex tokens."""

    if value is None or pd.isna(value):
        return ""

    return " ".join(
        f"{ord(ch):04X}"
        for ch in str(value)
    )


def normalise_emoji_glyph(value: object) -> str:
    """Conservatively normalise emoji glyphs for lookup matching.

    The function removes variation selectors FE0E/FE0F and dangling
    zero-width joiners, while preserving the visible base emoji sequence.
    """

    if value is None or pd.isna(value):
        return ""

    text = str(value)

    text = "".join(
        ch for ch in text
        if ch not in VARIATION_SELECTORS
    )

    text = text.strip(ZERO_WIDTH_JOINER)

    if text.endswith(ZERO_WIDTH_JOINER):
        text = text[:-1]

    return text


def codepoint_tokens(value: object) -> list[str]:
    """Return codepoint tokens excluding variation selectors and ZWJ."""

    if value is None or pd.isna(value):
        return []

    return [
        f"{ord(ch):04X}"
        for ch in str(value)
        if ch not in VARIATION_SELECTORS
        and ch != ZERO_WIDTH_JOINER
    ]


def codepoint_jaccard(a: object, b: object) -> float:
    """Return Jaccard similarity between two emoji codepoint sets."""

    a_set = set(codepoint_tokens(a))
    b_set = set(codepoint_tokens(b))

    if not a_set or not b_set:
        return 0.0

    return len(a_set & b_set) / len(a_set | b_set)


def _unicode_description(value: object) -> str:
    """Fallback description based on Unicode character names."""

    if value is None or pd.isna(value):
        return "-"

    names = []

    for ch in str(value):
        try:
            names.append(unicodedata.name(ch).lower())
        except ValueError:
            pass

    return " + ".join(names) if names else "-"


def prepare_emoji_lookup(
    lookup: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare a normalised emoji lookup table."""

    emoji_lookup = lookup.copy()

    emoji_lookup.columns = (
        emoji_lookup.columns
        .astype(str)
        .str.lower()
        .str.strip()
    )

    required_lookup_cols = {"emoji", "description"}
    missing_lookup = required_lookup_cols.difference(emoji_lookup.columns)

    if missing_lookup:
        raise ValueError(
            f"emoji_lookup.csv is missing required columns: {missing_lookup}"
        )

    emoji_lookup["emoji"] = emoji_lookup["emoji"].astype("string")
    emoji_lookup["emoji_description"] = (
        emoji_lookup["description"]
        .astype("string")
    )

    if "codepoints_hex" not in emoji_lookup.columns:
        emoji_lookup["codepoints_hex"] = emoji_lookup["emoji"].map(
            emoji_codepoints_hex
        )

    emoji_lookup["codepoints_hex"] = (
        emoji_lookup["codepoints_hex"]
        .astype("string")
        .str.upper()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    emoji_lookup["emoji_norm"] = emoji_lookup["emoji"].map(
        normalise_emoji_glyph
    )

    emoji_lookup["codepoints_norm"] = emoji_lookup["emoji_norm"].map(
        emoji_codepoints_hex
    )

    emoji_lookup = (
        emoji_lookup[
            [
                "emoji",
                "emoji_norm",
                "codepoints_hex",
                "codepoints_norm",
                "emoji_description",
            ]
        ]
        .dropna(subset=["emoji"])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return emoji_lookup


def add_emoji_descriptions(
    term_stats_df: pd.DataFrame,
    *,
    lookup: pd.DataFrame | None = None,
    config: EmojiLabellingConfig | None = None,
    models_dir: str | Path | None = None,
    emoji_lookup_file: str | None = None,
    min_similarity: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add emoji descriptions to a term-level dataframe.

    Returns
    -------
    tuple
        ``(out, coverage_df, method_counts_df, missing_df)``.
    """

    base_cfg = config or EmojiLabellingConfig()

    cfg = EmojiLabellingConfig(
        lookup_file=emoji_lookup_file or base_cfg.lookup_file,
        models_dir=models_dir if models_dir is not None else base_cfg.models_dir,
        term_col=base_cfg.term_col,
        upos_col=base_cfg.upos_col,
        emoji_upos=base_cfg.emoji_upos,
        min_similarity=(
            min_similarity if min_similarity is not None
            else base_cfg.min_similarity
        ),
        verbose=base_cfg.verbose,
    )

    _require_columns(term_stats_df, {cfg.term_col, cfg.upos_col})

    if cfg.verbose:
        print("Emoji description labelling started...")

    if lookup is None:
        lookup = load_emoji_lookup(
            models_dir=cfg.models_dir,
            lookup_file=cfg.lookup_file,
        )

    emoji_lookup = prepare_emoji_lookup(lookup)

    out = term_stats_df.copy()

    out = out.drop(
        columns=list(PREVIOUS_EMOJI_COLUMNS),
        errors="ignore",
    )

    out["emoji_description"] = pd.NA
    out["emoji_in_lookup"] = 0
    out["emoji_match_method"] = "-"
    out["emoji_match_similarity"] = pd.NA

    emoji_mask = out[cfg.upos_col].eq(cfg.emoji_upos)

    out.loc[emoji_mask, "emoji_norm"] = out.loc[
        emoji_mask,
        cfg.term_col,
    ].map(normalise_emoji_glyph)

    out.loc[emoji_mask, "codepoints_hex"] = out.loc[
        emoji_mask,
        cfg.term_col,
    ].map(emoji_codepoints_hex)

    out.loc[emoji_mask, "codepoints_norm"] = out.loc[
        emoji_mask,
        "emoji_norm",
    ].map(emoji_codepoints_hex)

    # Pass 1: exact glyph.
    lookup_exact = (
        emoji_lookup
        .drop_duplicates(subset="emoji", keep="first")
        .set_index("emoji")["emoji_description"]
    )

    missing = emoji_mask & out["emoji_description"].isna()

    out.loc[missing, "emoji_description"] = out.loc[
        missing,
        cfg.term_col,
    ].map(lookup_exact)

    hit = missing & out["emoji_description"].notna()

    out.loc[hit, "emoji_in_lookup"] = 1
    out.loc[hit, "emoji_match_method"] = "exact_glyph"
    out.loc[hit, "emoji_match_similarity"] = 1.0

    # Pass 2: exact codepoints.
    lookup_cp = (
        emoji_lookup
        .drop_duplicates(subset="codepoints_hex", keep="first")
        .set_index("codepoints_hex")["emoji_description"]
    )

    missing = emoji_mask & out["emoji_description"].isna()

    out.loc[missing, "emoji_description"] = out.loc[
        missing,
        "codepoints_hex",
    ].map(lookup_cp)

    hit = missing & out["emoji_description"].notna()

    out.loc[hit, "emoji_in_lookup"] = 1
    out.loc[hit, "emoji_match_method"] = "exact_codepoints"
    out.loc[hit, "emoji_match_similarity"] = 1.0

    # Pass 3a: normalised glyph.
    lookup_norm_glyph = (
        emoji_lookup
        .drop_duplicates(subset="emoji_norm", keep="first")
        .set_index("emoji_norm")["emoji_description"]
    )

    missing = emoji_mask & out["emoji_description"].isna()

    out.loc[missing, "emoji_description"] = out.loc[
        missing,
        "emoji_norm",
    ].map(lookup_norm_glyph)

    hit = missing & out["emoji_description"].notna()

    out.loc[hit, "emoji_in_lookup"] = 1
    out.loc[hit, "emoji_match_method"] = "normalised_glyph"
    out.loc[hit, "emoji_match_similarity"] = 1.0

    # Pass 3b: normalised codepoints.
    lookup_norm_cp = (
        emoji_lookup
        .drop_duplicates(subset="codepoints_norm", keep="first")
        .set_index("codepoints_norm")["emoji_description"]
    )

    missing = emoji_mask & out["emoji_description"].isna()

    out.loc[missing, "emoji_description"] = out.loc[
        missing,
        "codepoints_norm",
    ].map(lookup_norm_cp)

    hit = missing & out["emoji_description"].notna()

    out.loc[hit, "emoji_in_lookup"] = 1
    out.loc[hit, "emoji_match_method"] = "normalised_codepoints"
    out.loc[hit, "emoji_match_similarity"] = 1.0

    # Pass 4: conservative codepoint-similarity fallback.
    lookup_records = (
        emoji_lookup[["emoji", "emoji_description"]]
        .drop_duplicates()
        .to_dict("records")
    )

    missing_terms = out.loc[
        emoji_mask & out["emoji_description"].isna(),
        cfg.term_col,
    ]

    for idx, term in missing_terms.items():
        best_desc = pd.NA
        best_sim = 0.0

        for record in lookup_records:
            sim = codepoint_jaccard(term, record["emoji"])

            if sim > best_sim:
                best_sim = sim
                best_desc = record["emoji_description"]

        if best_sim >= cfg.min_similarity:
            out.at[idx, "emoji_description"] = best_desc
            out.at[idx, "emoji_in_lookup"] = 1
            out.at[idx, "emoji_match_method"] = "codepoint_similarity"
            out.at[idx, "emoji_match_similarity"] = round(best_sim, 3)

    # Final fallback: Unicode names, explicitly marked as outside lookup.
    missing = emoji_mask & out["emoji_description"].isna()

    out.loc[missing, "emoji_description"] = out.loc[
        missing,
        cfg.term_col,
    ].map(_unicode_description)

    out.loc[missing, "emoji_match_method"] = "unicode_name_fallback"
    out.loc[missing, "emoji_match_similarity"] = pd.NA

    # Non-emoji rows.
    out.loc[~emoji_mask, "emoji_description"] = pd.NA
    out.loc[~emoji_mask, "emoji_in_lookup"] = 0
    out.loc[~emoji_mask, "emoji_match_method"] = "-"
    out.loc[~emoji_mask, "emoji_match_similarity"] = pd.NA

    out["emoji_description"] = (
        out["emoji_description"]
        .fillna("-")
        .astype("string")
    )

    n_emoji = int(emoji_mask.sum())
    n_hit = int(out.loc[emoji_mask, "emoji_in_lookup"].sum())
    pct_hit = round(100 * n_hit / n_emoji, 1) if n_emoji > 0 else 0.0

    coverage_df = pd.DataFrame(
        {
            "emoji_terms": [n_emoji],
            "in_lookup": [n_hit],
            "pct_in_lookup": [pct_hit],
            "min_similarity": [cfg.min_similarity],
        }
    )

    missing_df = out.loc[
        emoji_mask & out["emoji_in_lookup"].eq(0),
        [cfg.term_col, "codepoints_hex", "emoji_description"],
    ].copy()

    method_counts_df = (
        out.loc[emoji_mask]
        .groupby("emoji_match_method", dropna=False)
        .size()
        .reset_index(name="n_terms")
        .sort_values("n_terms", ascending=False)
        .reset_index(drop=True)
    )

    out = out.drop(
        columns=["emoji_norm", "codepoints_hex", "codepoints_norm"],
        errors="ignore",
    )

    if cfg.verbose:
        print(
            "Emoji labelling complete: "
            f"{n_hit}/{n_emoji} emoji terms covered ({pct_hit}%)."
        )

    gc.collect()

    return out, coverage_df, method_counts_df, missing_df


def label_emojis(
    terms: pd.DataFrame,
    *,
    lookup: pd.DataFrame | None = None,
    term_col: str = "term",
    upos_col: str = "upos",
    description_col: str = "emoji_description",
    return_diagnostics: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Backward-compatible wrapper for emoji labelling."""

    config = EmojiLabellingConfig(
        term_col=term_col,
        upos_col=upos_col,
        verbose=False,
    )

    out, coverage, methods, missing = add_emoji_descriptions(
        terms,
        lookup=lookup,
        config=config,
    )

    if description_col != "emoji_description":
        out = out.rename(columns={"emoji_description": description_col})

    if return_diagnostics:
        return out, coverage, methods, missing

    return out
