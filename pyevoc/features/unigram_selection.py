"""Unigram cleaning and selection.

This module implements the unigram filtering stage used before PyEvoc
term-level indicator computation.

The selection logic follows the original PyEvoc analytical workflow:

- retain focal UPOS categories;
- construct analytical terms from lemmas or emoji tokens;
- remove stop words, interactional markers and low-content lexical items;
- remove URLs, punctuation-only tokens, numeric tokens and malformed clitics;
- retain valid emoji tokens;
- apply minimum document/user reliability filters;
- preserve token-level metadata columns.

The output remains a token-level dataframe enriched with the ``term`` column.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterable
import gc

import numpy as np
import pandas as pd
import regex

try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
except Exception:  # pragma: no cover
    ENGLISH_STOP_WORDS = frozenset()

try:
    import emoji as emoji_lib  # type: ignore
except Exception:  # pragma: no cover
    emoji_lib = None


DEFAULT_UPOS = {"NOUN", "ADJ", "EMOJI"}

INTERACTIONAL_MARKERS = {
    "lol", "lmao", "lmfao", "lmbo", "rofl", "rotfl", "lawl", "kek",
    "haha", "hahaha", "hehe", "yeah", "yea", "yep", "yup", "ok", "okay",
    "alright", "aight", "ikr", "ofc", "obv", "duh", "mhm", "nah", "naw",
    "nope", "nop", "meh", "uh", "uhh", "uhhh", "uhm", "um", "umm", "ummm",
    "er", "erm", "ah", "ahh", "eh", "ehh", "huh", "hmm", "hmmm", "omg",
    "omfg", "wtf", "wth", "wow", "whoa", "woah", "oof", "yikes", "ugh",
    "sheesh", "geez", "jeez", "gosh", "damn", "goddamn", "bruh", "aww",
    "awww", "yay", "hooray", "boo", "hiss", "btw", "tbh", "anyways",
    "anyhow", "soo", "sooo", "welp", "tho", "tho.", "bc", "imo", "imho",
    "afaik", "fwiw", "idk", "dunno", "idc", "wdym", "hbu", "wbu", "pls",
    "plz", "thx", "thnx", "thanx", "ty", "sry", "np", "nbd", "nvm", "nvmd",
    "rn", "asap", "irl", "rip", "thru", "ppl", "ur", "u", "r", "tldr",
    "tl;dr", "eli5", "til", "iirc", "ianal", "cmv", "dae", "tia", "inb4",
    "nsfw", "sfw", "oc", "op", "edit", "eta", "ps", "fyi", "psa", "ff",
    "ffs", "smh", "smdh", "jfc",
}

GENERIC_LOW_CONTENT_NOUNS = {
    "thing", "things", "way", "ways", "stuff", "lot", "lots", "kind",
    "kinds", "sort", "sorts", "type", "types", "case", "cases", "point",
    "points", "part", "parts", "place", "places", "someone", "somebody",
    "something", "anything", "nothing", "everything", "guy", "guys", "man",
    "men", "woman", "women", "kid", "kids", "ir",
}

GENERIC_LOW_CONTENT_ADJECTIVES = {
    "able", "unable", "likely", "possible", "similar", "different", "certain",
    "various", "several", "many", "few", "same", "other", "another", "such",
    "own",
}

DEFAULT_STOP_WORDS_EXTENDED = (
    set(ENGLISH_STOP_WORDS)
    .union(INTERACTIONAL_MARKERS)
    .union(GENERIC_LOW_CONTENT_NOUNS)
    .union(GENERIC_LOW_CONTENT_ADJECTIVES)
)

PUNCT_ONLY_RE = regex.compile(r"^\p{P}+$")
LATIN_TERM_RE = regex.compile(r"^[\p{Latin}'-]+$")


@dataclass(frozen=True)
class UnigramSelectionConfig:
    """Configuration for unigram cleaning and selection."""

    focal_upos: set[str] = field(default_factory=lambda: set(DEFAULT_UPOS))

    token_col: str = "text"
    display_col: str | None = None
    lemma_col: str = "lemma"
    upos_col: str = "upos"
    doc_col: str = "doc_id"
    user_col: str = "user_id"

    term_col: str = "term"

    stop_words: set[str] = field(default_factory=lambda: set(DEFAULT_STOP_WORDS_EXTENDED))

    min_chars: int = 2
    min_docs_per_term: int = 3
    min_users_per_term: int = 3

    exclude_url_token: bool = True
    require_latin_terms: bool = True
    remove_numeric_terms: bool = True
    remove_punct_only: bool = True

    lowercase_non_emoji: bool = True

    return_diagnostics: bool = False
    verbose: bool = True


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Validate that required columns are present."""

    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Token table is missing required columns: {missing}")


def is_punct_only(value: object) -> bool:
    """Return True if a value consists only of punctuation."""

    if value is None or pd.isna(value):
        return False
    return bool(PUNCT_ONLY_RE.fullmatch(str(value)))


def is_latin_term(value: object) -> bool:
    """Return True if a value contains only Latin letters, apostrophes or hyphens."""

    if value is None or pd.isna(value):
        return False
    return bool(LATIN_TERM_RE.fullmatch(str(value)))


def is_valid_emoji(value: object) -> bool:
    """Return True if a value is a valid emoji token."""

    if emoji_lib is None:
        return False
    if value is None or pd.isna(value):
        return False

    value = str(value)
    if value == "":
        return False

    return (
        value in emoji_lib.EMOJI_DATA
        or emoji_lib.is_emoji(value)
        or any(ch in emoji_lib.EMOJI_DATA for ch in value)
    )


def _choose_display_column(df: pd.DataFrame, config: UnigramSelectionConfig) -> str:
    """Choose the display column used for emoji terms."""

    if config.display_col is not None:
        if config.display_col not in df.columns:
            raise ValueError(f"Display column '{config.display_col}' not found.")
        return config.display_col

    if "token_display" in df.columns:
        return "token_display"

    return config.token_col


def build_unigram_terms(
    tokens: pd.DataFrame,
    *,
    config: UnigramSelectionConfig | None = None,
) -> pd.DataFrame:
    """Append the analytical unigram ``term`` column."""

    cfg = config or UnigramSelectionConfig()

    _require_columns(tokens, [cfg.token_col, cfg.lemma_col, cfg.upos_col])

    out = tokens.copy()
    display_col = _choose_display_column(out, cfg)

    out[cfg.upos_col] = out[cfg.upos_col].astype("string")
    out[cfg.lemma_col] = out[cfg.lemma_col].astype("string")
    out[display_col] = out[display_col].astype("string")

    is_emoji = out[cfg.upos_col].eq("EMOJI")

    out[cfg.term_col] = np.where(
        is_emoji,
        out[display_col],
        out[cfg.lemma_col],
    )

    out[cfg.term_col] = pd.Series(out[cfg.term_col], index=out.index, dtype="string")

    if cfg.lowercase_non_emoji:
        non_emoji = ~is_emoji
        out.loc[non_emoji, cfg.term_col] = out.loc[non_emoji, cfg.term_col].str.lower()

    return out


def unigram_selection_diagnostics(
    *,
    input_tokens: int,
    base_mask: pd.Series,
    lexical_mask: pd.Series,
    emoji_mask: pd.Series,
    initial_mask: pd.Series,
    cleaned: pd.DataFrame,
    config: UnigramSelectionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return diagnostic tables for the unigram selection step."""

    unique_terms = (
        int(cleaned[[config.term_col, config.upos_col]].drop_duplicates().shape[0])
        if len(cleaned)
        else 0
    )

    diagnostics = pd.DataFrame(
        {
            "stage": [
                "input_tokens",
                "after_base_filter",
                "retained_lexical_tokens_before_reliability",
                "retained_emoji_tokens_before_reliability",
                "tokens_after_initial_filter",
                "tokens_after_reliability_filter",
                "unique_terms_after_reliability_filter",
            ],
            "n": [
                int(input_tokens),
                int(base_mask.sum()),
                int((base_mask & lexical_mask).sum()),
                int((base_mask & emoji_mask).sum()),
                int(initial_mask.sum()),
                int(len(cleaned)),
                unique_terms,
            ],
        }
    )

    parameters = pd.DataFrame(
        {
            "criterion": [
                "focal_upos",
                "min_chars",
                "min_docs_per_term",
                "min_users_per_term",
                "stop_words_extended",
                "interactional_markers",
                "generic_low_content_nouns",
                "generic_low_content_adjectives",
                "require_latin_terms",
                "remove_numeric_terms",
                "remove_punct_only",
            ],
            "value": [
                ", ".join(sorted(config.focal_upos)),
                config.min_chars,
                config.min_docs_per_term,
                config.min_users_per_term,
                len(config.stop_words),
                len(INTERACTIONAL_MARKERS),
                len(GENERIC_LOW_CONTENT_NOUNS),
                len(GENERIC_LOW_CONTENT_ADJECTIVES),
                config.require_latin_terms,
                config.remove_numeric_terms,
                config.remove_punct_only,
            ],
        }
    )

    return diagnostics, parameters


def clean_unigram_tokens(
    tokens: pd.DataFrame,
    *,
    config: UnigramSelectionConfig | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clean and select unigram tokens."""

    cfg = config or UnigramSelectionConfig()

    _require_columns(
        tokens,
        [cfg.token_col, cfg.lemma_col, cfg.upos_col, cfg.doc_col, cfg.user_col],
    )

    if cfg.verbose:
        print("Cleaning tokens (unigrams)...")

    df = build_unigram_terms(tokens, config=cfg)

    df[cfg.doc_col] = df[cfg.doc_col].astype("string")
    df[cfg.user_col] = df[cfg.user_col].astype("string")
    df[cfg.upos_col] = df[cfg.upos_col].astype("string")

    term = df[cfg.term_col].astype("string")
    term_lower = term.str.lower()

    is_emoji = df[cfg.upos_col].eq("EMOJI")

    base_mask = (
        term.notna()
        & term.ne("")
        & df[cfg.upos_col].isin(cfg.focal_upos)
    )

    if cfg.exclude_url_token:
        base_mask = base_mask & ~term_lower.eq("url")

    lexical_mask = ~is_emoji
    lexical_mask = lexical_mask & ~term_lower.isin(cfg.stop_words)

    if cfg.remove_numeric_terms:
        lexical_mask = lexical_mask & ~term.str.contains(r"[0-9]", regex=True, na=False)

    lexical_mask = (
        lexical_mask
        & ~term.str.fullmatch(r"'+", na=False)
        & ~term.str.fullmatch(r"'?s", na=False)
        & ~term.str.fullmatch(r"s'", na=False)
        & ~term.str.fullmatch(r"'?t", na=False)
        & ~term.str.fullmatch(r"n'?t", na=False)
    )

    if cfg.remove_punct_only:
        lexical_mask = lexical_mask & ~term.map(is_punct_only)

    lexical_mask = lexical_mask & term.str.len().ge(cfg.min_chars)

    if cfg.require_latin_terms:
        lexical_mask = lexical_mask & term.map(is_latin_term)

    emoji_mask = is_emoji & term.map(is_valid_emoji)

    initial_mask = base_mask & (lexical_mask | emoji_mask)

    cleaned = df.loc[initial_mask].copy()

    if not cleaned.empty:
        term_reliability = (
            cleaned
            .groupby([cfg.term_col, cfg.upos_col], observed=True)
            .agg(
                n_docs_term=(cfg.doc_col, "nunique"),
                n_users_term=(cfg.user_col, "nunique"),
            )
            .reset_index()
        )

        reliable_terms = term_reliability.loc[
            (term_reliability["n_docs_term"] >= cfg.min_docs_per_term)
            & (term_reliability["n_users_term"] >= cfg.min_users_per_term),
            [cfg.term_col, cfg.upos_col],
        ]

        cleaned = cleaned.merge(
            reliable_terms,
            on=[cfg.term_col, cfg.upos_col],
            how="inner",
        )

    cleaned = cleaned.reset_index(drop=True)

    if cfg.verbose:
        print(f"Clean tokens retained: {len(cleaned):,}")

    diagnostics, parameters = unigram_selection_diagnostics(
        input_tokens=len(df),
        base_mask=base_mask,
        lexical_mask=lexical_mask,
        emoji_mask=emoji_mask,
        initial_mask=initial_mask,
        cleaned=cleaned,
        config=cfg,
    )

    gc.collect()

    if cfg.return_diagnostics:
        return cleaned, diagnostics, parameters

    return cleaned


def select_unigrams(
    tokens: pd.DataFrame,
    *,
    lemma_col: str = "lemma",
    upos_col: str = "upos",
    keep_upos: set[str] | None = None,
    min_chars: int = 2,
    lowercase: bool = True,
    token_col: str = "text",
    doc_col: str = "doc_id",
    user_col: str = "user_id",
    min_docs_per_term: int = 3,
    min_users_per_term: int = 3,
    return_diagnostics: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Backward-compatible wrapper for unigram selection."""

    config = UnigramSelectionConfig(
        focal_upos=set(keep_upos or DEFAULT_UPOS),
        token_col=token_col,
        lemma_col=lemma_col,
        upos_col=upos_col,
        doc_col=doc_col,
        user_col=user_col,
        min_chars=min_chars,
        lowercase_non_emoji=lowercase,
        min_docs_per_term=min_docs_per_term,
        min_users_per_term=min_users_per_term,
        return_diagnostics=return_diagnostics,
    )

    return clean_unigram_tokens(tokens, config=config)
