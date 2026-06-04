"""Basic text cleaning for social-media corpora.

The module performs conservative, emoji-safe normalisation of raw textual data.
It is designed for social-media corpora where URLs, contractions, emojis,
hashtags, mentions, punctuation, and non-standard orthography may coexist.

The default behaviour follows the original PyEvoc analytical pipeline:

- decode HTML entities;
- normalise apostrophes;
- expand English contractions;
- replace URLs with a stable placeholder;
- preserve emojis as independent grapheme-level units;
- reduce exaggerated vowel repetitions;
- remove control characters;
- space punctuation without removing Unicode symbols;
- normalise whitespace.
"""

from __future__ import annotations

from dataclasses import dataclass

import html
import re

import emoji
import pandas as pd
import regex
from tqdm.auto import tqdm


URL_PATTERN = re.compile(
    r"(https?://\S+|www\.[A-Za-z0-9./?=_-]+|[A-Za-z0-9.-]+\.(com|org|net|info|io|gov|edu)\S*)",
    flags=re.IGNORECASE,
)

APOSTROPHE_PATTERN = re.compile(r"[\u2019\u2018\u02BC\u2032]")

CONTROL_PATTERN = re.compile(r"[\x00-\x1F\x7F]+")

ELONGATED_VOWELS_PATTERN = re.compile(r"(?i)([aeiou])\1{2,}")

SPACE_PATTERN = re.compile(r"\s+")

# Unicode punctuation only. Unicode symbols, including emojis, are preserved.
PUNCT_PATTERN = regex.compile(r"([\p{P}])")


@dataclass(frozen=True)
class CleaningConfig:
    """Configuration for basic text cleaning."""

    lowercase: bool = False
    decode_html: bool = True
    replace_ampersand: bool = True
    normalise_apostrophes: bool = True
    expand_contractions: bool = True
    remove_urls: bool = True
    url_placeholder: str = "URL"
    space_emojis: bool = True
    reduce_elongated_vowels: bool = True
    remove_control_chars: bool = True
    space_punctuation: bool = True
    normalise_whitespace: bool = True
    show_progress: bool = True


def is_emoji_grapheme(value: object) -> bool:
    """Return True if a Unicode grapheme cluster is an emoji.

    The function is robust to multi-codepoint emojis and modifiers.
    """

    if value is None:
        return False

    x = str(value)

    return (
        x in emoji.EMOJI_DATA
        or emoji.is_emoji(x)
        or any(ch in emoji.EMOJI_DATA for ch in x)
    )


def separate_emojis(text: object) -> str:
    """Add spaces around emoji grapheme clusters without splitting them.

    Examples
    --------
    'great😂' -> 'great 😂'

    'ok👍🏽' -> 'ok 👍🏽'
    """

    value = "" if text is None else str(text)

    clusters = regex.findall(r"\X", value)

    out: list[str] = []

    for cluster in clusters:
        if is_emoji_grapheme(cluster):
            out.append(f" {cluster} ")
        else:
            out.append(cluster)

    return "".join(out)


def clean_text(
    text: object,
    *,
    config: CleaningConfig | None = None,
    lowercase: bool | None = None,
    remove_urls: bool | None = None,
) -> str:
    """Clean a single text.

    Parameters
    ----------
    text:
        Raw textual input.
    config:
        Optional cleaning configuration.
    lowercase:
        Optional override for ``config.lowercase``.
    remove_urls:
        Optional override for ``config.remove_urls``.

    Returns
    -------
    str
        Cleaned text.
    """

    cfg = config or CleaningConfig()

    if lowercase is not None:
        cfg = CleaningConfig(
            **{
                **cfg.__dict__,
                "lowercase": lowercase,
            }
        )

    if remove_urls is not None:
        cfg = CleaningConfig(
            **{
                **cfg.__dict__,
                "remove_urls": remove_urls,
            }
        )

    value = "" if text is None or pd.isna(text) else str(text)

    if cfg.decode_html:
        value = html.unescape(value)

    if cfg.replace_ampersand:
        value = value.replace("&", "and")

    value = value.replace('""', '"')

    if cfg.normalise_apostrophes:
        value = APOSTROPHE_PATTERN.sub("'", value)

    if cfg.expand_contractions:
        try:
            import contractions

            value = contractions.fix(value)
        except ImportError as exc:
            raise ImportError(
                "The 'contractions' package is required when "
                "expand_contractions=True. Install it with "
                "`pip install contractions`, or set "
                "expand_contractions=False."
            ) from exc

    if cfg.remove_urls:
        value = URL_PATTERN.sub(cfg.url_placeholder, value)

        # Clean URL placeholders when surrounded by brackets/punctuation.
        value = re.sub(
            rf"[()\[\]{{}}<>]+({re.escape(cfg.url_placeholder)})",
            r" \1",
            value,
        )
        value = re.sub(
            rf"({re.escape(cfg.url_placeholder)})[()\[\]{{}}<>]+",
            r"\1 ",
            value,
        )
        value = re.sub(
            rf"\[([^\]]+)\s+{re.escape(cfg.url_placeholder)}",
            rf"\1 {cfg.url_placeholder}",
            value,
        )
        value = re.sub(
            rf"\s*{re.escape(cfg.url_placeholder)}\s*",
            f" {cfg.url_placeholder} ",
            value,
        )

    if cfg.space_emojis:
        value = separate_emojis(value)

    if cfg.reduce_elongated_vowels:
        value = ELONGATED_VOWELS_PATTERN.sub(r"\1", value)

    if cfg.remove_control_chars:
        value = CONTROL_PATTERN.sub(" ", value)

    if cfg.space_punctuation:
        value = PUNCT_PATTERN.sub(r" \1 ", value)

    if cfg.normalise_whitespace:
        value = SPACE_PATTERN.sub(" ", value).strip()

    if cfg.lowercase:
        value = value.lower()

    return value


def clean_corpus(
    df: pd.DataFrame,
    text_col: str = "text",
    output_col: str = "clean_text",
    *,
    config: CleaningConfig | None = None,
    lowercase: bool | None = None,
    remove_urls: bool | None = None,
) -> pd.DataFrame:
    """Apply basic cleaning to a dataframe.

    Parameters
    ----------
    df:
        Input dataframe.
    text_col:
        Name of the source text column.
    output_col:
        Name of the output cleaned-text column.
    config:
        Optional cleaning configuration.
    lowercase:
        Optional override for ``config.lowercase``.
    remove_urls:
        Optional override for ``config.remove_urls``.

    Returns
    -------
    pandas.DataFrame
        Copy of the input dataframe with an additional cleaned-text column.
    """

    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in dataframe.")

    cfg = config or CleaningConfig()

    out = df.copy()

    if cfg.show_progress:
        tqdm.pandas(desc="Cleaning text", unit="doc")

        out[output_col] = out[text_col].progress_apply(
            lambda value: clean_text(
                value,
                config=cfg,
                lowercase=lowercase,
                remove_urls=remove_urls,
            )
        )
    else:
        out[output_col] = out[text_col].map(
            lambda value: clean_text(
                value,
                config=cfg,
                lowercase=lowercase,
                remove_urls=remove_urls,
            )
        )

    return out