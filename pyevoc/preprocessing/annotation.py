"""Stanza UD annotation utilities for PyEvoc.

This module converts a document-level dataframe into a token-level Universal
Dependencies table. The output intentionally follows the original notebook
schema used by the EVOC pipeline, because dependency-based collocation
extraction requires the original column names ``token``, ``head_token_id`` and
``dep_rel``.

Canonical output columns
------------------------
The core token-level columns are:

- ``doc_id``
- ``paragraph_id``
- ``sentence_id``
- ``token_id``
- ``position``
- ``token``
- ``lemma``
- ``upos``
- ``xpos``
- ``feats``
- ``head_token_id``
- ``dep_rel``

If present in the input dataframe, ``user_id`` and ``time`` are copied to every
token row. No duplicate aliases such as ``text``, ``head`` or ``deprel`` are
created here. If other modules still expect those aliases, they should be
updated to the canonical names above.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import gc
from typing import Iterable

import pandas as pd
from tqdm.auto import tqdm


try:
    import regex as _regex  # type: ignore
except Exception:  # pragma: no cover
    _regex = None

try:
    import emoji as _emoji  # type: ignore
except Exception:  # pragma: no cover
    _emoji = None


@dataclass(frozen=True)
class AnnotationConfig:
    """Configuration for Stanza-based UD annotation."""

    text_col: str = "text_clean"
    doc_id_col: str = "post_id"
    user_col: str = "user_id"
    time_col: str = "time"

    language: str = "en"
    processors: str = "tokenize,pos,lemma,depparse"
    tokenize_no_ssplit: bool = False
    use_gpu: bool = False

    batch_size: int = 128
    show_progress: bool = True

    save_parquet: bool = False
    output_dir: str | Path = "annotated_outputs"
    output_name: str | None = None
    save_emoji_sample: bool = True


# ---------------------------------------------------------------------------
# Emoji diagnostics
# ---------------------------------------------------------------------------

def contains_emoji_grapheme(value: object) -> bool:
    """Return True if a string contains at least one emoji grapheme."""

    if _emoji is None:
        return False

    text = "" if pd.isna(value) else str(value)

    if _regex is not None:
        clusters = _regex.findall(r"\X", text)
    else:
        clusters = list(text)

    return any(
        cluster in _emoji.EMOJI_DATA or _emoji.is_emoji(cluster)
        for cluster in clusters
    )


def count_emoji_documents(
    df: pd.DataFrame,
    *,
    text_col: str = "text_clean",
) -> int:
    """Count documents containing at least one emoji grapheme."""

    if text_col not in df.columns:
        return 0

    return int(df[text_col].astype(str).map(contains_emoji_grapheme).sum())


def count_emoji_tokens(
    anno_df: pd.DataFrame,
    *,
    token_col: str = "token",
) -> int:
    """Count annotated tokens containing at least one emoji grapheme."""

    if token_col not in anno_df.columns:
        return 0

    return int(anno_df[token_col].astype(str).map(contains_emoji_grapheme).sum())


def emoji_token_sample(
    anno_df: pd.DataFrame,
    *,
    n: int = 30,
    token_col: str = "token",
) -> pd.DataFrame:
    """Return a sample of emoji-bearing annotated tokens."""

    if token_col not in anno_df.columns:
        return pd.DataFrame()

    mask = anno_df[token_col].astype(str).map(contains_emoji_grapheme)

    cols = [
        "doc_id",
        "user_id",
        "time",
        "paragraph_id",
        "sentence_id",
        "token_id",
        "position",
        "token",
        "lemma",
        "upos",
        "xpos",
        "feats",
        "head_token_id",
        "dep_rel",
    ]

    existing = [col for col in cols if col in anno_df.columns]

    return anno_df.loc[mask, existing].head(n).copy()


# ---------------------------------------------------------------------------
# Stanza loading and validation
# ---------------------------------------------------------------------------

def _load_stanza_pipeline(config: AnnotationConfig):
    """Load the Stanza pipeline, downloading the language model if needed."""

    try:
        import stanza  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The 'stanza' package is required for annotation. "
            "Install it with `pip install stanza`."
        ) from exc

    kwargs = {
        "lang": config.language,
        "processors": config.processors,
        "tokenize_no_ssplit": config.tokenize_no_ssplit,
        "use_gpu": config.use_gpu,
        "verbose": False,
    }

    try:
        return stanza.Pipeline(**kwargs)
    except Exception:
        stanza.download(config.language, processors=config.processors)
        return stanza.Pipeline(**kwargs)


def validate_annotation_input(
    df: pd.DataFrame,
    config: AnnotationConfig = AnnotationConfig(),
) -> None:
    """Validate the input dataframe before annotation."""

    required = {config.doc_id_col, config.text_col}
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            "Input dataframe is missing required annotation columns: "
            f"{sorted(missing)}"
        )


def _canonical_columns(include_user: bool = True, include_time: bool = True) -> list[str]:
    """Return the canonical output-column order."""

    cols = ["doc_id"]

    if include_user:
        cols.append("user_id")

    if include_time:
        cols.append("time")

    cols.extend(
        [
            "paragraph_id",
            "sentence_id",
            "token_id",
            "position",
            "token",
            "lemma",
            "upos",
            "xpos",
            "feats",
            "head_token_id",
            "dep_rel",
        ]
    )

    return cols


def _empty_annotation_frame(
    *,
    include_user: bool = True,
    include_time: bool = True,
) -> pd.DataFrame:
    """Return an empty annotation dataframe with canonical columns."""

    return pd.DataFrame(columns=_canonical_columns(include_user, include_time))


# ---------------------------------------------------------------------------
# Main annotation functions
# ---------------------------------------------------------------------------

def annotate_with_stanza(
    df: pd.DataFrame,
    config: AnnotationConfig = AnnotationConfig(),
) -> pd.DataFrame:
    """Annotate a document-level dataframe with Stanza.

    The returned dataframe is the correct input for dependency-based
    collocation extraction. In particular, it contains ``token``,
    ``head_token_id`` and ``dep_rel``.
    """

    validate_annotation_input(df, config)

    nlp = _load_stanza_pipeline(config)

    include_user = config.user_col in df.columns
    include_time = config.time_col in df.columns

    n_docs = len(df)
    n_docs_with_emoji = count_emoji_documents(df, text_col=config.text_col)

    if config.show_progress:
        print(f"Starting Stanza annotation: {n_docs:,} documents.")
        print(f"Documents with emoji before annotation: {n_docs_with_emoji:,}")

    all_batches: list[pd.DataFrame] = []
    batch_size = max(1, int(config.batch_size))

    iterator: Iterable[int] = range(0, n_docs, batch_size)

    if config.show_progress:
        iterator = tqdm(iterator, desc="Annotating documents", unit="batch")

    for start in iterator:
        end = min(start + batch_size, n_docs)
        batch = df.iloc[start:end].copy()

        rows: list[dict[str, object]] = []

        required_batch_cols = [config.doc_id_col, config.text_col]

        if include_user:
            required_batch_cols.append(config.user_col)

        if include_time:
            required_batch_cols.append(config.time_col)

        batch = batch[required_batch_cols]

        for _, record in batch.iterrows():
            doc_id = record[config.doc_id_col]
            text = "" if pd.isna(record[config.text_col]) else str(record[config.text_col])

            if not text.strip():
                continue

            doc = nlp(text)
            global_position = 0

            for sent_id, sent in enumerate(doc.sentences, start=1):
                for word in sent.words:
                    global_position += 1

                    row = {
                        "doc_id": doc_id,
                        "paragraph_id": 1,
                        "sentence_id": sent_id,
                        "token_id": word.id,
                        "position": global_position,
                        "token": word.text,
                        "lemma": word.lemma,
                        "upos": word.upos,
                        "xpos": word.xpos,
                        "feats": word.feats,
                        "head_token_id": word.head,
                        "dep_rel": word.deprel,
                    }

                    if include_user:
                        row["user_id"] = record[config.user_col]

                    if include_time:
                        row["time"] = record[config.time_col]

                    rows.append(row)

        if rows:
            all_batches.append(pd.DataFrame(rows))

        del batch, rows
        gc.collect()

    if all_batches:
        anno_df = pd.concat(all_batches, ignore_index=True)
    else:
        anno_df = _empty_annotation_frame(
            include_user=include_user,
            include_time=include_time,
        )

    # Ensure all canonical columns exist and are ordered consistently.
    for col in _canonical_columns(include_user=include_user, include_time=include_time):
        if col not in anno_df.columns:
            anno_df[col] = pd.NA

    anno_df = anno_df[_canonical_columns(include_user=include_user, include_time=include_time)].copy()

    n_tokens_with_emoji = count_emoji_tokens(anno_df)

    if config.show_progress:
        print(f"Completed annotation: {len(anno_df):,} annotated tokens.")
        print(f"Tokens with emoji after annotation: {n_tokens_with_emoji:,}")

        if n_docs_with_emoji > 0 and n_tokens_with_emoji == 0:
            print(
                "WARNING: emojis were present before annotation but no "
                "emoji-bearing tokens were found afterwards. Check the cleaning "
                "step or Stanza tokenisation."
            )

    if config.save_parquet:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_name = config.output_name or "stanza_anno"
        out_file = output_dir / f"{output_name}.parquet"

        anno_df.to_parquet(out_file, index=False)

        if config.show_progress:
            print(f"Saved annotation to: {out_file}")

        sample = emoji_token_sample(anno_df)

        if config.save_emoji_sample and not sample.empty:
            sample_file = output_dir / f"{output_name}_emoji_token_sample.csv"
            sample.to_csv(sample_file, index=False, encoding="utf-8-sig")

            if config.show_progress:
                print(f"Saved emoji diagnostic sample to: {sample_file}")

    del all_batches
    gc.collect()

    return anno_df


def annotate_dataframe(
    df: pd.DataFrame,
    df_name: str = "df",
    *,
    doc_id_col: str = "post_id",
    text_col: str = "text_clean",
    user_col: str = "user_id",
    time_col: str = "time",
    batch_size: int = 128,
    save_parquet: bool = True,
    output_dir: str | Path = "annotated_outputs",
    show_progress: bool = True,
    use_gpu: bool = False,
) -> pd.DataFrame:
    """Notebook-compatible wrapper around :func:`annotate_with_stanza`.

    This mirrors the original notebook call style:

    ``anno_df = annotate_dataframe(df, "df")``

    and saves ``<df_name>_stanza_anno.parquet`` when ``save_parquet=True``.
    """

    config = AnnotationConfig(
        text_col=text_col,
        doc_id_col=doc_id_col,
        user_col=user_col,
        time_col=time_col,
        processors="tokenize,pos,lemma,depparse",
        tokenize_no_ssplit=False,
        use_gpu=use_gpu,
        batch_size=batch_size,
        show_progress=show_progress,
        save_parquet=save_parquet,
        output_dir=output_dir,
        output_name=f"{df_name}_stanza_anno",
        save_emoji_sample=True,
    )

    return annotate_with_stanza(df, config=config)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def annotation_diagnostics(
    anno_df: pd.DataFrame,
    *,
    doc_col: str = "doc_id",
    user_col: str = "user_id",
    time_col: str = "time",
) -> pd.DataFrame:
    """Return a compact diagnostic table for an annotated token dataframe."""

    rows: list[tuple[str, object]] = [
        ("Tokens", len(anno_df)),
        (
            "Documents",
            anno_df[doc_col].nunique(dropna=True)
            if doc_col in anno_df.columns
            else pd.NA,
        ),
        (
            "Users",
            anno_df[user_col].nunique(dropna=True)
            if user_col in anno_df.columns
            else pd.NA,
        ),
    ]

    if time_col in anno_df.columns:
        t = pd.to_datetime(anno_df[time_col], errors="coerce", utc=True)
        rows.extend(
            [
                ("Minimum timestamp", t.min()),
                ("Maximum timestamp", t.max()),
            ]
        )

    if "upos" in anno_df.columns:
        rows.append(("UPOS categories", anno_df["upos"].nunique(dropna=True)))

    if "lemma" in anno_df.columns:
        rows.append(("Unique lemmas", anno_df["lemma"].nunique(dropna=True)))

    if "dep_rel" in anno_df.columns:
        rows.append(("Dependency relations", anno_df["dep_rel"].nunique(dropna=True)))

    if "head_token_id" in anno_df.columns:
        head = pd.to_numeric(anno_df["head_token_id"], errors="coerce")
        rows.extend(
            [
                ("Tokens with dependency head", head.notna().sum()),
                ("Root tokens", head.eq(0).sum()),
            ]
        )

    if "token" in anno_df.columns:
        rows.append(("Tokens with emoji", count_emoji_tokens(anno_df)))

    return (
        pd.DataFrame(rows, columns=["Statistic", "Value"])
        .dropna(subset=["Value"])
        .reset_index(drop=True)
    )


def dependency_diagnostics(
    anno_df: pd.DataFrame,
    *,
    dep_rel_col: str = "dep_rel",
    head_col: str = "head_token_id",
) -> pd.DataFrame:
    """Return diagnostics for dependency columns used by collocations."""

    rows: list[tuple[str, object]] = [
        ("Rows", len(anno_df)),
        ("Has dep_rel", dep_rel_col in anno_df.columns),
        ("Has head_token_id", head_col in anno_df.columns),
    ]

    if dep_rel_col in anno_df.columns:
        rows.extend(
            [
                ("Non-null dep_rel", anno_df[dep_rel_col].notna().sum()),
                ("Unique dep_rel", anno_df[dep_rel_col].nunique(dropna=True)),
            ]
        )

    if head_col in anno_df.columns:
        head = pd.to_numeric(anno_df[head_col], errors="coerce")
        rows.extend(
            [
                ("Non-null head_token_id", head.notna().sum()),
                ("Root tokens", head.eq(0).sum()),
            ]
        )

    return pd.DataFrame(rows, columns=["Statistic", "Value"])


__all__ = [
    "AnnotationConfig",
    "annotate_with_stanza",
    "annotate_dataframe",
    "annotation_diagnostics",
    "dependency_diagnostics",
    "contains_emoji_grapheme",
    "count_emoji_documents",
    "count_emoji_tokens",
    "emoji_token_sample",
    "validate_annotation_input",
]
