"""Language filtering utilities for large-scale textual corpora.

PyEvoc uses fastText language identification as the primary detector and can
optionally apply a Lingua fallback to uncertain cases.

Large pretrained fastText models are not bundled with PyEvoc. Use
``download_fasttext_lid_model`` to download ``lid.176.bin`` or
``lid.176.ftz`` from the official fastText repository into
``~/.pyevoc/models``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve
import gc
import time

import pandas as pd
from tqdm.auto import tqdm

from pyevoc.utils.resources import user_model_path


FASTTEXT_LID_URL = (
    "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
)

FASTTEXT_LID_FTZ_URL = (
    "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
)


@dataclass(frozen=True)
class LanguageFilterConfig:
    """Configuration for language filtering."""

    text_col: str = "text"

    target_language: str = "en"

    min_probability: float = 0.90

    ambiguous_min_probability: float = 0.50
    ambiguous_max_probability: float = 0.90

    use_lingua_fallback: bool = True
    lingua_min_confidence: float = 0.80

    min_text_length: int = 3

    fasttext_batch_size: int = 50000
    lingua_batch_size: int = 10000

    prediction_col: str = "language"
    probability_col: str = "language_probability"

    lingua_prediction_col: str = "language_lingua"
    lingua_confidence_col: str = "language_lingua_confidence"

    keep_probability: bool = True
    keep_diagnostics: bool = False

    show_progress: bool = True
    verbose: bool = True


def _log(message: str, *, enabled: bool = True) -> None:
    """Internal silent logger.

    Verbose runs intentionally show progress bars only.
    """

    return None


def _human_mb(n_bytes: int) -> str:
    """Format bytes as megabytes."""

    return f"{n_bytes / (1024**2):.2f} MB"


def _format_seconds(seconds: float) -> str:
    """Format seconds as human-readable elapsed time."""

    if seconds < 60:
        return f"{seconds:.2f} s"

    minutes = int(seconds // 60)
    sec = seconds % 60

    return f"{minutes} min {sec:.2f} s"


def download_fasttext_lid_model(
    compact: bool = False,
    overwrite: bool = False,
) -> Path:
    """Download the official fastText language-identification model.

    Parameters
    ----------
    compact:
        If True, download ``lid.176.ftz`` instead of ``lid.176.bin``.
    overwrite:
        Re-download even if the file already exists.

    Returns
    -------
    pathlib.Path
        Local model path.
    """

    name = "lid.176.ftz" if compact else "lid.176.bin"
    url = FASTTEXT_LID_FTZ_URL if compact else FASTTEXT_LID_URL

    path = user_model_path(name)

    if path.exists() and not overwrite:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading fastText language-ID model to {path}")
    urlretrieve(url, path)

    if path.exists():
        print(f"Downloaded {_human_mb(path.stat().st_size)}")

    return path


def prepare_for_langid(text: object) -> str:
    """Prepare text for language identification."""

    value = "" if text is None or pd.isna(text) else str(text)
    value = value.replace("\n", " ").replace("\r", " ")
    value = " ".join(value.split())

    return value


def _load_fasttext_model(model_path: str | Path | None = None):
    """Load a fastText language-identification model."""

    try:
        import fasttext  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Install fasttext to use fastText language identification. "
            "For example: pip install fasttext"
        ) from exc

    path = Path(model_path) if model_path is not None else user_model_path(
        "lid.176.bin"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"fastText model not found at {path}. "
            "Run download_fasttext_lid_model() first."
        )

    return fasttext.load_model(str(path))


def _normalise_fasttext_label(label: object) -> str | None:
    """Normalise a fastText label."""

    if label is None:
        return None

    value = str(label)

    return value.replace("__label__", "")


def fasttext_predict_batch(
    texts: pd.Series,
    model: Any,
    *,
    batch_size: int = 50000,
    show_progress: bool = True,
    verbose: bool = True,
) -> tuple[list[str | None], list[float | None]]:
    """Predict languages with fastText in batches.

    The function first attempts fastText's list-based batch prediction.
    If that fails because of a fastText/NumPy compatibility issue, it falls
    back to row-wise prediction. Display is limited to one global progress bar.
    """

    labels: list[str | None] = []
    probs: list[float | None] = []

    n = len(texts)

    progress = (
        tqdm(total=n, desc="Detecting language with fastText", unit="doc")
        if show_progress
        else None
    )

    try:
        for batch_idx, start in enumerate(range(0, n, batch_size), start=1):
            end = min(start + batch_size, n)
            batch = texts.iloc[start:end].tolist()

            try:
                pred_labels, pred_probs = model.predict(
                    batch,
                    k=1,
                    threshold=0.0,
                )

                labels.extend(
                    [
                        _normalise_fasttext_label(lab[0]) if lab else None
                        for lab in pred_labels
                    ]
                )

                probs.extend(
                    [
                        float(prob[0]) if len(prob) else None
                        for prob in pred_probs
                    ]
                )

            except Exception:
                for value in batch:
                    try:
                        pred_label, pred_prob = model.predict(
                            str(value),
                            k=1,
                            threshold=0.0,
                        )

                        labels.append(
                            _normalise_fasttext_label(pred_label[0])
                            if pred_label
                            else None
                        )

                        probs.append(
                            float(pred_prob[0])
                            if len(pred_prob)
                            else None
                        )

                    except Exception:
                        labels.append(None)
                        probs.append(None)

            if progress is not None:
                progress.update(end - start)

    finally:
        if progress is not None:
            progress.close()

    if len(labels) != n or len(probs) != n:
        raise RuntimeError(
            "fastText prediction returned inconsistent lengths: "
            f"expected {n}, got labels={len(labels)}, probs={len(probs)}."
        )

    return labels, probs


def predict_language_fasttext(
    texts: pd.Series,
    model_path: str | Path | None = None,
    *,
    batch_size: int = 50000,
    show_progress: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Predict language labels and probabilities using fastText."""

    model = _load_fasttext_model(model_path)

    prepared = texts.fillna("").astype(str).map(prepare_for_langid)

    labels, probs = fasttext_predict_batch(
        prepared,
        model,
        batch_size=batch_size,
        show_progress=show_progress,
        verbose=verbose,
    )

    del model
    gc.collect()

    return pd.DataFrame(
        {
            "language": labels,
            "language_probability": probs,
        }
    )


def _load_lingua_detector():
    """Load a Lingua detector.

    Supports either ``linguars`` or ``lingua-language-detector`` when
    available.
    """

    try:
        from linguars import LanguageDetector  # type: ignore

        return ("linguars", LanguageDetector())

    except ImportError:
        pass

    try:
        from lingua import Language, LanguageDetectorBuilder  # type: ignore

        detector = (
            LanguageDetectorBuilder
            .from_languages(
                Language.ENGLISH,
                Language.FRENCH,
                Language.GERMAN,
                Language.SPANISH,
                Language.ITALIAN,
                Language.PORTUGUESE,
                Language.DUTCH,
            )
            .with_preloaded_language_models()
            .build()
        )

        return ("lingua", detector)

    except ImportError as exc:
        raise ImportError(
            "Lingua fallback requested, but neither 'linguars' nor "
            "'lingua-language-detector' is installed. Install one of them "
            "or set use_lingua_fallback=False."
        ) from exc


def lingua_detect_batch(
    texts: pd.Series,
    *,
    batch_size: int = 10000,
    show_progress: bool = True,
    verbose: bool = True,
) -> tuple[list[str | None], list[float | None]]:
    """Apply Lingua fallback to a series of texts.

    Display is limited to one global progress bar. The implementation appends
    exactly one language label and one confidence value per input text.
    """

    backend, detector = _load_lingua_detector()

    langs: list[str | None] = []
    confs: list[float | None] = []

    values = texts.fillna("").astype(str).reset_index(drop=True)
    n = len(values)

    progress = (
        tqdm(total=n, desc="Detecting language with Lingua", unit="doc")
        if show_progress
        else None
    )

    try:
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = values.iloc[start:end]

            for text in batch:
                lang_value: str | None = None
                conf_value: float | None = None

                try:
                    if text.strip():
                        if backend == "linguars":
                            detected = detector.detect(text)
                            confidence = detector.confidence(text)

                            lang_value = (
                                str(detected)
                                if detected is not None
                                else None
                            )

                            conf_value = (
                                float(confidence)
                                if confidence is not None
                                else None
                            )

                        else:
                            detected = detector.detect_language_of(text)

                            if detected is not None:
                                lang_value = getattr(
                                    detected,
                                    "name",
                                    str(detected),
                                )

                                confidence_values = (
                                    detector
                                    .compute_language_confidence_values(text)
                                )

                                for item in confidence_values:
                                    if item.language == detected:
                                        conf_value = float(item.value)
                                        break

                except Exception:
                    lang_value = None
                    conf_value = None

                langs.append(lang_value)
                confs.append(conf_value)

            if progress is not None:
                progress.update(end - start)

    finally:
        if progress is not None:
            progress.close()

    # Defensive alignment. This should not normally be triggered, but it avoids
    # hard failures with backend-specific edge cases.
    if len(langs) < n:
        langs.extend([None] * (n - len(langs)))
    elif len(langs) > n:
        langs = langs[:n]

    if len(confs) < n:
        confs.extend([None] * (n - len(confs)))
    elif len(confs) > n:
        confs = confs[:n]

    return langs, confs


def _normalise_lingua_label(value: object) -> str | None:
    """Normalise Lingua language labels to ISO-like lowercase labels."""

    if value is None or pd.isna(value):
        return None

    v = str(value).upper()

    if v in {"ENGLISH", "LANGUAGE.ENGLISH"}:
        return "en"

    if v in {"ITALIAN", "LANGUAGE.ITALIAN"}:
        return "it"

    if v in {"FRENCH", "LANGUAGE.FRENCH"}:
        return "fr"

    if v in {"GERMAN", "LANGUAGE.GERMAN"}:
        return "de"

    if v in {"SPANISH", "LANGUAGE.SPANISH"}:
        return "es"

    if v in {"PORTUGUESE", "LANGUAGE.PORTUGUESE"}:
        return "pt"

    if v in {"DUTCH", "LANGUAGE.DUTCH"}:
        return "nl"

    return v.lower()


def filter_by_language(
    df: pd.DataFrame,
    config: LanguageFilterConfig = LanguageFilterConfig(),
    model_path: str | Path | None = None,
) -> pd.DataFrame:
    """Return rows matching the target language.

    Decision rule
    -------------
    A document is retained when:

    1. fastText predicts the target language with probability greater than
       or equal to ``min_probability``;

    or, if Lingua fallback is enabled,

    2. fastText predicts the target language with probability in the
       ambiguous interval and Lingua confirms the same target language
       with confidence greater than or equal to ``lingua_min_confidence``.
    """

    if config.text_col not in df.columns:
        raise ValueError(
            f"Column '{config.text_col}' not found in dataframe."
        )

    out = df.reset_index(drop=True).copy()

    out["_text_langid"] = (
        out[config.text_col]
        .fillna("")
        .astype(str)
        .map(prepare_for_langid)
    )

    out = out[
        out["_text_langid"].str.len() >= config.min_text_length
    ].copy()

    _log(
        f"Rows retained after minimum length filter: {len(out):,}",
        enabled=config.verbose,
    )

    predictions = predict_language_fasttext(
        out["_text_langid"],
        model_path=model_path,
        batch_size=config.fasttext_batch_size,
        show_progress=config.show_progress,
        verbose=config.verbose,
    )

    out[config.prediction_col] = predictions["language"].values
    out[config.probability_col] = predictions[
        "language_probability"
    ].values

    high_confidence_mask = (
        (out[config.prediction_col] == config.target_language)
        & (
            pd.to_numeric(
                out[config.probability_col],
                errors="coerce",
            )
            >= config.min_probability
        )
    )

    probability = pd.to_numeric(
        out[config.probability_col],
        errors="coerce",
    )

    ambiguous_mask = probability.between(
        config.ambiguous_min_probability,
        config.ambiguous_max_probability,
        inclusive="left",
    )

    target_ambiguous_mask = (
        ambiguous_mask
        & (out[config.prediction_col] == config.target_language)
    )

    out["_is_target_language"] = high_confidence_mask

    if config.use_lingua_fallback:
        ambiguous_idx = out.index[target_ambiguous_mask]

        out[config.lingua_prediction_col] = pd.NA
        out[config.lingua_confidence_col] = pd.NA

        _log(
            f"Ambiguous target-language texts requiring Lingua fallback: "
            f"{len(ambiguous_idx):,}",
            enabled=config.verbose,
        )

        if len(ambiguous_idx) > 0:
            lingua_langs, lingua_confs = lingua_detect_batch(
                out.loc[ambiguous_idx, "_text_langid"],
                batch_size=config.lingua_batch_size,
                show_progress=config.show_progress,
                verbose=config.verbose,
            )

            out.loc[
                ambiguous_idx,
                config.lingua_prediction_col,
            ] = lingua_langs

            out.loc[
                ambiguous_idx,
                config.lingua_confidence_col,
            ] = lingua_confs

            lingua_norm = out[config.lingua_prediction_col].map(
                _normalise_lingua_label
            )

            lingua_conf = pd.to_numeric(
                out[config.lingua_confidence_col],
                errors="coerce",
            )

            lingua_confirmed_mask = (
                target_ambiguous_mask
                & (lingua_norm == config.target_language)
                & (lingua_conf >= config.lingua_min_confidence)
            )

            out["_is_target_language"] = (
                out["_is_target_language"] | lingua_confirmed_mask
            )

    retained = int(out["_is_target_language"].sum())

    _log(
        f"Documents retained as '{config.target_language}': "
        f"{retained:,} / {len(out):,}",
        enabled=config.verbose,
    )

    keep = out["_is_target_language"]

    if not config.keep_diagnostics:
        drop_cols = [
            "_text_langid",
            "_is_target_language",
        ]

        if not config.keep_probability:
            drop_cols.append(config.probability_col)

        if config.lingua_prediction_col in out.columns:
            drop_cols.append(config.lingua_prediction_col)

        if config.lingua_confidence_col in out.columns:
            drop_cols.append(config.lingua_confidence_col)

        out = out.drop(
            columns=drop_cols,
            errors="ignore",
        )

    return out.loc[keep].reset_index(drop=True)