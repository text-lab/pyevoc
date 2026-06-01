"""Language filtering utilities.

Large pretrained fastText models are not bundled with PyEvoc. Use
``download_fasttext_lid_model`` to download ``lid.176.bin`` from the official fastText
repository into ``~/.pyevoc/models``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve
import gzip
import shutil

import pandas as pd

from pyevoc.utils.resources import user_model_path

FASTTEXT_LID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
FASTTEXT_LID_FTZ_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"


@dataclass(frozen=True)
class LanguageFilterConfig:
    text_col: str = "text"
    target_language: str = "en"
    min_probability: float = 0.80
    prediction_col: str = "language"
    probability_col: str = "language_probability"
    keep_probability: bool = True


def download_fasttext_lid_model(compact: bool = False, overwrite: bool = False) -> Path:
    """Download the official fastText language-identification model.

    Parameters
    ----------
    compact:
        If ``True``, download ``lid.176.ftz`` instead of ``lid.176.bin``.
    overwrite:
        Re-download even if the file already exists.
    """
    name = "lid.176.ftz" if compact else "lid.176.bin"
    url = FASTTEXT_LID_FTZ_URL if compact else FASTTEXT_LID_URL
    path = user_model_path(name)
    if path.exists() and not overwrite:
        return path
    urlretrieve(url, path)
    return path


def predict_language_fasttext(texts: pd.Series, model_path: str | Path | None = None) -> pd.DataFrame:
    """Predict languages using fastText.

    Requires the optional ``fasttext`` Python package. Newlines are replaced because
    fastText expects one document per line.
    """
    try:
        import fasttext  # type: ignore
    except ImportError as exc:
        raise ImportError("Install fasttext to use predict_language_fasttext().") from exc

    path = Path(model_path) if model_path is not None else user_model_path("lid.176.bin")
    if not path.exists():
        raise FileNotFoundError(
            f"fastText model not found at {path}. Run download_fasttext_lid_model() first."
        )
    model = fasttext.load_model(str(path))
    labels, probs = [], []
    for value in texts.fillna("").astype(str):
        label, prob = model.predict(value.replace("\n", " "), k=1)
        labels.append(label[0].replace("__label__", "") if label else None)
        probs.append(float(prob[0]) if len(prob) else None)
    return pd.DataFrame({"language": labels, "language_probability": probs})


def filter_by_language(
    df: pd.DataFrame,
    config: LanguageFilterConfig = LanguageFilterConfig(),
    model_path: str | Path | None = None,
) -> pd.DataFrame:
    """Return rows matching the target language according to fastText."""
    predictions = predict_language_fasttext(df[config.text_col], model_path=model_path)
    out = df.reset_index(drop=True).copy()
    out[config.prediction_col] = predictions["language"]
    if config.keep_probability:
        out[config.probability_col] = predictions["language_probability"]
    keep = (out[config.prediction_col] == config.target_language) & (
        predictions["language_probability"] >= config.min_probability
    )
    return out.loc[keep].reset_index(drop=True)
