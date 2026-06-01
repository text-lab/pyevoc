"""Linguistic annotation utilities."""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class AnnotationConfig:
    text_col: str = "clean_text"
    doc_id_col: str = "doc_id"
    language: str = "en"
    processors: str = "tokenize,pos,lemma,ner"
    use_gpu: bool = False


def annotate_with_stanza(df: pd.DataFrame, config: AnnotationConfig = AnnotationConfig()) -> pd.DataFrame:
    """Annotate documents with Stanza and return a token-level table."""
    try:
        import stanza  # type: ignore
    except ImportError as exc:
        raise ImportError("Install stanza to use annotate_with_stanza().") from exc
    try:
        nlp = stanza.Pipeline(config.language, processors=config.processors, use_gpu=config.use_gpu, verbose=False)
    except Exception:
        stanza.download(config.language, processors=config.processors)
        nlp = stanza.Pipeline(config.language, processors=config.processors, use_gpu=config.use_gpu, verbose=False)

    rows = []
    for _, record in df.iterrows():
        doc_id = record[config.doc_id_col]
        text = str(record[config.text_col] or "")
        doc = nlp(text)
        for sent_i, sent in enumerate(doc.sentences, start=1):
            for token_i, word in enumerate(sent.words, start=1):
                rows.append({
                    "doc_id": doc_id,
                    "sentence_id": sent_i,
                    "token_id": token_i,
                    "text": word.text,
                    "lemma": word.lemma,
                    "upos": word.upos,
                    "xpos": word.xpos,
                    "deprel": word.deprel,
                    "head": word.head,
                })
    return pd.DataFrame(rows)
