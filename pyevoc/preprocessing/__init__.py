"""PyEvoc preprocessing layer.

This subpackage contains document-level and token-level preprocessing routines:
emoji-safe text cleaning, corpus diagnostics, language filtering, thematic
subsetting, and Stanza-based Universal Dependencies annotation.

The public names are loaded lazily so that API documentation can be generated
without importing optional runtime dependencies such as ``emoji``, ``fasttext``
or ``stanza`` unless the corresponding function is actually requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PUBLIC_OBJECTS: dict[str, str] = {'CleaningConfig': 'cleaning', 'is_emoji_grapheme': 'cleaning', 'separate_emojis': 'cleaning', 'clean_text': 'cleaning', 'clean_corpus': 'cleaning', 'CorpusStatisticsConfig': 'corpus_statistics', 'simple_tokenise': 'corpus_statistics', 'is_valid_stat_token': 'corpus_statistics', 'extract_tokens': 'corpus_statistics', 'corpus_statistics': 'corpus_statistics', 'LanguageFilterConfig': 'language_filtering', 'download_fasttext_lid_model': 'language_filtering', 'prepare_for_langid': 'language_filtering', 'fasttext_predict_batch': 'language_filtering', 'predict_language_fasttext': 'language_filtering', 'lingua_detect_batch': 'language_filtering', 'filter_by_language': 'language_filtering', 'ThematicFilterConfig': 'thematic_filtering', 'load_anchor_terms': 'thematic_filtering', 'direct_anchor_filter': 'thematic_filtering', 'derive_expansion_terms': 'thematic_filtering', 'build_thematic_subset': 'thematic_filtering', 'AnnotationConfig': 'annotation', 'validate_annotation_input': 'annotation', 'annotate_with_stanza': 'annotation', 'annotate_dataframe': 'annotation', 'annotation_diagnostics': 'annotation', 'dependency_diagnostics': 'annotation', 'contains_emoji_grapheme': 'annotation', 'count_emoji_documents': 'annotation', 'count_emoji_tokens': 'annotation', 'emoji_token_sample': 'annotation'}

__all__ = sorted(_PUBLIC_OBJECTS)

def __getattr__(name: str) -> Any:
    """Load public objects lazily from their implementation module."""
    try:
        module_name = _PUBLIC_OBJECTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value

def __dir__() -> list[str]:
    """Return the public API exposed by this subpackage."""
    return sorted(set(globals()) | set(__all__))
