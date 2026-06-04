"""PyEvoc preprocessing layer.

This subpackage contains the document-level and token-level preprocessing
routines used before feature construction and EVOC analysis. It includes
emoji-safe text cleaning, corpus diagnostics, language filtering, thematic
subsetting, and Stanza-based Universal Dependencies annotation.

The annotation module follows the notebook-compatible schema required by the
collocation pipeline, using ``token``, ``head_token_id`` and ``dep_rel`` as the
canonical dependency columns.
"""

# -----------------------------------------------------
# Text cleaning
# -----------------------------------------------------

from .cleaning import (
    CleaningConfig,
    is_emoji_grapheme,
    separate_emojis,
    clean_text,
    clean_corpus,
)

# -----------------------------------------------------
# Corpus-level statistics
# -----------------------------------------------------

from .corpus_statistics import (
    CorpusStatisticsConfig,
    simple_tokenise,
    is_valid_stat_token,
    extract_tokens,
    corpus_statistics,
)

# -----------------------------------------------------
# Language filtering
# -----------------------------------------------------

from .language_filtering import (
    LanguageFilterConfig,
    download_fasttext_lid_model,
    prepare_for_langid,
    fasttext_predict_batch,
    predict_language_fasttext,
    lingua_detect_batch,
    filter_by_language,
)

# -----------------------------------------------------
# Thematic filtering
# -----------------------------------------------------

from .thematic_filtering import (
    ThematicFilterConfig,
    load_anchor_terms,
    direct_anchor_filter,
    derive_expansion_terms,
    build_thematic_subset,
)

# -----------------------------------------------------
# Stanza UD annotation
# -----------------------------------------------------

from .annotation import (
    AnnotationConfig,
    validate_annotation_input,
    annotate_with_stanza,
    annotate_dataframe,
    annotation_diagnostics,
    dependency_diagnostics,
    contains_emoji_grapheme,
    count_emoji_documents,
    count_emoji_tokens,
    emoji_token_sample,
)

__all__ = [
    # Text cleaning
    "CleaningConfig",
    "is_emoji_grapheme",
    "separate_emojis",
    "clean_text",
    "clean_corpus",

    # Corpus-level statistics
    "CorpusStatisticsConfig",
    "simple_tokenise",
    "is_valid_stat_token",
    "extract_tokens",
    "corpus_statistics",

    # Language filtering
    "LanguageFilterConfig",
    "download_fasttext_lid_model",
    "prepare_for_langid",
    "fasttext_predict_batch",
    "predict_language_fasttext",
    "lingua_detect_batch",
    "filter_by_language",

    # Thematic filtering
    "ThematicFilterConfig",
    "load_anchor_terms",
    "direct_anchor_filter",
    "derive_expansion_terms",
    "build_thematic_subset",

    # Stanza UD annotation
    "AnnotationConfig",
    "validate_annotation_input",
    "annotate_with_stanza",
    "annotate_dataframe",
    "annotation_diagnostics",
    "dependency_diagnostics",
    "contains_emoji_grapheme",
    "count_emoji_documents",
    "count_emoji_tokens",
    "emoji_token_sample",
]
