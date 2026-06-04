"""
PyEvoc feature-engineering modules.

This subpackage contains:

- foregrounding and salience reconstruction
- unigram cleaning and selection
- emoji assignment
- term-level EVOC indices
- concreteness enrichment
- emoji description enrichment
"""

# =====================================================
# Foregrounding
# =====================================================

from .foregrounding import (
    ForegroundingConfig,
    ForegroundingWeights,
    add_foregrounding_indicators,
    add_positional_salience,
    add_salience_indicators,
    compute_structural_salience,
    foregrounding_diagnostics,
    metadata_coverage,
)

# =====================================================
# Term-level indices
# =====================================================

from .term_indices import (
    SalienceConfig,
    compute_term_indices,
    compute_term_statistics,
    term_statistics_summary,
)

# =====================================================
# Unigram selection
# =====================================================

from .unigram_selection import (
    UnigramSelectionConfig,
    clean_unigram_tokens,
    select_unigrams,
)

# =====================================================
# Emoji assignment
# =====================================================

from .emoji_assignment import (
    EmojiAssignmentConfig,
    assign_emoji_upos,
    emoji_assignment_diagnostics,
    emoji_summary,
)

# =====================================================
# Concreteness
# =====================================================

from .concreteness import (
    ConcretenessConfig,
    resolve_concreteness_lexicon_path,
    available_concreteness_lexicon_paths,
    load_concreteness_lexicon,
    add_concreteness_labels,
    label_concreteness,
)

# =====================================================
# Emoji descriptions
# =====================================================

from .emoji_labelling import (
    EmojiLabellingConfig,
    load_emoji_lookup,
    available_emoji_lookup_paths,
    add_emoji_descriptions,
    label_emojis,
)

# =====================================================
# Public API
# =====================================================

__all__ = [

    # Foregrounding
    "ForegroundingConfig",
    "ForegroundingWeights",
    "add_foregrounding_indicators",
    "add_positional_salience",
    "add_salience_indicators",
    "compute_structural_salience",
    "foregrounding_diagnostics",
    "metadata_coverage",

    # Term indices
    "SalienceConfig",
    "compute_term_indices",
    "compute_term_statistics",
    "term_statistics_summary",

    # Unigram selection
    "UnigramSelectionConfig",
    "clean_unigram_tokens",
    "select_unigrams",

    # Emoji assignment
    "EmojiAssignmentConfig",
    "assign_emoji_upos",
    "emoji_assignment_diagnostics",
    "emoji_summary",

    # Concreteness
    "ConcretenessConfig",
    "resolve_concreteness_lexicon_path",
    "available_concreteness_lexicon_paths",
    "load_concreteness_lexicon",
    "add_concreteness_labels",
    "label_concreteness",

    # Emoji descriptions
    "EmojiLabellingConfig",
    "load_emoji_lookup",
    "available_emoji_lookup_paths",
    "add_emoji_descriptions",
    "label_emojis",
]