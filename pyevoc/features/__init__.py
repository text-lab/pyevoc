"""PyEvoc feature-engineering layer.

This subpackage exposes foregrounding indicators, unigram selection, emoji
assignment, concreteness enrichment, emoji labelling and term-level EVOC
indices. Public objects are loaded lazily to keep documentation imports light.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PUBLIC_OBJECTS: dict[str, str] = {'ForegroundingConfig': 'foregrounding', 'ForegroundingWeights': 'foregrounding', 'add_foregrounding_indicators': 'foregrounding', 'add_positional_salience': 'foregrounding', 'add_salience_indicators': 'foregrounding', 'compute_structural_salience': 'foregrounding', 'foregrounding_diagnostics': 'foregrounding', 'metadata_coverage': 'foregrounding', 'SalienceConfig': 'term_indices', 'compute_term_indices': 'term_indices', 'compute_term_statistics': 'term_indices', 'term_statistics_summary': 'term_indices', 'UnigramSelectionConfig': 'unigram_selection', 'clean_unigram_tokens': 'unigram_selection', 'select_unigrams': 'unigram_selection', 'EmojiAssignmentConfig': 'emoji_assignment', 'assign_emoji_upos': 'emoji_assignment', 'emoji_assignment_diagnostics': 'emoji_assignment', 'emoji_summary': 'emoji_assignment', 'ConcretenessConfig': 'concreteness', 'resolve_concreteness_lexicon_path': 'concreteness', 'available_concreteness_lexicon_paths': 'concreteness', 'load_concreteness_lexicon': 'concreteness', 'add_concreteness_labels': 'concreteness', 'label_concreteness': 'concreteness', 'EmojiLabellingConfig': 'emoji_labelling', 'load_emoji_lookup': 'emoji_labelling', 'available_emoji_lookup_paths': 'emoji_labelling', 'add_emoji_descriptions': 'emoji_labelling', 'label_emojis': 'emoji_labelling'}

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
