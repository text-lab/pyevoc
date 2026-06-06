"""High-level settings dataclasses."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class PyEvocSettings:
    """Immutable top-level configuration for a PyEvoc analysis run.

    Attributes:
        alpha: Significance threshold used when filtering statistical tests.
        r_max: Maximum radius for the concentric EVOC map display.
        language: BCP-47 language code for the corpus being analysed.
    """

    alpha: float = 0.50
    r_max: float = 5.0
    language: str = "en"
