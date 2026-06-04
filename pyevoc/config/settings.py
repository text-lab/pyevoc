"""High-level settings dataclasses."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class PyEvocSettings:
    alpha: float = 0.50
    r_max: float = 5.0
    language: str = "en"
