"""Composable full-pipeline helpers.

This module intentionally exposes a light orchestration layer. Research users are expected
also to run modules step by step for transparency and reproducibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pyevoc.data.dataset import DatasetConfig, load_dataset
from pyevoc.preprocessing.cleaning import clean_corpus
from pyevoc.preprocessing.thematic_filtering import ThematicFilterConfig, build_thematic_subset


@dataclass(frozen=True)
class BasicPipelineConfig:
    dataset: DatasetConfig
    anchors: str | Path
    thematic: ThematicFilterConfig = ThematicFilterConfig()
    clean_text_col: str = "clean_text"


def build_clean_thematic_corpus(path: str | Path, config: BasicPipelineConfig) -> tuple[pd.DataFrame, dict]:
    corpus = load_dataset(path, config.dataset)
    corpus = clean_corpus(corpus, text_col="text", output_col=config.clean_text_col)
    subset, metadata = build_thematic_subset(corpus, config.anchors, config.thematic)
    return subset, metadata
