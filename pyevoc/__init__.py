"""PyEvoc: computational Hierarchical Evocation Analysis for digital corpora."""

from .data.dataset import DatasetConfig, load_dataset, standardise_dataset, corpus_summary

__version__ = "0.1.0"
__all__ = ["DatasetConfig", "load_dataset", "standardise_dataset", "corpus_summary"]