"""Semantic-tree visualisation placeholders."""
from __future__ import annotations


def build_semantic_tree_edges(collocations, root_col="term_a", target_col="term_b", weight_col="G2"):
    return collocations[[root_col, target_col, weight_col]].copy()
