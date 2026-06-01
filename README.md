<p align="center">
  <img src="assets/pyevoc_logo.png" width="35%">
</p>

<p align="center">
  <strong>A Python Framework for Hierarchical Evocation Analysis in Large-Scale Digital Corpora</strong>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <a href="docs/methodology.md"><img src="https://img.shields.io/badge/methodology-docs-green.svg" alt="Methodology"></a>
  <a href="https://doi.org/10.5281/zenodo.20493285"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.20493285.svg" alt="DOI"></a>
</p>

---

## Overview

**PyEvoc** is an open-source Python framework that operationalises the **Hierarchical Evocation Method (HEM)** for large-scale digital communication environments.

The framework extends classical approaches developed within **Social Representation Theory (SRT)** by reconstructing representational structures directly from naturally occurring online discourse — without relying on elicitation tasks. It combines lexical diffusion, positional salience, rhetorical foregrounding, semantic association, and temporal dynamics to identify the central and peripheral elements of public representations.

PyEvoc provides a complete, end-to-end workflow: from corpus ingestion and linguistic annotation, through EVOC quadrant assignment and collocation analysis, to temporal stability assessment and interactive visualisation.

> A detailed mathematical description of the framework is available in [docs/methodology.md](docs/methodology.md).

---

## Features

| Module | Capabilities |
|---|---|
| **Corpus Construction** | Flexible CSV ingestion, date filtering, metadata preservation, schema mapping |
| **Language Processing** | fastText language identification, Stanza annotation, lemmatisation, POS tagging, dependency parsing |
| **Thematic Extraction** | Anchor-based filtering, semantic expansion, domain-specific subcorpus generation |
| **Representational Analysis** | AFE/AOE reconstruction, EVOC quadrant assignment, central nucleus and peripheral structure identification |
| **Semantic Analysis** | Collocations, named entities, semantic trees, entity–term overlap |
| **Longitudinal Analysis** | Temporal EVOC structures, quadrant transitions, stability indices, Sankey evolution diagrams |
| **Reporting** | Interactive HTML outputs, publication-ready figures |

---

## Expected Input Structure

PyEvoc requires a `pandas.DataFrame` with at least four columns:

| Column | Description |
|---|---|
| `user_id` | User identifier |
| `document_id` | Document identifier |
| `time` | Datetime variable |
| `text` | Raw textual content |

Additional metadata columns are automatically preserved throughout the pipeline.

```python
import pandas as pd

df = pd.DataFrame({
    "user_id":     ["u1", "u2"],
    "document_id": ["d1", "d2"],
    "time":        ["2025-01-01", "2025-01-02"],
    "text":        ["Example text", "Another text"]
})
```

---

## Quick Start

```python
from pyevoc.dataset  import load_dataset
from pyevoc.language import language_filter
from pyevoc.thematic import thematic_filter

df = load_dataset(
    path="corpus.csv",
    text_column="text",
    user_column="user_id",
    id_column="document_id",
    time_column="time"
)

df        = language_filter(df)
subcorpus = thematic_filter(df, anchor_file="anchors.txt")
```

---

## Complete Workflow

```python
from pyevoc import *

# --- Ingestion ---
df = load_dataset(
    path="corpus.csv",
    text_column="text",
    user_column="user_id",
    id_column="document_id",
    time_column="time"
)

# --- Preprocessing ---
df        = language_filter(df)
subcorpus = thematic_filter(df, anchor_file="anchors.txt")
subcorpus = clean_text(subcorpus)
compute_subcorpus_statistics(subcorpus)

# --- Linguistic annotation ---
tokens = annotate_corpus(subcorpus)
tokens = assign_emojis(tokens)
tokens = compute_foregrounding(tokens)

# --- Term-level indicators ---
terms = compute_term_indices(tokens)
terms = label_concreteness(terms)
terms = label_emojis(terms)

# --- Representational mapping ---
quadrants = assign_quadrants(terms)

# --- Semantic analysis ---
compute_collocations(tokens)
compute_ner(tokens)

# --- Temporal analysis ---
analyse_temporal_stability(tokens)

# --- Output ---
export_html_reports(quadrants)
plot_evoc_map(quadrants)
plot_semantic_tree(tokens)
plot_emoji_map(quadrants)
plot_sankey(tokens)
```

---

## Computational Pipeline

<p align="center">
  <img src="assets/pipeline.png" width="50%">
</p>

The pipeline consists of 15 stages: dataset ingestion → language identification → thematic filtering → corpus diagnostics → linguistic annotation → emoji processing → structural foregrounding → term-level indicators → concreteness labelling → EVOC quadrant assignment → collocation extraction → named entity recognition → temporal stability analysis → interactive reporting → visual analytics.

---

## EVOC Quadrant Structure

<p align="center">
  <img src="assets/evoc_q.png" width="40%">
</p>

Lexical units are positioned in a two-dimensional space defined by **representational diffusion** (AFE) and **discursive salience** (AOE), yielding four analytically distinct zones:

| Zone | Diffusion | Salience | Interpretation |
|---|---|---|---|
| **Central Nucleus** | High | High | Stable, consensual core of the representation |
| **First Periphery** | High | Low | Widely shared but contextually flexible elements |
| **Contrast Zone** | Low | High | Minority positions or emerging framings |
| **Peripheral System** | Low | Low | Contextually variable, weakly structured elements |

Thresholds are computed separately for each POS category (nouns, adjectives, emojis) to avoid artefacts from grammatical frequency asymmetries.

---

## Example Outputs

<table>
  <tr>
    <td align="center"><strong>EVOC Map — Nouns</strong><br><img src="assets/evoctarget_N.png" width="100%"></td>
    <td align="center"><strong>EVOC Map — Adjectives</strong><br><img src="assets/evoctarget_A.png" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><strong>Semantic Tree — Nouns</strong><br><img src="assets/evoctree_N.png" width="100%"></td>
    <td align="center"><strong>Semantic Tree — Adjectives</strong><br><img src="assets/evoctree_A.png" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><strong>Emoji EVOC Map</strong><br><img src="assets/emoji_map.png" width="100%"></td>
    <td align="center"><strong>Temporal Stability (Sankey)</strong><br><img src="assets/temp_sankey.png" width="100%"></td>
  </tr>
</table>

---

## Package Structure

```text
PyEvoc/
├── pyevoc/           # Core library
├── models/           # Bundled resources (see below)
├── assets/           # Logo, figures
├── docs/             # methodology.md and additional documentation
├── examples/         # Worked examples
├── tests/            # Test suite
├── README.md
├── LICENSE
├── CITATION.cff
└── pyproject.toml
```

### Bundled Models

All required resources are distributed locally and loaded automatically:

```text
models/
├── lid.176.bin         # fastText language identification model
├── emoji_lookup.csv    # Emoji–description mapping
├── concreteness.csv    # Concreteness norms
└── ...
```

---

## Reproducibility

PyEvoc is designed to support transparent and reproducible computational social science research. The framework preserves metadata throughout the workflow, records processing parameters, exports intermediate outputs, and generates human-readable HTML reports alongside publication-ready figures.

---

## Citation

If you use PyEvoc in academic work, please cite:

```bibtex
@software{misuraca2026pyevoc,
  author       = {Misuraca, Michelangelo},
  title        = {PyEvoc: Computational Hierarchical Evocation Analysis for Digital Corpora},
  year         = {2026},
  version      = {0.1.0},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20493284},
  url          = {https://doi.org/10.5281/zenodo.20493284}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
