# PyEvoc

<p align="center">
  <img src="assets/pyevoc_logo.png" width="260">
</p>

<p align="center">
<b>PyEvoc</b><br>
A Python Framework for Hierarchical Evocation Analysis in Large-Scale Digital Corpora
</p>

---

## Overview

PyEvoc is an open-source Python framework designed to operationalise the Hierarchical Evocation Method (HEM) within large-scale digital communication environments.

The framework extends classical approaches developed within Social Representation Theory (SRT) by enabling the reconstruction of representational structures directly from naturally occurring online discourse. Rather than relying on elicitation tasks, PyEvoc analyses large textual corpora and combines lexical diffusion, positional salience, rhetorical foregrounding, semantic association, and temporal dynamics to identify central and peripheral elements of public representations.

The package provides a complete workflow from corpus ingestion to representational mapping, collocation analysis, temporal stability assessment, and interactive visualisation.

---

## Theoretical Background

The Hierarchical Evocation Method (HEM) is rooted in the structural approach to Social Representation Theory.

Classical HEM relies on two dimensions:

- Frequency of Evocation (AFE)
- Average Order of Evocation (AOE)

These indicators are used to identify the internal organisation of social representations through a four-quadrant structure composed of:

1. Central nucleus
2. First periphery
3. Contrast zone
4. Peripheral system

PyEvoc computationally reconstructs these dimensions from naturally occurring discourse.

### AFE: Diffusion

AFE measures the collective diffusion of a lexical unit.

In PyEvoc, diffusion is typically computed at the user level:

\[
AFE(w)=|\{u\in U:w\in V(u)\}|
\]

where a term is counted only once for each user, regardless of repetition.

This indicator captures representational consensus and collective dissemination.

### AOE: Salience

AOE is reconstructed through a composite salience indicator that integrates:

- positional prominence;
- structural foregrounding;
- rhetorical emphasis.

Lower AOE values indicate greater representational salience.

Together, AFE and AOE provide the basis for EVOC quadrant assignment.

---

## Computational Workflow

<p align="center">
<img src="assets/pipeline.jpeg" width="100%">
</p>

The PyEvoc pipeline consists of the following stages:

1. Dataset ingestion
2. Language identification
3. Thematic filtering
4. Corpus diagnostics
5. Linguistic annotation
6. Emoji processing
7. Structural foregrounding
8. Term-level indicators
9. Concreteness labelling
10. EVOC quadrant assignment
11. Collocation extraction
12. Named Entity Recognition
13. Temporal stability analysis
14. Interactive reporting
15. Visual analytics

---

## Main Features

### Corpus Construction

- Flexible CSV ingestion
- Date filtering
- Metadata preservation
- Generic schema mapping

### Language Processing

- fastText language identification
- Stanza linguistic annotation
- Lemmatisation
- POS tagging
- Dependency parsing

### Thematic Extraction

- Anchor-based filtering
- Semantic expansion
- Domain-specific subcorpus generation

### Representational Analysis

- AFE reconstruction
- AOE reconstruction
- EVOC quadrants
- Central nucleus identification
- Peripheral structure analysis

### Semantic Analysis

- Collocations
- Named entities
- Semantic trees
- Entity-term overlap

### Longitudinal Analysis

- Temporal EVOC structures
- Quadrant transitions
- Stability indices
- Sankey evolution diagrams

### Reporting

- HTML outputs
- Interactive graphics
- Publication-ready figures

---

## Expected Input Structure

PyEvoc requires a pandas DataFrame containing at least four columns.

| Column | Description |
|----------|------------|
| user_id | User identifier |
| document_id | Document identifier |
| time | Datetime variable |
| text | Raw textual content |

Example:

```python
import pandas as pd

df = pd.DataFrame({
    "user_id": ["u1", "u2"],
    "document_id": ["d1", "d2"],
    "time": ["2025-01-01", "2025-01-02"],
    "text": ["Example text", "Another text"]
})
```

Additional metadata columns are automatically retained.

---

## Quick Start

The following example illustrates the minimal workflow.

```python
from pyevoc.dataset import load_dataset
from pyevoc.language import language_filter
from pyevoc.thematic import thematic_filter

df = load_dataset(
    path="corpus.csv",
    text_column="text",
    user_column="user_id",
    id_column="document_id",
    time_column="time"
)

df = language_filter(df)

subcorpus = thematic_filter(
    df,
    anchor_file="anchors.txt"
)
```

---

## Complete Workflow Example

```python
from pyevoc import *

# Load dataset
df = load_dataset(
    path="corpus.csv",
    text_column="text",
    user_column="user_id",
    id_column="document_id",
    time_column="time"
)

# Language filtering
df = language_filter(df)

# Thematic filtering
subcorpus = thematic_filter(
    df,
    anchor_file="anchors.txt"
)

# Corpus diagnostics
compute_subcorpus_statistics(subcorpus)

# Basic cleaning
subcorpus = clean_text(subcorpus)

# Linguistic annotation
tokens = annotate_corpus(subcorpus)

# Emoji assignment
tokens = assign_emojis(tokens)

# Structural foregrounding
tokens = compute_foregrounding(tokens)

# Term-level indicators
terms = compute_term_indices(tokens)

# Concreteness labelling
terms = label_concreteness(terms)

# Emoji descriptions
terms = label_emojis(terms)

# EVOC quadrants
quadrants = assign_quadrants(terms)

# Collocations
compute_collocations(tokens)

# Named entities
compute_ner(tokens)

# Temporal stability
analyse_temporal_stability(tokens)

# Reports
export_html_reports(quadrants)

# Visualisations
plot_evoc_map(quadrants)
plot_semantic_tree(tokens)
plot_emoji_map(quadrants)
plot_sankey(tokens)
```

---

## EVOC Quadrants

<p align="center">
<img src="assets/evoc_q.png" width="75%">
</p>

The representational structure is organised into four quadrants.

### Central Nucleus

High diffusion and high salience.

Represents the most stable and collectively shared elements of a representation.

### First Periphery

High salience but lower diffusion.

Contains important representational elements that remain less consensual.

### Contrast Zone

Low diffusion and high salience.

May indicate subgroup-specific meanings or emerging interpretative positions.

### Peripheral System

Low salience and low diffusion.

Represents contextual, flexible, and evolving representational elements.

---

## Example Outputs

### EVOC Map – Nouns

<p align="center">
<img src="assets/evoctarget_N.jpeg" width="95%">
</p>

### EVOC Map – Adjectives

<p align="center">
<img src="assets/evoctarget_A.jpeg" width="95%">
</p>

### Semantic Tree – Nouns

<p align="center">
<img src="assets/evoctree_N.jpeg" width="95%">
</p>

### Semantic Tree – Adjectives

<p align="center">
<img src="assets/evoctree_A.jpeg" width="95%">
</p>

### Emoji EVOC Map

<p align="center">
<img src="assets/emoji_map.png" width="95%">
</p>

### Temporal Stability

<p align="center">
<img src="assets/temp_sankey.png" width="95%">
</p>

---

## Package Structure

```text
PyEvoc/
│
├── pyevoc/
├── models/
├── assets/
├── docs/
├── examples/
├── tests/
│
├── README.md
├── LICENSE
├── CITATION.cff
└── pyproject.toml
```

---

## Models Included

The package distributes all required resources locally.

```text
models/
├── lid.176.bin
├── emoji_lookup.csv
├── concreteness.csv
└── ...
```

These resources are automatically loaded when not explicitly specified by the user.

---

## Documentation

Detailed documentation is available in:

```text
docs/documentation.md
```

including:

- complete API reference;
- parameter descriptions;
- advanced workflows;
- reproducibility guidelines;
- output interpretation.

---

## Reproducibility

PyEvoc is designed to support transparent and reproducible computational social science research.

The framework:

- preserves metadata throughout the workflow;
- records processing parameters;
- exports intermediate outputs;
- produces publication-ready figures;
- generates human-readable HTML reports.

---

## Citation

If you use PyEvoc in academic work, please cite:

```text
Citation information will be added upon software release.
```

A Zenodo DOI will be assigned upon publication of the first stable release.

---

## License

MIT License.

---

## Authors

PyEvoc was developed to support computational applications of Social Representation Theory and Hierarchical Evocation Analysis in large-scale digital communication environments.