# PyEvoc: Step-by-Step Tutorial

This tutorial illustrates the complete PyEvoc workflow, from raw corpus ingestion to EVOC-based representational analysis and visualisation.

The package is organised into five main modules:

```text
pyevoc.data
pyevoc.preprocessing
pyevoc.features
pyevoc.analysis
pyevoc.visualisation
```

A complete reproducible example is available in:

```text
examples/pyevoc_step_by_step.ipynb
```

---

# 1. Dataset Loading and Standardisation

PyEvoc uses a standard corpus schema:

```text
user_id | doc_id | time | text
```

Original datasets may use arbitrary column names. These are mapped to the PyEvoc standard using a `DatasetConfig`.

Supported formats include:

- CSV
- TSV
- Excel (.xlsx, .xls)
- Parquet
- JSON

Example:

```python
from pyevoc.data.dataset import DatasetConfig, load_dataset

config = DatasetConfig(
    column_map={
        "account_ID": "user_id",
        "tweet_ID": "doc_id",
        "tweet_pub_time": "time",
        "tweet_text": "text",
    }
)

corpus = load_dataset("AGI.xlsx", config=config)
```

---

# 2. Language Filtering

PyEvoc implements a two-stage language-identification pipeline.

Stage 1:

- fastText language identification

Stage 2 (optional):

- Lingua validation for ambiguous cases

Supported languages include:

- English
- Italian
- French
- German
- Spanish
- Portuguese

Install the optional dependency:

```bash
pip install lingua-language-detector
```

Example:

```python
from pyevoc.preprocessing.language_filtering import (
    LanguageFilterConfig,
    download_fasttext_lid_model,
    filter_by_language,
)

model_path = download_fasttext_lid_model()

config = LanguageFilterConfig(
    target_language="en",
    use_lingua_fallback=True,
)

corpus_en = filter_by_language(
    corpus,
    config=config,
    model_path=model_path,
)
```

---

# 3. Thematic Filtering

The thematic filter extracts a domain-specific subcorpus using:

1. Anchor terms
2. Semantic expansion

Anchor terms are stored in a plain-text file:

```text
solar energy
wind power
renewable energy
green hydrogen
...
```

One term per line.

Example:

```python
from pyevoc.preprocessing.thematic_filtering import (
    ThematicFilterConfig,
    build_thematic_subset,
)

subcorpus, metadata = build_thematic_subset(
    corpus_en,
    anchor_file="anchors.txt",
    config=ThematicFilterConfig()
)
```

The procedure automatically:

- identifies anchor hits;
- computes semantic expansion terms;
- creates the thematic subset;
- records filtering diagnostics.

---

# 4. Corpus Statistics

Basic descriptive indicators can be computed before further processing.

```python
from pyevoc.preprocessing.corpus_statistics import corpus_statistics

stats = corpus_statistics(
    subcorpus,
    text_col="text",
)
```

Outputs include:

- number of documents;
- number of users;
- token counts;
- vocabulary size;
- lexical diversity indicators.

---

# 5. Text Cleaning

PyEvoc provides a configurable cleaning pipeline.

Typical operations include:

- lowercasing;
- URL replacement;
- contraction expansion;
- punctuation normalisation;
- emoji spacing.

Example:

```python
from pyevoc.preprocessing.cleaning import (
    CleaningConfig,
    clean_corpus,
)

clean_corpus(...)
```

The cleaned text is stored in a dedicated column.

---

# 6. Linguistic Annotation

PyEvoc uses Stanza for linguistic annotation.

Supported processors include:

```text
tokenize
pos
lemma
depparse
ner
```

Example:

```python
from pyevoc.preprocessing import (
    AnnotationConfig,
    annotate_with_stanza,
)

tokens = annotate_with_stanza(...)
```

The output is a token-level dataframe.

---

# 7. Emoji Assignment

Emoji tokens can be reassigned to a dedicated:

```text
UPOS = EMOJI
```

category.

Example:

```python
from pyevoc.features.emoji_assignment import assign_emoji_upos

tokens = assign_emoji_upos(tokens)
```

Optional diagnostics are also available.

---

# 8. Structural Foregrounding

PyEvoc reconstructs salience indicators inspired by the Hierarchical Evocation Method.

The framework combines:

- positional prominence;
- opening-sentence emphasis;
- quotations and lists;
- rhetorical intensification.

Example:

```python
from pyevoc.features.foregrounding import (
    add_salience_indicators,
)

tokens = add_salience_indicators(...)
```

Outputs include:

```text
r_pos
r_str
```

scores for each token.

---

# 9. Unigram Selection

Lexical units are filtered according to:

- part-of-speech category;
- document frequency;
- user frequency.

Example:

```python
from pyevoc.features.unigram_selection import select_unigrams

tokens_selected = select_unigrams(...)
```

Typical retained categories:

```python
{"NOUN", "ADJ", "EMOJI"}
```

---

# 10. Term-Level Indicators

PyEvoc computes:

- diffusion (AFE-like component);
- salience (AOE-like component);
- rank values.

Example:

```python
from pyevoc.features.term_indices import (
    compute_term_statistics,
)

term_stats = compute_term_statistics(...)
```

---

# 11. Concreteness Labelling

Terms can be matched against a concreteness lexicon.

Example:

```python
from pyevoc.features.concreteness import (
    load_concreteness_lexicon,
    add_concreteness_labels,
)
```

Resulting categories include:

```text
Concrete
Mixed
Abstract
```

---

# 12. Emoji Labelling

Emoji terms can be enriched with textual descriptions.

Example:

```python
from pyevoc.features.emoji_labelling import (
    add_emoji_descriptions,
)
```

Descriptions are obtained from:

1. internal emoji lookup tables;
2. Unicode metadata fallback.

---

# 13. EVOC Quadrants

PyEvoc automatically computes quadrant thresholds separately for each POS category.

Example:

```python
from pyevoc.analysis.quadrants import assign_evoc_quadrants

evoc_quadrants = assign_evoc_quadrants(...)
```

Output quadrants:

- Central nucleus
- First periphery
- Contrast zone
- Peripheral system

---

# 14. Collocations and Named Entities

Dependency-based collocations can be extracted from the annotated corpus.

Example:

```python
from pyevoc.analysis.collocations_entities import (
    extract_collocations_and_entities,
)

results = extract_collocations_and_entities(...)
```

Optional outputs include:

- collocations;
- named entities;
- overlap diagnostics.

---

# 15. Temporal Stability

PyEvoc supports longitudinal representational analysis.

Available modes:

- equal-day partitions;
- equal-frequency partitions;
- custom periods.

Example:

```python
from pyevoc.analysis.temporal_stability import (
    run_temporal_stability_analysis,
)
```

Outputs include:

- stability metrics;
- quadrant transitions;
- longitudinal diagnostics.

---

# 16. Visualisation

PyEvoc currently provides:

## EVOC target maps

```python
build_evoc_target_plot(...)
```

## Semantic trees

```python
build_evoc_collocation_tree_for_upos(...)
```

## Emoji EVOC maps

```python
build_emoji_evoc_plot(...)
```

## Temporal Sankey diagrams

```python
build_temporal_sankey_ordered(...)
```

Outputs can be exported as:

- PNG
- HTML

depending on the selected configuration.

---

# Workflow Summary

A complete PyEvoc analysis typically follows the sequence:

```text
Dataset
   ↓
Language filtering
   ↓
Thematic filtering
   ↓
Cleaning
   ↓
Linguistic annotation
   ↓
Emoji assignment
   ↓
Foregrounding indicators
   ↓
Unigram selection
   ↓
Term statistics
   ↓
EVOC quadrants
   ↓
Collocations
   ↓
Temporal stability
   ↓
Visualisation
```

For a fully reproducible implementation, see:

```text
examples/pyevoc_step_by_step.ipynb
```