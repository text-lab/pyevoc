# Methodology

## Introduction

PyEvoc implements a computational adaptation of the Hierarchical Evocation Method (HEM) grounded in Social Representation Theory (SRT).

Traditional HEM relies on elicitation procedures in which participants explicitly generate and rank lexical associations. In contrast, digital communication environments generate large volumes of naturally occurring discourse in which representational salience is not directly observable and must be inferred from interactional behaviour.

PyEvoc extends the logic of HEM to large-scale textual corpora by reconstructing the concepts of Average Frequency of Evocation (AFE) and Average Order of Evocation (AOE) through diffusion, positional prominence, and rhetorical foregrounding.

The framework is designed for social media data, online discussions, news comments, survey responses, forums, and other large textual collections.

---

# Notation

Let

D = {d₁, d₂, ..., dN} 

denote the collection of documents.

Let

U = {u₁, u₂, ..., uM} 

be the set of distinct users.

Each document is associated with:

x(d) = textual content    t(d) = timestamp    u(d) = author 

Let

V = {w₁, w₂, ..., wK} 

denote the vocabulary retained after preprocessing.

---

# Positional Salience

The first component of representational salience is positional prominence.

For a lexical unit occurring in document d containing Ld tokens:

r_pos(w,d) = 1                                    if Ld = 1  r_pos(w,d) = 1 - (position(w,d)-1)/(Ld-1)        if Ld > 1 

where:

- position(w,d) is the token position within the document;
- earlier occurrences receive greater salience.

This formulation assumes that terms introduced earlier in discourse tend to occupy more prominent communicative positions.

---

# Structural Salience

A second component captures rhetorical foregrounding.

Four binary indicators are considered:

I_first I_emph I_list I_intens 

representing whether a token occurs:

- in the opening sentence;
- in emphasised spans;
- in lists or quotations;
- in typographically intensified segments.

Structural salience is computed as:

r_str(w,d) = η₁ I_first(w,d) + η₂ I_emph(w,d) + η₃ I_list(w,d) + η₄ I_intens(w,d) 

subject to:

η₁ + η₂ + η₃ + η₄ = 1 

and

0 ≤ ηj ≤ 1 

for all j.

Default implementation:

η₁ = 0.40 η₂ = 0.30 η₃ = 0.15 η₄ = 0.15 

These values can be modified by the user.

---

# Average Salience

For each lexical unit w:

r̄_pos(w) = (1 / |I(w)|) Σ r_pos(w,d) 

and

r̄_str(w) = (1 / |I(w)|) Σ r_str(w,d) 

where I(w) denotes the set of all occurrences of w.

---

# Composite Salience

The two dimensions are integrated into a composite salience score:

S(w) = π r̄_pos(w) + (1−π) r̄_str(w) 

with

0 ≤ π ≤ 1 

Default implementation:

π = 0.50 

This weighting gives equal importance to positional and structural prominence.

---

# Reconstruction of AOE

Classical HEM relies on the Average Order of Evocation.

PyEvoc reconstructs this concept through a rank-like transformation:

R(w) = 1 + (1 − S(w))(Rmax − 1) 

where:

- larger salience implies lower rank values;
- lower rank values indicate greater representational prominence.

The corpus-level AOE threshold is:

AOE = (1 / |V|) Σ R(w) 

This threshold separates highly salient lexical units from less salient ones.

---

# Reconstruction of AFE

Representational diffusion is measured through user-level circulation.

For each lexical unit:

F_user(w) = |{u ∈ U : w ∈ V(u)}| / |U| 

where:

- V(u) is the vocabulary used by user u;
- each user contributes at most once.

This measure captures the breadth of collective sharing independently of raw repetition.

---

# Alternative Comment-Based Diffusion

PyEvoc also supports a document-based formulation:

F_comm(w) = |{d ∈ D : w ∈ d}| / |D| 

This version measures diffusion across documents rather than users.

---

# AFE Threshold

The Average Frequency of Evocation threshold is computed as:

AFE = (1 / |V|) Σ F_user(w) 

where:

- terms above the threshold are considered highly diffused;
- terms below the threshold are considered weakly diffused.

Thresholds are computed separately for each POS category.

---

# EVOC Quadrants

Lexical units are assigned to four representational regions.

Let:

θF(p) 

be the POS-specific AFE threshold.

Let:

θR(p) 

be the POS-specific AOE threshold.

For a lexical unit w:

## Central Nucleus

F_user(w) ≥ θF(p) and R(w) ≤ θR(p) 

Represents highly diffused and highly salient elements.

---

## First Periphery

F_user(w) ≥ θF(p) and R(w) > θR(p) 

Represents widely shared but less salient elements.

---

## Contrast Zone

F_user(w) < θF(p) and R(w) ≤ θR(p) 

Represents salient but weakly diffused elements.

---

## Peripheral System

F_user(w) < θF(p) and R(w) > θR(p) 

Represents low-salience and low-diffusion elements.

---

# POS-Specific Thresholding

PyEvoc computes thresholds separately for different lexical categories.

Current implementation supports:

- Nouns
- Adjectives
- Emojis

This choice reflects the fact that different grammatical categories fulfil different communicative functions and exhibit distinct diffusion and salience distributions.

### Nouns

Typically associated with objectification processes and referential stability.

### Adjectives

Primarily encode evaluative and normative dimensions.

### Emojis

Capture affective, interactional, and paralinguistic information.

POS-specific thresholding reduces artefacts caused by grammatical frequency asymmetries and improves interpretability.

---

# Temporal Stability Analysis

PyEvoc includes longitudinal diagnostics for assessing representational evolution.

Available indicators include:

- quadrant transitions;
- nucleus continuity;
- diffusion stability;
- rank stability;
- temporal Sankey diagrams;
- period-specific EVOC structures.

These measures allow researchers to investigate the emergence, persistence, transformation, and disappearance of representational elements over time.

---

# Parameterisation

The framework adopts default values for:

π η₁ η₂ η₃ η₄ 

based on theoretical considerations concerning positional prominence and rhetorical foregrounding.

These values should be interpreted as operational parameters rather than universal constants.

Future developments may include dedicated sensitivity analyses exploring the robustness of representational structures under alternative parameterisations.

---

# References

Abric, J.-C. (2003). La recherche du noyau central et de la zone muette des représentations sociales.

Vergès, P. (1992). L'évocation de l'argent: Une méthode pour la définition du noyau central d'une représentation.

Moliner, P., Rateau, P., & Cohen-Scali, V. (2002). Les représentations sociales.

PyEvoc extends these principles to large-scale digital corpora through computational reconstruction of salience and diffusion indicato
