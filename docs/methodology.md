# PyEvoc — Methodology

This document describes the computational and representational framework underlying **PyEvoc**, a Python library for the analysis of social representations in large-scale social media discourse. The approach extends the classical **Hierarchical Evocation Method (HEM)** to naturally occurring, user-generated corpora by reformulating its core constructs — evocation frequency and evocation order — through discursive diffusion and textual salience.

---

## Table of Contents

1. [Theoretical Background](#1-theoretical-background)
2. [Corpus and Notation](#2-corpus-and-notation)
3. [Discursive Salience](#3-discursive-salience)
   - 3.1 [Positional Salience](#31-positional-salience)
   - 3.2 [Structural Salience](#32-structural-salience)
   - 3.3 [Composite Salience Score](#33-composite-salience-score)
   - 3.4 [Rank Transformation](#34-rank-transformation)
4. [Representational Diffusion](#4-representational-diffusion)
   - 4.1 [User-Level Diffusion](#41-user-level-diffusion)
   - 4.2 [Comment-Level Diffusion](#42-comment-level-diffusion)
5. [Structural Thresholds: AFE and AOE](#5-structural-thresholds-afe-and-aoe)
6. [Four-Quadrant Representational Structure](#6-four-quadrant-representational-structure)
7. [POS-Specific Thresholding](#7-pos-specific-thresholding)
8. [Summary](#8-summary)
9. [References](#9-references)

---

## 1. Theoretical Background

The **Hierarchical Evocation Method** (Vergès, 1992) is a classical technique in Social Representations Theory (SRT) for mapping the internal structure of a social representation. In traditional elicitation studies, participants freely associate words in response to an inductor term, ranking each association by order of importance. The joint distribution of *average frequency of evocation* (AFE) and *average order of evocation* (AOE) is then used to partition the lexical space into four structurally distinct zones: the **central nucleus**, the **first periphery**, the **contrast zone**, and the **peripheral system**.

PyEvoc adapts this logic to large-scale online discourse, where explicit elicitation is unavailable and representational salience must instead be inferred from naturally occurring communicative behaviour. The framework operationalises:

- **AFE** → *representational diffusion*: the breadth of collective circulation of a lexical unit across distinct users.
- **AOE** → *discursive salience*: the prominence of a lexical unit within individual posts, derived from its textual position and rhetorical marking.

---

## 2. Corpus and Notation

Let $\mathcal{D} = \{d_1, d_2, \dots, d_N\}$ denote the collection of social media posts, and $\mathcal{U} = \{u_1, u_2, \dots, u_M\}$ the set of distinct users producing them. Each post $d_i$ is associated with:

- $x(d_i)$: textual content,
- $t(d_i)$: timestamp,
- $u(d_i)$: user identifier.

Let $\mathcal{V} = \{w_1, w_2, \dots, w_K\}$ denote the vocabulary of retained lexical units after preprocessing and linguistic annotation (tokenisation, lemmatisation, POS tagging, stopword removal).

---

## 3. Discursive Salience

Discursive salience approximates the cognitive prominence of a lexical unit within a post by combining two complementary components: **positional salience** and **structural salience**.

### 3.1 Positional Salience

Tokens appearing earlier in a post are assumed to reflect greater communicative foregrounding. For a token occurrence $w$ within a document $d$ containing $L_d$ tokens, positional salience is defined as:

$$
r_{\mathrm{pos}}(w,d)=
\begin{cases}
1, & L_d = 1, \\
1 - \dfrac{\mathrm{position}(w,d)-1}{L_d-1}, & L_d > 1.
\end{cases}
$$

This assigns a salience of 1 to the first token and decreases linearly to 0 for the last token.

### 3.2 Structural Salience

A second component captures rhetorical foregrounding via discourse-specific communicative markers. Let the following binary indicators be defined for token $w$ in document $d$:

| Indicator | Meaning |
|---|---|
| $I_{\mathrm{first}}(w,d)$ | $w$ appears in the opening sentence |
| $I_{\mathrm{emph}}(w,d)$ | $w$ appears within emphasised spans (e.g., bold, italic) |
| $I_{\mathrm{list}}(w,d)$ | $w$ appears inside a list or quotation |
| $I_{\mathrm{intens}}(w,d)$ | $w$ appears in typographically intensified segments (e.g., all-caps) |

Structural salience is then:

$$
r_{\mathrm{str}}(w,d) =
\eta_1 I_{\mathrm{first}}(w,d)
+
\eta_2 I_{\mathrm{emph}}(w,d)
+
\eta_3 I_{\mathrm{list}}(w,d)
+
\eta_4 I_{\mathrm{intens}}(w,d),
$$

where the weights $\eta_j \in [0,1]$ satisfy $\sum_{j=1}^{4} \eta_j = 1$ and are configurable by the user.

### 3.3 Composite Salience Score

For each lexical unit $w \in \mathcal{V}$, average positional and structural salience are computed over all its corpus occurrences $I(w)$:

$$
\bar{r}_{\mathrm{pos}}(w) = \frac{1}{|I(w)|} \sum_{(w,d)\in I(w)} r_{\mathrm{pos}}(w,d),
\qquad
\bar{r}_{\mathrm{str}}(w) = \frac{1}{|I(w)|} \sum_{(w,d)\in I(w)} r_{\mathrm{str}}(w,d).
$$

These are combined into a single **composite salience score**:

$$
S(w) = \pi\,\bar{r}_{\mathrm{pos}}(w) + (1-\pi)\,\bar{r}_{\mathrm{str}}(w), \qquad \pi \in [0,1],
$$

where $\pi$ is a user-configurable mixing parameter controlling the relative weight of positional versus structural information.

### 3.4 Rank Transformation

To preserve the interpretative logic of AOE — where lower ranks indicate greater salience — the composite score is transformed into a rank-like indicator:

$$
R(w) = 1 + \bigl(1 - S(w)\bigr)(R_{\max} - 1),
$$

where $R_{\max}$ is the maximum admissible rank value. Lower $R(w)$ corresponds to higher discursive salience.

---

## 4. Representational Diffusion

Diffusion measures how broadly a lexical unit circulates across the discursive community, operationalising the AFE at the collective rather than individual level.

### 4.1 User-Level Diffusion

The primary formulation privileges **user-level diffusion** to capture the breadth of collective sharing independently of corpus size or posting frequency:

$$
F_{\mathrm{user}}(w) = \frac{\bigl|\{u \in \mathcal{U} : w \in \mathcal{V}(u)\}\bigr|}{|\mathcal{U}|},
$$

where $\mathcal{V}(u)$ is the set of lexical units used at least once by user $u$. This measures the *proportion of distinct users* who employ a given term.

### 4.2 Comment-Level Diffusion

An alternative formulation computes diffusion at the post level:

$$
F_{\mathrm{comm}}(w) = \frac{\bigl|\{d \in \mathcal{D} : w \in d\}\bigr|}{|\mathcal{D}|}.
$$

This measures the proportion of posts containing the term. User-level diffusion is the default and recommended setting, as it is less sensitive to prolific individual users dominating the frequency signal.

---

## 5. Structural Thresholds: AFE and AOE

Following the logic of classical HEM, two corpus-level thresholds are computed to partition the lexical space.

The **Average Frequency of Evocation (AFE)** is defined as the mean diffusion score across the retained vocabulary:

$$
\mathrm{AFE} = \frac{1}{|\mathcal{V}|} \sum_{w \in \mathcal{V}} F_{\mathrm{user}}(w).
$$

The **Average Order of Evocation (AOE)** is the mean rank value across the retained vocabulary:

$$
\mathrm{AOE} = \frac{1}{|\mathcal{V}|} \sum_{w \in \mathcal{V}} R(w).
$$

In the empirical implementation, both thresholds are computed **separately within each POS category** to account for the distributional asymmetries between grammatical classes (see [Section 7](#7-pos-specific-thresholding)).

---

## 6. Four-Quadrant Representational Structure

The joint distribution of diffusion and salience defines a two-dimensional representational space. Using the AFE and AOE as structural thresholds, each lexical unit is assigned to one of four analytically distinct zones.

For each term $w$ belonging to POS category $p$, with POS-specific thresholds $\theta_F(p)$ (AFE) and $\theta_R(p)$ (AOE):

$$
\text{Quadrant}(w) =
\begin{cases}
\textbf{Central Nucleus}, 
 & F_{\mathrm{user}}(w)\ge \theta_F(p) \;\text{ and }\; R(w) \le \theta_R(p),\\[6pt]
\textbf{First Periphery}, 
 & F_{\mathrm{user}}(w)\ge \theta_F(p) \;\text{ and }\; R(w) > \theta_R(p),\\[6pt]
\textbf{Contrast Zone}, 
 & F_{\mathrm{user}}(w)< \theta_F(p) \;\text{ and }\; R(w) \le \theta_R(p),\\[6pt]
\textbf{Peripheral System},
 & F_{\mathrm{user}}(w)< \theta_F(p) \;\text{ and }\; R(w) > \theta_R(p).
\end{cases}
$$

The substantive interpretation of each zone is as follows:

| Zone | Diffusion | Salience | Interpretation |
|---|---|---|---|
| **Central Nucleus** | High | High | Stable, consensual core elements of the representation |
| **First Periphery** | High | Low | Widely shared but contextually flexible elements |
| **Contrast Zone** | Low | High | Salient minority positions or emerging framings |
| **Peripheral System** | Low | Low | Weakly structured, contextually variable elements |

---

## 7. POS-Specific Thresholding

A key methodological feature of PyEvoc is the application of **POS-specific thresholds**. Lexical units belonging to different grammatical categories fulfil distinct communicative and representational functions in online discourse and consequently exhibit different frequency and salience distributions. Conflating them under a single global threshold would introduce artefactual prominence effects driven by grammatical asymmetries rather than representational structure.

PyEvoc focuses on three primary POS categories:

- **Nouns** — support objectification processes by stabilising socially recognisable referents.
- **Adjectives** — encode evaluative and normative dimensions of the representation.
- **Emojis** — convey affective and interactional information in digital communicative contexts.

For each category $p$, the thresholds $\theta_F(p)$ and $\theta_R(p)$ are computed from the within-category distributions of $F_{\mathrm{user}}$ and $R$, respectively. This preserves the interpretability of the representational structure and ensures that quadrant assignment reflects genuine representational dynamics rather than artefacts of grammatical frequency.

---

## 8. Summary

The table below summarises the correspondence between classical HEM constructs and their computational counterparts in PyEvoc:

| Classical HEM | PyEvoc Operationalisation |
|---|---|
| Frequency of evocation | User-level diffusion $F_{\mathrm{user}}(w)$ |
| Order of evocation | Composite salience rank $R(w)$ |
| AFE threshold | Mean $F_{\mathrm{user}}$ within POS category |
| AOE threshold | Mean $R$ within POS category |
| Quadrant assignment | Joint thresholding on $F$ and $R$ |

The framework is designed for digital discourse environments where social representations emerge from fragmented, decentralised, and continuously evolving communicative exchanges. By integrating the SRT interpretative logic with scalable computational procedures, PyEvoc enables systematic exploration of how public understandings, symbolic framings, and evaluative orientations surrounding complex social objects are collectively constructed and negotiated in online environments.

---

## 9. References

- Moscovici, S. (1961). *La psychanalyse, son image et son public*. Presses Universitaires de France.
- Vergès, P. (1992). L'evocation de l'argent: Une méthode pour la définition du noyau central d'une représentation. *Bulletin de Psychologie*, 45(405), 203–209.
- Abric, J.-C. (1994). *Pratiques sociales et représentations*. Presses Universitaires de France.
