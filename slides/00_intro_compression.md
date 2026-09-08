---
title: Fundamentos de Aprendizagem Automática
subtitle: Learning as Compression & Information Foundations
---

# Course Introduction

## Welcome to FAA
* **Curricular Unit:** Fundamentos de Aprendizagem Automática (41010)
* **Target:** Master's / Advanced Undergraduate in Informatics & Computer Engineering
* **Structure:** 3 hours per week (1h Theory/Presentation + 2h Hands-on Practice)
* **Instructor:** Mário Antunes (`mario.antunes@ua.pt`)
* **Philosophy:** From first principles to modern frontiers:
  - Foundations grounded in Information Theory and Compression
  - Comprehensive perspective across the learning paradigms
  - Hands-on mastery with Python and modern ML ecosystems

## Evaluation Scheme
* **Discrete Evaluation Mode:**
  - **Theoretical Component (50%):**
    - Mid-Term Test: **25%** (Class 8, Week of 05-06 Nov 2026)
    - Final Test: **25%** (Class 14, Week of 17-18 Dec 2026)
  - **Practical Component (50%):**
    - Small Project 1: **25%** (Due Class 8)
    - Small Project 2: **25%** (Due Class 14)
* **Final Evaluation Mode:**
  - Final Exam: **50%**
  - Comprehensive Project: **50%**

# What is Learning?

## The Core Question
* What does it mean for a machine to *learn*?
* Classic definition (Tom Mitchell, 1997):
  > "A computer program is said to learn from experience $E$ with respect to some class of tasks $T$ and performance measure $P$, if its performance at tasks in $T$, as measured by $P$, improves with experience $E$."
* Is learning just memorization?
  - A lookup table has perfect performance on seen data ($E$).
  - Does it generalize to unseen instances?

## Generalization vs Memorization
* True learning requires **generalization**: identifying underlying regularities.
* The dilemma:
  - Complex models can memorize noise (**overfitting**).
  - Simple models may fail to capture true patterns (**underfitting**).
* How do we formalize the preference for simplicity?
  - Enter **Information Theory** and **Algorithmic Information Theory**.

# Learning as Compression

## The Compression View of Learning
* Fundamental thesis: **Learning is compression**.
* If a dataset can be compressed into a concise algorithm or model, regularities have been discovered.
* Random data (pure noise) cannot be compressed.
* If a hypothesis $H$ explains dataset $D$ with fewer bits than raw $D$, then $H$ captures the regularities of the data distribution.

## Kolmogorov Complexity
* The **Kolmogorov Complexity** $K(x)$ of a string $x$ is the length of the shortest computer program that outputs $x$:
  $$K(x) = \min_{p} \{ |p| : U(p) = x \}$$
* $K(x)$ is non-computable, but serves as the theoretical ideal for compression and induction.
* Solomonoff Induction: Prediction by weighting all computable hypotheses inversely to their program lengths ($2^{-|p|}$).

## Minimum Description Length (MDL)
* Practical computable formulation proposed by Jorma Rissanen (1978).
* To explain dataset $D$ with hypothesis $H$, total code length is:
  $$L(D, H) = L(H) + L(D \mid H)$$
* Where:
  - $L(H)$: Length in bits needed to describe the model / hypothesis (Model Complexity).
  - $L(D \mid H)$: Length in bits needed to describe the data given the model (Residual / Error).
* Minimizing $L(D, H)$ automatically balances goodness of fit with model complexity (**Occam's Razor**).

## Connection to Modern Machine Learning
* Regularized loss in modern ML:
  $$\mathcal{L}_{total}(\theta) = \underbrace{\mathcal{L}_{data}(D; \theta)}_{L(D \mid H)} + \lambda \underbrace{\Omega(\theta)}_{L(H)}$$
* Negative log-likelihood is Shannon code length:
  $$-\log_2 P(D \mid \theta) \equiv L(D \mid H)$$
* Parameter regularization ($L_1, L_2$) corresponds to prior description length of weights.

# Practical Session Preview

## Today's Lab
* Hands-on exploration in `practice/01_compression_mdl.md`:
  1. Measuring entropy of sequences and texts.
  2. Polynomial regression under varying degrees.
  3. Measuring model complexity vs residual error (BIC, AIC, MDL).
  4. Visualizing how compression prevents overfitting.
