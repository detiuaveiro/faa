---
title: Small Project 1 - Tribal ML Benchmark & Population Optimization
---

# Small Project 1: Tribal Machine Learning & Population Optimization

## 1. Context & Objectives
Pedro Domingos' *The Master Algorithm* frames Machine Learning through five core tribes:
1. **Symbolists** (Rule induction, Decision Trees)
2. **Connectionists** (Multi-Layer Perceptrons, Neural Nets)
3. **Evolutionaries** (Genetic Algorithms, Differential Evolution)
4. **Bayesians** (Naive Bayes, Bayesian Networks)
5. **Analogizers** ($k$-NN, Support Vector Machines)

In this project, students will:
* Benchmark implementations from the 5 tribes on complex non-linear classification and regression tasks.
* Investigate the evolutionary tribe in depth: evaluate population-based optimization algorithms (Differential Evolution, Particle Swarm Optimization).
* Implement and evaluate modern population initialization methods:
  - Uniform Random Initialization
  - Opposition-Based Learning (OBL)
  - Empty Space Search (ESS)
  - **OBLESA** (Opposition-Based Learning with Empty Space Search)
* Perform knee detection (using the `knee` library) on performance trade-off curves.

## 2. Deliverables
1. **Source Code & Notebooks:** Clean, documented implementation conforming to `ruff` linting standards.
2. **Technical Report (PDF, max 6 pages):**
   - Empirical comparison of the 5 tribes.
   - Exploration vs exploitation analysis of population optimizers.
   - Benchmark comparing random init vs OBL vs ESS vs OBLESA across varying dimensions.
   - Analysis of knee points in hyperparameter sensitivity.

## 3. Important Dates & Submission
* **Release:** Week 3 (October 01-02, 2026)
* **Submission Deadline:** Class 8 (November 05-06, 2026)
* **Weight:** 25% of final course grade.
