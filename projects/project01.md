---
title: "Project 1 — The Five Classes of Learners"
---

# 1. Goal

Pedro Domingos (*The Master Algorithm*, 2015) groups machine learning into **five classes of learners**, each with its own idea of what learning is. You will pick **one learner from each class**, train the five on **one dataset**, and compare them with a methodology you define and defend. The question is not "which model won", but **why**: what each class assumes about the data, and what the plots and the numbers say about those assumptions.

Choose a task: **classification** or **regression**. Use the same task for all five learners.

# 2. The Five Classes of Learners

| Class | Learns by | Typical learners (scikit-learn, Keras, pyBlindOpt) | Taught |
|:-----------------|:-----------------|:---------------------------|:------|
| **Symbolists** | inducing rules by inverse deduction | decision tree, rule list, extra-trees; boosted trees only if you justify them as symbolic | Class 06--07 |
| **Connectionists** | adjusting the weights of a network by backpropagation | MLP (Keras with the JAX backend, or `MLPClassifier`/`MLPRegressor`), logistic regression as a one-neuron network | Class 03, 05 |
| **Evolutionaries** | evolving a population of candidate models | the *parameters*, the *features* or the *rules* of a model found by a genetic algorithm or differential evolution (`pyBlindOpt`), optimizing the metric directly | Class 02 |
| **Bayesians** | updating probabilities with Bayes' rule | Gaussian or categorical Naive Bayes, Naive Bayes regression, Bayesian ridge, a model fitted by MAP | Class 04 |
| **Analogizers** | comparing a new case with similar known cases | $k$-NN, support vector machine or SVR with a kernel | Class 05--06 |

Notes:

* One learner **per class**, five in total. Different settings of the same learner (two trees with different depths) count as one.
* Tuning a hyper-parameter with an evolutionary algorithm does **not** make a learner evolutionary: here the evolution must be the way the model is *learned*.
* A class may have borderline members (an ensemble, a regularized linear model). You may use one, but your README must say why it belongs to the class.
* Some classes are taught after the release (the last ones in Class 06--07): start with the ones you know.

# 3. What To Do

Follow these steps, **in this order**; they map to the sections of your README (Section 5).

1. **Choose the dataset** (Section 4): the task, the target, what a good model is worth in the real problem.
2. **Data visualization.** Look at the data before any model: the target (distribution, classes balance), the features (distributions, scales, outliers, missing values), the relations (correlations, the features against the target). Every plot must lead to a decision.
3. **Evaluation methodology.** Define how you will compare the learners: the split (hold-out, stratified, $k$-fold, repeated), the metrics and why they match the problem, the baseline (the mean, the majority class), and how you quantify the spread so that a difference is not noise.
4. **Preprocessing.** Define what the data need (missing values, categorical encoding, scaling, transformations). Every statistic is learned on the training part only, **inside** each fold; each learner may need a different preprocessing, and you explain why.
5. **Models.** Fit and tune each learner on validation data only. Report the search space and the tuning cost, not only the best result.
6. **Evaluation and conclusion.** Compare the five learners under the same methodology; analyse errors and costs, and relate the outcome to what the data looked like in step 2.

Constraints:

* Use the course stack: **polars** for the data, **numpy** and **jax** for computation, **matplotlib** and **seaborn** for plots, **scikit-learn** and **keras** for the models, **pyBlindOpt** for the evolutionary learner.
* Everything must run in a few minutes on a laptop (the datasets below do). Fix every random seed.
* A test set (or the outer folds) is used **once**, at the end, for the final numbers.

# 4. Datasets

Use one of the suggestions, which are real, small and already in the repository (`datasets/`, as `.csv.zst` files that polars reads directly; sources and columns in `datasets/README.md`), or propose your own. A dataset of your own must be real, public, run on a laptop and be agreed beforehand with your instructor.

| Dataset | Task | Size | What makes it interesting |
|:--------------|:----------------|:----------|:-----------------------------|
| `breast_cancer` | classification | 569 × 30 | clean and easy: a ceiling effect, strongly correlated features, the cost of a missed tumour |
| `german_credit` | classification | 1000 × 20 | categorical and numeric features; wrong decisions have different costs (5 : 1) |
| `heart_disease` | classification | 303 × 13 | very small, categorical, **missing values**: the variance of the estimate is the problem |
| `spambase` | classification | 4601 × 57 | very skewed features, many zeros, duplicates, a threshold to choose |
| `digits` | classification, 10 classes | 1797 × 64 | multi-class; images as vectors; $k$-NN and SVM shine |
| `california_housing` | regression | 20640 × 8 | geographic structure, a capped target, skew (class 03); heavy for $k$-NN and SVR: subsample and say so |
| `concrete` | regression | 1030 × 8 | a smooth non-linear function with interactions |
| `abalone` | regression | 4177 × 8 | a categorical feature and a noisy target |
| `auto_mpg` | regression | 398 × 7 | tiny, with missing values, correlated features |
| `wine_quality` | regression (or classification) | 6497 × 11 | an integer target on a few levels, unbalanced |
| `diabetes` | regression | 442 × 10 | tiny and noisy; low ceiling for every learner |

Justify your choice: why this dataset, what the five classes are *expected* to do on it (a hypothesis you can confirm or reject), and what its limits are.

# 5. Submission

A repository (or a zip) with:

1. **The code**: a Jupyter notebook **or** Python files, runnable from top to bottom with the course environment (`make venv`), reading the dataset from `datasets/`. Keep the notebook clean: no unused cells, outputs allowed.
2. **`README.md`** with *all* the information, with these sections:

| Section | Content |
|:------------|:--------------------------------------|
| **1. Identification** | names and student numbers, course, date, the dataset and the task in one line |
| **2. Models** | the five learners, one per class: what it is, why you chose it for this dataset, its hyper-parameters and search space |
| **3. Dataset** | the choice and its justification, the source, the size and features; the data visualization with its findings |
| **4. Methodology** | the evaluation methodology (split, folds, metrics, baseline, spread) and the preprocessing (steps, and where they are fitted); the tuning protocol; how the pipeline avoids leakage |
| **5. Evaluation and conclusion** | the results table (mean and spread, with the baseline), the plots, the analysis of errors, the answer to your hypotheses, the limits, and what you would do next |

**Plots and diagrams are mandatory**, at least:

* a **diagram** of the pipeline (data, visualization, preprocessing, folds, tuning, test) with the five learners;
* the **data visualization** of Section 3 (target, features, relations);
* a **comparison plot** of the five learners with their spread (not only means);
* a **diagnostic plot** for the best learner (confusion matrix and threshold curve, or predicted against actual and residuals);
* a **learning or validation curve** for at least one learner.

Every figure has a caption and is discussed in the text. Save the figures in the repository and link them from the `README.md`.

**Assessment.** We look at: the quality of the justifications; the correctness of the methodology (no leakage, a baseline, a spread); the quality and the use of the plots; the five classes represented honestly; the conclusions supported by the numbers; reproducibility.

# 6. Dates

* **Release:** Class 03 (01/02 October 2026)
* **Checkpoint:** Class 06 (22/23 October 2026): dataset, data visualization and methodology
* **Submission deadline:** Class 08 (05/06 November 2026)
* **Weight:** 25% of the final grade
