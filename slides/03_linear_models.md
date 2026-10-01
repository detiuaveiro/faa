---
title: Fundamentos de Aprendizagem Automática
subtitle: "Class 03 — Linear Models"
---

# From Optimization to Models

## Where We Left Off

```{=latex}
\begin{center}
\begin{tikzpicture}
  \node[fillbox=cblue, text width=2.4cm] (m) {\textbf{model} $f_\theta$\\ a family of functions};
  \node[fillbox=cred, text width=2.4cm, right=14mm of m] (l) {\textbf{loss} $\mathcal{L}$\\ what "wrong" costs};
  \node[fillbox=corange, text width=2.4cm, right=14mm of l] (o) {\textbf{optimizer}\\ finds $\theta^\star$};
  \draw[flow] (m) -- node[above, note] {scored by} (l);
  \draw[flow] (o) -- node[above, note] {minimizes} (l);
\end{tikzpicture}
\end{center}
```

* Class 01: a good model **compresses** the data; we measure it on **unseen** data
* Class 02: training is $\theta^\star = \arg\min_\theta\, \mathcal{L}_{data}(\theta) + \lambda\,\Omega(\theta)$
* Today: three classic models, each one *a choice of $f_\theta$ and $\mathcal{L}$*

## Three Models, One Recipe

```{=latex}
\renewcommand{\arraystretch}{1.4}
```

| Model | Predicts | Loss | Fit by |
|:----------------|:------------|:----------------------|:------------------|
| Linear regression | a number | squared error | closed form or GD |
| Perceptron | a class | $\max(0, -y\,f(x))$ | online update |
| Logistic regression | a probability | cross-entropy | GD (no closed form) |

* All three are trained with **`jax.grad`** in the lab: we write the loss, JAX gives the gradient
* Polynomial features and regularization can be added to any of them
* Every one is compared with its **scikit-learn** twin: same split, same scaling, a printed difference

## Two Datasets

:::: {.columns}
::: {.column width="50%"}
**Spam detection** (classification)

* Spambase: **4601** e-mails, **57** features
* Word and character frequencies, capital runs
* **39.4%** spam
* Most values are 0, a few huge
* 571 duplicate rows
:::
::: {.column width="50%"}
**House prices** (regression)

* California housing: **20640** districts, **8** features
* Income, age, rooms, occupancy, latitude, longitude
* Target: median value (100 000 USD)
* 992 districts capped at 5.0
* Up to 1243 people per house
:::
::::

* Real, noisy, small: every demo runs in seconds (`datasets/*.csv.zst`, read with polars)

## Project 1 Is Released

* **The five classes of learners:** symbolists, connectionists, evolutionaries, Bayesians, analogizers
* Pick **one dataset** (suggestions in `datasets/`, classification or regression) and **one learner per class**; justify every choice
* Today's tools are what the report is judged on: data visualization, evaluation methodology, preprocessing, the right metrics
* Submission: the code and a `README.md` with identification, models, dataset, methodology, evaluation and conclusion, **with plots and diagrams**
* Specification: `projects/project01.pdf`; due at Class 08

# Data, Methodology and Preprocessing

## Three Steps Before Any Model

```{=latex}
\begin{center}
\begin{tikzpicture}[node distance=5mm]
  \node[fillbox=cblue, text width=2.3cm] (v) {\textbf{1. Data}\\ \textbf{visualization}\\ look first};
  \node[fillbox=corange, text width=2.3cm, right=of v] (m) {\textbf{2. Evaluation}\\ \textbf{methodology}\\ split, metric};
  \node[fillbox=cgreen, text width=2.3cm, right=of m] (p) {\textbf{3.}\\ \textbf{Preprocessing}\\ training only};
  \node[fillbox=cred, text width=1.1cm, right=of p] (mo) {\textbf{models}\\ fit, tune};
  \draw[flow] (v) -- (m); \draw[flow] (m) -- (p); \draw[flow] (p) -- (mo);
\end{tikzpicture}
\end{center}
```

* **Data visualization:** the target, the features, their relations; every plot ends in a **decision** (a logarithm, a metric, a split)
* **Evaluation methodology:** fixed *before* the results: the split, the metric that matches the cost of errors, a baseline, the spread
* **Preprocessing:** whatever is learned from data (means, vocabularies, imputed values) is fitted on the **training part only**
* The same three steps for **both** datasets, in this order

## What Does the Target Look Like?

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, ybar, width=0.62\textwidth, height=3.1cm, bar width=14pt, xmin=0, xmax=5.7, ymin=0, ymax=4800, xtick={0,1,2,3,4,5}, ytick={0,2000,4000}, xlabel={median house value (100 000 USD)}, ylabel={districts}]
  \addplot[fill=cblue!55, draw=cblue] coordinates {(0.25,199) (0.75,3397) (1.25,3960) (1.75,4329) (2.25,2926) (2.75,1963) (3.25,1214) (3.75,881) (4.25,477) (4.75,302)};
  \addplot[fill=cred!70, draw=cred] coordinates {(5.25,992)};
  \node[font=\scriptsize, cred, anchor=south east, align=center] at (axis cs:5.7,1400) {capped:\\ 992};
\end{axis}
\end{tikzpicture}
\end{center}
```

* A **histogram** counts the examples per range of values: the *shape* of what we predict
* A long tail lets a few rows dominate the squared error; a pile-up means **censored** data
* Here: right-skewed, with a **wall at 5.0** (992 districts, 4.8%): the price was capped, no model can see above it

## What Does One Feature Look Like?

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, ybar, width=0.62\textwidth, height=3.1cm, bar width=14pt, xmin=0, xmax=6.6, ymin=0, ymax=12500, xtick={0,1,2,3,4,5,6}, xticklabels={0,1,2,3,4,5,6+}, ytick={0,5000,10000}, xlabel={people per house (\texttt{AveOccup})}, ylabel={districts}]
  \addplot[fill=cblue!55, draw=cblue] coordinates {(0.25,0) (0.75,3) (1.25,149) (1.75,1481) (2.25,4403) (2.75,6581) (3.25,4319) (3.75,1954) (4.25,978) (4.75,426) (5.25,161) (5.75,70) (6.25,115)};
\end{axis}
\end{tikzpicture}
\end{center}
```

* The same histogram, now for an **input**: the average number of people per house
* It reveals scale and **outliers**: least squares, distances and penalties all react to a few extreme rows
* Here: the median is 2.8 and 92% of the districts are below 4, but the maximum is **1243**, 441 times the median

## Does a Feature Predict the Target?

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, width=0.62\textwidth, height=3.1cm, only marks, mark=*, mark size=0.6pt, mark options={fill=cblue, draw=cblue, opacity=0.5}, xmin=0, xmax=12, ymin=0, ymax=5.4, xlabel={median income (10 000 USD)}, ylabel={median value}]
  \addplot coordinates {(2.82,1.83) (4.36,4.69) (2.80,0.88) (3.97,1.12) (6.09,2.98) (3.67,1.38) (2.55,0.96) (2.38,1.28) (5.74,3.09) (4.76,1.91) (3.81,1.39) (1.64,0.60) (3.78,1.84) (1.31,1.36) (2.50,0.89) (2.10,1.88) (3.80,2.05) (1.62,0.79) (3.75,1.62) (4.07,2.12) (3.94,1.43) (3.96,3.46) (2.66,1.75) (1.81,1.17) (4.74,2.63) (1.88,0.66) (3.10,1.40) (6.75,2.04) (2.87,2.45) (4.18,4.87) (6.26,3.07) (4.28,2.38) (3.14,1.37) (6.58,3.49) (4.61,1.84) (3.61,1.15) (3.16,1.30) (4.76,1.38) (2.82,1.10) (4.80,2.62) (3.96,1.61) (4.89,1.78) (3.35,1.80) (1.83,2.25) (2.25,1.15) (2.18,1.62) (4.41,2.82) (3.64,3.28) (4.86,2.66) (1.96,0.75) (3.18,1.26) (2.76,0.72) (2.85,0.41) (2.73,1.49) (3.36,4.20) (8.54,4.51) (6.07,3.12) (4.32,1.30) (5.84,2.17) (2.95,1.49) (5.59,2.09) (6.58,2.45) (1.92,0.94) (3.98,1.89) (4.60,2.42) (5.87,4.04) (2.52,3.50) (5.02,3.24) (5.21,2.02) (4.40,1.52) (3.94,2.13) (1.76,2.06) (2.37,2.19) (9.07,3.67) (7.52,2.92) (6.68,4.11) (5.58,2.18) (3.47,1.39) (5.62,3.17) (3.18,2.50) (2.55,0.92) (6.31,2.59) (1.45,0.71) (3.14,0.75) (3.54,0.87) (5.57,3.43) (2.02,1.88) (5.07,2.34) (4.79,2.91) (4.73,1.65) (6.81,3.28) (1.56,0.82) (2.28,0.79) (4.75,2.96) (3.58,1.10) (3.17,1.10) (2.63,0.66) (3.98,2.67) (3.64,1.47) (2.93,2.76) (15.00,5.00) (5.13,3.46) (2.22,1.81) (5.02,1.65) (6.58,4.08) (4.22,1.76) (4.24,1.57) (4.98,2.09) (2.82,1.68) (2.45,1.53) (2.81,0.95) (4.04,1.33) (3.16,1.38) (3.24,1.18) (4.23,1.67) (5.63,2.73) (4.58,3.27) (2.39,0.82) (3.83,1.49) (3.23,2.19) (3.35,4.93) (2.36,1.28) (1.81,0.59) (1.99,2.25) (5.34,2.70) (5.96,3.51) (3.85,3.19) (6.50,4.84) (2.69,1.96) (3.29,2.32) (3.39,0.82) (3.55,2.71) (2.36,1.98) (3.25,2.31) (3.58,0.93) (3.03,2.19) (3.46,2.17) (6.18,2.30) (1.83,1.38) (2.67,1.53) (2.56,0.77) (6.54,2.73) (3.23,1.62) (8.19,5.00) (6.47,3.48) (3.26,1.24) (2.32,1.59) (1.76,0.55) (1.93,1.12) (4.78,1.01) (4.38,3.41) (8.77,3.66) (3.52,2.65) (3.74,3.20) (2.41,2.14) (3.94,1.37) (4.43,1.58) (2.60,1.22) (3.24,2.12) (5.22,2.14) (2.94,3.04) (2.29,2.34) (3.18,0.85) (6.03,2.89) (4.18,3.64) (2.64,2.41) (2.95,1.28) (3.25,3.55) (2.88,1.97) (3.62,1.56) (1.90,0.93) (2.79,1.18) (7.38,1.72) (5.94,4.22) (2.46,1.57) (4.55,2.81) (2.28,2.47) (4.37,2.26) (2.34,1.71) (1.53,0.51) (3.87,2.32) (1.37,0.52) (4.72,2.90) (2.28,0.59) (2.79,1.11) (5.03,2.11) (4.39,2.31) (5.13,3.42) (4.51,1.13) (4.11,1.16) (4.12,2.33) (1.99,2.62) (0.96,0.60) (2.81,0.83) (15.00,5.00) (2.62,1.49) (8.08,3.81) (1.63,1.62) (1.71,1.08) (6.33,5.00) (7.64,5.00) (0.71,1.38) (3.24,2.53) (1.62,0.47) (4.06,1.42) (3.72,2.30) (4.05,2.86) (1.93,3.00) (3.53,1.52) (3.46,1.45) (3.37,0.98) (4.61,1.39) (8.74,5.00) (2.71,1.66) (5.34,1.61) (3.07,4.12) (3.09,1.10) (8.27,5.00) (10.40,5.00) (1.71,0.52) (2.32,0.92) (2.97,1.53) (1.26,0.85) (6.80,3.25) (7.13,2.79) (6.59,3.29) (5.30,2.55) (5.48,2.09) (5.04,2.24) (2.42,1.93) (4.67,2.23) (3.40,2.16) (4.70,1.74) (2.31,1.54) (4.56,1.27) (2.95,0.83) (2.75,0.97) (5.25,1.61) (8.27,3.74) (4.89,3.34) (6.42,4.47) (1.62,0.50) (2.27,0.89) (3.75,0.87) (3.41,0.83) (3.73,2.61) (2.67,2.59) (2.23,2.38) (6.78,5.00) (3.43,4.57) (3.50,2.32) (2.19,1.44) (3.70,1.78) (2.62,1.01) (2.61,0.94) (4.12,0.88) (2.23,0.38) (3.94,1.71) (3.70,1.75) (3.97,3.00) (3.21,1.05) (3.28,1.95) (5.21,2.41) (4.02,0.88) (1.95,0.90) (7.07,3.36) (0.50,0.57) (2.44,0.79) (1.69,0.88) (2.02,0.58) (4.12,1.60) (3.17,1.25) (3.54,0.71) (3.35,2.21) (3.36,1.94) (4.64,2.17) (6.37,2.42) (3.76,1.84) (5.43,2.23) (2.83,2.57) (2.63,1.71) (8.04,3.82) (3.14,3.50) (4.52,4.46) (7.41,3.74) (4.48,3.41) (4.29,1.85) (4.06,1.60) (3.20,1.80) (6.10,3.44) (4.18,3.50) (2.16,0.86) (2.48,1.36) (3.52,1.15) (5.92,2.39) (1.62,1.30) (2.52,1.19) (2.36,1.12) (5.28,3.33) (5.98,2.45) (6.45,2.71) (5.48,2.96) (2.38,1.22) (3.35,2.94) (7.92,5.00) (2.84,1.82) (4.98,1.58) (2.33,5.00) (3.81,1.61) (5.62,2.72) (2.61,0.70) (3.68,1.36) (2.58,1.33) (9.15,4.19) (2.06,1.20) (2.53,1.23) (5.29,3.56) (6.83,3.84) (1.44,1.03) (15.00,1.31) (3.22,1.26) (5.92,2.37) (3.29,1.29) (1.04,0.62) (3.03,1.04) (3.35,2.20) (8.15,5.00) (4.05,1.82) (4.74,4.19) (5.13,1.82) (3.42,1.38) (0.94,1.12) (3.08,3.38) (5.37,2.61) (5.19,1.54) (5.74,3.98) (2.53,0.62) (5.29,2.65) (1.60,1.01) (2.72,0.85) (4.14,2.12) (1.40,1.05) (1.84,0.63) (8.96,4.00) (6.64,2.96) (2.31,0.62) (1.71,1.07) (2.97,1.63) (3.72,3.33) (2.78,1.29) (5.05,3.95) (4.49,2.19) (1.60,0.81) (4.74,2.17) (4.66,1.59) (2.88,1.12) (1.60,1.69) (3.60,4.64) (7.62,5.00) (4.11,2.80) (5.08,2.19) (3.38,4.00) (3.58,1.78) (5.58,2.35) (4.58,2.12) (2.93,2.58) (4.99,2.62) (2.02,0.90) (6.16,2.78) (5.59,1.68) (2.44,1.88) (4.02,1.39) (1.99,1.65) (5.63,2.31) (4.95,5.00) (4.90,1.51) (2.31,1.39) (5.50,2.80) (5.29,4.55) (3.33,2.71) (4.42,2.15) (2.56,1.29) (3.69,1.11) (3.86,1.23) (0.86,0.25) (5.00,2.83) (4.86,2.03) (12.44,5.00) (2.81,1.58) (5.40,5.00) (3.60,1.33) (1.55,0.54) (2.18,1.02) (3.59,1.78) (6.19,2.67) (4.80,1.33) (2.96,2.04) (2.12,1.75) (5.41,2.66) (3.39,4.71) (3.36,1.03) (2.29,5.00) (2.54,1.09) (4.10,2.12) (2.72,0.70) (2.65,1.25) (1.75,0.68) (5.09,2.62) (5.62,2.75) (3.99,5.00) (5.90,3.51) (2.73,1.63) (3.96,2.06) (7.34,3.67) (4.60,2.21) (1.73,2.88) (3.20,1.72) (5.17,2.63) (3.98,1.62) (4.93,1.80) (4.40,2.27) (3.30,2.23) (2.58,1.60) (4.41,2.39) (2.60,1.73) (3.50,2.17) (1.51,0.88) (5.09,1.48) (4.06,2.75) (3.58,2.46) (2.80,1.52) (5.36,2.36) (2.95,1.85) (5.02,1.90) (6.68,2.75) (3.00,0.71) (3.33,1.23) (8.16,3.48) (2.72,0.65) (2.94,1.16) (4.53,3.07) (3.92,1.54) (1.62,0.85) (6.61,3.12) (6.07,2.47) (4.56,1.84) (2.66,1.80) (5.55,2.08) (2.44,1.62) (2.86,2.06)};
\end{axis}
\end{tikzpicture}
\end{center}
```

* A **scatter plot** draws one point per example: the *relation* of a feature with the target
* It tells whether a **line is plausible**, how noisy the relation is, and where it breaks
* Here (450 districts): the price rises with income ($r = 0.69$), the cloud **fans out**, and the cap shows as a flat row at 5.0

## Which Features Repeat Each Other?

```{=latex}
\begin{center}
\begin{tikzpicture}[x=0.6cm,y=0.33cm]
\fill[cred!100!white] (0,0) rectangle (1,-1); \node[font=\fontsize{5}{5}\selectfont] at (0.5,-0.5) {1.00};
\fill[cblue!12!white] (1,0) rectangle (2,-1); \node[font=\fontsize{5}{5}\selectfont] at (1.5,-0.5) {-0.12};
\fill[cred!55!white] (2,0) rectangle (3,-1); \node[font=\fontsize{5}{5}\selectfont] at (2.5,-0.5) {0.55};
\fill[cblue!13!white] (3,0) rectangle (4,-1); \node[font=\fontsize{5}{5}\selectfont] at (3.5,-0.5) {-0.13};
\fill[cblue!0!white] (4,0) rectangle (5,-1); \node[font=\fontsize{5}{5}\selectfont] at (4.5,-0.5) {-0.00};
\fill[cblue!2!white] (5,0) rectangle (6,-1); \node[font=\fontsize{5}{5}\selectfont] at (5.5,-0.5) {-0.02};
\fill[cblue!8!white] (6,0) rectangle (7,-1); \node[font=\fontsize{5}{5}\selectfont] at (6.5,-0.5) {-0.08};
\fill[cblue!2!white] (7,0) rectangle (8,-1); \node[font=\fontsize{5}{5}\selectfont] at (7.5,-0.5) {-0.02};
\fill[cred!69!white] (8,0) rectangle (9,-1); \node[font=\fontsize{5}{5}\selectfont] at (8.5,-0.5) {0.69};
\fill[cblue!12!white] (0,-1) rectangle (1,-2); \node[font=\fontsize{5}{5}\selectfont] at (0.5,-1.5) {-0.12};
\fill[cred!100!white] (1,-1) rectangle (2,-2); \node[font=\fontsize{5}{5}\selectfont] at (1.5,-1.5) {1.00};
\fill[cblue!22!white] (2,-1) rectangle (3,-2); \node[font=\fontsize{5}{5}\selectfont] at (2.5,-1.5) {-0.22};
\fill[cblue!13!white] (3,-1) rectangle (4,-2); \node[font=\fontsize{5}{5}\selectfont] at (3.5,-1.5) {-0.13};
\fill[cblue!24!white] (4,-1) rectangle (5,-2); \node[font=\fontsize{5}{5}\selectfont] at (4.5,-1.5) {-0.24};
\fill[cblue!1!white] (5,-1) rectangle (6,-2); \node[font=\fontsize{5}{5}\selectfont] at (5.5,-1.5) {-0.01};
\fill[cred!1!white] (6,-1) rectangle (7,-2); \node[font=\fontsize{5}{5}\selectfont] at (6.5,-1.5) {0.01};
\fill[cblue!11!white] (7,-1) rectangle (8,-2); \node[font=\fontsize{5}{5}\selectfont] at (7.5,-1.5) {-0.11};
\fill[cred!11!white] (8,-1) rectangle (9,-2); \node[font=\fontsize{5}{5}\selectfont] at (8.5,-1.5) {0.11};
\fill[cred!55!white] (0,-2) rectangle (1,-3); \node[font=\fontsize{5}{5}\selectfont] at (0.5,-2.5) {0.55};
\fill[cblue!22!white] (1,-2) rectangle (2,-3); \node[font=\fontsize{5}{5}\selectfont] at (1.5,-2.5) {-0.22};
\fill[cred!100!white] (2,-2) rectangle (3,-3); \node[font=\fontsize{5}{5}\selectfont] at (2.5,-2.5) {1.00};
\fill[cred!47!white] (3,-2) rectangle (4,-3); \node[font=\fontsize{5}{5}\selectfont] at (3.5,-2.5) {0.47};
\fill[cblue!12!white] (4,-2) rectangle (5,-3); \node[font=\fontsize{5}{5}\selectfont] at (4.5,-2.5) {-0.12};
\fill[cblue!1!white] (5,-2) rectangle (6,-3); \node[font=\fontsize{5}{5}\selectfont] at (5.5,-2.5) {-0.01};
\fill[cred!15!white] (6,-2) rectangle (7,-3); \node[font=\fontsize{5}{5}\selectfont] at (6.5,-2.5) {0.15};
\fill[cblue!7!white] (7,-2) rectangle (8,-3); \node[font=\fontsize{5}{5}\selectfont] at (7.5,-2.5) {-0.07};
\fill[cred!24!white] (8,-2) rectangle (9,-3); \node[font=\fontsize{5}{5}\selectfont] at (8.5,-2.5) {0.24};
\fill[cblue!13!white] (0,-3) rectangle (1,-4); \node[font=\fontsize{5}{5}\selectfont] at (0.5,-3.5) {-0.13};
\fill[cblue!13!white] (1,-3) rectangle (2,-4); \node[font=\fontsize{5}{5}\selectfont] at (1.5,-3.5) {-0.13};
\fill[cred!47!white] (2,-3) rectangle (3,-4); \node[font=\fontsize{5}{5}\selectfont] at (2.5,-3.5) {0.47};
\fill[cred!100!white] (3,-3) rectangle (4,-4); \node[font=\fontsize{5}{5}\selectfont] at (3.5,-3.5) {1.00};
\fill[cblue!15!white] (4,-3) rectangle (5,-4); \node[font=\fontsize{5}{5}\selectfont] at (4.5,-3.5) {-0.15};
\fill[cblue!11!white] (5,-3) rectangle (6,-4); \node[font=\fontsize{5}{5}\selectfont] at (5.5,-3.5) {-0.11};
\fill[cred!9!white] (6,-3) rectangle (7,-4); \node[font=\fontsize{5}{5}\selectfont] at (6.5,-3.5) {0.09};
\fill[cred!2!white] (7,-3) rectangle (8,-4); \node[font=\fontsize{5}{5}\selectfont] at (7.5,-3.5) {0.02};
\fill[cblue!9!white] (8,-3) rectangle (9,-4); \node[font=\fontsize{5}{5}\selectfont] at (8.5,-3.5) {-0.09};
\fill[cblue!0!white] (0,-4) rectangle (1,-5); \node[font=\fontsize{5}{5}\selectfont] at (0.5,-4.5) {-0.00};
\fill[cblue!24!white] (1,-4) rectangle (2,-5); \node[font=\fontsize{5}{5}\selectfont] at (1.5,-4.5) {-0.24};
\fill[cblue!12!white] (2,-4) rectangle (3,-5); \node[font=\fontsize{5}{5}\selectfont] at (2.5,-4.5) {-0.12};
\fill[cblue!15!white] (3,-4) rectangle (4,-5); \node[font=\fontsize{5}{5}\selectfont] at (3.5,-4.5) {-0.15};
\fill[cred!100!white] (4,-4) rectangle (5,-5); \node[font=\fontsize{5}{5}\selectfont] at (4.5,-4.5) {1.00};
\fill[cred!20!white] (5,-4) rectangle (6,-5); \node[font=\fontsize{5}{5}\selectfont] at (5.5,-4.5) {0.20};
\fill[cblue!14!white] (6,-4) rectangle (7,-5); \node[font=\fontsize{5}{5}\selectfont] at (6.5,-4.5) {-0.14};
\fill[cred!11!white] (7,-4) rectangle (8,-5); \node[font=\fontsize{5}{5}\selectfont] at (7.5,-4.5) {0.11};
\fill[cblue!2!white] (8,-4) rectangle (9,-5); \node[font=\fontsize{5}{5}\selectfont] at (8.5,-4.5) {-0.02};
\fill[cblue!2!white] (0,-5) rectangle (1,-6); \node[font=\fontsize{5}{5}\selectfont] at (0.5,-5.5) {-0.02};
\fill[cblue!1!white] (1,-5) rectangle (2,-6); \node[font=\fontsize{5}{5}\selectfont] at (1.5,-5.5) {-0.01};
\fill[cblue!1!white] (2,-5) rectangle (3,-6); \node[font=\fontsize{5}{5}\selectfont] at (2.5,-5.5) {-0.01};
\fill[cblue!11!white] (3,-5) rectangle (4,-6); \node[font=\fontsize{5}{5}\selectfont] at (3.5,-5.5) {-0.11};
\fill[cred!20!white] (4,-5) rectangle (5,-6); \node[font=\fontsize{5}{5}\selectfont] at (4.5,-5.5) {0.20};
\fill[cred!100!white] (5,-5) rectangle (6,-6); \node[font=\fontsize{5}{5}\selectfont] at (5.5,-5.5) {1.00};
\fill[cblue!13!white] (6,-5) rectangle (7,-6); \node[font=\fontsize{5}{5}\selectfont] at (6.5,-5.5) {-0.13};
\fill[cred!15!white] (7,-5) rectangle (8,-6); \node[font=\fontsize{5}{5}\selectfont] at (7.5,-5.5) {0.15};
\fill[cblue!26!white] (8,-5) rectangle (9,-6); \node[font=\fontsize{5}{5}\selectfont] at (8.5,-5.5) {-0.26};
\fill[cblue!8!white] (0,-6) rectangle (1,-7); \node[font=\fontsize{5}{5}\selectfont] at (0.5,-6.5) {-0.08};
\fill[cred!1!white] (1,-6) rectangle (2,-7); \node[font=\fontsize{5}{5}\selectfont] at (1.5,-6.5) {0.01};
\fill[cred!15!white] (2,-6) rectangle (3,-7); \node[font=\fontsize{5}{5}\selectfont] at (2.5,-6.5) {0.15};
\fill[cred!9!white] (3,-6) rectangle (4,-7); \node[font=\fontsize{5}{5}\selectfont] at (3.5,-6.5) {0.09};
\fill[cblue!14!white] (4,-6) rectangle (5,-7); \node[font=\fontsize{5}{5}\selectfont] at (4.5,-6.5) {-0.14};
\fill[cblue!13!white] (5,-6) rectangle (6,-7); \node[font=\fontsize{5}{5}\selectfont] at (5.5,-6.5) {-0.13};
\fill[cred!100!white] (6,-6) rectangle (7,-7); \node[font=\fontsize{5}{5}\selectfont] at (6.5,-6.5) {1.00};
\fill[cblue!92!white] (7,-6) rectangle (8,-7); \node[font=\fontsize{5}{5}\selectfont] at (7.5,-6.5) {-0.92};
\fill[cblue!14!white] (8,-6) rectangle (9,-7); \node[font=\fontsize{5}{5}\selectfont] at (8.5,-6.5) {-0.14};
\fill[cblue!2!white] (0,-7) rectangle (1,-8); \node[font=\fontsize{5}{5}\selectfont] at (0.5,-7.5) {-0.02};
\fill[cblue!11!white] (1,-7) rectangle (2,-8); \node[font=\fontsize{5}{5}\selectfont] at (1.5,-7.5) {-0.11};
\fill[cblue!7!white] (2,-7) rectangle (3,-8); \node[font=\fontsize{5}{5}\selectfont] at (2.5,-7.5) {-0.07};
\fill[cred!2!white] (3,-7) rectangle (4,-8); \node[font=\fontsize{5}{5}\selectfont] at (3.5,-7.5) {0.02};
\fill[cred!11!white] (4,-7) rectangle (5,-8); \node[font=\fontsize{5}{5}\selectfont] at (4.5,-7.5) {0.11};
\fill[cred!15!white] (5,-7) rectangle (6,-8); \node[font=\fontsize{5}{5}\selectfont] at (5.5,-7.5) {0.15};
\fill[cblue!92!white] (6,-7) rectangle (7,-8); \node[font=\fontsize{5}{5}\selectfont] at (6.5,-7.5) {-0.92};
\fill[cred!100!white] (7,-7) rectangle (8,-8); \node[font=\fontsize{5}{5}\selectfont] at (7.5,-7.5) {1.00};
\fill[cblue!5!white] (8,-7) rectangle (9,-8); \node[font=\fontsize{5}{5}\selectfont] at (8.5,-7.5) {-0.05};
\fill[cred!69!white] (0,-8) rectangle (1,-9); \node[font=\fontsize{5}{5}\selectfont] at (0.5,-8.5) {0.69};
\fill[cred!11!white] (1,-8) rectangle (2,-9); \node[font=\fontsize{5}{5}\selectfont] at (1.5,-8.5) {0.11};
\fill[cred!24!white] (2,-8) rectangle (3,-9); \node[font=\fontsize{5}{5}\selectfont] at (2.5,-8.5) {0.24};
\fill[cblue!9!white] (3,-8) rectangle (4,-9); \node[font=\fontsize{5}{5}\selectfont] at (3.5,-8.5) {-0.09};
\fill[cblue!2!white] (4,-8) rectangle (5,-9); \node[font=\fontsize{5}{5}\selectfont] at (4.5,-8.5) {-0.02};
\fill[cblue!26!white] (5,-8) rectangle (6,-9); \node[font=\fontsize{5}{5}\selectfont] at (5.5,-8.5) {-0.26};
\fill[cblue!14!white] (6,-8) rectangle (7,-9); \node[font=\fontsize{5}{5}\selectfont] at (6.5,-8.5) {-0.14};
\fill[cblue!5!white] (7,-8) rectangle (8,-9); \node[font=\fontsize{5}{5}\selectfont] at (7.5,-8.5) {-0.05};
\fill[cred!100!white] (8,-8) rectangle (9,-9); \node[font=\fontsize{5}{5}\selectfont] at (8.5,-8.5) {1.00};
\node[font=\tiny, anchor=east] at (-0.1,-0.5) {MedInc}; \node[font=\tiny, rotate=90, anchor=east] at (0.5,-9.1) {MedInc};
\node[font=\tiny, anchor=east] at (-0.1,-1.5) {Age}; \node[font=\tiny, rotate=90, anchor=east] at (1.5,-9.1) {Age};
\node[font=\tiny, anchor=east] at (-0.1,-2.5) {Rooms}; \node[font=\tiny, rotate=90, anchor=east] at (2.5,-9.1) {Rooms};
\node[font=\tiny, anchor=east] at (-0.1,-3.5) {Bedrms}; \node[font=\tiny, rotate=90, anchor=east] at (3.5,-9.1) {Bedrms};
\node[font=\tiny, anchor=east] at (-0.1,-4.5) {Pop}; \node[font=\tiny, rotate=90, anchor=east] at (4.5,-9.1) {Pop};
\node[font=\tiny, anchor=east] at (-0.1,-5.5) {Occup}; \node[font=\tiny, rotate=90, anchor=east] at (5.5,-9.1) {Occup};
\node[font=\tiny, anchor=east] at (-0.1,-6.5) {Lat}; \node[font=\tiny, rotate=90, anchor=east] at (6.5,-9.1) {Lat};
\node[font=\tiny, anchor=east] at (-0.1,-7.5) {Lon}; \node[font=\tiny, rotate=90, anchor=east] at (7.5,-9.1) {Lon};
\node[font=\tiny, anchor=east] at (-0.1,-8.5) {price}; \node[font=\tiny, rotate=90, anchor=east] at (8.5,-9.1) {price};
\end{tikzpicture}
\end{center}
```

* **Pearson's $r$** for every pair: $+1$ together, $-1$ opposite, $0$ no *linear* relation
* High $r$ with the target: **signal**; high $r$ between inputs: **collinearity**, unstable weights (regularization)
* Here: income and price $0.69$; latitude and longitude $-0.92$; rooms and bedrooms $0.47$

## Does Location Matter?

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, width=4.4cm, height=3.6cm, xmin=-124.6, xmax=-114, ymin=32.4, ymax=42.1, xlabel={longitude}, ylabel={latitude}, scatter, only marks, scatter src=explicit, mark=square*, mark size=1.5pt, colormap/viridis, point meta min=0.5, point meta max=4, colorbar, colorbar style={width=0.2cm, ytick={1,2,3,4}, yticklabel style={font=\tiny}}, xtick={-122,-118}, ytick={34,38,42}]
  \addplot[scatter, scatter src=explicit] coordinates {(-124.25,40.75) [0.90] (-124.25,41.75) [0.87] (-123.75,39.25) [1.47] (-123.75,40.25) [0.80] (-123.25,38.75) [1.54] (-123.25,39.25) [1.18] (-123.25,40.75) [0.66] (-122.75,37.75) [3.70] (-122.75,38.25) [2.20] (-122.75,38.75) [1.47] (-122.75,39.25) [1.02] (-122.75,40.75) [0.95] (-122.75,41.75) [0.69] (-122.25,36.75) [2.55] (-122.25,37.25) [3.75] (-122.25,37.75) [2.64] (-122.25,38.25) [1.76] (-122.25,38.75) [1.81] (-122.25,39.25) [0.78] (-122.25,39.75) [0.72] (-122.25,40.25) [0.76] (-122.25,40.75) [0.96] (-122.25,41.25) [0.75] (-121.75,36.25) [3.33] (-121.75,36.75) [2.41] (-121.75,37.25) [2.68] (-121.75,37.75) [2.63] (-121.75,38.25) [1.38] (-121.75,38.75) [1.51] (-121.75,39.25) [0.78] (-121.75,39.75) [0.95] (-121.75,40.75) [0.74] (-121.25,36.25) [1.30] (-121.25,36.75) [2.00] (-121.25,37.25) [1.23] (-121.25,37.75) [1.16] (-121.25,38.25) [1.34] (-121.25,38.75) [1.36] (-121.25,39.25) [1.41] (-121.25,39.75) [0.92] (-121.25,40.25) [1.10] (-120.75,35.25) [2.27] (-120.75,35.75) [1.80] (-120.75,36.75) [0.70] (-120.75,37.25) [0.99] (-120.75,37.75) [1.33] (-120.75,38.25) [1.14] (-120.75,38.75) [1.47] (-120.75,39.25) [1.50] (-120.75,39.75) [0.93] (-120.75,40.25) [0.75] (-120.25,34.75) [1.81] (-120.25,36.25) [0.70] (-120.25,36.75) [0.70] (-120.25,37.25) [0.86] (-120.25,37.75) [1.16] (-120.25,38.25) [1.22] (-120.25,38.75) [1.29] (-120.25,39.25) [1.64] (-120.25,39.75) [0.76] (-119.75,34.25) [3.29] (-119.75,36.25) [0.77] (-119.75,36.75) [0.86] (-119.75,37.25) [1.16] (-119.75,37.75) [1.13] (-119.75,38.75) [1.25] (-119.25,34.25) [2.40] (-119.25,35.25) [0.85] (-119.25,35.75) [0.64] (-119.25,36.25) [0.79] (-119.25,36.75) [0.80] (-119.25,37.25) [1.17] (-118.75,34.25) [3.03] (-118.75,34.75) [1.45] (-118.75,35.25) [0.74] (-118.75,36.25) [0.91] (-118.75,37.75) [1.75] (-118.25,33.75) [2.19] (-118.25,34.25) [2.60] (-118.25,34.75) [1.48] (-118.25,35.25) [0.79] (-118.25,35.75) [0.81] (-118.25,37.25) [1.16] (-117.75,33.25) [3.25] (-117.75,33.75) [2.50] (-117.75,34.25) [1.79] (-117.75,34.75) [1.18] (-117.75,35.25) [1.01] (-117.75,35.75) [0.86] (-117.25,32.75) [1.93] (-117.25,33.25) [2.18] (-117.25,33.75) [1.48] (-117.25,34.25) [1.13] (-117.25,34.75) [0.97] (-116.75,32.75) [1.78] (-116.75,33.25) [2.03] (-116.75,33.75) [1.23] (-116.75,34.25) [1.33] (-116.75,34.75) [0.70] (-116.25,33.75) [1.34] (-116.25,34.25) [0.74] (-115.75,32.75) [0.78] (-115.75,33.25) [0.59] (-115.25,32.75) [0.89] (-114.75,33.75) [0.83]};
\end{axis}
\end{tikzpicture}
\end{center}
```

* A **map**: position on the axes, the target as colour; it exposes structure no single feature shows
* Neighbours with similar values mean a **non-linear, joint** effect: a plane cannot follow a coastline
* Here: the Bay Area and Los Angeles stand out (polynomial features, later; trees, Class 06)

## How Balanced Are the Classes?

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, xbar, width=0.6\textwidth, height=2.6cm, bar width=12pt, symbolic y coords={spam, ham}, ytick=data, xmin=0, xmax=0.8, xlabel={fraction of e-mails}, nodes near coords, nodes near coords style={font=\scriptsize, /pgf/number format/fixed, /pgf/number format/precision=3}, enlarge y limits=0.5]
  \addplot[fill=cblue!55, draw=cblue] coordinates {(0.606,ham) (0.394,spam)};
\end{axis}
\end{tikzpicture}
\end{center}
```

* A **bar chart of the class counts** is the first plot of any classification problem
* It sets the **baseline** (always answer the majority class), decides the use of **stratified** splits, and warns when accuracy may mislead
* Here: 39.4% spam, so "everything is ham" is already 60.6% accurate: imbalanced, but not rare

## How Sparse Are the Features?

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, ybar, width=0.62\textwidth, height=3.1cm, bar width=14pt, xmin=0, xmax=1, ymin=0, xtick={0,0.2,0.4,0.6,0.8,1}, xlabel={fraction of e-mails where the feature is 0}, ylabel={features}]
  \addplot[fill=cblue!55, draw=cblue] coordinates {(0.05,3) (0.15,0) (0.25,1) (0.35,0) (0.45,3) (0.55,2) (0.65,2) (0.75,8) (0.85,19) (0.95,19)};
\end{axis}
\end{tikzpicture}
\end{center}
```

* For every feature we plot the **fraction of zeros**: how dense the data are
* Mostly-zero features mean "word absent": the mean says little, the standard deviation is set by a few rare values, and some models (Naive Bayes, Class 04) use *presence* itself
* Here: **77.4%** of the entries are 0, and 19 of the 57 features are 0 in more than 90% of the e-mails

## Which Features Separate the Classes?

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, xbar, width=0.6\textwidth, height=3.4cm, bar width=5pt, symbolic y coords={!,\$,free,remove,george,hp}, ytick=data, y dir=reverse, xmin=0, xmax=1.5, xlabel={mean frequency (\%)}, legend pos=north east, enlarge y limits=0.12]
  \addplot[fill=cred!70, draw=cred] coordinates {(0.514,!) (0.174,\$) (0.518,free) (0.275,remove) (0.002,george) (0.017,hp)}; \addlegendentry{spam}
  \addplot[fill=cblue!55, draw=cblue] coordinates {(0.110,!) (0.012,\$) (0.074,free) (0.009,remove) (1.265,george) (0.895,hp)}; \addlegendentry{ham}
\end{axis}
\end{tikzpicture}
\end{center}
```

* Compare a feature's **mean in each class**: a large gap means the feature carries signal
* It tells us which inputs matter, and which are **artefacts** a model should not lean on
* Here: spam says `!`, `$`, `free`, `remove`; `george` and `hp` appear only in ham: one HP Labs mailbox, a pattern that will not transfer to other mailboxes

## Do the Classes Differ in Scale?

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, ybar, width=0.62\textwidth, height=3.1cm, bar width=5pt, xmin=0, xmax=4.5, ymin=0, xtick={0,1,2,3,4}, xlabel={capital letters in runs, total ($\log_{10}$ axis)}, ylabel={e-mails (\%)}, legend pos=north east]
  \addplot[fill=cblue!55, draw=cblue] coordinates {(0.25,1.7) (0.75,13.1) (1.25,20.8) (1.75,31.2) (2.25,20.5) (2.75,9.9) (3.25,2.5) (3.75,0.3) (4.25,0.0)}; \addlegendentry{ham}
  \addplot[fill=cred!70, draw=cred] coordinates {(0.25,0.3) (0.75,0.4) (1.25,4.2) (1.75,22.4) (2.25,36.1) (2.75,23.6) (3.25,11.4) (3.75,1.5) (4.25,0.1)}; \addlegendentry{spam}
\end{axis}
\end{tikzpicture}
\end{center}
```

* **Overlaid histograms**, one per class, show how well a feature separates them and which scale it lives on
* A feature that spans orders of magnitude cannot be drawn, or fitted, on a linear scale: the tail hides everything else
* Here: the total of capital letters needs a $\log_{10}$ axis; spam sits about one order of magnitude to the right

## From Plots to Decisions

| Plot | It measures | Decision |
|:------------------|:-------------------------|:-----------------------------|
| target histogram | shape of the target, censoring | loss and metric; a floor for the error |
| feature histogram | scale, outliers | logarithm, scaling |
| scatter | form of one relation | linear or not, features to add |
| correlation matrix | linear association, collinearity | regularization, drop duplicates |
| map | joint structure | polynomial features, other models |
| class bars | class proportions | stratified split, MCC, baseline |
| zero fractions | sparsity | $\log(1+x)$, which model |
| class means | signal and artefacts | features to trust |

* A plot that changes no decision was not worth drawing

## Methodology: Why Hold Out Data?

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=4cm, width=0.7\textwidth, xmin=1, xmax=4, ymin=1e-4, ymax=1e5, ymode=log, xtick={1,2,3,4}, xlabel={polynomial degree}, ylabel={MSE (log)}, legend pos=north west]
  \addplot[cblue, mark=*, mark size=1.5pt] coordinates {(1,0.478) (2,0.310) (3,0.135) (4,0.0003)}; \addlegendentry{training set}
  \addplot[cred, mark=*, mark size=1.5pt] coordinates {(1,0.4665) (2,0.460) (3,159.8) (4,20289)}; \addlegendentry{held-out test set}
\end{axis}
\end{tikzpicture}
\end{center}
```

* House prices, 300 training examples, polynomial degrees 1 to 4 (demo R4)
* The **training** error says "perfect"; only **unseen** data shows the truth
* Every number we report must come from data the model did not train on

## Methodology: One Hold-Out Split

```{=latex}
\begin{center}
\begin{tikzpicture}
  \fill[cblue!55] (0,0) rectangle (8,0.7); \fill[corange!75] (8,0) rectangle (10,0.7);
  \node at (4,0.35) {training set: fit the parameters ($80\%$)}; \node at (9,0.35) {test ($20\%$)};
  \draw[decorate, decoration={brace, amplitude=4pt}] (0,-0.1) -- node[below=4pt, note] {16512 districts} (8,-0.1);
  \draw[decorate, decoration={brace, amplitude=4pt}] (10,-0.1) -- node[below=4pt, note] {4128} (8,-0.1);
\end{tikzpicture}
\end{center}
```

* Shuffle, then cut: the model never sees the test rows while training
* Cheap, but the estimate depends on **which** rows landed in the test set

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, ybar, height=3.3cm, width=0.6\textwidth, ymin=0, xlabel={test MSE of one random 80/20 split}, ylabel={splits}, bar width=8pt, xtick={0.42,0.44,0.46,0.48}, xmin=0.41, xmax=0.49]
  \addplot[fill=cblue!55, draw=cblue] coordinates {(0.4179,4) (0.4238,6) (0.4297,13) (0.4356,24) (0.4415,29) (0.4474,37) (0.4533,30) (0.4592,18) (0.4651,13) (0.471,12) (0.4769,10) (0.4829,4)};
\end{axis}
\end{tikzpicture}
\end{center}
```

## Methodology: One Split Is Not Enough

* The same linear model on **200 random splits**: test MSE from **0.415** to **0.486** (mean 0.449, std 0.015)
* With 4128 test rows the spread is small; with a few hundred rows it would be large
* **Repeated hold-out** and **cross-validation** turn one number into a mean *and* a spread

| Method | Mean test MSE | Spread |
|:--------------------|:-------:|:-----------------------|
| one 80/20 split | 0.449 | $\pm 0.015$ over 200 splits |
| 5-fold CV | 0.449 | std of folds 0.019, 95% CI $\pm 0.024$ |
| 10-fold CV | 0.449 | std of folds 0.028, 95% CI $\pm 0.020$ |

* Class 01: confidence intervals, corrected resampled $t$-test, McNemar

## Methodology: $k$-Fold Cross-Validation

```{=latex}
\begin{center}
\begin{tikzpicture}
  \foreach \r in {1,...,5} {
    \node[anchor=east, font=\scriptsize] at (-0.1,-\r*0.55) {round \r};
    \foreach \c in {1,...,5} {
      \ifnum\c=\r \fill[corange!75] ({(\c-1)*1.5},{-\r*0.55-0.22}) rectangle ({\c*1.5-0.05},{-\r*0.55+0.22});
      \else \fill[cblue!45] ({(\c-1)*1.5},{-\r*0.55-0.22}) rectangle ({\c*1.5-0.05},{-\r*0.55+0.22}); \fi
    }
    \node[anchor=west, font=\scriptsize] at (7.6,-\r*0.55) {score$_\r$};
  }
  \node[font=\scriptsize, cblue] at (1.5,0.2) {train}; \node[font=\scriptsize, corange] at (4.5,0.2) {validation (one fold per round)};
\end{tikzpicture}
\end{center}
```

* Every example is used for validation **exactly once**; report the mean and the spread of the $k$ scores
* Use it to **choose** hyper-parameters (degree, $\lambda$, threshold): a separate **test** set is touched once, at the end

## Methodology: Stratified, Grouped, Temporal

```{=latex}
\begin{center}
\begin{tikzpicture}[x=0.5cm, y=0.5cm]
  \node[font=\scriptsize, anchor=east] at (-0.2,2) {\textbf{stratified}};
  \foreach \i in {0,...,19} { \pgfmathparse{mod(\i,5)<2 ? 1 : 0}\ifnum\pgfmathresult=1 \fill[cred!70] (\i,1.6) rectangle ++(0.8,0.7); \else \fill[cblue!50] (\i,1.6) rectangle ++(0.8,0.7); \fi }
  \node[font=\scriptsize, anchor=east] at (-0.2,0) {\textbf{grouped}};
  \foreach \i/\g in {0/1,1/1,2/1,3/2,4/2,5/3,6/3,7/3,8/3,9/4,10/4,11/5,12/5,13/5,14/6,15/6,16/7,17/7,18/7,19/7} { \pgfmathtruncatemacro{\sh}{\g*12+10} \fill[cgreen!\sh] (\i,-0.4) rectangle ++(0.8,0.7); }
  \node[font=\scriptsize, anchor=east] at (-0.2,-2) {\textbf{temporal}};
  \fill[cblue!50] (0,-2.4) rectangle (14.8,-1.7); \fill[corange!75] (15,-2.4) rectangle (19.8,-1.7);
\end{tikzpicture}
\end{center}
```

* **Stratified:** keeps the class proportions; on 100 e-mails the spam fraction of train and test differs by **0.092** on average with a random split, **0.015** stratified (lab B1)
* **Grouped:** correlated rows must stay together, or the test set leaks
* **Temporal:** never test on data older than the training data

## Preprocessing: Why Scaling Matters

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, ymode=log, width=0.62\textwidth, height=3.8cm, xmin=0, xmax=100, ymin=1e-6, ymax=10, xlabel={gradient descent step}, ylabel={loss $-$ optimum (log)}, legend pos=north east]
  \addplot[cblue, mark=*, mark size=1pt] coordinates {(0,5.16e+00) (2,1.32e-01) (5,5.35e-02) (10,2.22e-02) (20,7.30e-03) (30,2.76e-03) (50,4.07e-04) (75,3.74e-05) (100,3.43e-06)}; \addlegendentry{standardized}
  \addplot[cred, mark=*, mark size=1pt] coordinates {(0,5.16e+00) (2,3.68e+00) (5,2.36e+00) (10,1.38e+00) (20,9.20e-01) (30,8.58e-01) (50,8.38e-01) (75,8.23e-01) (100,8.10e-01)}; \addlegendentry{raw units}
\end{axis}
\end{tikzpicture}
\end{center}
```

* Inputs in different units (latitude near 35, income near 4) make the loss a **long thin valley**: condition number **41** standardized, $\mathbf{3\cdot10^{8}}$ raw (Class 02)
* Gradient descent takes a tiny step to survive the steep direction and crawls along the flat one: after 100 steps the raw fit is still **0.81** above the optimum
* Penalties ($\lambda\lVert w\rVert^2$), distances ($k$-NN) and kernels treat all inputs alike: they need **comparable scales**

## Preprocessing: Scale on Training Only

```{=latex}
\begin{center}
\begin{tikzpicture}[node distance=6mm]
  \node[fillbox=cblue, text width=2.6cm] (tr) {training set};
  \node[fillbox=corange, text width=3.2cm, right=of tr] (f) {learn $\mu_j$, $\sigma_j$ (and the model)};
  \node[fillbox=cgray, text width=2.4cm, right=of f] (a) {apply to \textbf{both} sets};
  \draw[flow] (tr) -- (f); \draw[flow] (f) -- (a);
  \node[fillbox=cred, text width=8cm, below=8mm of f] (bad) {\textbf{wrong:} $\mu_j$, $\sigma_j$ from all the data $\Rightarrow$ the test set shapes the model};
\end{tikzpicture}
\end{center}
```

$$z_j = \frac{x_j - \mu_j}{\sigma_j}, \qquad \mu_j, \sigma_j \text{ estimated on the training rows only}$$

* Any statistic learned from data (mean, vocabulary, bins, threshold) stays **inside** the training part (data leakage, Class 01)

## Preprocessing: Why a Logarithm Matters

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, ybar, width=5.2cm, height=3.4cm, bar width=7pt, xmin=0, xmax=6.6, ymin=0, xtick={0,2,4,6}, ytick=\empty, xlabel={\texttt{AveOccup} (raw)}]
  \addplot[fill=cblue!55, draw=cblue] coordinates {(0.25,0) (0.75,3) (1.25,149) (1.75,1481) (2.25,4403) (2.75,6581) (3.25,4319) (3.75,1954) (4.25,978) (4.75,426) (5.25,161) (5.75,70) (6.25,115)};
\end{axis}
\begin{axis}[faa, ybar, at={(6.2cm,0)}, width=5.2cm, height=3.4cm, bar width=7pt, xmin=-1, xmax=7, ymin=0, ytick=\empty, xlabel={$\ln$ \texttt{AveOccup}}]
  \addplot[fill=cgreen!55, draw=cgreen] coordinates {(-0.75,0) (-0.25,3) (0.25,394) (0.75,8554) (1.25,10897) (1.75,725) (2.25,37) (2.75,20) (3.25,1) (3.75,3) (4.25,2) (4.75,0) (5.25,1) (5.75,0) (6.25,2) (6.75,0)};
\end{axis}
\end{tikzpicture}
\end{center}
```

* Squared error is dominated by the **largest values**: one district with 1243 people per house outweighs thousands of ordinary ones
* A logarithm turns *ratios* into *differences* and pulls the tail in ($\log(1+x)$ keeps 0 at 0)
* Housing, least squares, test MSE: **0.497** raw, **0.421** with the log ($R^2$ 0.623 to 0.681)
* Spam, logistic regression, 5-fold MCC: **0.831** raw, **0.877** with $\log(1+x)$

## Preprocessing: Missing Values, Categories

| Problem | What to do | Learned from |
|:-------------------|:--------------------------------------|:------------------|
| Missing number | impute the **median** (+ a 0/1 "was missing" column if absence means something) | training rows |
| Category | **one-hot**: a 0/1 column per level; an unseen level gives zeros | training levels |
| Many levels | group the rare ones; target encoding needs nested folds | training folds |

* **Standard steps of the preprocessing toolset**, like scaling: `SimpleImputer`, `OneHotEncoder`, `ColumnTransformer` (scikit-learn); `fill_null`, `to_dummies` (polars)
* Dropping incomplete rows changes the **population** you evaluate on: say so, or impute
* The median of **all** rows is the same leak as scaling with all rows

## The Pipeline, End to End

```{=latex}
\begin{center}
\begin{tikzpicture}[node distance=5mm]
  \node[fillbox=cgray, text width=0.9cm] (all) {all data};
  \node[fillbox=cblue, text width=1.2cm, right=6mm of all] (tr) {training part};
  \node[fillbox=cgreen, text width=2.0cm, right=6mm of tr] (pp) {\textbf{3.}\\ preprocess\\ (per fold)};
  \node[fillbox=cpurple, text width=1.2cm, right=4mm of pp] (md) {\textbf{4.} fit, tune};
  \node[fillbox=cred, text width=1.3cm, right=6mm of md] (fin) {\textbf{5.} refit, test once};
  \node[fillbox=corange, text width=1.7cm, below=9mm of tr] (te) {\textbf{2.} test set (locked)};
  \begin{scope}[on background layer]
    \node[draw, dashed, rounded corners, fit=(pp)(md), inner sep=4pt, label={[note]above:inside every $k$-fold round}] {};
  \end{scope}
  \draw[flow] (all) -- (tr); \draw[flow] (tr) -- (pp); \draw[flow] (pp) -- (md); \draw[flow] (md) -- (fin); \draw[flow] (all.south) |- (te.west); \draw[flow] (te.east) -| (fin.south);
\end{tikzpicture}
\end{center}
```

1. **Data visualization:** decides the transformations and the metric
2. **Evaluation methodology:** split once, keep the test set **locked**, choose folds and metric
3. **Preprocessing:** fitted on the training folds only (scale, log, impute, encode)
4. **Model:** fit and tune on validation folds
5. **Report:** refit, test **once**, with baseline and spread

## Live Demo: Data, Preprocessing, Splitting

```{=latex}
\begin{center}
\vspace{8mm}
{\Huge\bfseries DEMO R1:}\\[5mm]
{\LARGE\ttfamily 01\_linear\_regression.ipynb}
\end{center}
```

# Linear Regression

## The Model: a Hyperplane

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=4.2cm, width=0.6\textwidth, xmin=0, xmax=5, ymin=0, ymax=6, xlabel={$x$}, ylabel={$y$}]
  \addplot[cblue, domain=0:5] {1.1*x};
  \foreach \px/\py in {1/1, 2/3, 3/2, 4/5} {
    \edef\tmp{\noexpand\draw[cred, very thick] (axis cs:\px,\py) -- (axis cs:\px,{1.1*\px});}\tmp
    \edef\tmp{\noexpand\fill (axis cs:\px,\py) circle (2.2pt);}\tmp
  }
  \node[font=\scriptsize, cred, anchor=south east, align=right] at (axis cs:4.95,0) {red: residuals\\ $e_i = y_i - \hat y_i$};
\end{axis}
\end{tikzpicture}
\end{center}
```

$$\hat y = w^\top x + b = w_1 x_1 + \dots + w_d x_d + b$$

* $d$ weights and one bias: $d + 1$ numbers **are** the model
* Each weight: the change of $\hat y$ per unit of feature $j$, the others fixed
* Housing: $d = 8$; with standardized inputs the weights are comparable

## The Loss: Mean Squared Error

$$\mathcal{L}(w, b) = \frac{1}{N}\sum_{i=1}^{N} \big(w^\top x_i + b - y_i\big)^2$$

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=3.8cm, width=0.55\textwidth, xmin=-0.2, xmax=2.2, ymin=0, ymax=12, xlabel={slope $w$ (with $b = 0$)}, ylabel={$\mathcal{L}$}]
  \addplot[cblue, domain=-0.2:2.2, samples=60] {(30*x^2 - 66*x + 39)/4};
  \addplot[only marks, mark=*, cred] coordinates {(1.1,0.675)};
  \node[font=\scriptsize, cred, anchor=south] at (axis cs:1.1,6) {$w^\star = 1.1$, $\mathcal{L} = 0.675$};
  \draw[->, cred] (axis cs:1.1,5.9) -- (axis cs:1.1,1.0);
\end{axis}
\end{tikzpicture}
\end{center}
```

* A **quadratic** in $(w, b)$: a convex bowl with one minimum (Class 02)
* Squaring: errors cannot cancel, large errors cost more
* Same four points as the previous slide

## Worked Example: Least Squares by Hand

Points $(1,1), (2,3), (3,2), (4,5)$, so $\bar x = 2.5$ and $\bar y = 2.75$

$$w = \frac{\sum (x_i - \bar x)(y_i - \bar y)}{\sum (x_i - \bar x)^2} = \frac{5.5}{5} = 1.1, \qquad b = \bar y - w \bar x = 0$$

| $x$ | $y$ | $\hat y$ | $e$ |
|:-:|:-:|--:|--:|
| 1 | 1 | 1.1 | $-0.1$ |
| 2 | 3 | 2.2 | $0.8$ |
| 3 | 2 | 3.3 | $-1.3$ |
| 4 | 5 | 4.4 | $0.6$ |

$$\text{MSE} = \frac{0.01 + 0.64 + 1.69 + 0.36}{4} = 0.675$$



## The Gradient and the Normal Equation

$$\nabla_w \mathcal{L} = \frac{2}{N}\, X^\top (X w + b - y) \qquad \frac{\partial \mathcal{L}}{\partial b} = \frac{2}{N}\sum_i (\hat y_i - y_i)$$

* Gradient = features $\times$ residuals: each weight moves to explain the residual it can see
* Set it to zero. With $A = [X\; \mathbf{1}]$ and $\theta = (w, b)$:

$$A^\top A\,\theta = A^\top y \quad\Longrightarrow\quad \theta^\star = (A^\top A)^{-1} A^\top y$$

* The **normal equations**: a linear system. Solve it (`np.linalg.solve`), never invert
* Exists whenever $A^\top A$ is invertible: no perfectly collinear features, and $N \ge d + 1$
* In the lab: **you never type this gradient**; `jax.grad` of the loss gives it

## Worked Example: Gradient Descent

Same four points, start $w = b = 0$, $\eta = 0.05$

| step | $w$ | $b$ | $\mathcal{L}$ | $\partial_w \mathcal{L}$ | $\partial_b \mathcal{L}$ |
|:-:|--:|--:|--:|--:|--:|
| 0 | 0.000 | 0.000 | 9.750 | $-16.500$ | $-5.500$ |
| 1 | 0.825 | 0.275 | 0.940 | $-2.750$ | $-0.825$ |
| 2 | 0.963 | 0.316 | 0.699 | $-0.481$ | $-0.055$ |
| 3 | 0.987 | 0.319 | 0.692 | $-0.107$ | $0.071$ |

* Step 0: $\partial_w \mathcal{L} = \frac{2}{4}\sum_i (0 - y_i)x_i = -0.5 \cdot 33 = -16.5$, so $w_1 = 0 - 0.05 \cdot (-16.5) = 0.825$
* Heading for $(w, b) = (1.1, 0)$ and $\mathcal{L} = 0.675$: the answer of the closed form
* Linear regression is the one model where we can **check** the iterative answer against the exact one

## Closed Form or Gradient Descent?

| | Normal equation | Gradient descent |
|:------------------------|:------------------------------|:----------------------------------|
| Cost | $O(N d^2 + d^3)$ once | $O(N d)$ per step, many steps |
| Needs | $A^\top A$ invertible | any differentiable loss |
| Tuning | none | learning rate, steps |
| Large $d$ | hopeless ($10^6$ features) | works (mini-batches, Class 02) |
| Other losses (Huber, L1, ...) | no closed form | change one line |

* For 8 housing features the closed form is instant; **all other models today have no closed form** (or only a variant)
* Both give the same weights: difference **$6\cdot10^{-15}$** (normal equation vs scikit-learn), **$2\cdot10^{-6}$** (300 GD steps)

## The Learning Rate Comes from the Curvature

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=4.1cm, width=0.7\textwidth, ymode=log, xmin=0, xmax=100, ymin=1e-5, ymax=100, xlabel={step}, ylabel={loss $-$ optimum (log)}, legend pos=outer north east]
  \addplot[cblue, mark=*, mark size=1pt] coordinates {(0,5.16) (10,2.33) (20,1.1) (30,0.555) (40,0.306) (50,0.19) (60,0.133) (70,0.102) (80,0.0843) (90,0.0726) (100,0.0641)}; \addlegendentry{$\eta = 0.02$}
  \addplot[corange, mark=*, mark size=1pt] coordinates {(0,5.16) (10,0.164) (20,0.0625) (30,0.0392) (40,0.0271) (50,0.0199) (60,0.0152) (70,0.012) (80,0.00955) (90,0.00769) (100,0.00622)}; \addlegendentry{$\eta = 0.1$}
  \addplot[cgreen, mark=*, mark size=1pt] coordinates {(0,5.16) (10,0.0257) (20,0.00923) (30,0.00394) (40,0.00171) (50,0.000741) (60,0.000322) (70,0.00014) (80,6.05e-05) (90,2.63e-05) (100,1.14e-05)}; \addlegendentry{$\eta = 0.4$}
  \addplot[cred, mark=*, mark size=1pt] coordinates {(0,5.16) (10,0.0354) (20,0.0421) (30,0.0828) (40,0.182) (50,0.407) (60,0.913) (70,2.05) (80,4.61) (90,10.3) (100,23.2)}; \addlegendentry{$\eta = 0.49$}
\end{axis}
\end{tikzpicture}
\end{center}
```

* The Hessian of the MSE is constant: $H = \frac{2}{N} A^\top A$. Housing: eigenvalues **0.102** to **4.166**, condition number **41**
* Stable only for $\eta < 2/\lambda_{\max} = 0.48$: $\eta = 0.49$ diverges, $\eta = 0.02$ crawls along the flat direction (Class 02)

## Linear Regression in JAX

```python
import jax, jax.numpy as jnp

def predict(params, X):
    return X @ params["w"] + params["b"]

def mse_loss(params, X, y):
    return jnp.mean((predict(params, X) - y) ** 2)

params = {"w": jnp.zeros(8), "b": jnp.zeros(())}
grad = jax.jit(jax.grad(mse_loss))       # derived, not typed
for _ in range(300):
    g = grad(params, X_train, y_train)
    params = jax.tree.map(lambda p, gp: p - 0.4 * gp, params, g)
```

* Parameters are a **pytree** (a dictionary): `jax.grad` returns the gradient with the same structure
* Change the loss (add a penalty, use Huber) and **nothing else** changes: this is the lab's method

## Why Squared Error? The Probabilistic View

$$y = w^\top x + b + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2)$$

$$\Longrightarrow\quad p(y \mid x) = \mathcal{N}\big(w^\top x + b,\ \sigma^2\big)$$

$$-\log \prod_i p(y_i \mid x_i) = \frac{1}{2\sigma^2}\sum_i (y_i - \hat y_i)^2 + N\log\sigma + \text{const}$$

* Minimizing the MSE is **maximum likelihood** under Gaussian noise (Class 01: the loss is a code length)
* Laplace noise gives the absolute error (MAE, robust to outliers); Gaussian noise punishes outliers heavily
* The same idea gives **cross-entropy** for classes (a few slides ahead)

## Result: House Prices

:::: {.columns}
::: {.column width="52%"}
```{=latex}
\begin{tikzpicture}
\begin{axis}[faa, xbar, height=5.2cm, width=\textwidth, symbolic y coords={Longitude, Latitude, AveOccup, AveRooms, Population, HouseAge, AveBedrms, MedInc}, ytick=data, xmin=-1.1, xmax=1.1, xlabel={weight (standardized inputs)}, bar width=6pt, enlarge y limits=0.06, y tick label style={font=\tiny}]
  \addplot[fill=cblue!55, draw=cblue] coordinates {(-0.829,Longitude) (-0.886,Latitude) (-0.258,AveOccup) (-0.169,AveRooms) (0.037,Population) (0.141,HouseAge) (0.185,AveBedrms) (0.836,MedInc)};
\end{axis}
\end{tikzpicture}
```
:::
::: {.column width="48%"}
Test set (4128 districts)

| | RMSE | $R^2$ |
|:--|--:|--:|
| mean baseline | 1.148 | 0.000 |
| linear | **0.649** | **0.681** |

RMSE 0.649 = **64 900 USD** typical error
:::
::::

* Income raises the price; latitude and longitude (the coast) dominate the rest
* About 32% of the variance is unexplained: what a hyperplane cannot see (and the cap at 5.0)

## Live Demo: Linear Regression Pipeline

```{=latex}
\begin{center}
\vspace{8mm}
{\Huge\bfseries DEMO R2:}\\[5mm]
{\LARGE\ttfamily 01\_linear\_regression.ipynb}
\end{center}
```

# Regression Metrics

## How Wrong Are the Predictions?

$$e_i = y_i - \hat y_i \qquad \text{(the residual)}$$

$$\text{MSE} = \frac{1}{n}\sum_i e_i^2 \qquad \text{RMSE} = \sqrt{\text{MSE}} \qquad \text{MAE} = \frac{1}{n}\sum_i |e_i|$$

* **MSE:** the loss we optimize; squared unit, punishes large errors
* **RMSE:** back in the unit of $y$, dominated by the large errors
* **MAE:** every error counts in proportion; robust to outliers
* Always **RMSE $\ge$ MAE**: the gap measures how uneven the errors are
* Lower is better; none is meaningful without a **baseline**

## $R^2$: Compared with Predicting the Mean

$$R^2 = 1 - \frac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i - \bar y)^2} = 1 - \frac{\text{MSE of the model}}{\text{MSE of the mean baseline}}$$

```{=latex}
\begin{center}
\begin{tikzpicture}
  \draw[->] (0,0) -- (7,0) node[right, font=\scriptsize] {$R^2$};
  \foreach \x/\l in {1/{$<0$}, 2.6/{$0$}, 4.6/{$0.68$}, 6.4/{$1$}} { \draw (\x,0.1) -- (\x,-0.1) node[below, font=\scriptsize] {\l}; }
  \node[note, above] at (1,0.15) {worse than\\ the mean}; \node[note, above] at (2.6,0.15) {= the mean\\ baseline}; \node[note, above, cblue] at (4.6,0.15) {housing,\\ linear}; \node[note, above] at (6.4,0.15) {perfect};
\end{tikzpicture}
\end{center}
```

* **Fraction of the variance explained** by the model; 0 is the mean baseline, 1 is perfect
* It can be **negative**: a constant prediction of 10 for values near 3 gives $R^2 = -98$
* Scale-free, but never enough alone: it says nothing about the size of the errors in USD

## sMAPE: a Symmetric Relative Error

$$\text{sMAPE} = \frac{100}{n}\sum_i \frac{2\,|y_i - \hat y_i|}{|y_i| + |\hat y_i|}\quad [\%]$$

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=3.6cm, width=0.62\textwidth, xmin=0, xmax=300, ymin=0, ymax=200, xlabel={prediction $\hat y$ (true value $y = 100$)}, ylabel={error (\%)}, legend pos=north east]
  \addplot[cblue, domain=0:300, samples=120] {200*abs(100-x)/(100+x)}; \addlegendentry{sMAPE}
  \addplot[cgray, dashed, domain=0:300, samples=120] {abs(100-x)}; \addlegendentry{percentage error}
\end{axis}
\end{tikzpicture}
\end{center}
```

* True value 100: prediction 50 gives **66.7%**, prediction 150 gives **40.0%** (same error, plain percentage error 50% for both)
* So it punishes **under-prediction** more; unstable near 0: use it for prices

## Worked Example: the Four Metrics

Residuals of the least-squares line: $e = (-0.1,\ 0.8,\ -1.3,\ 0.6)$, $\ y = (1, 3, 2, 5)$, $\ \bar y = 2.75$

| Metric | Computation | Value |
|:--------|:------------------------------------------|--------:|
| MSE | $(0.01 + 0.64 + 1.69 + 0.36)/4$ | 0.675 |
| RMSE | $\sqrt{0.675}$ | 0.822 |
| MAE | $(0.1 + 0.8 + 1.3 + 0.6)/4$ | 0.700 |
| $R^2$ | $1 - 2.7 / 8.75$ | 0.691 |
| sMAPE | mean of $\frac{2|e_i|}{y_i + \hat y_i}$: $9.5\%,\ 30.8\%,\ 49.1\%,\ 12.8\%$ | 25.5% |

* $\sum(y_i - \bar y)^2 = 3.0625 + 0.0625 + 0.5625 + 5.0625 = 8.75$
* Four numbers, four questions: how big (RMSE), how typical (MAE), how much explained ($R^2$), how relative (sMAPE)

## Metrics Disagree: Outliers

| Predictions | MSE | RMSE | MAE | sMAPE | $R^2$ |
|:--------------------|-----:|-----:|-----:|------:|------:|
| errors $\pm 0.1$ | 0.010 | 0.100 | 0.100 | 3.5% | 0.98 |
| same, **one** error of $+3$ | 1.93 | 1.39 | 0.70 | 16.5% | $-2.86$ |

* True values $2, 2.5, 3, 3.5, 4$; a single prediction 3 units off
* **MSE $\times$ 193**, MAE $\times$ 7: squaring lets one point dominate
* Choose by the **cost of a mistake**: quadratic cost (safety margins) $\Rightarrow$ MSE/RMSE; linear cost, dirty data $\Rightarrow$ MAE
* Report at least one absolute and one relative measure, plus the baseline

## Baselines: Mean, Median, Mode

| Constant answer | Value | MSE | MAE | sMAPE | $R^2$ |
|:------------------|------:|------:|------:|-------:|-------:|
| mean | 2.069 | **1.319** | 0.904 | 44.6% | 0.000 |
| median | 1.797 | 1.392 | **0.875** | **43.4%** | $-0.056$ |
| mode | 5.000 | 9.915 | 2.932 | 89.8% | $-6.520$ |
| *linear regression* | | *0.421* | *0.472* | *25.6%* | *0.681* |

* The **mean** minimizes the squared error, the **median** the absolute error: each wins *its own* metric, so the baseline must be chosen with the metric
* The **mode** is the most frequent value: here the cap at 5.0, a terrible constant for a continuous target
* A model is only worth reporting if it beats the best baseline **for the metric you report**

## Result: House Prices

| Model | MSE | RMSE | MAE | sMAPE | $R^2$ |
|:------------------|------:|------:|------:|-------:|------:|
| mean baseline | 1.319 | 1.148 | 0.904 | 44.6% | 0.000 |
| linear regression | **0.421** | **0.649** | **0.472** | **25.6%** | **0.681** |

* Units: 100 000 USD. The typical error falls from **114 800** USD to **64 900** USD
* RMSE (0.649) above MAE (0.472): a few districts are badly wrong (the capped ones, the very expensive)
* sMAPE 25.6%: a quarter of the price, on average. Useful, not precise: the next slides try to do better

## Live Demo: Metrics and Baselines

```{=latex}
\begin{center}
\vspace{8mm}
{\Huge\bfseries DEMO R3:}\\[5mm]
{\LARGE\ttfamily 01\_linear\_regression.ipynb}
\end{center}
```

# Polynomial Feature Expansion

## Still Linear, in the Parameters

$$\hat y = w_0 + w_1 x + w_2 x^2 + w_3 x^3 = w_0 + w^\top \phi(x)$$

$$\phi(x) = (x,\ x^2,\ x^3)$$

* The model is a **curve** in $x$ but a **hyperplane** in the new features $\phi(x)$
* The loss is still a quadratic in $w$: the same normal equation, the same `jax.grad`, the same metrics
* With several inputs: **all monomials** up to degree $d$, e.g. degree 2 for $(x_1, x_2)$:

$$\phi(x_1, x_2) = (x_1,\ x_2,\ x_1^2,\ x_1 x_2,\ x_2^2)$$

* The products $x_1 x_2$ are **interactions**: the effect of one feature depends on another


## The Number of Features Explodes

$$\#\text{features} = \binom{n + d}{d} - 1 \qquad (n \text{ inputs, degree } d)$$

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=3.8cm, width=0.6\textwidth, ymode=log, xmin=1, xmax=4, ymin=5, ymax=1000, xtick={1,2,3,4}, xlabel={degree $d$}, ylabel={features (log)}, height=3.2cm, nodes near coords, nodes near coords style={font=\tiny, /pgf/number format/fixed, /pgf/number format/precision=0, anchor=south}]
  \addplot[cblue, mark=*, mark size=1.8pt] coordinates {(1,8) (2,44) (3,164) (4,494)};
  \addplot[cred, dashed, no marks] coordinates {(1,300) (4,300)};
  \node[font=\tiny, cred, anchor=north west] at (axis cs:1.6,280) {300 training examples};
\end{axis}
\end{tikzpicture}
\end{center}
```

* The 8 housing features: 8, **44**, **164**, **494** columns (degrees 1 to 4)
* Beyond $N$ columns the model **interpolates** the training set
* **Standardize after expanding** ($x^4$ has a very different scale)

## Curves of Growing Flexibility

```{=latex}
\begin{center}
\begin{tikzpicture}
\pgfplotsset{small/.style={faa, width=0.30\textwidth, height=3.4cm, xmin=-1.5, xmax=2.3, ymin=0, ymax=6, xtick=\empty, ytick=\empty, title style={font=\scriptsize}}}
\def\pts{(-1.21,1.23) (-1.16,0.778) (-1.01,0.934) (-0.781,0.833) (-0.756,2.21) (-0.698,0.938) (-0.672,2.85) (-0.63,5) (-0.515,1.7) (-0.494,2.4) (-0.426,1.68) (-0.0799,1.35) (-0.0611,1.35) (0.0117,2.1) (0.106,3.01) (0.117,1.43) (0.137,2.68) (0.351,3.07) (0.644,2.67) (0.678,4.39) (0.961,1.65) (1.13,1.93) (1.19,2.12) (1.46,5) (2.06,5)}
\begin{axis}[small, at={(0,0)}, title={degree 1: underfits}]
  \addplot[only marks, mark=*, mark size=1pt, black] coordinates {\pts};
  \addplot[cred, no marks] coordinates {(-1.41,1.12) (-0.79,1.65) (-0.17,2.18) (0.45,2.71) (1.08,3.24) (1.7,3.77) (2.26,4.24)};
\end{axis}
\begin{axis}[small, at={(3.9cm,0)}, title={degree 3: smooth}]
  \addplot[only marks, mark=*, mark size=1pt, black] coordinates {\pts};
  \addplot[cred, no marks] coordinates {(-1.41,0.381) (-1.16,1.1) (-0.915,1.61) (-0.666,1.96) (-0.417,2.18) (-0.167,2.31) (0.082,2.38) (0.331,2.43) (0.581,2.49) (0.83,2.61) (1.08,2.82) (1.33,3.16) (1.58,3.66) (1.83,4.36) (2.08,5.3) (2.26,6.17)};
\end{axis}
\begin{axis}[small, at={(7.8cm,0)}, title={degree 9: memorizes}]
  \addplot[only marks, mark=*, mark size=1pt, black] coordinates {\pts};
  \addplot[cred, no marks] coordinates {(-1.41,0.125) (-1.35,1.49) (-1.23,1.3) (-1.1,0.619) (-0.977,0.729) (-0.853,1.43) (-0.728,2.14) (-0.603,2.48) (-0.479,2.36) (-0.354,1.97) (-0.23,1.59) (-0.105,1.48) (0.0197,1.76) (0.144,2.36) (0.269,3.08) (0.394,3.65) (0.518,3.83) (0.643,3.52) (0.767,2.85) (0.892,2.1) (1.02,1.65) (1.14,1.87) (1.27,2.86) (1.39,4.3) (1.52,5.43) (1.64,5.21) (1.76,3.01) (1.83,1.37) (1.89,-0.108) (1.95,-0.658) (2.01,0.97) (2.08,6.64)};
\end{axis}
\end{tikzpicture}
\end{center}
```

* 25 districts: value against (standardized) median income, fitted with polynomials of degree 1, 3 and 9
* Degree 1 misses the bend, degree 9 chases the noise and swings wildly at the edges

## Training Error Versus Test Error

| degree | features | train MSE | test MSE |
|:-:|--:|--:|--:|
| 1 | 8 | 0.478 | **0.467** |
| 2 | 44 | 0.310 | **0.460** |
| 3 | 164 | 0.135 | 159.8 |
| 4 | 494 | 0.0003 | 20 289 |

* 300 training examples of the 8 housing features; the **training** error falls at every degree
* At degree 4 there are more features than examples: the fit is perfect on the training set, useless on the test set
* Degree 3 is *worse* than degree 2 by a factor of **350**: outlying districts, cubed, dominate
* The **test** (or validation) error picks the degree, never the training error

## More Data Tames a Flexible Model

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, ybar, height=3.9cm, width=0.75\textwidth, ymode=log, log origin=infty, ymin=0.2, ymax=50000, symbolic x coords={degree 1, degree 2, degree 3, degree 4}, xtick=data, ylabel={test MSE (log)}, bar width=10pt, legend style={at={(0.5,1.02)}, anchor=south, legend columns=2}, enlarge x limits=0.15]
  \addplot[fill=cred!55, draw=cred] coordinates {(degree 1,0.4665) (degree 2,0.460) (degree 3,159.8) (degree 4,20289)}; \addlegendentry{300 examples}
  \addplot[fill=cblue!55, draw=cblue] coordinates {(degree 1,0.4213) (degree 2,0.3901) (degree 3,1.0043) (degree 4,9.6725)}; \addlegendentry{3000 examples}
\end{axis}
\end{tikzpicture}
\end{center}
```

* Ten times more data: degree 3 goes from **160** to **1.0**, degree 4 from **20 000** to **9.7**
* Flexibility only pays when the data can *support* it: at 3000 examples degree 2 (**0.390**) finally beats the line (**0.421**)
* The other cure keeps the data and **constrains the model**: regularization

## Live Demo: Polynomial Features

```{=latex}
\begin{center}
\vspace{8mm}
{\Huge\bfseries DEMO R4:}\\[5mm]
{\LARGE\ttfamily 01\_linear\_regression.ipynb}
\end{center}
```

# Regularization

## Penalize Large Weights

$$\min_{w,\,b}\;\; \underbrace{\frac{1}{N}\sum_i (\hat y_i - y_i)^2}_{\text{fit the data}} \;+\; \lambda\,\underbrace{\Omega(w)}_{\text{keep the weights small}}$$

| | $\Omega(w)$ | Name | Effect |
|:----------|:------------------------|:--------------|:------------------------|
| $L_2$ | $\lVert w\rVert_2^2 = \sum_j w_j^2$ | **ridge** ($L_2$ regularization, weight decay) | shrinks all weights smoothly |
| $L_1$ | $\lVert w\rVert_1 = \sum_j \lvert w_j\rvert$ | **lasso** ($L_1$ regularization) | shrinks and sets many to **exactly 0** |
| both | $\rho\lVert w\rVert_1 + \frac{1-\rho}{2}\lVert w\rVert_2^2$ | **elastic net** | sparse and stable |

* $\lambda \ge 0$ is a **hyper-parameter**: $0$ is least squares, $\infty$ predicts the mean
* The bias $b$ is **not** penalized; **standardize first**, the penalty treats all weights alike

## Ridge ($L_2$) Regression

$$\mathcal{L}_{ridge} = \mathcal{L}_{L_2} = \frac{1}{N}\lVert y - Xw - b\rVert^2 + \lambda\lVert w\rVert_2^2$$

$$w^\star = (X^\top X + \lambda N I)^{-1} X^\top y \qquad (\text{centred data})$$

* Adding $\lambda N I$ to $X^\top X$ makes it **invertible and well conditioned**: highly correlated polynomial features no longer blow up
* Gradient descent: the penalty adds $2\lambda w$ to the gradient, a "weight decay" towards 0
* One feature, centred: $w_{ridge} = \dfrac{w_{ols}}{1 + \lambda N / \sum x_i^2}$, a **shrinkage factor** below 1
* $\lambda = 0$: least squares; moderate $\lambda$: smooth and stable; $\lambda \to \infty$: $w \to 0$, the mean (more bias, less variance)

## Lasso ($L_1$) and Elastic Net

$$\mathcal{L}_{lasso} = \mathcal{L}_{L_1} = \frac{1}{N}\lVert y - Xw - b\rVert^2 + \lambda\lVert w\rVert_1$$

* No closed form ($\lvert w\rvert$ has a kink at 0), but still **convex**: coordinate descent (scikit-learn) or a **proximal** step, descend on the MSE, then **soft-threshold**

$$\text{soft}(w, t) = \text{sign}(w)\,\max(\lvert w\rvert - t,\, 0)$$

* A weight pulled less by the data than by the penalty **stays at 0**: **feature selection**
* `jax.grad` of $\lvert w\rvert$ is a *subgradient*: weights hover near 0, never exactly; the proximal step gives exact zeros (lab E3)
* **Elastic net** keeps correlated features together; lasso picks one

## The Geometry: Why $L_1$ Gives Zeros

```{=latex}
\begin{center}
\begin{tikzpicture}[scale=0.8]
\def\cx{1.6}\def\cy{0.8}
\begin{scope}
  \draw[->, cgray] (-0.6,0) -- (3.6,0) node[right, font=\scriptsize] {$w_1$}; \draw[->, cgray] (0,-0.6) -- (0,2.4) node[above, font=\scriptsize] {$w_2$};
  \foreach \r in {0.332,0.7,1.3} { \draw[cblue, rotate around={150:(\cx,\cy)}] (\cx,\cy) ellipse ({2*\r} and {0.3*\r}); }
  \fill[cblue] (\cx,\cy) circle (1.6pt);
  \draw[very thick] (0,0) circle (1.5);
  \fill[cred] (1.07,1.05) circle (2.4pt);
  \node[note, below] at (1.5,-1.5) {ridge ($L_2$): contact off the axes\\ (both weights non-zero)};
\end{scope}
\begin{scope}[xshift=5.6cm]
  \draw[->, cgray] (-0.6,0) -- (3.6,0) node[right, font=\scriptsize] {$w_1$}; \draw[->, cgray] (0,-0.6) -- (0,2.4) node[above, font=\scriptsize] {$w_2$};
  \foreach \r in {0.7,1.0818,1.5} { \draw[cblue, rotate around={150:(\cx,\cy)}] (\cx,\cy) ellipse ({2*\r} and {0.3*\r}); }
  \fill[cblue] (\cx,\cy) circle (1.6pt);
  \draw[very thick] (1.5,0) -- (0,1.5) -- (-1.5,0) -- (0,-1.5) -- cycle;
  \fill[cred] (0,1.5) circle (2.4pt);
  \node[note, below] at (1.5,-1.5) {lasso ($L_1$): contact at a corner\\ ($w_1 = 0$ exactly)};
\end{scope}
\end{tikzpicture}
\end{center}
```

* Blue: contours of the squared error (centre = least squares). Black: the region $\lVert w\rVert_p \le t$; the solution is where they touch
* The $L_1$ region has **corners on the axes**: contours tend to meet it there

## The Prior View: MAP

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=3.4cm, width=0.62\textwidth, xmin=-3, xmax=3, ymin=0, ymax=1.1, xlabel={weight $w_j$}, ylabel={prior density}, ytick=\empty, height=3cm, legend style={at={(0.5,1.05)}, anchor=south, legend columns=2}]
  \addplot[cblue, domain=-3:3, samples=120] {exp(-x^2/2)/1.0}; \addlegendentry{Gaussian (ridge)}
  \addplot[cred, domain=-3:3, samples=120] {exp(-abs(x)*1.4)}; \addlegendentry{Laplace (lasso)}
\end{axis}
\end{tikzpicture}
\end{center}
```

$$\hat w_{MAP} = \arg\min_w\; \underbrace{-\log p(y \mid X, w)}_{\text{data: MSE}} \;\underbrace{-\;\log p(w)}_{\text{prior: penalty}}$$

* Gaussian prior $\Rightarrow$ $L_2$; Laplace prior $\Rightarrow$ $L_1$ (a peak at 0: most weights are 0)
* Class 01 (MDL): the penalty is the **code length of the model**, the MSE that of the errors
* $\lambda N$ = noise variance / prior variance

## Regularization Paths

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=3.8cm, width=0.44\textwidth, xmode=log, xmin=0.01, xmax=10000, ymin=-3, ymax=3, xlabel={ridge ($L_2$) $\alpha$ (log)}, ylabel={coefficient}, title={\scriptsize ridge: smooth shrinkage}, title style={yshift=-1mm}]
  \addplot[cblue, no marks] coordinates {(0.01,-1.8) (0.0316,-1.82) (0.1,-1.63) (0.316,-1.24) (1,-0.761) (3.16,-0.399) (10,-0.2) (31.6,-0.104) (100,-0.0501) (316,-0.0115) (1e3,0.0101) (3.16e3,0.0144) (1e4,0.0102)};
  \addplot[corange, no marks] coordinates {(0.01,1.6) (0.0316,1.97) (0.1,1.72) (0.316,1.14) (1,0.544) (3.16,0.163) (10,0.00537) (31.6,-0.0314) (100,-0.0194) (316,0.00204) (1e3,0.0143) (3.16e3,0.0153) (1e4,0.0104)};
  \addplot[cgreen, no marks] coordinates {(0.01,2.66) (0.0316,2.05) (0.1,1.52) (0.316,0.964) (1,0.522) (3.16,0.279) (10,0.151) (31.6,0.0818) (100,0.0487) (316,0.0282) (1e3,0.0125) (3.16e3,0.00351) (1e4,0.000393)};
  \addplot[cred, no marks] coordinates {(0.01,-2.01) (0.0316,-2.04) (0.1,-1.78) (0.316,-1.22) (1,-0.629) (3.16,-0.252) (10,-0.0876) (31.6,-0.0294) (100,-0.00773) (316,0.000207) (1e3,0.00348) (3.16e3,0.00417) (1e4,0.00289)};
  \addplot[cpurple, no marks] coordinates {(0.01,-1.88) (0.0316,-1.27) (0.1,-1.04) (0.316,-0.806) (1,-0.488) (3.16,-0.271) (10,-0.16) (31.6,-0.0864) (100,-0.0316) (316,-0.00371) (1e3,0.00497) (3.16e3,0.00632) (1e4,0.0049)};
\end{axis}
\end{tikzpicture}
\begin{tikzpicture}
\begin{axis}[faa, height=3.8cm, width=0.44\textwidth, xmode=log, xmin=0.001, xmax=1, ymin=-1.3, ymax=1.5, xlabel={lasso ($L_1$) $\alpha$ (log)}, title={\scriptsize lasso: weights reach exactly 0}, title style={yshift=-1mm}]
  \addplot[cblue, no marks] coordinates {(1,0) (0.0412,0) (0.0242,0) (0.0143,0) (0.00838,-0.0782) (0.00492,-0.173) (0.00289,-0.587) (0.0017,-0.883) (0.001,-1.16)};
  \addplot[corange, no marks] coordinates {(1,0) (0.00492,0) (0.00289,0) (0.0017,0.548) (0.001,1.31)};
  \addplot[cgreen, no marks] coordinates {(1,0) (0.0242,0) (0.0143,0.0773) (0.00838,0.168) (0.00492,0.201) (0.00289,0.174) (0.0017,0.389) (0.001,0.724)};
  \addplot[cred, no marks] coordinates {(1,0) (0.00838,0) (0.00492,-0.0241) (0.00289,-0.0888) (0.0017,-0.374) (0.001,-1.05)};
  \addplot[cpurple, no marks] coordinates {(1,0) (0.0242,0) (0.0143,-0.091) (0.00838,-0.146) (0.00492,-0.228) (0.00289,-0.326) (0.0017,-0.657) (0.001,-0.992)};
\end{axis}
\end{tikzpicture}
\end{center}
```

* 300 examples, 164 degree-3 features, five coefficients; regularization is **stronger to the right** in both plots
* Ridge: every weight shrinks smoothly. Lasso: weights become exactly 0 one after the other; non-zero weights: $\alpha = 1$: 0 of 164, $0.07$: 7, $0.014$: 20, $0.001$: 44

## Choosing $\lambda$

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=4cm, width=0.7\textwidth, xmode=log, ymode=log, xmin=0.01, xmax=10000, ymin=0.15, ymax=40, xlabel={ridge ($L_2$) $\alpha$ (log, stronger $\rightarrow$)}, ylabel={MSE (log)}, legend pos=north east]
  \addplot[cblue, mark=*, mark size=1pt] coordinates {(0.01,0.21) (0.0316,0.234) (0.1,0.262) (0.316,0.288) (1,0.312) (3.16,0.333) (10,0.356) (31.6,0.387) (100,0.43) (316,0.48) (1e3,0.541) (3.16e3,0.647) (1e4,0.878)}; \addlegendentry{train}
  \addplot[cred, mark=*, mark size=1pt] coordinates {(0.01,21.8) (0.0316,15.3) (0.1,5.21) (0.316,1.06) (1,0.454) (3.16,0.476) (10,0.469) (31.6,0.461) (100,0.495) (316,0.551) (1e3,0.59) (3.16e3,0.615) (1e4,0.751)}; \addlegendentry{test}
  \draw[cgray, dashed] (axis cs:1,0.15) -- (axis cs:1,40);
\end{axis}
\end{tikzpicture}
\end{center}
```

* A U-shaped **validation** curve: too small $\alpha$ overfits (test 21.8), too large underfits (test 0.75); training error rises monotonically
* Choose $\lambda$ on a **validation set or by cross-validation** (`RidgeCV`, `LassoCV`); the test set is touched once
* Search on a **logarithmic grid** ($10^{-4}, \dots, 10^{4}$); refine around the best value

## Conventions: $\lambda$ and scikit-learn

| Model | scikit-learn minimizes | Our $\lambda$ (MSE = a mean) |
|:----------|:--------------------------------------------------------|:-----------------|
| Ridge | $\lVert y - Xw\rVert^2 + \alpha\lVert w\rVert^2$ | $\alpha = \lambda N$ |
| Lasso | $\frac{1}{2N}\lVert y - Xw\rVert^2 + \alpha\lVert w\rVert_1$ | $\alpha = \lambda/2$ |
| Logistic | $C\sum_i \ell_i + \frac{1}{2}\lVert w\rVert^2$ | $C = 1/(\lambda N)$ (loss $+\ \frac{\lambda}{2}\lVert w\rVert^2$) |

* The same idea, three scalings: a mismatch is the usual reason two "identical" models disagree
* Lab: after mapping $\lambda$, our ridge equals `Ridge` to **$3\cdot10^{-16}$** and our logistic equals `LogisticRegression` to **$2\cdot10^{-6}$**
* Lasso weights differ from scikit-learn's by up to 0.15--0.20 while the **objective differs by $10^{-5}$**: nearly collinear features have many equally good solutions

## Result: Taming a Degree-3 Model

| Model (164 features, 300 examples) | Test MSE | Non-zero weights |
|:--------------------------------|--------:|--------:|
| plain least squares | 159.8 | 164 |
| ridge ($\alpha = 1$, best on test) | **0.454** | 164 |
| ridge, $\alpha$ by 5-fold CV (31.6) | 0.461 | 164 |
| lasso ($\alpha = 0.0203$, best on test) | 0.494 | **15** |
| lasso, $\alpha$ by 5-fold CV (0.01) | 0.547 | 19 |
| elastic net ($\rho = 0.5$) | 0.535 | 39 |
| *(for reference) the plain line, 8 features* | *0.467* | *8* |

* Regularization turns a disastrous model into a good one; the lasso does it with **15** of 164 features
* Choosing $\alpha$ on the test set (the "best" rows) is optimistic: the CV rows are the honest numbers

## Live Demo: Ridge, Lasso, Elastic Net

```{=latex}
\begin{center}
\vspace{8mm}
{\Huge\bfseries DEMO R5:}\\[5mm]
{\LARGE\ttfamily 01\_linear\_regression.ipynb}
\end{center}
```

# The Perceptron

## A Neuron That Draws a Line

```{=latex}
\begin{center}
\begin{tikzpicture}[node distance=9mm]
  \foreach \i/\y in {1/1.2, 2/0.4, 3/-0.4} { \node[dot=cgray, minimum size=8pt, label=left:{$x_\i$}] (x\i) at (0,\y) {}; }
  \node[circle, draw, thick, fill=cblue!15, minimum size=1.2cm] (s) at (3.2,0.4) {$\sum$};
  \node[note, above=0mm of s] {$w^\top x + b$};
  \node[fillbox=corange, right=12mm of s, text width=1.6cm] (h) {sign};
  \node[right=8mm of h] (o) {$\hat y \in \{-1, +1\}$};
  \foreach \i in {1,2,3} { \draw[flow] (x\i) -- node[above, note, pos=0.4] {$w_\i$} (s); }
  \draw[flow] (s) -- (h); \draw[flow] (h) -- (o);
\end{tikzpicture}
\end{center}
```

$$\hat y = \text{sign}(w^\top x + b) \qquad y \in \{-1, +1\}$$

* The first learning machine (Rosenblatt, 1958): a weighted vote, then a hard decision
* The **decision boundary** $w^\top x + b = 0$ is a hyperplane; $w$ is its normal, pointing to the $+1$ side
* The **margin** $m = y\,(w^\top x + b)$ is positive when the point is correctly classified
* It is the building block of neural networks (Class 05)

## The Perceptron Loss and Its Update

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=3.4cm, width=0.7\textwidth, xmin=-3, xmax=3, ymin=-0.3, ymax=3.3, xlabel={margin $m = y\,(w^\top x + b)$}, ylabel={loss}]
  \addplot[cred, domain=-3:0] {-x}; \addplot[cred, domain=0:3] {0};
  \addplot[cgray, dashed] coordinates {(-3,1) (0,1) (0.001,0) (3,0)};
  \node[cred, font=\scriptsize, anchor=south west] at (axis cs:0.25,0.1) {perceptron loss};
  \node[cgray, font=\scriptsize, anchor=south west] at (axis cs:0.25,1.15) {0/1 loss (dashed)};
\end{axis}
\end{tikzpicture}
\end{center}
```

$$\ell = \max(0,\ -m), \qquad \nabla_w \ell = \begin{cases} -y\,x & m \le 0\ \text{(a mistake)} \\ 0 & m > 0\end{cases}$$

* A gradient step of length 1: on a mistake, $w \leftarrow w + y\,x$ and $b \leftarrow b + y$; otherwise nothing
* **The perceptron rule is a subgradient step** on this loss: in the lab, `jax.grad` of one example *is* the update


## Worked Example: the Perceptron by Hand

:::: {.columns}
::: {.column width="58%"}
Points $(2,1)^+,\ (1,2)^+,\ (0,-2)^-,\ (0,1)^-$; $\ w = (0,0),\ b = 0$

| epoch | point | $y$ | score | new $w$ | new $b$ |
|:-:|:--|:-:|--:|:--|:-:|
| 1 | $(2,1)$ | $+$ | 0 | $(2,1)$ | 1 |
| 1 | $(0,1)$ | $-$ | 2 | $(2,0)$ | 0 |
| 2 | $(0,-2)$ | $-$ | 0 | $(2,2)$ | $-1$ |
| 2 | $(0,1)$ | $-$ | 1 | $(2,1)$ | $-2$ |
| 3 | all four | | | no change | |
:::
::: {.column width="42%"}
```{=latex}
\begin{tikzpicture}
\begin{axis}[faa, height=4.4cm, width=\textwidth, xmin=-1.5, xmax=3, ymin=-3, ymax=3, xlabel={$x_1$}, ylabel={$x_2$}]
  \addplot[cblue, domain=-1.5:3] {2-2*x};
  \addplot[only marks, mark=*, cred, mark size=2.4pt] coordinates {(2,1) (1,2)};
  \addplot[only marks, mark=square*, cblue, mark size=2.4pt] coordinates {(0,-2) (0,1)};
\end{axis}
\end{tikzpicture}
```
:::
::::

* Only the mistakes are listed (a **score** of 0 counts as a mistake); the other visits change nothing
* Final boundary $2x_1 + x_2 - 2 = 0$: all four points are correct, so the algorithm stops

## Convergence, and When It Fails

| Data | Mistakes per epoch | Outcome |
|:--------------------|:----------------------------------|:-------------|
| separable blobs | 2, 0, 0, 0, ... | **converges** |
| overlapping blobs | 20, 14, 14, 14, 11, 14, 10, 14, ... | never stops |
| XOR, raw features | 98, 100, 98, 92, 96, ... | accuracy 0.42 |
| XOR, degree-2 features | 16, 2, 0, 0, ... | **converges** |

* **Novikoff (1962):** on linearly separable data with margin $\gamma$ and $\lVert x\rVert \le R$, at most $(R/\gamma)^2$ updates, whatever the order
* On overlapping data it cycles: the final weights are arbitrary, and there is no "best" line in sight
* The exact behaviour is shared with scikit-learn: our `jax.grad` perceptron and `Perceptron(penalty=None, eta0=1, shuffle=False)` give **identical** weights (difference **0.0**)

## XOR: Beyond a Straight Line

```{=latex}
\begin{center}
\begin{tikzpicture}
\pgfplotsset{sm/.style={faa, width=0.36\textwidth, height=3.6cm, xmin=-1.2, xmax=1.2, ymin=-1.2, ymax=1.2, xtick=\empty, ytick=\empty, title style={font=\scriptsize}}}
\begin{axis}[sm, at={(0,0)}, title={raw $(x_1, x_2)$: no line works}]
  \addplot[only marks, mark=*, cred, mark size=2pt] coordinates {(-0.7,0.6) (-0.4,0.9) (-0.9,0.3) (0.6,-0.7) (0.9,-0.4) (0.3,-0.9)};
  \addplot[only marks, mark=square*, cblue, mark size=2pt] coordinates {(0.7,0.6) (0.4,0.9) (0.9,0.3) (-0.6,-0.7) (-0.9,-0.4) (-0.3,-0.9)};
\end{axis}
\begin{axis}[sm, at={(6.3cm,0)}, title={with $x_1 x_2$: a threshold works}, xlabel={$x_1$}, ylabel={$x_1 x_2$}, xtick=\empty, ytick=\empty, ymin=-1, ymax=1]
  \addplot[only marks, mark=*, cred, mark size=2pt] coordinates {(-0.7,-0.42) (-0.4,-0.36) (-0.9,-0.27) (0.6,-0.42) (0.9,-0.36) (0.3,-0.27)};
  \addplot[only marks, mark=square*, cblue, mark size=2pt] coordinates {(0.7,0.42) (0.4,0.36) (0.9,0.27) (-0.6,0.42) (-0.9,0.36) (-0.3,0.27)};
  \addplot[cgray, dashed, no marks] coordinates {(-1.2,0) (1.2,0)};
\end{axis}
\end{tikzpicture}
\end{center}
```

* XOR: class 1 when the coordinates have **different signs** (Minsky and Papert, 1969)
* Raw features: chance level (accuracy **0.42**); adding $x_1 x_2$ makes it **separable**
* Same for logistic regression: accuracy 0.655 on raw XOR, **1.0** with degree-2 features (lab)
* Learning the features instead is the hidden layer (Class 05)

## The Perceptron on Spam

* One pass over the 3680 training e-mails (log frequencies, standardized), a step of 1:
  * **394 mistakes** in the epoch (10.7%)
  * test accuracy **0.912**
* Fast, online, no probabilities, no learning rate to tune
* Sensitive to the **order** of the examples and to feature scale (a large feature dominates the update)
* No notion of *how confident* a prediction is: a point just inside the boundary looks the same as one far away
* Its smoother cousin, next: **logistic regression**, which turns the same score $w^\top x + b$ into a probability

## Live Demo: The Perceptron

```{=latex}
\begin{center}
\vspace{8mm}
{\Huge\bfseries DEMO P1:}\\[5mm]
{\LARGE\ttfamily 02\_perceptron.ipynb}
\end{center}
```

# Generalized Linear Models

## Generalized Linear Models: Three Parts

$$g\big(\mathbb{E}[y \mid x]\big) = w^\top x + b$$

| Part | Role | Linear regression |
|:--------------------|:------------------------------------|:-----------------|
| linear predictor | $\eta = w^\top x + b$: what we learn | the same |
| **link** $g$ | connects the mean of $y$ to $\eta$ | identity |
| **distribution** of $y$ | how $y$ varies around its mean (exponential family) | Gaussian |

* We already know one member: linear regression is a Gaussian $y$ with the identity link
* A **GLM** keeps the linear predictor and changes the other two parts to fit the *type* of $y$: a probability, a count, a class
* The loss is always the **negative log-likelihood** of the chosen distribution (Class 04 derives it)

## Why a Link? The Mean Must Stay in Its Range

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, width=3.6cm, height=2.8cm, xmin=-4, xmax=4, ymin=-4, ymax=4, xlabel={$\eta$}, title={\scriptsize identity: $\mu = \eta$}, xtick={-4,0,4}, ytick={-4,0,4}]
  \addplot[cblue, domain=-4:4] {x};
\end{axis}
\begin{axis}[faa, at={(4.1cm,0)}, width=3.6cm, height=2.8cm, xmin=-4, xmax=4, ymin=0, ymax=1, xlabel={$\eta$}, title={\scriptsize logit link: $\mu = \sigma(\eta)$}, xtick={-4,0,4}, ytick={0,0.5,1}]
  \addplot[cred, domain=-4:4, samples=80] {1/(1+exp(-x))};
\end{axis}
\begin{axis}[faa, at={(8.2cm,0)}, width=3.6cm, height=2.8cm, xmin=-4, xmax=4, ymin=0, ymax=8, xlabel={$\eta$}, title={\scriptsize log link: $\mu = e^{\eta}$}, xtick={-4,0,4}, ytick={0,4,8}]
  \addplot[cgreen, domain=-4:4, samples=80] {exp(x)};
\end{axis}
\end{tikzpicture}
\end{center}
```

| Model | $y \mid x$ | Link $g$ | Mean $\mu$ lives in |
|:--------------------|:-----------|:---------|:----------------|
| Linear regression | Gaussian | identity | $\mathbb{R}$ |
| Logistic regression | Bernoulli | logit | $(0, 1)$ |
| Poisson regression | Poisson | $\log$ | $(0, \infty)$ |

* The score $\eta$ is any real number; the **inverse link** maps it to a valid mean (a probability, a rate)

## The Canonical Link: One Gradient for the Family

$$\mathcal{L} = -\tfrac{1}{N}\sum_i \log p(y_i \mid x_i) \quad\Rightarrow\quad \nabla_w\,\mathcal{L} = \tfrac{1}{N}\, X^\top(\hat\mu - y)$$

* With the **canonical link** (identity, logit, log) the gradient is always *features $\times$ residual*: the same JAX code serves the whole family
* Linear regression: $\hat\mu = \hat y$, residual $\hat y - y$ (the gradient seen before); convex in $w$ for every canonical-link GLM
* scikit-learn: `PoissonRegressor`, `GammaRegressor`, `TweedieRegressor` (and the two we know)
* The **perceptron** has no probability model: it is *not* a GLM

# Logistic Regression

## From a Score to a Probability

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=3.8cm, width=0.62\textwidth, xmin=-4, xmax=4, ymin=-0.6, ymax=1.6, xlabel={score $z = w^\top x + b$}, height=3.2cm, legend style={at={(0.02,0.98)}, anchor=north west}]
  \addplot[cgray, dashed, domain=-4:4] {0.5 + 0.22*x}; \addlegendentry{linear regression on labels}
  \addplot[cblue, domain=-4:4, samples=100] {1/(1+exp(-x))}; \addlegendentry{$\sigma(z)$}
  \addplot[only marks, mark=*, black, mark size=1.4pt] coordinates {(-3.2,0) (-2.5,0) (-1.8,0) (-1.1,0) (-0.5,0) (0.6,1) (1.2,1) (1.9,1) (2.6,1) (3.3,1)};
\end{axis}
\end{tikzpicture}
\end{center}
```

$$p(y = 1 \mid x) = \sigma(w^\top x + b), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}$$

* A line fitted to 0/1 labels leaves $[0, 1]$: not a probability (schematic data)
* The GLM for a yes/no $y$: **Bernoulli** distribution with the **logit link**
* The sigmoid maps any score into $(0, 1)$; the boundary $p = 0.5$ is still $w^\top x + b = 0$; the **threshold** is a choice

## Odds and Log-Odds

| $p$ | odds $\frac{p}{1-p}$ | log-odds $z$ |
|:-:|--:|--:|
| 0.01 | 0.010 | $-4.60$ |
| 0.10 | 0.111 | $-2.20$ |
| 0.50 | 1 | $0$ |
| 0.90 | 9 | $2.20$ |
| 0.99 | 99 | $4.60$ |

$$\log\frac{p}{1-p} = w^\top x + b$$

* The model is **linear in the log-odds**: each weight adds evidence for or against spam
* $e^{w_j}$ is the **odds ratio**: the odds get multiplied by $e^{w_j}$ when feature $j$ grows by one standard deviation
* Spam: `char_freq_!` $+0.74$ ($\times 2.09$). Ham: `word_freq_hp` $-1.37$ ($\times 0.25$)

## Cross-Entropy: Maximum Likelihood for Labels

Each label is a Bernoulli variable with $P(y = 1) = p_i$; the likelihood of the data and its negative log:

$$-\log \prod_i p_i^{y_i}(1-p_i)^{1-y_i} = \sum_i \ell_i, \qquad \ell_i = -y_i \log p_i - (1 - y_i)\log(1 - p_i)$$

$$\ell_i = \log\big(1 + e^{z_i}\big) - y_i z_i \qquad (z_i = w^\top x_i + b)$$

* The **binary cross-entropy**: the code length of the labels (Class 01)
* With the logit $z$ and `jnp.logaddexp(0, z)` it never overflows
* Confident and right: $\ell \approx 0$; confident and **wrong**: $\ell$ grows like $|z|$

## Why Not the Squared Error?

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=3.5cm, width=0.42\textwidth, xmin=-6, xmax=6, ymin=0, ymax=3.2, xlabel={logit $z$}, title={\scriptsize loss for $y = 1$}, title style={yshift=-1mm}, legend style={at={(0.98,0.98)}, anchor=north east, font=\tiny}]
  \addplot[cblue, domain=-6:6, samples=100] {ln(1+exp(-x))}; \addlegendentry{cross-entropy}
  \addplot[cred, domain=-6:6, samples=100] {(1/(1+exp(-x))-1)^2}; \addlegendentry{squared error}
\end{axis}
\end{tikzpicture}
\begin{tikzpicture}
\begin{axis}[faa, height=3.5cm, width=0.42\textwidth, xmin=-6, xmax=6, ymin=-1.1, ymax=0.3, xlabel={logit $z$}, title={\scriptsize gradient $\partial \ell / \partial z$}, title style={yshift=-1mm}, legend style={at={(0.98,0.02)}, anchor=south east}]
  \addplot[cblue, domain=-6:6, samples=100] {-(1-1/(1+exp(-x)))};
  \addplot[cred, domain=-6:6, samples=100] {2*(1/(1+exp(-x))-1)*(1/(1+exp(-x)))*(1-1/(1+exp(-x)))};
\end{axis}
\end{tikzpicture}
\end{center}
```

* For a confidently **wrong** prediction ($z \ll 0$) the cross-entropy gradient is $-1$: a strong push
* The squared error's gradient **vanishes** there: the model is stuck exactly when it is most wrong
* The cross-entropy is **convex** in $(w, b)$; the squared error on a sigmoid is not

## The Gradient Is a Residual Again

$$\nabla_w \mathcal{L} = \frac{1}{N} X^\top (p - y), \qquad \frac{\partial \mathcal{L}}{\partial b} = \frac{1}{N}\sum_i (p_i - y_i)$$

Worked example: one e-mail, feature $x = 1.5$, weights $w = 1.2$, $b = -0.5$

| | $y = 1$ (spam) | $y = 0$ (ham) |
|:--|--:|--:|
| score $z = 1.2 \cdot 1.5 - 0.5$ | 1.300 | 1.300 |
| probability $p = \sigma(z)$ | 0.786 | 0.786 |
| loss $-\log p_y$ | **0.241** | **1.541** |
| $\partial \ell / \partial w = (p - y)\,x$ | $-0.321$ | $+1.179$ |
| $\partial \ell / \partial b = p - y$ | $-0.214$ | $+0.786$ |

* The canonical-link result of the GLM section, for the Bernoulli: (prediction $-$ target) $\times$ feature. Not typed in the lab: `jax.grad` produces it

## Training It: Convex but No Closed Form

* $\nabla \mathcal{L} = 0$ is **nonlinear** in $w$: no closed form. The loss is convex: GD, Adam or L-BFGS reach the global minimum
* Newton's method on it is IRLS (iteratively reweighted least squares)
* **Separable data:** the loss keeps decreasing as $\lVert w\rVert \to \infty$. **Regularization fixes it**

$$\mathcal{L}_\lambda = \frac{1}{N}\sum_i \ell_i + \frac{\lambda}{2}\lVert w\rVert^2, \qquad C = \frac{1}{\lambda N}$$

* Spam ($\lambda = 10^{-3}$): our weights equal scikit-learn's to **$2\cdot10^{-6}$**

## Result: Spam Detection

| Model | Accuracy | Precision | Recall | $F_1$ | MCC |
|:------------------------|-----:|-----:|-----:|-----:|-----:|
| majority class ("all ham") | 0.606 | 0 | 0 | 0 | 0 |
| perceptron, 1 epoch | 0.912 | 0.875 | 0.906 | 0.890 | 0.817 |
| logistic regression ($C = 0.1$) | **0.937** | **0.913** | **0.928** | **0.921** | **0.869** |

* Test set: 921 e-mails (363 spam). Log frequencies, standardized with the training statistics
* The logistic model is better than a single pass of the perceptron, *and* it outputs a probability per e-mail
* 92.8% of the spam is caught; **8.7%** of the e-mails flagged as spam are actually ham: the price of the default threshold of 0.5
* Next: what the numbers mean, and how to choose the threshold

## Non-Linear Boundaries: Features and Regularization

```{=latex}
\begin{center}
\begin{tikzpicture}
\pgfplotsset{mo/.style={faa, width=0.30\textwidth, height=3.5cm, xmin=-1.7, xmax=2.7, ymin=-1.3, ymax=1.9, xtick=\empty, ytick=\empty, title style={font=\scriptsize, align=center}}}
\def\mzero{(0.695,0.427) (-1.13,-0.611) (0.912,0.47) (0.75,0.34) (-0.398,0.297) (-0.824,1.02) (1.01,0.149) (0.746,0.987) (0.179,0.718) (-1.16,1.37) (0.488,1.03) (-0.0355,1.43) (-0.0366,0.454) (0.976,0.705) (-0.5,0.121) (0.755,0.819) (0.811,-0.0376) (-0.691,0.714) (-0.133,0.826) (0.527,0.601) (-0.162,1.18) (-0.697,0.478) (-1.03,0.927) (-0.439,0.0263) (0.902,0.491) (0.398,0.345) (0.0105,0.88) (-0.221,1.06) (-1.12,0.502) (1.13,0.349) (-1.54,0.318) (-0.241,1.14) (0.419,0.873) (-1.04,0.736) (-1.15,1.03) (-0.939,0.703) (1.04,0.31) (0.367,0.918) (-0.677,0.887)}
\def\mone{(1.68,-0.8) (-0.25,0.244) (1.77,-0.31) (2,-0.186) (0.882,-0.467) (1.3,-0.723) (-0.0105,-0.368) (1.78,-0.45) (1.59,0.00796) (-0.0115,-0.233) (2.19,0.00383) (0.877,-0.41) (1.25,0.0502) (0.178,0.376) (2.07,-0.122) (1.79,-0.0814) (-0.175,0.197) (0.817,-0.288) (-0.0473,0.378) (1.72,0.331) (0.645,0.0984) (-0.338,0.59) (0.692,-0.642) (0.197,-0.0921) (1.52,-0.378) (0.174,-0.443) (1.21,-0.363) (0.586,-0.227) (2.44,0.0221) (1.35,-0.502) (0.807,-0.63) (-0.122,-0.0811) (1.48,-0.544) (1.05,-0.335)}
\begin{axis}[mo, at={(0,0)}, title={degree 1\\ train 0.84, test 0.848}]
  \addplot[only marks, mark=*, cred, mark size=1pt] coordinates {\mzero}; \addplot[only marks, mark=square*, cblue, mark size=1pt] coordinates {\mone};
  \addplot[black, thick, no marks] coordinates {(-1.7,-0.474) (-0.44,-0.046) (0.87,0.399) (2.7,1.02)};
\end{axis}
\begin{axis}[mo, at={(3.9cm,0)}, title={degree 15, $C = 10^6$\\ train 0.955, test 0.859}]
  \addplot[only marks, mark=*, cred, mark size=1pt] coordinates {\mzero}; \addplot[only marks, mark=square*, cblue, mark size=1pt] coordinates {\mone};
  \addplot[black, thick, no marks] coordinates {(1.48,2) (1.3,1.7) (1.08,1.23) (1.17,0.866) (1.33,0.558) (1.27,0.22) (1.18,0.0653) (1.43,-0.181) (1.47,-0.312) (1.09,-0.159) (0.773,0.0477) (0.591,0.347) (0.462,0.461) (0.0854,0.402) (-0.138,0.681) (-0.475,0.663) (-0.277,0.382) (-0.406,0.0477) (-0.537,-0.286) (-0.616,-0.656) (-0.266,-0.74) (0.135,-0.585) (0.354,-0.796) (0.324,-1.41)};
\end{axis}
\begin{axis}[mo, at={(7.8cm,0)}, title={degree 15, $C = 0.1$\\ train 0.860, test 0.876}]
  \addplot[only marks, mark=*, cred, mark size=1pt] coordinates {\mzero}; \addplot[only marks, mark=square*, cblue, mark size=1pt] coordinates {\mone};
  \addplot[black, thick, no marks] coordinates {(1.18,2) (1.12,1.67) (1.19,1.35) (1.34,1.08) (1.49,0.804) (1.52,0.597) (1.42,0.428) (1.22,0.328) (0.94,0.273) (0.613,0.244) (0.312,0.221) (0.0101,0.18) (-0.241,0.116) (-0.467,0.0256) (-0.666,-0.093) (-0.819,-0.227) (-0.935,-0.392) (-0.991,-0.603) (-0.965,-0.832) (-0.894,-1.04) (-0.816,-1.22) (-0.732,-1.41)};
\end{axis}
\end{tikzpicture}
\end{center}
```

* "Two moons", 200 noisy training points (a sample is drawn), 2000 test points; polynomial features + logistic regression
* Unregularized degree 15: an intricate boundary that fits the noise (train 0.955, test 0.859); with $C = 0.1$ it stays smooth and **generalizes better** (test 0.876)
* The best unregularized model here is degree 3 (test 0.887): flexibility and constraint both need tuning

## More Than Two Classes: One Class Against the Rest

* A logistic regression is **binary**. For $K$ classes, train $K$ binary models: "class $k$ **against all the others**"
* Predict $\hat k = \arg\max_k\, p_k(x)$ (separate models: the $p_k$ need not sum to 1)
* scikit-learn: `OneVsRestClassifier(LogisticRegression())`; the same wrapper works for the perceptron
* Alternative: the **multinomial** logit (softmax) is the GLM for a categorical $y$: one weight vector per class, fitted *together*, probabilities sum to 1 (neural-network outputs, Class 05)

## Live Demo: Logistic Regression

```{=latex}
\begin{center}
\vspace{8mm}
{\Huge\bfseries DEMO L1:}\\[5mm]
{\LARGE\ttfamily 03\_logistic\_regression.ipynb}
\end{center}
```

# Classification Metrics

## The Confusion Matrix

```{=latex}
\begin{center}
\begin{tikzpicture}
  \node[font=\scriptsize] at (2.5,2.55) {\textbf{predicted}};
  \node[font=\scriptsize, rotate=90] at (-1.35,0.8) {\textbf{actual}};
  \node[font=\scriptsize] at (1.5,2.15) {spam}; \node[font=\scriptsize] at (3.5,2.15) {ham};
  \node[font=\scriptsize, anchor=east] at (0.45,1.3) {spam}; \node[font=\scriptsize, anchor=east] at (0.45,0.3) {ham};
  \draw[fill=cgreen!45] (0.5,0.8) rectangle (2.5,1.8); \node[align=center, font=\scriptsize] at (1.5,1.3) {\textbf{TP} = 337\\ spam caught};
  \draw[fill=cred!35] (2.5,0.8) rectangle (4.5,1.8); \node[align=center, font=\scriptsize] at (3.5,1.3) {\textbf{FN} = 26\\ spam missed};
  \draw[fill=corange!40] (0.5,-0.2) rectangle (2.5,0.8); \node[align=center, font=\scriptsize] at (1.5,0.3) {\textbf{FP} = 32\\ ham hidden};
  \draw[fill=cgreen!25] (2.5,-0.2) rectangle (4.5,0.8); \node[align=center, font=\scriptsize] at (3.5,0.3) {\textbf{TN} = 526\\ ham delivered};
  \node[note, anchor=west, text width=4.2cm] at (5.2,0.8) {921 test e-mails\\ 363 spam, 558 ham\\ (positive class: spam)};
\end{tikzpicture}
\end{center}
```

* Every metric of this section is a function of these **four counts** (the logistic model at threshold 0.5)
* The two errors are **not** equally bad: a false positive hides a real e-mail, a false negative shows one spam
* Multi-class: an $K \times K$ matrix; the diagonal is correct

## Accuracy, Precision and Recall

$$\text{accuracy} = \frac{TP + TN}{n} = \frac{337 + 526}{921} = 0.937$$

$$\text{precision} = \frac{TP}{TP + FP} = \frac{337}{369} = 0.913$$

$$\text{recall} = \frac{TP}{TP + FN} = \frac{337}{363} = 0.928$$

* **Accuracy:** the fraction of correct decisions; misleading when the classes are imbalanced
* **Precision:** of the e-mails flagged as spam, how many really are? (cost of a false positive)
* **Recall** (sensitivity, true-positive rate): of the spam, how much was caught? (cost of a false negative)


## The Accuracy Paradox

| Test set: all 558 ham + 11 spam (1.9% spam) | Accuracy | Precision | Recall | $F_1$ | MCC |
|:-------------------------|-----:|-----:|-----:|-----:|-----:|
| "everything is ham" | **0.981** | 0 | 0 | 0 | 0 |
| our classifier | 0.942 | 0.238 | 0.909 | 0.377 | 0.449 |

* The useless filter has the **higher accuracy**: it never fails on the 98% that are ham
* Our classifier finds **10 of 11** spam, at the price of 32 false alarms (precision 0.238)
* Class 01: "the metric must match the goal". With rare positives, look at precision, recall and MCC, never accuracy alone
* Also: always compare with the **majority-class baseline** (0.606 on the balanced-ish spam data)

## $F_1$: One Number for Precision and Recall

$$F_1 = \frac{2PR}{P + R} = \frac{2\,TP}{2\,TP + FP + FN} = \frac{674}{732} = 0.921$$

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=3.4cm, width=0.5\textwidth, xmin=0, xmax=1, ymin=0, ymax=1, xlabel={precision}, ylabel={recall}, title={\scriptsize $F_1$ contours}, title style={yshift=-1mm}, xtick={0,0.5,1}, ytick={0,0.5,1}]
  \foreach \f in {0.3,0.5,0.7,0.9} { \edef\tmp{\noexpand\addplot[cblue!70, domain={\f/(2-\f)}:1, samples=40] {\f*x/(2*x-\f)};}\tmp }
  \node[font=\tiny, cblue] at (axis cs:0.93,0.75) {0.9}; \node[font=\tiny, cblue] at (axis cs:0.85,0.4) {0.7};
\end{axis}
\end{tikzpicture}
\end{center}
```

* The **harmonic mean**: low if *either* is low (precision 1, recall 0.01 gives $F_1 = 0.02$, not 0.5)
* Ignores the **true negatives**: $F_1$ can look fine for a model that is useless on the negative class

## MCC: the Whole Matrix in One Number

$$\text{MCC} = \frac{TP\cdot TN - FP\cdot FN}{\sqrt{(TP{+}FP)(TP{+}FN)(TN{+}FP)(TN{+}FN)}}$$

Ours: $\dfrac{337\cdot 526 - 32\cdot 26}{\sqrt{369 \cdot 363 \cdot 558 \cdot 552}} = \dfrac{176\,430}{203\,120} = 0.869$

* The correlation of predicted and actual labels: $+1$ perfect, $0$ chance, $-1$ total disagreement
* Uses all four cells: **symmetric** in the classes, robust to imbalance

| "Everything is spam" (test set, 39.4% spam) | Value |
|:---------------------------|-----:|
| accuracy | 0.394 |
| precision / recall | 0.394 / 1.000 |
| $F_1$ | **0.565** |
| MCC | **0.000** |

* $F_1$ = 0.565 flatters a useless filter; MCC = 0 tells the truth

## The Threshold Is a Decision

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=4.1cm, width=0.72\textwidth, xmin=0, xmax=1, ymin=0.3, ymax=1.02, xlabel={decision threshold on $P(\text{spam})$}, legend style={at={(0.5,0.02)}, anchor=south, legend columns=4}]
  \addplot[corange, mark=none] coordinates {(0.02,0.565) (0.08,0.726) (0.14,0.792) (0.2,0.831) (0.26,0.865) (0.32,0.886) (0.38,0.909) (0.44,0.91) (0.5,0.913) (0.56,0.931) (0.62,0.951) (0.68,0.959) (0.74,0.963) (0.8,0.965) (0.86,0.965) (0.92,0.98) (0.98,0.988)}; \addlegendentry{precision}
  \addplot[cblue, mark=none] coordinates {(0.02,0.997) (0.08,0.986) (0.14,0.975) (0.2,0.972) (0.26,0.972) (0.32,0.967) (0.38,0.964) (0.44,0.942) (0.5,0.928) (0.56,0.923) (0.62,0.904) (0.68,0.898) (0.74,0.868) (0.8,0.829) (0.86,0.769) (0.92,0.664) (0.98,0.446)}; \addlegendentry{recall}
  \addplot[cgreen, mark=none] coordinates {(0.02,0.721) (0.08,0.836) (0.14,0.874) (0.2,0.896) (0.26,0.916) (0.32,0.925) (0.38,0.936) (0.44,0.926) (0.5,0.921) (0.56,0.927) (0.62,0.927) (0.68,0.927) (0.74,0.913) (0.8,0.892) (0.86,0.856) (0.92,0.791) (0.98,0.615)}; \addlegendentry{$F_1$}
  \addplot[cred, mark=none] coordinates {(0.02,0.528) (0.08,0.729) (0.14,0.791) (0.2,0.827) (0.26,0.86) (0.32,0.875) (0.38,0.893) (0.44,0.876) (0.5,0.869) (0.56,0.879) (0.62,0.881) (0.68,0.884) (0.74,0.864) (0.8,0.836) (0.86,0.791) (0.92,0.723) (0.98,0.565)}; \addlegendentry{MCC}
\end{axis}
\end{tikzpicture}
\end{center}
```

* Raising the threshold: fewer e-mails flagged, **precision up, recall down**. Lowering it: the reverse
* $F_1$ and MCC peak around 0.4--0.7; the best threshold for *our costs* may be far from it

## Choosing the Threshold

| Threshold | Precision | Recall | $F_1$ | MCC | Ham hidden |
|:-:|--:|--:|--:|--:|--:|
| 0.50 | 0.913 | 0.928 | 0.921 | 0.869 | 32 |
| 0.90 | 0.973 | 0.705 | 0.818 | 0.749 | 7 |
| 0.99 | 0.993 | 0.386 | 0.556 | 0.521 | **1** |

* If hiding a real e-mail costs, say, 20 times a missed spam, the third row is the sensible one, at the price of **61% of the spam** getting through
* **Choose the threshold on validation data** with the costs in hand; the test set only reports the result (in the lab: precision 0.99 on the *training* data gives 0.974 on test)
* Threshold-free summaries: the **ROC curve** (recall against false-positive rate) and the **precision-recall curve**; AUC of ours: **0.981**
* Prefer PR curves when positives are rare

## Which Metric When?

| Situation | Look at | Why |
|:------------------------------------|:------------------------|:--------------------------------|
| balanced classes, equal costs | accuracy, MCC | simple, all cells |
| rare positives | precision, recall, PR curve | accuracy is inflated by negatives |
| a missed positive is costly (screening) | **recall** | do not miss sick patients |
| a false alarm is costly (spam filter) | **precision** | do not hide real e-mails |
| one number, any imbalance | **MCC** | uses TP, TN, FP, FN |
| ranking quality, threshold not fixed | ROC AUC, average precision | independent of the threshold |

* Regression counterpart: MSE/RMSE for quadratic costs, MAE for robust, sMAPE for relative, $R^2$ against the mean
* Whatever the metric: baseline, spread over folds, and a **test set used once**

## Live Demo: Classification Metrics

```{=latex}
\begin{center}
\vspace{8mm}
{\Huge\bfseries DEMO C1:}\\[5mm]
{\LARGE\ttfamily 04\_classification\_metrics.ipynb}
\end{center}
```

# Summary

## The Models Side by Side

| | Linear regression | Perceptron | Logistic regression |
|:--------------|:---------------|:---------------|:---------------|
| Output | a number | a class | a probability |
| Loss | squared error | $\max(0, -m)$ | cross-entropy |
| Training | closed form or GD | online updates | GD (convex) |
| Boundary | (hyperplane fit) | linear | linear |
| Assumes | linear mean, Gaussian noise | separability | linear log-odds |
| Result | $R^2 = 0.681$ (housing) | accuracy 0.912 (spam) | accuracy 0.937 (spam) |

* Regularization ($L_2$, $L_1$) applies to every model that has weights; polynomial features make any of them non-linear
* Next class: where the losses come from (**maximum likelihood**) and a different kind of classifier (**Naive Bayes**)

## Choosing and Checking

* **Start simple:** a baseline, then a linear model; add flexibility only if validation asks
* **A probability, not just a class:** logistic regression
* **Many features, few examples:** regularize; lasso to select features
* **Never trust a number without:** a baseline, a spread, a test set used once, the right metric
* **Watch for:** unscaled features, leakage, imbalance, a threshold left at 0.5

```{=latex}
\begin{center}
\begin{tikzpicture}[node distance=5mm]
  \node[fillbox=cgray, text width=1.8cm] (a) {visualize, split, preprocess};
  \node[fillbox=cblue, text width=1.8cm, right=of a] (b) {fit\\ (JAX)};
  \node[fillbox=cgreen, text width=2.2cm, right=of b] (c) {tune $\lambda$, degree, threshold};
  \node[fillbox=corange, text width=2.2cm, right=of c] (d) {test once, right metric};
  \draw[flow] (a) -- (b); \draw[flow] (b) -- (c); \draw[flow] (c) -- (d);
\end{tikzpicture}
\end{center}
```

## Key Takeaways

* **A model is (hypothesis, loss, optimizer):** write the loss in `jax.numpy`, `jax.grad` does the rest
* **Linear regression:** MSE, closed form or GD; **polynomial features** add flexibility (and overfitting)
* **Regularization:** ridge shrinks, lasso selects; $\lambda$ is chosen on validation data
* **GLM:** linear predictor + link + distribution; the canonical-link gradient is always *features $\times$ residual*. **Logistic regression** is the Bernoulli GLM (cross-entropy); the **perceptron** is a subgradient step, not a GLM
* **Metrics:** MSE, MAE, sMAPE, $R^2$; confusion matrix, precision, recall, $F_1$, **MCC**; the threshold is a decision
* **Methodology:** visualize, split once, preprocess on training only, tune by CV, test once, always a baseline

## Next Class: Probabilistic Models

* **Where do the losses come from?** Maximum likelihood: MSE is Gaussian noise, cross-entropy is Bernoulli labels
* **MAP:** a prior on the parameters; ridge is a Gaussian prior, Laplace smoothing a Beta prior
* **Naive Bayes:** Bayes' rule and independence; discrete and continuous features, for classes **and** numbers, trained by counting
* **Probabilities you can trust:** calibration, log-loss, and decisions with costs

## References (1/2)

* G. James, D. Witten, T. Hastie, R. Tibshirani, J. Taylor, *An Introduction to Statistical Learning with Applications in Python*, 2023: chapters 3 (linear regression), 4 (classification), 6 (regularization)
* C. Bishop, *Pattern Recognition and Machine Learning*, 2006: chapters 3 and 4 (linear models)
* P. McCullagh, J. Nelder, *Generalized Linear Models*, 2nd ed., 1989

## References (2/2)

* R. Tibshirani, "Regression shrinkage and selection via the lasso", *JRSS B*, 1996
* H. Zou, T. Hastie, "Regularization and variable selection via the elastic net", *JRSS B*, 2005
* D. Chicco, G. Jurman, "The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation", *BMC Genomics*, 2020
* F. Rosenblatt, "The perceptron: a probabilistic model for information storage and organization in the brain", *Psychological Review*, 1958
* M. Minsky, S. Papert, *Perceptrons*, 1969
