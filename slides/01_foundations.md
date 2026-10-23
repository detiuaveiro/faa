---
title: Fundamentos de Aprendizagem Automática
subtitle: "Class 01 — Learning as Compression, Pitfalls & Taxonomies"
---

# Course and Class

## Professor

:::: {.columns}
::: {.column width="62%"}
* **Mário Antunes**
* Associate Professor, UA
* Researcher, Instituto de Telecomunicações
* `mario.antunes@ua.pt`
* Office: 19.2.15
* Research: machine learning, time series, optimization
* Maintains the Python libraries of this course
:::
::: {.column width="38%"}
![](mantunes.jpg){width=100%}
:::
::::

## A Fundamentals Course

```{=latex}
\begin{center}
\begin{tikzpicture}
  \fill[cblue!20]   (-2.2,0) -- (2.2,0) -- (1.6,1.0) -- (-1.6,1.0) -- cycle;
  \fill[corange!25] (-1.6,1.0) -- (1.6,1.0) -- (1.0,2.0) -- (-1.0,2.0) -- cycle;
  \fill[cgreen!25]  (-1.0,2.0) -- (1.0,2.0) -- (0.4,3.0) -- (-0.4,3.0) -- cycle;
  \node at (0,0.5) {\textbf{Foundations}};
  \node at (0,1.5) {\textbf{Classic models}};
  \node at (0,2.5) {\textbf{Modern}};
  \node[note, font=\small, anchor=west, text width=6.4cm, align=left] at (2.4,0.5) {compression, optimization, evaluation};
  \node[note, font=\small, anchor=west, text width=6.4cm, align=left] at (2.4,1.5) {linear, SVM, neural nets,\\ trees, ensembles};
  \node[note, font=\small, anchor=west, text width=6.4cm, align=left] at (2.4,2.5) {autoencoders, Transformers, JEPA};
\end{tikzpicture}
\end{center}
```

* Understand why learning works before using the latest architectures
* Every model in the course rests on the same foundations

## Course Topics and How They Connect

```{=latex}
\begin{center}
\begin{tikzpicture}
  \node[fillbox=cblue, text width=10.2cm] (f) {\textbf{Foundations}: Classes 01--02\\ compression, optimization, evaluation, learning types};
  \node[fillbox=corange, text width=3.2cm, below=6mm of f.south west, anchor=north west] (s) {\textbf{Supervised}\\ 03 Linear\\ 04 Probabilistic\\ 05 SVM, neural nets\\ 06 Trees, KNN\\ 07 Ensembles};
  \node[fillbox=cgreen, text width=3.2cm, anchor=north west] (u) at ($(s.north east)+(0.3,0)$) {\textbf{Unsupervised}\\ 09 Blind signals\\ 10 Clustering};
  \node[fillbox=cpurple, text width=3.2cm, anchor=north west] (ss) at ($(u.north east)+(0.3,0)$) {\textbf{Semi- and self-supervised}\\ 11 Autoencoders\\ 12 Transformers,\\ JEPA};
  \node[fillbox=cred, text width=10.2cm, below=6mm of s.south west, anchor=north west] (rl) {\textbf{Reinforcement learning}: Class 13, learning from rewards};
  \draw[flow] (f.south -| s.north) -- (s.north);
  \draw[flow] (f.south -| u.north) -- (u.north);
  \draw[flow] (f.south -| ss.north) -- (ss.north);
\end{tikzpicture}
\end{center}
```

* Every later class reuses the foundations of Classes 01--02

## Class Format and Environment

```{=latex}
\begin{center}
\begin{tikzpicture}
  \fill[cblue!60] (0,1.5) rectangle (6.8,2.2); \node[white] at (3.4,1.85) {2 h exposition: slides, live demos};
  \fill[cgreen!60] (6.8,1.5) rectangle (10.2,2.2); \node at (8.5,1.85) {1 h practice};
  \foreach \h in {0,1,2,3} { \draw[cgray] (3.4*\h,1.45) -- (3.4*\h,1.3) node[below, note] {\h\,h}; }
  \node[fillbox=cgray, text width=1.9cm] at (1.0,-0.5) {NumPy\\ JAX};
  \node[fillbox=corange, text width=1.9cm] at (3.05,-0.5) {scikit-learn Keras 3};
  \node[fillbox=cgreen, text width=1.9cm] at (5.1,-0.5) {polars};
  \node[fillbox=cblue, text width=1.9cm] at (7.15,-0.5) {matplotlib seaborn};
  \node[fillbox=cpurple, text width=1.9cm] at (9.2,-0.5) {JupyterLab};
\end{tikzpicture}
\end{center}
```

```bash
git clone https://github.com/detiuaveiro/faa.git && cd faa
export KERAS_BACKEND=jax    # add to ~/.bashrc
make venv && source venv/bin/activate
```

* Schedule, tests and grading: the course README, section 5

# What Is Learning?

## The Core Question and a Classic Definition

```{=latex}
\begin{center}
\begin{tikzpicture}[node distance=9mm]
  \node[fillbox=cgray] (d) {\textbf{Data}\\ past examples};
  \node[fillbox=cblue, right=of d] (l) {\textbf{Learning}\\ algorithm};
  \node[fillbox=corange, right=of l] (m) {\textbf{Model}\\ a function};
  \node[fillbox=cgreen, right=of m] (p) {\textbf{Predictions}\\ new cases};
  \draw[flow] (d) -- (l); \draw[flow] (l) -- (m); \draw[flow] (m) -- (p);
  \node[note, below=5mm of l.south east, text width=9cm, align=center] {A program learns if its performance at a task\\ improves with experience (after T. Mitchell, 1997)};
\end{tikzpicture}
\end{center}
```

* It receives examples from the past
* It must act well on cases it has never seen
* Improvement needs a measure of performance

## Notation N1: Data and Terms

<!-- defines: pattern, noise, training set, loss, variance -->

```{=latex}
\renewcommand{\arraystretch}{1.3}
```

Example 1 is a noisy sine: 50 points with one feature.

| Symbol | Meaning | Example 1 |
|:--------------|:-------------------------------------|:----------------------------|
| $n$, $d$ | number of examples, of features | 50, 1 |
| $X$, $x_i$ | data matrix, example $i$ | $x_1 = 0.774$ |
| $y$, $y_i$, $\hat{y}_i$ | targets and prediction, per example | $y_1 = -0.785$, $\hat{y}_1 = -1.004$ |
| $f$, $\hat{f}$ | true **pattern**, learned estimate: the **model** | $f(x_1) = -0.989$ |
| $\varepsilon$, $\sigma^2$ | **noise**, noise variance: the random part | $\sigma^2 = 0.09$ |
| $\theta$, $m$ | parameters, their number | cubic: $m = 4$ |
| $D$ | dataset: the **training set** of pairs | 50 pairs |
| $\mathcal{L}$ | **loss**: one number that scores the errors | squared error |

## Data = Pattern + Noise

Each target is the pattern plus noise: $y_i = f(x_i) + \varepsilon_i$.

![](plots/01/d1_data.pdf)

* We never see $f$, only the sum; the noise has $\sigma^2 = 0.09$
* Example: $-0.989 + 0.204 = -0.785$ for $i = 1$
* We see $n$ pairs, not the whole process; the goal is new $x$

## Two Models: a Line and a Polynomial

<!-- defines: polynomial, degree -->

```{=latex}
\begin{center}
\begin{tikzpicture}[x=1cm, y=1cm]
  \begin{scope}
    \draw[->] (0,0) -- (2.6,0); \draw[->] (0,0) -- (0,1.5);
    \draw[cblue, very thick, domain=0.1:2.5] plot (\x, {0.2+0.5*\x});
    \node[note, font=\footnotesize] at (1.3,-0.45) {$q = 1$: a line};
  \end{scope}
  \begin{scope}[xshift=3.8cm]
    \draw[->] (0,0) -- (2.6,0); \draw[->] (0,0) -- (0,1.5);
    \draw[corange, very thick, domain=0.1:2.5, samples=30] plot (\x, {1.3-0.45*(\x-1.3)*(\x-1.3)});
    \node[note, font=\footnotesize] at (1.3,-0.45) {$q = 2$: a parabola};
  \end{scope}
  \begin{scope}[xshift=7.6cm]
    \draw[->] (0,0) -- (2.6,0); \draw[->] (0,0) -- (0,1.5);
    \draw[cgreen, very thick, domain=0.1:2.5, samples=40] plot (\x, {0.75+0.55*sin(\x*115)});
    \node[note, font=\footnotesize] at (1.3,-0.45) {$q = 3$: a cubic};
  \end{scope}
\end{tikzpicture}
\end{center}
```

* Line: $\hat{y} = w x + b$, with $m = 2$ parameters $(w, b)$
* Degree $q$: $\hat{y} = \theta_0 + \theta_1 x + \dots + \theta_q x^q$, $m = q + 1$
* A least-squares fit gives the cubic of Example 1 its 4 numbers: $-0.322$, $11.854$, $-33.375$, $21.862$

## Errors and Mean Squared Error

Error $e_i = y_i - \hat{y}_i$; the loss is their mean square.

![](plots/01/d1_errors.pdf)

```{=latex}
\setlength{\abovedisplayskip}{1pt}\setlength{\belowdisplayskip}{1pt}\setlength{\abovedisplayshortskip}{1pt}\setlength{\belowdisplayshortskip}{1pt}
```

<!-- derivation: error-mse -->
$$
\begin{aligned}
\mathcal{L}(\theta) &= \tfrac{1}{n}\textstyle\sum_{i=1}^{n} e_i^2 && \text{(definition of the MSE loss)}
\end{aligned}
$$

* Five errors: sum 0.2553, squares 0.2759, loss 0.0552

## Fitting Means Minimizing the Loss

Fit a line to Example 1: the loss depends on its two parameters $\theta = (w, b)$.

:::: {.columns}
::: {.column width="58%"}
![](plots/01/d3_loss_surface.pdf)
:::
::: {.column width="42%"}
* The star is the lowest loss
* $\min$: 0.1962
* $\arg\min$: $w = -2.219$, $b = 1.068$
* The best parameters are the fit
:::
::::

```{=latex}
\setlength{\abovedisplayskip}{1pt}\setlength{\belowdisplayskip}{1pt}\setlength{\abovedisplayshortskip}{1pt}\setlength{\belowdisplayshortskip}{1pt}
```

<!-- derivation: argmin -->
$$
\begin{aligned}
\theta^\star &= \arg\min_{\theta}\, \mathcal{L}(\theta) && \text{(definition of the best parameters)}
\end{aligned}
$$

## Why Not Try Every Value?

```{=latex}
\begin{center}
\begin{tikzpicture}
  \foreach \i in {0,...,9} { \fill[cblue] (\i*0.3,0) circle (2pt); }
  \node[note] at (1.35,-0.5) {1 parameter: $g$ tries};
  \foreach \i in {0,...,9} { \foreach \j in {0,...,9} { \fill[corange] (4+\i*0.2,\j*0.2-0.6) circle (1.2pt); } }
  \node[note] at (4.9,-1.1) {2 parameters: $g^2$ tries};
  \node[fillbox=cred, text width=3.5cm] at (9.3,0.3) {1000 parameters:\\ $g^{1000}$ tries\\ \textbf{impossible}};
\end{tikzpicture}
\end{center}
```

* A grid with $g$ values per parameter needs $g^m$ evaluations
* The demo grid: $g = 200$, 40000 evaluations for two parameters
* Real models have thousands to billions of parameters; Class 02 searches smarter

## Memorization and Generalization

<!-- defines: memorization, generalization, test error -->

A lookup table copies the closest of 50 stored examples.

![](plots/01/d1_lookup_vs_cubic.pdf){width=82%}

* **Memorization**: exact recall, training error 0.000
* **Generalization**: quality on 1000 new points
* Test error: table 0.164, cubic 0.105
* Rigid models miss the pattern; flexible ones copy errors

## DEMO D1: 01_learning_as_compression

`\vspace{8mm}\begin{center}{\Huge\bfseries DEMO D1}\\[4mm]{\LARGE\ttfamily 01\_learning\_as\_compression.ipynb}\end{center}`{=latex}

# Learning as Compression

## To Learn Is to Compress

```{=latex}
\begin{center}
\begin{tikzpicture}
  \fill[cgray!40] (0,0) rectangle (2.2,3); \node[align=center] at (1.1,1.5) {raw data\\ 50 points\\ 100 numbers};
  \draw[flow, very thick] (2.6,1.5) -- node[above, note] {learn} (4,1.5);
  \fill[cblue!50] (4.4,1.6) rectangle (6.6,2.4); \node[align=center] at (5.5,2) {model\\ 4 numbers};
  \fill[corange!50] (4.4,0.6) rectangle (6.6,1.4); \node[align=center] at (5.5,1) {errors\\ 50 numbers};
  \draw[decorate, decoration={brace, amplitude=5pt}] (6.8,2.4) -- node[right=6pt, note, align=left] {rebuilds\\ the data} (6.8,0.6);
\end{tikzpicture}
\end{center}
```

* A short model plus its errors can rebuild the data
* The model is a compressed description of the data
* A lookup table is as big as the data: nothing learned
* Pure noise has no short description: nothing to learn

## Notation N2: Information

```{=latex}
\renewcommand{\arraystretch}{1.3}
```

| Symbol | Meaning | Example |
|:---------------------|:-----------------------------------|:--------------------|
| $p(x)$ | probability of outcome $x$ (mass or density) | $p(\text{rain}) = 0.2$ |
| $p(y \mid x)$ | probability of $y$ given $x$ | spam given a word |
| $h(x)$ | surprise of outcome $x$, in bits | $h(\text{rain}) = 2.3219$ |
| $H(X)$ | entropy: average surprise of a source $X$ | fair coin: 1 bit |
| $\ell(M)$ | bits to describe the model $M$ | $q = 5$: 16.9 |
| $\ell(D \mid M)$ | bits for the data $D$ given $M$ (its errors) | $q = 5$: $-110.4$ |

## Surprise and the Logarithm

The surprise of an outcome is $h(x) = -\log_2 p(x)$ bits.

![](plots/01/d2_surprise.pdf)

* $p(x)$ lies between 0 (never) and 1 (always)
* A rare outcome surprises more; a certain one, 0 bits
* One bit is one yes/no answer: $\log_2 8 = 3$ questions

## Shannon Entropy

Entropy is the average surprise of a source: how unpredictable its next outcome is.

```{=latex}
\begin{center}
\begin{tikzpicture}
  \node[fillbox=cblue, text width=1.7cm] (a) {for each\\ outcome $x$};
  \node[fillbox=corange, text width=2.1cm, right=5mm of a] (b) {its surprise\\ $-\log_2 p(x)$};
  \node[fillbox=cgreen, text width=2.3cm, right=5mm of b] (c) {weight by how\\ often: $p(x)$};
  \node[fillbox=cpurple, text width=1.6cm, right=5mm of c] (d) {add up over all $x$};
  \draw[flow] (a) -- (b); \draw[flow] (b) -- (c); \draw[flow] (c) -- (d);
\end{tikzpicture}
\end{center}
```

```{=latex}
\setlength{\abovedisplayskip}{1pt}\setlength{\belowdisplayskip}{1pt}\setlength{\abovedisplayshortskip}{1pt}\setlength{\belowdisplayshortskip}{1pt}
```

<!-- derivation: entropy -->
$$
\begin{aligned}
H(X) &= -\sum_{x} p(x)\,\log_2 p(x) && \text{(definition of entropy)}
\end{aligned}
$$

* High entropy: unpredictable; low entropy: predictable
* Claude Shannon introduced it in 1948

## Entropy Is the Limit of Compression

No code beats the entropy on average (Shannon, 1948).

![](plots/01/d2_entropy_coin.pdf){width=82%}

* Fair coin: $H = 1$ bit; compressors zlib, bz2 need 1.2403, 1.2921
* Biased: $H = 0.9 \times 0.152 + 0.1 \times 3.32 = 0.47$

## DEMO D2: 01_learning_as_compression

`\vspace{8mm}\begin{center}{\Huge\bfseries DEMO D2}\\[4mm]{\LARGE\ttfamily 01\_learning\_as\_compression.ipynb}\end{center}`{=latex}

## Occam's Razor and the MDL Trade-Off

<!-- defines: minimum description length -->

Total length: $\ell(D, M) = \ell(M) + \ell(D \mid M)$, model bits plus error bits.

![](plots/01/d3_parts.pdf){width=82%}

* Occam's razor: among equal fits, prefer the simplest
* Shortest total wins (Rissanen, 1978)
* Constants are dropped: a length can be negative

## MDL in Practice: Gaussian Errors

For Gaussian (bell-shaped) errors the two lengths have closed forms. With $m$ parameters and $n$ examples:

```{=latex}
\begin{center}
\begin{tikzpicture}
  \node[fillbox=cblue, text width=4.4cm, align=left] (a) {model cost $\ell(M)$\\ each parameter costs\\ $(1/2)\log_2 n$ bits; grows with $m$};
  \node[fillbox=corange, text width=4.4cm, align=left, right=6mm of a] (b) {error cost $\ell(D \mid M)$\\ $(n/2)\log_2 \mathrm{MSE}$ bits\\ shrinks as the fit improves};
\end{tikzpicture}
\end{center}
```

```{=latex}
\setlength{\abovedisplayskip}{1pt}\setlength{\belowdisplayskip}{1pt}\setlength{\abovedisplayshortskip}{1pt}\setlength{\belowdisplayshortskip}{1pt}
```

<!-- derivation: bic -->
$$
\begin{aligned}
\ell(M) + \ell(D \mid M) &= \tfrac{m}{2}\log_2 n + \tfrac{n}{2}\log_2 \mathrm{MSE} && \text{(by Rissanen)}
\end{aligned}
$$

* This is the Bayesian information criterion (BIC), in bits
* It needs only the training data; the derivation is in the appendix

## Worked Example: Choosing a Degree

Example 1, $n = 50$: total length for each degree $q$.

![](plots/01/d3_mdl_curve.pdf){width=74%}

* Minimum $q = 5$: $-93.5$ bits; $q = 3$: $-92.4$, one bit more
* Over 20 repeated draws the choice runs from 3 to 7
* With 15 points it picks $q = 14$: it fails

## Likelihood: How Probable Are the Data?

<!-- defines: likelihood -->

The **likelihood** of the data $D$ is $p(D \mid \theta)$: their probability under the parameters $\theta$.

```{=latex}
\begin{center}
\begin{tikzpicture}[node distance=7mm]
  \node[fillbox=cblue, text width=3.2cm] (m) {model: biased coin\\ $p(\text{heads}) = 0.9$};
  \node[fillbox=corange, text width=2.2cm, right=of m] (d) {data: one toss, heads};
  \node[fillbox=cgreen, text width=3.6cm, right=of d] (l) {likelihood $0.9$\\ code length $0.152$ bits};
  \draw[flow] (m) -- (d); \draw[flow] (d) -- (l);
\end{tikzpicture}
\end{center}
```

* A higher likelihood means a shorter code for the data
* $\ell(D \mid M) = -\log_2 p(D \mid \theta)$: maximum likelihood = shortest code

## Regularization: Paying for Complexity

<!-- defines: regularization -->

**Regularization** adds a penalty to the loss: $\mathcal{L}_{\lambda}(\theta) = \mathcal{L}(\theta) + \lambda\,\Omega(\theta)$.

```{=latex}
\begin{center}
\begin{tikzpicture}
  \fill[corange!65] (0,0) rectangle (5,0.8); \node at (2.5,0.4) {data loss $\mathcal{L}(\theta)$};
  \fill[cblue!55] (5,0) rectangle (7.5,0.8); \node at (6.25,0.4) {$\lambda\,\Omega(\theta)$};
  \node[note, font=\small] at (2.5,-0.35) {$\approx \ell(D \mid M)$};
  \node[note, font=\small] at (6.25,-0.35) {$\approx \ell(M)$};
  \draw[decorate, decoration={brace, amplitude=5pt}] (0,0.9) -- node[above=5pt, note, font=\small] {regularized loss $\mathcal{L}_{\lambda}(\theta)$} (7.5,0.9);
\end{tikzpicture}
\end{center}
```

* $\Omega(\theta)$ measures complexity, for example the sum of squared parameters
* $\lambda \ge 0$ sets its weight: $\lambda = 0$ is the plain fit
* The penalty plays the part of the model cost $\ell(M)$

## DEMO D3: 01_learning_as_compression

`\vspace{8mm}\begin{center}{\Huge\bfseries DEMO D3}\\[4mm]{\LARGE\ttfamily 01\_learning\_as\_compression.ipynb}\end{center}`{=latex}

# Evaluating a Model

## Training Loss Is Not the Goal

Example 2: 569 tumours, malignant ($y = 1$) or benign. Accuracy: share correct.

![](plots/01/d4_train_vs_new.pdf){width=82%}

* Black boxes here: a decision tree (Class 06), logistic regression (Class 03)
* 143 unseen patients: tree 0.958, training 1.000; logistic regression 0.965, 0.988

## The Confusion Matrix

```{=latex}
\begin{center}
\begin{tikzpicture}
  \node[font=\scriptsize] at (2.5,2.35) {\textbf{predicted}};
  \node[font=\scriptsize, rotate=90] at (-1.25,0.8) {\textbf{actual}};
  \node[font=\scriptsize] at (1.5,1.95) {malignant}; \node[font=\scriptsize] at (3.5,1.95) {benign};
  \node[font=\scriptsize, anchor=east] at (0.45,1.3) {malignant}; \node[font=\scriptsize, anchor=east] at (0.45,0.3) {benign};
  \draw[fill=cgreen!45] (0.5,0.8) rectangle (2.5,1.8); \node[align=center, font=\scriptsize] at (1.5,1.3) {\textbf{TP} = 48\\ true positive};
  \draw[fill=cred!35] (2.5,0.8) rectangle (4.5,1.8); \node[align=center, font=\scriptsize] at (3.5,1.3) {\textbf{FN} = 5\\ missed};
  \draw[fill=corange!40] (0.5,-0.2) rectangle (2.5,0.8); \node[align=center, font=\scriptsize] at (1.5,0.3) {\textbf{FP} = 1\\ false alarm};
  \draw[fill=cgreen!25] (2.5,-0.2) rectangle (4.5,0.8); \node[align=center, font=\scriptsize] at (3.5,0.3) {\textbf{TN} = 89\\ true negative};
\end{tikzpicture}
\end{center}
```

* The tree on the 143 test patients; the positive class is malignant
* Each cell counts one kind of right or wrong decision
* Accuracy is $(48 + 89) / 143 = 0.9580$

## Precision, Recall, and F1

```{=latex}
\begin{center}
\begin{tikzpicture}
  \node[fillbox=corange, text width=3.3cm] (p) {\textbf{precision}\\ flagged tumours that are malignant\\ 48 of 49 = 0.9796};
  \node[fillbox=cblue, text width=3.3cm, right=3mm of p] (r) {\textbf{recall}\\ malignant tumours that are found\\ 48 of 53 = 0.9057};
  \node[fillbox=cgreen, text width=2.4cm, right=3mm of r] (f) {\textbf{F1}\\ balance of both\\ 0.9412};
\end{tikzpicture}
\end{center}
```

* precision $= TP/(TP + FP)$
* recall $= TP/(TP + FN)$
* $F_1 = 2PR/(P + R)$, for precision $P$, recall $R$
* Screening needs recall; a spam filter needs precision

## Baseline and the Accuracy Paradox

<!-- defines: baseline, balanced accuracy -->

A **baseline** always predicts the most common class. A toy: breast cancer thinned to 5% malignant.

![](plots/01/d4_paradox.pdf){width=82%}

* Always benign: accuracy 0.949, recall 0.00
* Balanced accuracy, mean recall of both classes: 0.50
* Example 2: baseline 0.6294, tree 0.958

## Hold Out Data: Training, Validation, and Test Sets

<!-- defines: validation set, test set -->

```{=latex}
\begin{center}
\begin{tikzpicture}
  \fill[cblue!55] (0,0) rectangle (5,0.8); \node at (2.5,0.4) {training};
  \fill[cgreen!55] (5,0) rectangle (7.5,0.8); \node at (6.25,0.4) {validation};
  \fill[corange!70] (7.5,0) rectangle (10,0.8); \node at (8.75,0.4) {test};
  \node[note, font=\small] at (2.5,-0.4) {fit the parameters};
  \node[note, font=\small] at (6.25,-0.4) {compare, tune};
  \node[note, font=\small] at (8.75,-0.4) {final, \textbf{once}};
\end{tikzpicture}
\end{center}
```

* Split the data before training
* Example 2: 426 patients train the model, 143 test it
* Every decision based on the test score makes it optimistic

## One Split Is Not Enough

![](plots/01/d4_splits.pdf)

* Same tree, same data, 50 random splits: 0.860 to 0.958
* Mean 0.928, a spread of 9.8 points
* A single number hides this spread

## k-Fold Cross-Validation and Its Variants

```{=latex}
\begin{center}
\begin{tikzpicture}[yscale=0.5]
  \foreach \r in {1,...,5} {
    \node[anchor=east, font=\scriptsize] at (-0.1,-\r) {round \r};
    \foreach \c in {1,...,5} {
      \ifnum\c=\r \fill[corange!75] ({(\c-1)*1.6},-\r-0.4) rectangle ({\c*1.6-0.05},-\r+0.4);
      \else \fill[cblue!45] ({(\c-1)*1.6},-\r-0.4) rectangle ({\c*1.6-0.05},-\r+0.4); \fi
    }
    \node[anchor=west, font=\scriptsize] at (8.1,-\r) {score \r};
  }
  \fill[cblue!45] (0,-6.4) rectangle (0.4,-5.9); \node[anchor=west, font=\scriptsize] at (0.45,-6.15) {train};
  \fill[corange!75] (2,-6.4) rectangle (2.4,-5.9); \node[anchor=west, font=\scriptsize] at (2.45,-6.15) {test fold};
\end{tikzpicture}
\end{center}
```

* Split the data into $k$ parts (folds); here $k = 5$
* Each round tests one fold; every example is tested once
* Variants keep class shares (stratified), a patient's records together (grouped) or time order (temporal)

## Notation N3a: Scores and Intervals

<!-- defines: standard deviation -->

```{=latex}
\renewcommand{\arraystretch}{1.3}
```

Example 2: decision tree, 10-fold cross-validation.

| Symbol | Meaning | Example 2 |
|:-----------------------------|:-----------------------------------------|:--------------------|
| $k$ | number of folds | 10 |
| $s_i$ | score (accuracy) on fold $i$ | 0.9123 for $i = 1$ |
| $\bar{s}$ | mean of the fold scores | 0.9262 |
| $s$ | standard deviation of the fold scores | 0.0245 |
| $\mathrm{SE}$ | standard error of the mean, $s / \sqrt{k}$ | 0.0078 |
| $\mathrm{CI}$ | 95% confidence interval for the true accuracy | [0.9087, 0.9437] |

## Variance and Standard Deviation

$s^2 = \frac{1}{k-1}\sum_{i=1}^{k}(s_i - \bar{s})^2$, $s = \sqrt{s^2}$.

![](plots/01/d4_variance.pdf)

* Divide by $k - 1$: the mean used up one score
* Simulated, true variance 1: $\div 5$: 0.801, $\div 4$: 1.001

## The Confidence Interval

$\mathrm{SE} = s / \sqrt{k}$ and $\mathrm{CI} = \bar{s} \pm t_{0.975,\,k-1}\,\mathrm{SE}$.

![](plots/01/d4_folds.pdf)

* $t_{0.975,\,k-1}$: 97.5% point of Student's $t$, $k - 1$ degrees of freedom
* Degrees of freedom: scores free once the mean is fixed

## Worked Example: Confidence Interval

:::: {.columns}
::: {.column width="52%"}
![](plots/01/d4_ci.pdf)
:::
::: {.column width="48%"}
1. $\bar{s} = 0.9262$, $s = 0.0245$
2. $\mathrm{SE} = 0.0078$
3. $t_{0.975,\,9} = 2.262$
4. Half-width $= 0.0175$
:::
::::

* Report 0.9262 $\pm$ 0.0175: the interval is [0.9087, 0.9437]
* 95% of such intervals hold the truth
* Limit: folds share training data: too narrow

## Is the Difference Real? Hypothesis Testing

<!-- defines: null hypothesis, p-value, significance level -->

```{=latex}
\begin{center}
\begin{tikzpicture}[x=0.9cm, y=1.5cm]
  \fill[cred!35] (1.4,0) -- plot[domain=1.4:3.2, samples=30] (\x, {exp(-(\x)^2/1.4)}) -- (3.2,0) -- cycle;
  \draw[cblue, very thick, domain=-3.2:3.2, samples=60] plot (\x, {exp(-(\x)^2/1.4)});
  \draw[->] (-3.4,0) -- (3.6,0) node[right, note] {statistic};
  \draw[cred, thick] (1.4,0) -- (1.4,1.0) node[above, note] {observed};
  \node[cblue, note, anchor=east] at (-1.2,0.8) {results if nothing is real};
  \node[cred, note, anchor=west] at (2.0,0.45) {tail area = $p_{\text{val}}$};
\end{tikzpicture}
\end{center}
```

* Null hypothesis $H_0$: nothing real, only luck
* p-value: chance of a result this extreme if $H_0$ holds
* Reject $H_0$ when $p_{\text{val}} < \alpha = 0.05$, the significance level
* It is not the chance that $H_0$ is true

## Worked Example: a Coin and a Classifier

Null hypothesis: a fair coin. We see 9 heads in 10 tosses.

![](plots/01/d4_null.pdf)

* 11 of 1024 outcomes are this extreme: $p_{\text{val}} = 0.01074$
* The tree gets 137 of 143 test patients right; null: the baseline's 0.6268
* Its p-value is 4.964e-21: not luck

## Comparing Two Models: Pair the Scores

$\delta_j = s_j^{A} - s_j^{B}$: A tree, B logistic regression.

![](plots/01/d4_paired.pdf)

* Mean difference $\bar{\delta} = -0.0492$: B wins in 8 of 10 folds
* False positive rate: plain $t$-test 31.5%, corrected 0.0%

## Which Test When?

```{=latex}
\begin{center}
\begin{tikzpicture}[node distance=5mm]
  \node[fillbox=cgray, text width=4.4cm] (q) {How was the model evaluated?};
  \node[fillbox=cblue, text width=4.4cm, below left=4mm and -1.0cm of q] (cv) {repeated $k$-fold cross-validation};
  \node[fillbox=corange, text width=4.4cm, below right=4mm and -1.0cm of q] (ts) {one held-out test set};
  \node[fillbox=cblue, text width=4.4cm, below=4mm of cv] (t) {\textbf{corrected resampled t-test}};
  \node[fillbox=corange, text width=4.4cm, below=4mm of ts] (m) {\textbf{McNemar's test}};
  \draw[flow] (q) -- (cv); \draw[flow] (q) -- (ts); \draw[flow] (cv) -- (t); \draw[flow] (ts) -- (m);
\end{tikzpicture}
\end{center}
```

* One score per fold or split: compare the scores
* One prediction per patient: compare who is right

## Notation N3b: Tests

```{=latex}
\renewcommand{\arraystretch}{1.1}
```

| Symbol | Meaning | Example 2 |
|:-----------|:-------------------------------------|:----------------------|
| $J$ | number of scores | 50 (5 $\times$ 10-fold) |
| $\bar{\delta}$, $s_\delta$ | mean score difference, its standard deviation | $-0.0520$, 0.0327 |
| $n_{\text{train}}$, $n_{\text{test}}$ | examples to train, to test in one split | ratio 0.1111 |
| $n_{11}$, $n_{00}$ | both models right, both wrong | 520, 7 |
| $n_{10}$, $n_{01}$ | only A right, only B right | 7, 35 |
| $N$ | disagreements, $n_{10} + n_{01}$ | 42 |
| $T$, $\chi^2$ | t statistic, chi-squared statistic | $-4.39$, 18.67 |
| $p_{\text{val}}$, $\alpha$ | p-value, significance level | 6.01e-05, 0.05 |

## The Corrected Resampled t-Test

Is the mean difference of the $J$ scores real?

```{=latex}
\begin{center}
\begin{tikzpicture}[node distance=5mm]
  \node[fillbox=cgray, text width=2.4cm] (a) {spread of the differences};
  \node[fillbox=corange, text width=3.3cm, right=of a] (b) {inflate by $1/J + n_{\text{test}}/n_{\text{train}}$};
  \node[fillbox=cgreen, text width=2.4cm, right=of b] (c) {variance of the mean difference};
  \draw[flow] (a) -- (b); \draw[flow] (b) -- (c);
  \end{tikzpicture}
\end{center}
```

```{=latex}
\setlength{\abovedisplayskip}{1pt}\setlength{\belowdisplayskip}{1pt}\setlength{\abovedisplayshortskip}{1pt}\setlength{\belowdisplayshortskip}{1pt}
```

<!-- derivation: corrected-t -->
$$
\begin{aligned}
T &= \bar{\delta} \Big/ \sqrt{\left(\tfrac{1}{J} + \tfrac{n_{\text{test}}}{n_{\text{train}}}\right) s_\delta^2} && \text{(by Nadeau and Bengio 2003)}
\end{aligned}
$$

* Example 2, $J = 50$: $\bar{\delta} = -0.0520$, $s_\delta = 0.0327$
* $n_{\text{test}} / n_{\text{train}} = 0.1111$
* $T = -4.39$, p-value 6.01e-05 (naive $T = -11.24$); conservative: power 59.5% against 96.5%

## McNemar's Test: the Statistic

Do A and B err on different patients? 569 out-of-fold predictions.

```{=latex}
\begin{center}
\begin{tikzpicture}
  \node[font=\scriptsize] at (3.0,2.15) {\textbf{B right} \hspace{1.6cm} \textbf{B wrong}};
  \node[font=\scriptsize, anchor=east] at (0.45,1.5) {\textbf{A right}}; \node[font=\scriptsize, anchor=east] at (0.45,0.5) {\textbf{A wrong}};
  \draw[fill=cgreen!25] (0.5,1.0) rectangle (2.5,2.0); \node[font=\scriptsize] at (1.5,1.5) {$n_{11} = 520$};
  \draw[fill=corange!40] (2.5,1.0) rectangle (4.5,2.0); \node[font=\scriptsize] at (3.5,1.5) {$n_{10} = 7$};
  \draw[fill=corange!40] (0.5,0.0) rectangle (2.5,1.0); \node[font=\scriptsize] at (1.5,0.5) {$n_{01} = 35$};
  \draw[fill=cred!25] (2.5,0.0) rectangle (4.5,1.0); \node[font=\scriptsize] at (3.5,0.5) {$n_{00} = 7$};
\end{tikzpicture}
\end{center}
```

```{=latex}
\setlength{\abovedisplayskip}{1pt}\setlength{\belowdisplayskip}{1pt}\setlength{\abovedisplayshortskip}{1pt}\setlength{\belowdisplayshortskip}{1pt}
```

<!-- derivation: mcnemar-chi2 -->
$$
\begin{aligned}
\chi^2 &= \tfrac{\left(n_{10} - n_{01}\right)^2}{n_{10} + n_{01}} && \text{(by McNemar 1947)}
\end{aligned}
$$

* Only the $N = 42$ disagreements count: if A = B, fair coin tosses
* $\chi^2 = 28^2 / 42 = 18.67$, p-value 1.56e-05
* Edwards 1948, $-1$ in the square: 17.36, 3.10e-05

## McNemar's Test: Meaning and Limit

![](plots/01/d4_chi2.pdf)

* A wins 7, B wins 35: B is better; exact p-value 1.51e-05
* Limit: for $N < 25$ use the exact test, not $\chi^2$
* Test set: table 133, 4, 5, 1, $N = 9$, exact p-value 1.0

## DEMO D4: 02_evaluation_validation

`\vspace{8mm}\begin{center}{\Huge\bfseries DEMO D4}\\[4mm]{\LARGE\ttfamily 02\_evaluation\_validation.ipynb}\\[3mm]{\large D4a--D4d live; D4e--D4g read in the lab}\end{center}`{=latex}

# Why Models Fail

## Underfitting, a Good Fit, Overfitting

<!-- defines: underfitting, overfitting -->

![](plots/01/d5_fits.pdf)

* Underfitting: a line misses the wave, test 0.3028
* Overfitting: degree 15 copies the sample, test 8.0205

## Error Versus Model Complexity

![](plots/01/d5_error_curve.pdf)

* Training error always falls as $q$ grows
* Test error is lowest at $q = 3$: 0.1050
* Beyond it the model fits noise: test error rises

## Bias and Variance

```{=latex}
\begin{center}
\begin{tikzpicture}[scale=0.5]
  \foreach \bx/\title/\cx/\cy/\spread in {0/{low bias,\\ low variance}/0/0/0.25, 5/{low bias,\\ high variance}/0/0/1.0, 10/{high bias,\\ low variance}/0.9/0.8/0.25} {
    \begin{scope}[shift={(\bx,0)}]
      \foreach \r/\c in {1.8/cgray!15, 1.2/cgray!30, 0.6/cred!40} { \fill[\c] (0,0) circle (\r); \draw[cgray] (0,0) circle (\r); }
      \foreach \i in {1,...,9} { \fill[cblue] ({\cx+\spread*rand},{\cy+\spread*rand}) circle (2.5pt); }
      \node[font=\small, align=center, text width=2.6cm] at (0,-2.7) {\title};
    \end{scope}
  }
\end{tikzpicture}
\end{center}
```

* Each dot is one fit on a new training set; the bullseye is $f$
* **Bias**: error left after averaging many fits, $\bar{f} - f$
* **Variance**: how much the fit changes with the sample
* Example 1, 200 training sets: line $q = 1$, bias$^2$ 0.2000, variance 0.0157; $q = 9$: 0.0006, 0.1298

## The Bias-Variance Decomposition

```{=latex}
\begin{center}
\begin{tikzpicture}
  \draw[thick] (0,0) -- (9,0);
  \foreach \x/\lab/\c in {1.5/{$f$, the truth}/cgreen, 4.5/{$\bar{f}$, mean of the fits}/cblue, 7.5/{$\hat{f}$, one fit}/corange} { \fill[\c] (\x,0) circle (3pt); \node[font=\small, above=3pt] at (\x,0) {\lab}; }
  \draw[decorate, decoration={brace, mirror, amplitude=4pt}] (1.5,-0.15) -- node[below=4pt, font=\small] {$\Delta_1 = f - \bar{f}$} (4.5,-0.15);
  \draw[decorate, decoration={brace, mirror, amplitude=4pt}] (4.5,-0.15) -- node[below=4pt, font=\small] {$\Delta_2 = \bar{f} - \hat{f}$} (7.5,-0.15);
\end{tikzpicture}
\end{center}
```

```{=latex}
\footnotesize\setlength{\abovedisplayskip}{1pt}\setlength{\belowdisplayskip}{1pt}\setlength{\abovedisplayshortskip}{1pt}\setlength{\belowdisplayshortskip}{1pt}
```

<!-- derivation: bias-variance
defs: {y: "f + ε", Δ_1: "f - f_bar", Δ_2: "f_bar - f_hat"}
tag: "derivation: bias-variance" -->
$$
\begin{aligned}
(y - \hat{f})^2 &= (\Delta_1 + \Delta_2 + \varepsilon)^2 && \text{(substitute)} \\
 &= \Delta_1^2 + \Delta_2^2 + \varepsilon^2 + 2(\Delta_1\Delta_2 + \Delta_1\varepsilon + \Delta_2\varepsilon) && \text{(expand)} \\
\mathbb{E}\big[(y - \hat{f})^2\big] &= \Delta_1^2 + \mathbb{E}\big[\Delta_2^2\big] + \sigma^2 && \text{(assume)} \\
 &= \mathrm{bias}^2 + \mathrm{variance} + \sigma^2 && \text{(definition)}
\end{aligned}
$$

Assumptions behind row 3 (they kill the cross terms):

* $\mathbb{E}[\varepsilon] = 0$; $\varepsilon$ is independent of $\hat{f}$
* $\mathbb{E}[\hat{f}] = \bar{f}$, so $\mathbb{E}[\Delta_2] = 0$; $f$, $\bar{f}$ are fixed

## The Trade-Off and What to Do About It

![](plots/01/d5_tradeoff.pdf)

* More complexity lowers bias and raises variance
* Lowest sum at $q = 3$: 0.0046 + 0.0084 + 0.09

## DEMO D5: 03_pitfalls

`\vspace{8mm}\begin{center}{\Huge\bfseries DEMO D5}\\[4mm]{\LARGE\ttfamily 03\_pitfalls.ipynb}\end{center}`{=latex}

## No Free Lunch and Inductive Bias

Over all possible problems, no learner beats another (Wolpert, 1996).

![](plots/01/d6_ranking.pdf){width=82%}

* Each model assumes something about the pattern
* Circles: logistic regression 0.461, k-NN 0.997; k-NN copies the nearest neighbours (Class 06)

## DEMO D6: 03_pitfalls

`\vspace{8mm}\begin{center}{\Huge\bfseries DEMO D6}\\[4mm]{\LARGE\ttfamily 03\_pitfalls.ipynb}\end{center}`{=latex}

## Data Leakage and the Right Pipeline

<!-- defines: data leakage -->

Leakage: information from outside the training data reaches the model.

![](plots/01/d7_leak.pdf){width=82%}

* Random labels, 100 patients, 10 000 features: select the 20 best
* Selecting before the split: 0.870; inside each fold, the right way: 0.490

## DEMO D7: 03_pitfalls

`\vspace{8mm}\begin{center}{\Huge\bfseries DEMO D7}\\[4mm]{\LARGE\ttfamily 03\_pitfalls.ipynb}\end{center}`{=latex}

## Correlation and the Ice-Cream Example

$r$: how two variables move together; $\bar{x}$, $\bar{y}$ are means.

![](plots/01/d8_scatter.pdf){width=72%}

```{=latex}
\setlength{\abovedisplayskip}{1pt}\setlength{\belowdisplayskip}{1pt}\setlength{\abovedisplayshortskip}{1pt}\setlength{\belowdisplayshortskip}{1pt}
```

<!-- derivation: correlation -->
$$
\begin{aligned}
r &= \tfrac{\textstyle\sum_{i}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\textstyle\sum_{i}(x_i - \bar{x})^2\,\textstyle\sum_{i}(y_i - \bar{y})^2}} && \text{(definition)}
\end{aligned}
$$

* Ice-cream sales and drownings: $r = 0.838$

## The Hidden Common Cause

```{=latex}
\begin{center}
\begin{tikzpicture}[node distance=14mm]
  \node[fillbox=cred, text width=2.4cm] (t) {\textbf{temperature}\\ (confounder)};
  \node[fillbox=cblue, text width=2.4cm, below left=of t] (i) {ice-cream sales};
  \node[fillbox=corange, text width=2.4cm, below right=of t] (d) {drownings};
  \draw[flow, very thick] (t) -- node[left, note, font=\footnotesize] {hot days} (i);
  \draw[flow, very thick] (t) -- node[right, note, font=\footnotesize] {more swimming} (d);
  \draw[cgray, dashed, thick, <->] (i) -- node[below, note, font=\footnotesize] {correlated, no causal link} (d);
\end{tikzpicture}
\end{center}
```

* A confounder causes both variables
* Ice-cream weight alone: $+1.2107$ drownings per 100 sales
* With temperature added, a regression with two inputs (Class 03): $-0.0186$, about zero
* Halving sales: the model predicts 384.2 drownings, reality gives 1351.2

## DEMO D8: 03_pitfalls

`\vspace{8mm}\begin{center}{\Huge\bfseries DEMO D8}\\[4mm]{\LARGE\ttfamily 03\_pitfalls.ipynb}\end{center}`{=latex}

## Shortcuts and Distribution Shift

<!-- defines: shortcut, distribution shift -->

A **shortcut** predicts in training, not in use. A sample that misses a group shifts the data too.

![](plots/01/d9_accuracy.pdf){width=76%}

* Both features 0.945, then 0.545; core only 0.750, 0.746
* Shifts: $p(x)$ covariate, $p(y)$ label, $p(y \mid x)$ concept

## DEMO D9: 03_pitfalls

`\vspace{8mm}\begin{center}{\Huge\bfseries DEMO D9}\\[4mm]{\LARGE\ttfamily 03\_pitfalls.ipynb}\end{center}`{=latex}

## Five Questions to Ask of Any Result

```{=latex}
\begin{center}
\begin{tikzpicture}[node distance=3mm]
  \node[fillbox=cblue, text width=1.8cm, minimum height=1.6cm] (a) {\textbf{Baseline?}\\ better than the simplest};
  \node[fillbox=cgreen, text width=1.8cm, minimum height=1.6cm, right=of a] (b) {\textbf{Split?}\\ tested on unseen data};
  \node[fillbox=cpurple, text width=1.8cm, minimum height=1.6cm, right=of b] (c) {\textbf{Leakage?}\\ nothing from the test};
  \node[fillbox=corange, text width=1.8cm, minimum height=1.6cm, right=of c] (d) {\textbf{Cause?}\\ more than correlation};
  \node[fillbox=cred, text width=1.8cm, minimum height=1.6cm, right=of d] (e) {\textbf{Shift?}\\ same data in use};
\end{tikzpicture}
\end{center}
```

* Ask all five before you trust a number

# Kinds of Learning

## Where the Learning Signal Comes From

<!-- defines: supervised, unsupervised, semi-supervised, self-supervised, reinforcement learning -->

```{=latex}
\begin{center}\small\renewcommand{\arraystretch}{1.35}
\begin{tabular}{@{}l L{3.1cm} L{2.6cm} c@{}}
\toprule
Type & Tumour images with ... & Learning signal & Classes \\
\midrule
\textcolor{cblue}{\textbf{Supervised}} & a diagnosis each & the diagnosis & 03--07 \\
\textcolor{cgreen!70!black}{\textbf{Unsupervised}} & no diagnosis & structure of the data & 09--10 \\
\textcolor{cpurple}{\textbf{Semi-supervised}} & a few diagnoses & labels + structure & 11 \\
\textcolor{corange}{\textbf{Self-supervised}} & no diagnosis & targets built from the data & 11--12 \\
\textcolor{cred}{\textbf{Reinforcement}} & treatment outcomes & rewards & 13 \\
\bottomrule
\end{tabular}
\end{center}
```

* Types differ in what tells the model it does well
* Self-supervised: hide part of the data, predict it

## Supervised and Unsupervised Learning

```{=latex}
\begin{center}
\begin{tikzpicture}
  \begin{scope}
    \foreach \px/\py in {0.3/0.4, 0.8/1.1, 1.2/0.3, 0.5/1.6} { \fill[cblue] (\px,\py) circle (2.5pt); }
    \foreach \px/\py in {2.6/1.2, 3.0/2.2, 2.3/2.4, 3.2/0.8} { \fill[corange] (\px,\py) rectangle ++(0.15,0.15); }
    \draw[cred, very thick] (1.1,2.7) -- (2.1,0);
    \node[note, font=\small, text width=3.4cm] at (1.7,-0.6) {\textbf{supervised}: labels, a boundary};
  \end{scope}
  \begin{scope}[xshift=5.4cm]
    \foreach \cx/\cy/\c in {0.5/0.5/cblue, 1.9/1.6/cgreen} { \foreach \i in {1,...,7} { \fill[\c] ({\cx+0.3*rand},{\cy+0.3*rand}) circle (1.8pt); } }
    \node[note, font=\small, text width=3.4cm] at (1.3,-0.6) {\textbf{unsupervised}: no labels, groups};
  \end{scope}
\end{tikzpicture}
\end{center}
```

* Supervised: every example has its answer $y$
* Classification: $y$ a category; regression: $y$ a number
* Unsupervised: only inputs $x$, no labels
* Clustering: Class 10; sources, reduction: Class 09

## Semi-Supervised Learning

Few labels, many unlabelled examples.

![](plots/01/d10_accuracy.pdf){width=78%}

* Cluster assumption: dense groups share a label
* Self-training labels what it is sure of; label spreading passes labels to neighbours
* At 10% labels: 0.894, 0.924, 0.950 (supervised, self-training, spreading)

## Reinforcement Learning

![](plots/01/d10_policy.pdf){width=76%}

| Symbol | Meaning |
|:-------------------------------------|:-----------------------------------------------------|
| $S_t$, $a_t$, $R_{t+1}$ | state, action, reward at time $t$ |
| policy | rule that picks the action |
| episode, return | one run from start to end, its summed reward |
| pit, goal | cells that end an episode: penalty, reward |

## DEMO D10: 04_taxonomies

`\vspace{8mm}\begin{center}{\Huge\bfseries DEMO D10}\\[4mm]{\LARGE\ttfamily 04\_taxonomies.ipynb}\end{center}`{=latex}

# Wrap-Up

## Key Takeaways, the Lab and Class 02

```{=latex}
\begin{center}
\begin{tikzpicture}[node distance=3mm]
  \node[fillbox=cblue, text width=3.3cm, minimum height=1.2cm] (a) {\textbf{1. Fit}\\ minimize a loss over the parameters};
  \node[fillbox=cblue, text width=3.3cm, minimum height=1.2cm, right=of a] (b) {\textbf{2. Compression}\\ keep the pattern, drop the noise};
  \node[fillbox=cblue, text width=3.3cm, minimum height=1.2cm, right=of b] (c) {\textbf{3. Evaluation}\\ splits, intervals, tests};
  \node[fillbox=corange, text width=3.3cm, minimum height=1.2cm, below=of a] (d) {\textbf{4. Trade-off}\\ bias against variance};
  \node[fillbox=cred, text width=3.3cm, minimum height=1.2cm, right=of d] (e) {\textbf{5. Pitfalls}\\ leakage and shift};
  \node[fillbox=cgreen, text width=3.3cm, minimum height=1.2cm, right=of e] (f) {\textbf{6. Learning types}\\ the learning signal};
\end{tikzpicture}
\end{center}
```

* Lab, 60 minutes: Parts A to F in class, Part G at home
* Class 02: learning as search over parameters, blind and gradient-based

## References

```{=latex}
\footnotesize
```

* C. E. Shannon, "A Mathematical Theory of Communication", 1948
* A. N. Kolmogorov, "Three Approaches to the Definition of Information", 1965
* J. Rissanen, "Modeling by Shortest Data Description", *Automatica*, 1978
* D. H. Wolpert, "The Lack of A Priori Distinctions Between Learning Algorithms", 1996
* T. M. Mitchell, *Machine Learning*, 1997
* Q. McNemar, "Note on the Sampling Error of the Difference Between Correlated Proportions", 1947
* A. L. Edwards, "Note on the Correction for Continuity", *Psychometrika*, 1948
* C. Nadeau and Y. Bengio, "Inference for the Generalization Error", *Machine Learning*, 2003

# Appendix

## Where BIC Comes From: the Error Cost

Each Gaussian error $e_i$ has a surprise $h(e_i)$; the $n$ surprises add up.

```{=latex}
\begin{center}
\begin{tikzpicture}[x=0.8cm, y=0.45cm]
  \draw[->] (-2.8,0) -- (2.9,0) node[right, note] {$e$};
  \draw[cblue, very thick, domain=-2.6:2.6, samples=50] plot (\x, {1.4*exp(-(\x)^2/1.5)});
  \draw[cred, thick] (1.3,0) -- (1.3,{1.4*exp(-(1.3)^2/1.5)});
  \node[cred, note, font=\footnotesize, anchor=west] at (1.4,0.9) {rare error: many bits};
  \node[cblue, note, font=\footnotesize, anchor=east] at (-1.4,1.3) {small error: few bits};
\end{tikzpicture}
\end{center}
```

```{=latex}
\setlength{\abovedisplayskip}{1pt}\setlength{\belowdisplayskip}{1pt}\setlength{\abovedisplayshortskip}{1pt}\setlength{\belowdisplayshortskip}{1pt}
```

<!-- derivation: bic-gauss
sympy: {r2: "-log(exp(-e**2/(2*σ**2))/sqrt(2*pi*σ**2))/log(2)", r3: "log(2*pi*σ**2)/(2*log(2)) + e**2/(2*σ**2*log(2))"}
tag: "derivation: bic-gauss" -->
$$
\begin{aligned}
h(e) &= -\log_2 p(e) && \text{(definition of surprise)} \\
 &= -\log_2 \frac{\exp\left(-e^2 / (2\sigma^2)\right)}{\sqrt{2\pi\sigma^2}} && \text{(by the Gaussian density)} \\
 &= \tfrac{1}{2}\log_2(2\pi\sigma^2) + \frac{e^2}{2\sigma^2\ln 2} && \text{(log of a product)}
\end{aligned}
$$

* Small errors have a high density: few bits

## Where BIC Comes From: Sum Over the Examples

**So far:** surprise of one error. **Goal:** code length of all errors.

```{=latex}
\setlength{\abovedisplayskip}{1pt}\setlength{\belowdisplayskip}{1pt}\setlength{\abovedisplayshortskip}{1pt}\setlength{\belowdisplayshortskip}{1pt}
```

<!-- derivation: bic-sum -->
$$
\begin{aligned}
\ell(D \mid M) &= \sum_{i=1}^{n} h(e_i) && \text{(definition)} \\
 &= \tfrac{n}{2}\log_2(2\pi\sigma^2) + \frac{\sum_{i} e_i^2}{2\sigma^2\ln 2} && \text{(linearity of the sum)} \\
 &= \tfrac{n}{2}\log_2(2\pi\sigma^2) + \frac{n\,\mathrm{MSE}}{2\sigma^2\ln 2} && \text{(definition of MSE)}
\end{aligned}
$$

## Where BIC Comes From: the Constant Drops Out

**So far:** code length of the errors. **Goal:** the BIC form.

```{=latex}
\setlength{\abovedisplayskip}{1pt}\setlength{\belowdisplayskip}{1pt}\setlength{\abovedisplayshortskip}{1pt}\setlength{\belowdisplayshortskip}{1pt}
```

<!-- derivation: bic-constant
defs: {σ: "sqrt(MSE)", c: "n/2*log(2*pi)/log(2) + n/(2*log(2))"}
tag: "derivation: bic-constant" -->
$$
\begin{aligned}
\ell(D \mid M) &= \tfrac{n}{2}\log_2(2\pi\sigma^2) + \frac{n\,\mathrm{MSE}}{2\sigma^2\ln 2} && \text{(definition)} \\
 &= \tfrac{n}{2}\log_2(2\pi\,\mathrm{MSE}) + \tfrac{n}{2\ln 2} && \text{(substitute)} \\
 &= \tfrac{n}{2}\log_2 \mathrm{MSE} + c && \text{(definition of } c\text{)}
\end{aligned}
$$

* $\sigma^2 = \mathrm{MSE}$: the fitted variance; $c$ is shared by all models

## Kolmogorov Complexity

$K(u)$ is the length of the shortest program that prints the string $u$.

```{=latex}
\begin{center}
\begin{tikzpicture}[every node/.style={font=\ttfamily\small}]
  \node[anchor=west] at (0,1.5) {ABABABABABABABABABABABAB};
  \node[anchor=west, cgreen!60!black] at (7.2,1.5) {\normalfont $\rightarrow$ "AB $\times$ 12"};
  \node[anchor=west] at (0,0.8) {THE CAT SAT ON THE MAT.};
  \node[anchor=west, corange] at (7.2,0.8) {\normalfont $\rightarrow$ some savings};
  \node[anchor=west] at (0,0.1) {Q7\#pZ2!mK9vX@4rT\&1wL8};
  \node[anchor=west, cred] at (7.2,0.1) {\normalfont $\rightarrow$ no shorter description};
\end{tikzpicture}
\end{center}
```

* Structure has a short description; randomness has none
* Pure noise leaves nothing to learn

## Kolmogorov Complexity Versus Compression

![](plots/01/d2_entropy_sources.pdf)

* Repeating two letters 2000 times: 18 characters of code
* $K(u)$ is not computable; a compressor such as zlib gives an upper bound

## The Source Coding Theorem

```{=latex}
\begin{center}
\begin{tikzpicture}
  \draw[thick, ->] (0,0) -- (9,0) node[right, note] {bits per outcome};
  \fill[cblue] (2.0,0) circle (3pt) node[above=4pt, note, font=\small] {entropy $H(X)$};
  \fill[corange] (6.0,0) circle (3pt) node[above=4pt, note, font=\small] {average code length};
  \draw[flow, very thick] (2.1,-0.5) -- node[below, note, font=\small] {no code is shorter} (5.9,-0.5);
\end{tikzpicture}
\end{center}
```

```{=latex}
\setlength{\abovedisplayskip}{1pt}\setlength{\belowdisplayskip}{1pt}\setlength{\abovedisplayshortskip}{1pt}\setlength{\belowdisplayshortskip}{1pt}
```

<!-- derivation: source-coding -->
$$
\begin{aligned}
\mathbb{E}\big[\ell(X)\big] &\ge H(X) && \text{(by Shannon 1948)}
\end{aligned}
$$

* $\ell(X)$ counts the bits of the code for outcome $X$
* A trusted statement: the proof is in every information-theory text
