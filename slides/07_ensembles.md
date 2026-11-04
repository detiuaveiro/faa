---
title: Fundamentos de Aprendizagem Automática
subtitle: "Class 07 — Ensemble Models"
bibliography: "references.bib"
suppress-bibliography: true
---

# One Model at a Time

## Where We Left Off

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cblue, text width=2.4cm] (a) {\textbf{linear, probabilistic}\\ Classes 03--04};
  \node[fillbox=cgreen, text width=2.4cm, right=8mm of a] (b) {\textbf{margin, network}\\ Class 05};
  \node[fillbox=corange, text width=2.4cm, right=8mm of b] (c) {\textbf{tree, neighbours}\\ Class 06};
  \node[fillbox=cred, text width=5cm, below=7mm of b] (d) {\textbf{Class 07:} many models, \emph{one} decision};
\end{tikzpicture}
\end{center}
```

* Until now: **one** model, tuned
* Today: first see where one model fails, then combine models

## Roadmap of the Class

| Part | Question | Methods |
|:-------------------------|:--------------------------------|:---------------------------------|
| **Shallow models** | where does one model fail? | six models, 17 sets |
| **Why combining helps** | when does a crowd win? | majority, mean |
| **Voting** | how to combine by hand? | hard, soft, weighted, k-NN |
| **Bagging** | how to cut variance? | bagged trees, random forest |
| **Boosting** | how to cut bias? | AdaBoost, gradient boosting |
| **Meta models** | how to learn the mix? | stacking, neural stacking |
| **Tuning** | how to set the knobs? | random search, Optuna |
| **Real tables** | which family wins? | 11 tables, ranks |
| **Is the gain real?** | is it luck? | McNemar, permutation, Wilcoxon |

<!-- defines: ensemble, member, shrinkage, variance, weighted -->

## Notation N7: Members

Each symbol is new in this class; earlier symbols keep their meaning.

| symbol | meaning | shape | example |
|:----------------------|:--------------------------------------|:-----------:|:------------------|
| $B$ | number of members | scalar | $B = 100$ trees |
| $b$ | index of a member or round | scalar | $b = 1,\dots,B$ |
| $\hat f_b$ | member $b$ | function | one tree |
| $\hat f_{\text{ens}}$ | the ensemble | function | the vote |
| $\hat p_b(k \mid x)$ | probability member $b$ gives to class $k$ | scalar | $0.8$ |
| $u_b$ | weight of member $b$ in a vote | scalar | $0.316$ |
| $\eta$ | shrinkage (the learning rate of Class 02) | scalar | $0.1$ |

## Notation N7b: Voters, Variance, Weights

| symbol | meaning | shape | example |
|:----------------|:--------------------------------------------|:---------:|:---------------------|
| $\gamma$ | chance that one voter is right | scalar | $0.6$ |
| $a_B$ | chance that the majority of $B$ voters is right | scalar | $a_3 = 0.648$ |
| $v$, $V_B$ | variance of one member, of the mean of $B$ | scalar | $V_5 = 0.680$ |
| $\rho$ | correlation of two members' errors | scalar | $\rho = 0.6$ |
| $\xi$ | share of examples left out of a bootstrap sample | scalar | $0.368$ |
| $w_i^{(b)}$ | weight of example $i$ in round $b$ | scalar | $1/n$ at the start |
| $\text{err}_b$ | weighted error of member $b$ | scalar | $0.20$ |
| $\beta_b$ | weight of member $b$ in AdaBoost | scalar | $0.693$ |
| $E_b$, $Z_b$ | exponential loss of member $b$, weight normaliser | scalar | $0.800$ |

## Notation N7c: Examples and Meta Models

| symbol | meaning | shape | example |
|:--------------|:-------------------------------------|:-----------:|:----------------------------|
| $n$ | number of examples (Class 01) | scalar | $600$ |
| $x_i$ | example $i$ (Class 01) | vector | one point of a plane |
| $y_i$ | targets (Class 01) | scalar | $+1$ or $-1$ |
| $\hat y$ | prediction | scalar | $0.8$ |
| $z_A$ | last hidden layer of network A | vector | 128 values |
| $z_B$ | last hidden layer of network B | vector | 64 values |

<!-- defines: MCC, MAE -->

## Notation N7d: Scores

| symbol | meaning | shape | example |
|:--------------------|:-------------------------------------------|:------------:|:----------------|
| $TP$ | confusion-matrix counts (Class 01) | scalar | $30$ |
| $TN$ | confusion-matrix counts (Class 01) | scalar | $50$ |
| $FP$ | confusion-matrix counts (Class 01) | scalar | $10$ |
| $FN$ | confusion-matrix counts (Class 01) | scalar | $10$ |
| $G$ | the product under the root of MCC | scalar | $5760000$ |
| $\text{MCC}$ | Matthews correlation coefficient | scalar | $0.583$ |

## Shallow Models

A **shallow model** is one fitted model that combines nothing.

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cblue, text width=2.9cm] (a) {\textbf{logistic}, or linear\\ Class 03};
  \node[fillbox=cblue, text width=2.9cm, right=3mm of a] (b) {\textbf{naive Bayes}, or polynomial\\ Classes 03--04};
  \node[fillbox=cblue, text width=2.9cm, right=3mm of b] (c) {\textbf{tree}, depth 5\\ Class 06};
  \node[fillbox=cblue, text width=2.9cm, below=3mm of a] (d) {\textbf{1-NN}\\ Class 06};
  \node[fillbox=cblue, text width=2.9cm, below=3mm of b] (e) {\textbf{SVM}, RBF kernel\\ Class 05};
  \node[fillbox=cblue, text width=2.9cm, below=3mm of c] (f) {\textbf{MLP}, one hidden layer\\ Class 05};
\end{tikzpicture}
\end{center}
```

* Nearest neighbours (NN) with $k > 1$ are left out: k-NN is already a vote

## Two Scores: MCC and MAE

**Matthews correlation** (MCC) for classes, **mean absolute error** (MAE) for values.

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cgreen, text width=1.5cm] (tp) {TP = 30};
  \node[fillbox=cred, text width=1.5cm, right=1mm of tp] (fn) {FN = 10};
  \node[fillbox=cred, text width=1.5cm, below=1mm of tp] (fp) {FP = 10};
  \node[fillbox=cgreen, text width=1.5cm, right=1mm of fp] (tn) {TN = 50};
\end{tikzpicture}
\end{center}
```

<!-- derivation: mcc -->
$$
\begin{aligned}
G &= {\scriptstyle (TP+FP)(TP+FN)(TN+FP)(TN+FN)} && \text{(definition)} \\
\text{MCC} &= \frac{TP\,TN - FP\,FN}{\sqrt{G}} && \text{(definition)}
\end{aligned}
$$

* Here: $\sqrt{G} = 2400$, so MCC $= 1400 / 2400 = 0.583$
* MAE: the mean of $|y_i - \hat y_i|$, in the units of $y$

## Why Not Accuracy?

:::: {.columns}
::: {.column width="50%"}
![](plots/07/e1_scores.pdf)
:::
::: {.column width="48%"}
* A rare class: 0.335 of the points
* Always negative: accuracy 0.665, MCC 0
* MCC 1 is perfect, 0 is a coin
* Errors 0.1, 0.1, 0.1, 0.1, 2: MAE 0.48, root mean squared error (RMSE) 0.90
:::
::::

## The Data Sets

Generated sets with default models show **where a model breaks**, not which is best.

| group | sets | train |
|:------------|:-----------------------------------------------------------------|:------------:|
| planes | moons, circles, XOR, skewed blobs, spiral, checkerboard | 600 |
| curves | cubic, sine, exponential, chirp, step, sawtooth | 120 |
| tables | 20 features, Hastie, kinked border, two Friedman sets | 1000 |

* 10 % flipped labels on four planes: MCC stops at **0.80**
* Noise of sd 0.10 on the curves: MAE stops at **0.08**

## Curved Data Breaks Straight Frontiers

Colour: probability of class 1. Red frame: best model of the row.

![](plots/07/e1_planes_a.pdf)

* Logistic regression: MCC 0.00 on circles, **a line cannot close a loop**
* 1-NN draws islands around the flipped labels

## Two Regions per Class Break the Rest

![](plots/07/e1_planes_b.pdf)

* Exclusive or (XOR): logistic and naive Bayes are at MCC 0.05 and 0.00
* Spiral: best MCC 0.63, far below the ceiling of 0.80
* Support vector machine (SVM), multilayer perceptron (MLP): too smooth here

## Small Cells and a Rare Class

![](plots/07/e1_planes_c.pdf)

* Checkerboard: only 1-NN follows the cells (0.75)
* Skewed blobs: the models are close, 0.64 to 0.80

## The Board of Classification

Mean test MCC of five seeds. Grey cells fill up in the next parts.

![](plots/07/e1_board_c.pdf)

* **Each model has a set that breaks it**; none wins everywhere

## Oscillations Break Global Models

![](plots/07/e1_curves_b.pdf)

* Degree 3 has too few bends: MAE 0.52 on the sine
* Support vector regression (SVR) misses the fast chirp: MAE 0.46

## Jumps Break Smooth Models

![](plots/07/e1_curves_c.pdf)

* Step: the tree wins with 0.11, smooth models round the corners
* Sawtooth: **no model is near 0.08**, the best is 1-NN with 0.14

## The Board of Regression

Mean test MAE of five seeds. The colour compares a cell with the best of its row.

![](plots/07/e1_board_r.pdf)

* Friedman 1: polynomial 0.40, the MLP 0.50, 1-NN 2.72
* The ranking changes from row to row, as in classification

## Do the Errors Overlap?

:::: {.columns}
::: {.column width="50%"}
![](plots/07/e1_overlap.pdf)
:::
::: {.column width="48%"}
* 0.317 of the points: right for all six
* 0.030: wrong for all six
* **The models disagree on 0.652**
* That is the room a combination can use
:::
::::

## DEMO E1: `01_shallow_models.ipynb`

# Why Combining Helps

## The Crowd Wins If Voters Are Independent

$B$ voters, each right with chance $\gamma$; the majority is right with the binomial sum.

![](plots/07/e2_condorcet.pdf)

* $\gamma = 0.6$: $B = 3$ gives $0.648$, $B = 11$ gives $0.753$, $B = 101$ gives $0.979$

## Condorcet for Three Voters

<!-- derivation: condorcet
defs: {γ: "0.6"}
tag: "derivation: condorcet" -->
$$
\begin{aligned}
a_3 &= \gamma^3 + 3\gamma^2(1-\gamma) && \text{(definition)} \\
&= 0.6^3 + 3\cdot 0.6^2\cdot 0.4 && \text{(substitute } \gamma = 0.6\text{)}
\end{aligned}
$$

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cgreen, text width=2.2cm] (a) {3 right: $0.216$};
  \node[fillbox=cgreen, text width=2.2cm, right=3mm of a] (b) {2 right: $0.432$};
  \node[fillbox=cred, text width=2.2cm, right=3mm of b] (c) {1 or 0 right: $0.352$};
\end{tikzpicture}
\end{center}
```

* Assumption: the voters err **independently**
* Copies do not help: if half the voters copy, $B = 101$ gives $0.790$

## Averaging Reduces Variance

For regression: the mean of $B$ predictions, each with variance $v$ and pairwise correlation $\rho$:

<!-- derivation: variance-of-mean -->
$$
\begin{aligned}
V_B &= \frac{B\,v + B(B-1)\,\rho\,v}{B^2} && \text{(definition)} \\
&= \rho\,v + \frac{(1-\rho)\,v}{B} && \text{(simplify)}
\end{aligned}
$$

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cblue, text width=1.3cm] (a) {$\hat f_1$};
  \node[fillbox=cblue, text width=1.3cm, right=2mm of a] (b) {$\hat f_2$};
  \node[fillbox=cblue, text width=1.3cm, right=2mm of b] (c) {$\hat f_B$};
  \node[fillbox=corange, text width=2cm, right=8mm of c] (m) {mean: $V_B$};
  \draw[flow] (c) -- (m);
\end{tikzpicture}
\end{center}
```

* With $v = 1$, $\rho = 0.6$: $B = 5$ gives $0.680$, $B = 50$ gives $0.608$

## The Floor Is the Correlation

![](plots/07/e2_variance.pdf)

* $\rho = 0$, $B = 50$: variance $0.020$; $\rho = 0.3$: $0.314$; $\rho = 0.6$: $0.608$
* More members cannot remove the shared part $\rho v$
* **Diverse members matter more than many members**

## Four Families of Ensembles

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cblue, text width=4.6cm] (v) {\textbf{Voting}\\ fit different models, combine the outputs};
  \node[fillbox=cgreen, text width=4.6cm, right=4mm of v] (a) {\textbf{Bagging}\\ one learner on resamples, in parallel: cut \emph{variance}};
  \node[fillbox=corange, text width=4.6cm, below=4mm of v] (b) {\textbf{Boosting}\\ members in sequence, each fixes the last: cut \emph{bias}};
  \node[fillbox=cpurple, text width=4.6cm, below=4mm of a] (c) {\textbf{Meta models}\\ a model learns the combination};
\end{tikzpicture}
\end{center}
```

* Same data, same scores: each family adds a row to the same figures

# Voting

## Hard Vote, Soft Vote, Weighted Vote

The members are the **six shallow models**, each fitted on the whole training set.

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cblue, text width=2.8cm] (a) {\textbf{Hard}: each member names a class, the majority wins};
  \node[fillbox=cgreen, text width=2.8cm, right=3mm of a] (b) {\textbf{Soft}: average the probabilities, take the largest};
  \node[fillbox=corange, text width=2.8cm, right=3mm of b] (c) {\textbf{Weighted}: a soft vote, better members count more};
\end{tikzpicture}
\end{center}
```

* Regression: the **median** plays the hard vote, the **mean** the soft vote

<!-- derivation: soft-vote -->
$$
\begin{aligned}
\hat f_{\text{ens}}(x) &= \arg\max_k \sum_b u_b\, \hat p_b(k \mid x) && \text{(definition)}
\end{aligned}
$$

## Where the Weights Come From

$u_b$: the MCC of member $b$ in 5-fold cross-validation on the training set, floored at 0.

![](plots/07/e2_weights.pdf)

* Spiral: 1-NN gets 0.316, logistic 0.082; dotted: equal votes
* Regression: $u_b$ is the inverse of the cross-validated MAE

## k-NN Is a Vote

k-NN with $k > 1$ is a vote of the $k$ nearest points: each neighbour is one voter.

![](plots/07/e2_knn.pdf)

* Spiral: $k = 1$ gives 0.635, $k = 15$ gives **0.802**, $k = 101$ gives 0.082
* **Weighted k-NN:** a closer neighbour counts more
* Too many neighbours vote across the frontier

## Voting on the Spiral

The vote panels say which shallow models they use.

![](plots/07/e2_gallery_spiral.pdf)

* 15-NN follows the spiral and ignores the flipped labels (0.80)
* The soft vote (0.65) beats each member

## Voting Fills Five Columns

![](plots/07/e2_board_c.pdf)

* Spiral 0.802, skewed blobs 0.817, moons 0.780: new best scores
* Tables: the votes stay below the SVM

## Is the Voting Gain Real?

Gain: best voter minus best shallow model. Green: won on all five seeds.

![](plots/07/e2_gain_c.pdf)

* **Four wins of nine:** spiral $+0.167$, checkerboard $+0.039$, skewed blobs $+0.030$, moons $+0.023$
* Circles and XOR sit at the ceiling: nothing is left to gain

## Voting in Regression

![](plots/07/e2_board_r.pdf)

* The MLP is at the floor on smooth curves: the vote can only lose
* Only the sawtooth gains: 5-NN by distance 0.132 against 0.142
* **A vote cannot beat a member that is already right**

## DEMO E2: `02_voting.ipynb`

# Bagging

## The Bootstrap Sample

To get $B$ different training sets from one, draw $n$ examples **with replacement**.

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cgray, text width=2.2cm] (d) {data $D$\\ $n$ examples};
  \node[fillbox=cblue, text width=2.2cm, right=12mm of d, yshift=0.7cm] (d1) {sample $D_1$\\ some twice, some never};
  \node[fillbox=cblue, text width=2.2cm, below=2mm of d1] (d2) {sample $D_2$};
  \node[fillbox=cblue, text width=2.2cm, below=2mm of d2] (d3) {sample $D_B$};
  \draw[flow] (d.east) -- (d1.west); \draw[flow] (d.east) -- (d2.west); \draw[flow] (d.east) -- (d3.west);
\end{tikzpicture}
\end{center}
```

* Each sample holds about two thirds of the distinct examples

## How Many Distinct Examples?

<!-- derivation: bootstrap-share -->
$$
\begin{aligned}
\xi &= \Big(1-\frac1n\Big)^{n} && \text{(definition)} \\
\xi &\to e^{-1} && \text{(identity: limit)}
\end{aligned}
$$

![](plots/07/e3_share.pdf)

* $\xi = 0.368$: all $n$ draws miss example $i$
* That left-out share is the **out-of-bag** (OOB) set of the member

## Bagging in Action: the Checkerboard

![](plots/07/e3_bagging.pdf)

* One deep tree 0.72; $B = 3, 10, 100$: MCC 0.81, 0.82, 0.82
* `BaggingClassifier` with 100 trees: 0.830; by hand: 0.819

## Variance Cut, Bias Kept

The decomposition needs the squared error: 200 training sets, noise $0.25$.

![](plots/07/e3_biasvar.pdf){height=4.3cm}

* One tree: bias$^2$ 0.105, variance 1.038
* Bagged: 0.131, 0.140; forest ($m = 5$): 0.125, 0.097
* Variance falls by a factor 7; the bias barely moves

## Out-of-Bag Score: Free Validation

Predict each example with the trees that did **not** train on it.

![](plots/07/e3_oob.pdf)

* $B = 5, 10, 100$: OOB 0.699, 0.735, 0.775; test 0.789, 0.798, 0.803
* OOB is **pessimistic**: each vote uses a third of the trees

## From Bagging to a Random Forest

Bagged trees are correlated: they all pick the same strong feature first.

**Random forest:** at each split consider only $m$ of the $d$ features.

![](plots/07/e3_maxfeat.pdf){height=3.2cm}

* $m = 1$: $\rho = 0.110$, 0.707; $m = 10$: 0.768; $m = 20$: $\rho = 0.277$, 0.746

## Bagging Joins the Gallery

![](plots/07/e3_gallery_checker.pdf)

* Bagged trees reach **0.83**, the best so far: squares suit straight cuts

## Bagging Fills Two Columns

![](plots/07/e3_board_c.pdf)

* Checkerboard: bagged trees 0.818 against 15-NN by distance 0.789
* Spiral: 0.727 at most, below 15-NN (0.802)
* Hastie: trees stay near 0.70, the SVM has 0.902

## DEMO E3: `03_bagging_forests.ipynb`

# Boosting

## The Idea: Fix the Mistakes in Turn

Train a weak model, find what it gets wrong, make those examples count more, train the next.

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cblue, text width=1.8cm] (a) {member 1\\ equal weights};
  \node[fillbox=corange, text width=1.8cm, right=9mm of a] (b) {member 2\\ errors weigh more};
  \node[fillbox=corange, text width=1.8cm, right=9mm of b] (c) {member 3\\ errors weigh more};
  \node[fillbox=cgreen, text width=1.8cm, right=9mm of c] (d) {weighted vote};
  \draw[flow] (a) -- (b); \draw[flow] (b) -- (c); \draw[flow] (c) -- (d);
\end{tikzpicture}
\end{center}
```

* The members are **stumps**: trees with one question (Class 06)
* Bagging cuts variance with deep trees; boosting cuts **bias** with shallow ones

## AdaBoost on Ten Points: Round 1

Ten points; every weight starts at $w_i^{(1)} = 1/10$.

![](plots/07/e4_weights.pdf){height=2.6cm}

* Stump $x_1 \le 0.053$ misses points 2 and 4: $\text{err}_1 = 0.2$
* Wrong points: $0.10 \to 0.25$; the others $0.10 \to 1/16$

## The Member Weight

<!-- derivation: beta -->
$$
\begin{aligned}
E_b &= (1-\text{err}_b)\,e^{-\beta_b} + \text{err}_b\,e^{\beta_b} && \text{(definition)} \\
e^{2\beta_b} &= \frac{1-\text{err}_b}{\text{err}_b} && \text{(set the derivative to zero)} \\
\beta_b &= \tfrac12\ln\frac{1-\text{err}_b}{\text{err}_b} && \text{(solve for $\beta_b$)}
\end{aligned}
$$

![](plots/07/e4_beta.pdf){height=1.9cm}

* $\text{err}_1 = 0.2$: $\beta_1 = \tfrac12\ln 4 = 0.693$, $E_1 = 0.800$

## The Weight Update

<!-- derivation: update -->
$$
\begin{aligned}
w_i^{(b+1)} &= \frac{w_i^{(b)}\,e^{-\beta_b\, y_i\, \hat f_b(x_i)}}{Z_b} && \text{(definition)}
\end{aligned}
$$

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cgreen, text width=3cm] (a) {right point: weight shrinks};
  \node[fillbox=cred, text width=3cm, right=4mm of a] (b) {wrong point: weight grows};
\end{tikzpicture}
\end{center}
```

* $Z_b$ makes the weights sum to 1; labels $y_i, \hat f_b(x_i) \in \{-1,+1\}$
* Round 2: $\text{err}_2 = 0.125$, $\beta_2 = 0.973$; round 3: $0.214$, $0.650$
* The vote $\operatorname{sign}\sum_b \beta_b \hat f_b(x)$ is right on 10 of 10

## AdaBoost in Action: the Circles

Red rings: the weight of each point.

![](plots/07/e4_adaboost.pdf)

* After rounds 1, 2, 5, 50: test MCC 0.28, 0.28, 0.58, 0.72
* The largest weight reaches 4.8 times the starting $1/n$
* Hand-made and `AdaBoostClassifier` agree on every test point

## More Rounds: When to Stop

![](plots/07/e4_rounds.pdf)

* Stumps: test MCC 0.754 at 300 rounds, train 0.790
* Depth 3: best 0.743 at 10 rounds, then 0.689 with train 0.991
* Boosting **can overfit**: the rounds are a hyperparameter

## Gradient Boosting: Fit the Residuals

Think of adding a member as a **step of gradient descent in the space of functions**.

* Start from a constant $F_0$ (the mean)
* Squared loss: the negative gradient is the residual $r_i = y_i - F_{b-1}(x_i)$
* Fit a small tree $\hat f_b$ to the residuals; set $F_b = F_{b-1} + \eta\,\hat f_b$
* Eight points, $\eta = 0.5$, stumps: mean squared error (MSE) $0.461 \to 0.372 \to 0.337$

## Boosting Joins the Gallery

Spiral, one seed: the row of trees now has five panels.

![](plots/07/e4_gallery_spiral.pdf)

* Trees reach 0.68 to 0.73: staircases on a round frontier
* 15-NN stays the best (0.80)

## Boosting Fills Three Columns

![](plots/07/e4_board_r.pdf)

* Step: AdaBoost 0.108 wins; sawtooth: gradient boosting 0.130
* Friedman kinks: XGBoost 1.05, the MLP 0.73

## DEMO E4: `04_boosting.ipynb`

# Meta Models

## Learn the Combination

A **meta model** takes the outputs of the members as its features.

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cblue, text width=1.8cm] (a) {1-NN};
  \node[fillbox=cblue, text width=1.8cm, below=2mm of a] (b) {tree};
  \node[fillbox=cblue, text width=1.8cm, below=2mm of b] (c) {SVM};
  \node[fillbox=corange, text width=2.2cm, right=14mm of b] (m) {meta model\\ logistic regression};
  \node[fillbox=cgreen, text width=1.6cm, right=10mm of m] (o) {$\hat y$};
  \draw[flow] (a.east) -- (m.west); \draw[flow] (b.east) -- (m.west); \draw[flow] (c.east) -- (m.west); \draw[flow] (m) -- (o);
\end{tikzpicture}
\end{center}
```

* This is **stacking**: the weights of a vote are learned, not chosen

## The Leakage Trap

* A member predicts its own training points too well
* Feed those predictions to the meta model and it trusts the members too much
* **Out-of-fold predictions:** each example is predicted by a member that did not see it
* A **blend** uses one hold-out set instead

## Stacking in Action: Leak or No Leak

![](plots/07/e5_stack.pdf)

* In-fold features: cross-validation (CV) MCC **1.000**, test 0.677
* Out-of-fold features: CV 0.650, test 0.683: an honest estimate
* Weights in-fold 4.2, 4.2, 0.65: the SVM is almost ignored

<!-- defines: neural stacking, frozen -->

## Neural Stacking: Stack What Networks Learned

Stacking reads predictions. **Neural stacking** reads the last hidden layers.

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cblue, text width=2.5cm] (a) {\textbf{NN A}\\ ReLU 128, 128, 128};
  \node[fillbox=cgreen, text width=2.5cm, below=4mm of a] (b) {\textbf{NN B}\\ GELU 64, 64, 64, 64};
  \node[fillbox=cgray, text width=1.5cm, right=7mm of a] (oa) {output removed};
  \node[fillbox=cgray, text width=1.5cm, right=7mm of b] (ob) {output removed};
  \node[fillbox=cpurple, text width=2cm, right=24mm of a, yshift=-7mm] (h) {\textbf{head}\\ 32 units, output};
  \draw[flow] (a) -- (oa); \draw[flow] (b) -- (ob);
  \draw[flow] (a.south east) -- (h.west); \draw[flow] (b.north east) -- (h.west);
\end{tikzpicture}
\end{center}
```

* Blue and green are **frozen**: `fit` changes only the head

## The Recipe

1. Build two networks of different topology
2. Train both on the task; 20 % of the data stops them early
3. Remove the output layers and freeze both
4. Concatenate the last hidden layers; add one hidden layer and an output
5. Train only this head, with the same early stopping

* Tables use tanh (256, 64) and swish (128, 128, 32)
* Regression trains on squared error and reports MAE

## The Formula

$z_A(x)$, $z_B(x)$: last hidden layers of the frozen networks. $g$: the head, the only trainable part.

<!-- derivation: neural-stack -->
$$
\begin{aligned}
\hat y(x) &= g\big(\,[\,z_A(x),\, z_B(x)\,]\,\big) && \text{(definition)}
\end{aligned}
$$

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cblue, text width=1.6cm] (a) {$z_A(x)$};
  \node[fillbox=cgreen, text width=1.6cm, right=2mm of a] (b) {$z_B(x)$};
  \node[fillbox=cpurple, text width=1.2cm, right=8mm of b] (g) {$g$};
  \node[fillbox=cgray, text width=1.2cm, right=8mm of g] (y) {$\hat y$};
  \draw[flow] (b) -- (g); \draw[flow] (g) -- (y);
\end{tikzpicture}
\end{center}
```

* A hidden layer is a set of features built for this task
* Two topologies give two feature sets: **diversity again**

## The Complete Gallery: Spiral

![](plots/07/e5_gallery_spiral.pdf)

## The Complete Gallery: Sawtooth

![](plots/07/e5_gallery_saw.pdf)

## The Board Is Complete

![](plots/07/e5_board_c.pdf)

* Generated sets, defaults: voting 3, meta 3, bagging 1, boosting 1
* Neural stack: first on XOR (0.725), 0.812 on the checkerboard

## The Regression Board Is Complete

![](plots/07/e5_board_r.pdf)

* Smooth curves stay with the shallow MLP and the polynomial: the floor is reached
* Friedman kinks: neural stack 0.525, the best of all

## Does the Neural Stack Win?

Neural stack minus the best other model, five seeds.

![](plots/07/e5_gain_r.pdf)

* Friedman kinks: $+0.045$ over NN B, won on 4 of 5 seeds
* On one input, with 120 points, simple models are better
* Friedman 1: NN B alone wins (0.304 against 0.338)

## DEMO E5: `05_stacking_neural.ipynb`

# Tuning

## Parameters and Hyperparameters

`fit` learns the parameters. You choose the hyperparameters before `fit`.

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cblue, text width=4.4cm] (a) {\textbf{parameters}\\ learned by \texttt{fit}: the splits of a tree};
  \node[fillbox=corange, text width=4.4cm, right=4mm of a] (b) {\textbf{hyperparameters}\\ chosen first: depth, $\eta$, $B$, $k$};
\end{tikzpicture}
\end{center}
```

* Ensembles have many hyperparameters
* An untuned comparison says little: this is why the earlier boards used defaults only to find weaknesses

## Three Ways to Search

Radial basis function (RBF) SVM, 25 trials; yellow is best.

![](plots/07/e6_space.pdf)

* A grid tries every combination; most points are wasted
* Random points cover each hyperparameter better (Bergstra and Bengio)
* **Tree-structured Parzen estimator** (TPE): near good trials

## Learning from Past Trials

**TPE** splits past trials into good and bad.

![](plots/07/e6_search.pdf)

* After 5 trials: grid 0.000, random 0.811, TPE 0.828
* After 25 trials: 0.846, 0.848, 0.851
* Optuna runs this search in a few lines

## Pruning: Stop a Bad Trial Early

XGBoost, 30 trials. Each trial reports its score every 25 trees.

![](plots/07/e6_pruning.pdf)

* Median pruner: stop a trial below the median of earlier trials
* Same best MCC, 0.893: 3750 trees against 9000

## The Best Score Is Optimistic

Picking the best of many trials also picks lucky noise.

![](plots/07/e6_nested.pdf)

* Heart disease, 303 patients: inner 0.701, honest 0.654 at 100 trials
* More trials give a larger gap: 0.005, 0.030, 0.047

## Nested Cross-Validation

Two loops: the inner one tunes, the outer one only scores.

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cblue, text width=3.4cm] (o) {\textbf{outer fold}\\ training part};
  \node[fillbox=cgreen, text width=2.2cm, right=4mm of o] (t) {\textbf{outer fold}\\ test part};
  \node[fillbox=corange, text width=3.4cm, below=4mm of o] (i) {\textbf{inner CV}\\ tune on the training part};
  \draw[flow] (o) -- (i);
\end{tikzpicture}
\end{center}
```

* The test part never touches the tuning
* The same rule as the leak in stacking

## Does Tuning Pay?

Default against 25 Optuna trials, on at most 1500 rows, five folds.

![](plots/07/e6_default.pdf)

* k-NN gains most: 0.475 to 0.601 on wine
* Boosting loses on wine: 0.609 to 0.579 with XGBoost
* Tune on enough data: 1500 of 6497 rows was too few

## DEMO E6: `06_tuning.ipynb`

# Real Tables

## From Toy Data to Real Tables

Generated data showed where models break. Now: which family wins in practice?

* 11 real tables, **all rows**, 5-fold CV, the same folds for every model
* Optuna tunes on the training folds only; boosting stops early
* Stacking uses five strong members: SVM, forest, boosting, k-NN, linear

## Classification on Real Tables

![](plots/07/e7_board_c.pdf)

* Spambase: stacking 0.913, forest 0.903, one tree 0.813
* Wine: stacking 0.632, SVM 0.379
* Small, noisy tables: German credit, 0.394, leaves little to gain

## Regression on Real Tables

![](plots/07/e7_board_r.pdf)

* California housing: stacking 0.301, HistGB 0.304, one tree 0.418
* Concrete: XGBoost 2.619, stacking 2.799, SVM 3.819
* Diabetes, 442 rows: neural stack 42.988, linear 44.270

## Which Model Ranks First?

Rank the models on each table, then average; a standard comparison (Demšar).

![](plots/07/e7_rank.pdf)

* Friedman test $p < 10^{-9}$: the models differ
* Right of the dotted line: clearly behind stacking

## How Much Does Each Family Gain?

Best model of a family against the best shallow model, on each table.

![](plots/07/e7_family.pdf){height=3.3cm}

* Median gain: meta 5.4 %, boosting 3.2 %; up to 31.4 % on concrete
* Small tables gain little: more members need more data

# Is the Gain Real?

## McNemar: Only the Disagreements Count

Two fitted models, one test set: rows both get right say nothing [@dietterich1998].

:::: {.columns}
::: {.column width="42%"}
![](plots/07/e8_table.pdf)
:::
::: {.column width="58%"}
* Equal models: each disagreement is a coin flip
* Spambase: 11 against 13, $p = 0.839$
* MCC 0.912 and 0.915: no real gap
:::
::::

## Which Gains Survive?

Stacking against six rivals on a 30 % hold-out, with Holm's correction per table.

![](plots/07/e8_grid.pdf){height=3.7cm}

* Green: permutation test on the metric: swap predictions on random rows
* Better in 35 of 66, never worse; forest and HistGB fall on 4 of 11 tables
* McNemar agrees on 29 of 30: it counts errors, not MCC

## Regression: No Right or Wrong

No 2×2 table exists: McNemar's twin, the sign test, asks which model came **closer**.

![](plots/07/e8_interval.pdf){height=2.9cm}

* Sign test: closer more often; Wilcoxon: smaller by rank; permutation: lower MAE
* Wine quality: k-NN is closer on 1133 of 1950 rows, yet MAE ties ($p = 0.138$)
* California: $p = 0.001$ for a 0.51 % gain: real, but tiny

## McNemar or Friedman?

Neither is better: one family of tests, at two levels [@demsar2006].

\begin{center}
\begin{tikzpicture}[font=\footnotesize, every node/.style={minimum height=5.5mm, inner sep=1pt}]
\node[anchor=west] at (0,1.2) {\textbf{Samples}};
\node at (5.0,1.2) {\textbf{Two models}};
\node at (8.1,1.2) {\textbf{Many models}};
\draw[gray] (0,0.9) -- (9.6,0.9);
\node[anchor=west] at (0,0.55) {rows, right or wrong};
\node at (5.0,0.55) {McNemar};
\node at (8.1,0.55) {Cochran's Q};
\node[anchor=west] at (0,0) {rows, loss of each row};
\node at (5.0,0) {sign test, Wilcoxon};
\node at (8.1,0) {Friedman on rows};
\node[anchor=west] at (0,-0.55) {tables, one score each};
\node at (5.0,-0.55) {sign test, Wilcoxon};
\node at (8.1,-0.55) {Friedman on tables};
\end{tikzpicture}
\end{center}

* Spambase: Cochran's Q and Friedman on rows both give 124.9
* McNemar: better **here**? Friedman: better **across tables**?

## Do Not Test on the Folds

Folds share training rows; at 0.00 neither model is better [@nadeau2003].

![](plots/07/e8_calibration.pdf){height=3.3cm}

* No true gap: plain t-test rejects 0.528, corrected 0.06, McNemar 0.042
* At 0.40 the corrected test finds 0.475, McNemar 0.168

## Across Tables: After Friedman

A mean-rank pairwise test depends on the other models in the pool [@benavoli2016].

* Wilcoxon with Holm: 11 of 15 models behind stacking; Nemenyi: 6
* Stacking was picked **after** E7: this correction does not pay for that
* All 120 pairs with 11 tables: smallest possible Holm $p$ is 0.117
* Bayesian signed-rank: P(better by over 1 %) at least 0.967 [@benavoli2017]

## DEMO E7: `07_real_tables.ipynb`

Then `08_significance.ipynb`: the tests of this section.

# Summary

## Summary: the Map

| | voting | bagging | boosting | meta |
|:----------|:---------------------|:---------------------|:----------------------|:-----------------|
| idea | combine outputs | average resamples | fix errors in turn | learn the mix |
| cuts | variance | variance | bias | both |
| costs | keeps bad members | memory | tuning, noise | time, leakage |

## Key Takeaways: Ensembles

* A crowd helps only if members are **better than chance and different**
* On real tables stacking ranks first, then boosting, forests and votes
* Averaging leaves a floor $\rho v$: lower $\rho$ with diverse members
* k-NN with $k > 1$ is a vote; bagging needs unstable learners
* Boosting needs a stopping rule; stacking needs out-of-fold predictions
* Tune with nested CV: the best validation score is optimistic
* Test gains: McNemar on one test set, permutation for MCC and MAE
* Near the ceiling no ensemble wins

## Lab 07: Build It from Scratch

| Part | Task | twin |
|:---------:|:-----------------------------------------|:---------------------------------------|
| A | Boundary gallery and a vote of three models | `VotingClassifier` |
| B | Bagging by hand: bootstrap, vote, OOB | `BaggingClassifier`, `RandomForestClassifier` |
| C | AdaBoost with stumps | `AdaBoostClassifier` |
| D | Gradient boosting, shrinkage, early stopping | `GradientBoostingRegressor` |
| E | Stacking with out-of-fold predictions | `StackingClassifier` |
| F | Challenge: pseudo-residuals with `jax.grad` | `HistGradientBoostingClassifier` |

* **In class:** Parts A to D (60 minutes); **at home:** Parts E and F
* Guide `practice/07_ensembles.pdf`; notebook `lab07_ensembles.ipynb`

<!-- cites: @breiman1996 @breiman2001 @freund1997 @friedman2001 @bergstra2012 @akiba2019 @demsar2006 @wolpert1992 @chen2016 @hastie2009 @grinsztajn2022 -->

<!-- refs:begin -->
## References

:::: {.columns}
::: {.column width="50%"}
\fontsize{5.5}{6.4}\selectfont\setlength{\parskip}{2.5pt}

Akiba, Takuya, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. 2019. “Optuna: A Next-Generation Hyperparameter Optimization Framework.” ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2623–31.

Benavoli, Alessio, Giorgio Corani, Janez Demšar, and Marco Zaffalon. 2017. “Time for a Change: A Tutorial for Comparing Multiple Classifiers Through Bayesian Analysis.” Journal of Machine Learning Research 18 (77): 1–36.

Benavoli, Alessio, Giorgio Corani, and Francesca Mangili. 2016. “Should We Really Use Post-Hoc Tests Based on Mean-Ranks?” Journal of Machine Learning Research 17 (5): 1–10.

Bergstra, James, and Yoshua Bengio. 2012. “Random Search for Hyper-Parameter Optimization.” Journal of Machine Learning Research 13: 281–305.

Breiman, Leo. 1996. “Bagging Predictors.” Machine Learning 24 (2): 123–40.

Breiman, Leo. 2001. “Random Forests.” Machine Learning 45 (1): 5–32.

Chen, Tianqi, and Carlos Guestrin. 2016. “XGBoost: A Scalable Tree Boosting System.” ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–94.

Demšar, Janez. 2006. “Statistical Comparisons of Classifiers over Multiple Data Sets.” Journal of Machine Learning Research 7: 1–30.

:::
::: {.column width="50%"}
\fontsize{5.5}{6.4}\selectfont\setlength{\parskip}{2.5pt}

Dietterich, Thomas G. 1998. “Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms.” Neural Computation 10 (7): 1895–923.

Freund, Yoav, and Robert E. Schapire. 1997. “A Decision-Theoretic Generalization of on-Line Learning and an Application to Boosting.” Journal of Computer and System Sciences 55 (1): 119–39.

Friedman, Jerome H. 2001. “Greedy Function Approximation: A Gradient Boosting Machine.” The Annals of Statistics 29 (5): 1189–232.

Grinsztajn, Léo, Edouard Oyallon, and Gaël Varoquaux. 2022. “Why Do Tree-Based Models Still Outperform Deep Learning on Typical Tabular Data?” Advances in Neural Information Processing Systems 35.

Hastie, Trevor, Robert Tibshirani, and Jerome Friedman. 2009. The Elements of Statistical Learning. 2nd ed. Springer.

Nadeau, Claude, and Yoshua Bengio. 2003. “Inference for the Generalization Error.” Machine Learning 52 (3): 239–81.

Wolpert, David H. 1992. “Stacked Generalization.” Neural Networks 5 (2): 241–59.

:::
::::
<!-- refs:end -->
