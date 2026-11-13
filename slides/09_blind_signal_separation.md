---
title: Fundamentos de Aprendizagem Automática
subtitle: "Class 09 — Blind Signal Separation"
bibliography: "references.bib"
suppress-bibliography: true
---

# One Family: Decompose the Data

## Where We Left Off

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cblue, text width=2.4cm] (a) {\textbf{supervised}\\ Classes 03--07: labels $y_i$};
  \node[fillbox=cgreen, text width=2.4cm, right=8mm of a] (b) {\textbf{no labels, one model}\\ Class 08: $k$-means, PCA preview};
  \node[fillbox=cred, text width=5cm, below=7mm of b] (d) {\textbf{Class 09:} decompose $X$ into latent factors};
\end{tikzpicture}
\end{center}
```

* Today there are no labels and no clusters to find
* One data matrix, three ways to pull it apart

## Roadmap of the Class

| Part | Question | Methods |
|:---------------------------|:-------------------------|:-------------------------------------|
| **The problem** | what does *blind* mean? | $X = AS$, limits |
| **principal component analysis (PCA)** | keep the most spread? | eigenvectors, scree |
| **independent component analysis (ICA)** | make the parts independent? | kurtosis, FastICA |
| **NMF** | make the parts non-negative? | multiplicative updates |
| **Applications** | sound, images, documents, ratings | latent semantic indexing (LSI), collaborative filtering |
| **Use cases** | one task, three methods | topics, strokes, denoising, blink removal, imputation |

<!-- defines: source, mixture, mixing matrix, latent factor, loading, score -->

## Terminology

A **source** is a hidden signal; a **mixture** is what a sensor records.

The **mixing matrix** $A$ says how sources combine.

| term | meaning | example |
|:----------------------|:--------------------------------------------|:------------------------|
| source | hidden signal | one speaker |
| mixture | what is recorded | microphone output |
| mixing matrix | how sources combine | $X = AS$ |
| loading | a learned column of the factorisation; for PCA, one eigenvector | an NMF part |
| component | one direction of PCA | PC1 |
| score | how much of a component an example has | coordinate of PC1 |
| singular value $\sigma_j$ | size of the $j$-th rank-one term of $X$ | $\lambda_j = \sigma_j^2/(n-1)$ |
| rank-one term | a column times a row, $w_j h_j$ | one stroke of a digit |

<!-- defines: loading, score, eigenvalue, singular value, rank-one term -->

## Notation N9: Factors and Mixtures

Each symbol is new in this class; earlier symbols keep their meaning.

| symbol | meaning | shape | example |
|:------------------|:---------------------------------------|:-------------:|:--------------------|
| $S$ | sources (rows) | $c \times n$ | two sines |
| $A$ | mixing matrix | $d \times c$ | $2 \times 2$ |
| $X$ | data matrix $= AS$ | $d \times n$ | two microphones |
| $c$ | number of components (rank) | scalar | $c = 2$ |
| $W$ | NMF parts (columns) | $d \times c$ | pixel parts |
| $H$ | NMF factors | $c \times n$ | how much of each part |
| $R$ | ratings matrix (users $\times$ items) | $n \times m$ | $6 \times 8$ |
| $w_j$ | PCA direction (unit vector) | $d$ | first PC |
| $C$ | covariance of the centred data, $C = XX^\top/(n-1)$ | $d \times d$ | $2 \times 2$ |
| $I$ | identity matrix | $d \times d$ | ones on the diagonal |
| $\lambda, \lambda_e$ | eigenvalue $e$ of the covariance | scalar | $3.2$ |

## Notation N9b: SVD and Rank-One Terms

The singular value decomposition (SVD) of the centred data.

| symbol | meaning | shape | example |
|:------------|:-----------------------------------------|:---------------:|:---------------------|
| $U$ | left singular vectors: $X = U D_\sigma B^\top$ | $d \times c$ | 64 pixels |
| $D_\sigma$ | diagonal of the singular values $\sigma_j$ | $c \times c$ | $\sigma_1 = 35.4$ |
| $B$ | right singular vectors | $n \times c$ | 1797 digits |
| $u_j$ | column $j$ of $U$ | $d$ | a loading image |
| $b_j$ | column $j$ of $B$ | $n$ | unit-scale weights |
| $X_k$ | the first $k$ rank-one terms of $X$ | $d \times n$ | $k = 16$ |

## Notation N9c: Coordinates and Scores

| symbol | meaning | shape | example |
|:--------------|:-------------------------------------|:--------------:|:------------------------|
| $Z$ | coordinates of the examples, $Z = W^\top X$ | $c \times n$ | PC1 values |
| $a_j$ | column $j$ of $A$ | $d$ | one voice at the sensors |
| $s_j$ | row $j$ of $S$ | $n$ | one voice over time |
| $\rho$ | correlation of an estimate with its source | scalar | 0.996 |
| $\text{SNR}$ | signal-to-noise ratio of an estimate, in dB | scalar | 20.6 |

<!-- defines: signal-to-noise ratio, SNR -->

# The Problem: Separating Without Labels

## Two Microphones, Two Sines

![](plots/09/p1_mixtures.pdf)

* Left: the two **sources**; right: the mixtures

## What Is Blind?

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cblue, text width=1.6cm] (s) {$S$};
  \node[fillbox=cgreen, text width=1.6cm, right=9mm of s] (a) {$A$};
  \node[fillbox=corange, text width=1.6cm, right=9mm of a] (x) {$X$};
  \draw[flow] (s) -- node[above]{$\times$} (a);
  \draw[flow] (a) -- (x);
  \node[fillbox=cgray, text width=1.4cm, below=6mm of a] (q) {both unknown};
\end{tikzpicture}
\end{center}
```

* We observe only $X$; $A$ and $S$ are both unknown
* $X = AS$: the columns of $A$ mix the $c$ sources
* Separation is possible only up to **scale** and **order** of the sources

## The Shared Skeleton

Every method of this class fits the same shape.

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cblue, text width=2.6cm] (a) {data matrix $X$};
  \node[fillbox=cgreen, text width=2.6cm, right=5mm of a] (b) {parts $W$};
  \node[fillbox=corange, text width=2.6cm, right=5mm of b] (c) {weights $H$};
  \draw[->] (a) -- node[above]{$\approx$} (b);
  \draw[->] (b) -- node[above]{$\times$} (c);
\end{tikzpicture}
\end{center}
```

<!-- derivation: skeleton -->
$$
\begin{aligned}
X &\approx \sum_{j=1}^{c} w_j\, h_j && \text{(definition)} \\
&= W H && \text{(definition of } W \text{ and } H\text{)}
\end{aligned}
$$

* $W$: one column $w_j$ per component (a direction, a part, a topic)
* $H$: row $h_j$ holds how much of component $j$ every example has
* Each product $w_j h_j$ is a $d \times n$ matrix of rank one
* The methods differ in **one constraint** on $W$ or $H$

## Three Constraints, Three Methods

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cblue, text width=2.8cm] (a) {\textbf{PCA}\\ $w_j$ orthogonal};
  \node[fillbox=cgreen, text width=2.8cm, right=5mm of a] (b) {\textbf{ICA}\\ $h_j$ independent};
  \node[fillbox=corange, text width=2.8cm, right=5mm of b] (c) {\textbf{NMF}\\ all entries $\geq 0$};
\end{tikzpicture}
\end{center}
```

* Choose by what the data *is*, not by what is fashionable

## Why Gaussian Sources Kill It

* If the sources are Gaussian, $AS$ is Gaussian for **every** invertible $A$
* Every rotation of a Gaussian cloud is an equally good answer
* Demo 02: Gaussian sources, 0.997 correlation — kurtosis is 0, nothing to maximise
* Non-Gaussianity is the only signal that survives the mixture

<!-- defines: kurtosis -->

## How We Score a Separation

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cblue, text width=2.0cm] (a) {estimates\\ any order, any sign};
  \node[fillbox=cgreen, text width=2.0cm, right=5mm of a] (b) {match each to its source};
  \node[fillbox=corange, text width=2.0cm, right=5mm of b] (c) {correlation $\rho$};
  \node[fillbox=cgray, text width=2.0cm, right=5mm of c] (d) {SNR in dB};
  \draw[->] (a) -- (b);
  \draw[->] (b) -- (c);
  \draw[->] (c) -- (d);
\end{tikzpicture}
\end{center}
```

<!-- derivation: snr -->
$$
\begin{aligned}
\text{SNR} &= 10 \log_{10} \frac{\rho^2}{1 - \rho^2} && \text{(definition)}
\end{aligned}
$$

* $\rho$ = 0.5, 0.7, 0.9, 0.99, 0.999: $-4.8$, $-0.2$, 6.3, 16.9, 27.0
* Every 10 dB cuts the error power tenfold

# PCA

## The Idea: Keep the Most Spread

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cblue, text width=3.6cm] (a) {\textbf{maximise variance}\\ along one direction $w_1$};
  \node[fillbox=cgreen, text width=3.6cm, right=5mm of a] (b) {\textbf{minimise error}\\ reconstruct from $c$ directions};
\end{tikzpicture}
\end{center}
```

* Variance maximisation and error minimisation are the same problem
* The solution is an **eigenproblem** on the covariance

## The Geometry in One Picture

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \draw[rotate=28,cblue,thick] (0,0) ellipse (2.4 and 1.0);
  \draw[->,thick,cgreen] (0,0) -- (28:2.7) node[above right]{$w_1$: most spread};
  \draw[->,thick,corange] (0,0) -- (118:1.5);
  \node[corange] at (118:1.95) {$w_2$: orthogonal to $w_1$};
\end{tikzpicture}
\end{center}
```

* The cloud is an ellipse; its axes are the eigenvectors of the covariance
* The long axis wins: it holds $\lambda_1$ of the total spread $\text{tr}(C)$
* Orthogonality is a *consequence* of least squares, not a goal we chose

## Variance Is Reconstruction Error

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cblue, text width=3.8cm] (a) {\textbf{minimise error}\\ $\|X - ww^\top X\|^2$};
  \node[fillbox=cgreen, text width=3.8cm, right=6mm of a] (b) {\textbf{maximise variance}\\ $w^\top C w = \lambda_1$};
  \draw[->] (a) -- node[above]{$\Leftrightarrow$} (b);
\end{tikzpicture}
\end{center}
```

<!-- derivation: pca-error -->
$$
\begin{aligned}
\|X - ww^\top X\|_F^2 &= (n{-}1)\,\text{tr}\!\big[(I - ww^\top)\, C\big] && \text{(definition of } C\text{)} \\
&= (n{-}1)\big[\text{tr}(C) - \text{tr}(ww^\top C)\big] && \text{(identity, } \text{tr is linear)} \\
&= (n{-}1)\big[\text{tr}(C) - w^\top C w\big] && \text{(identity, cyclic trace)}
\end{aligned}
$$

(assumption) unit $w$, centred $X$; (result) minimising error = maximising $w^\top C w$

## The Eigen Equation, From the Lagrangian

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cblue, text width=2.6cm] (l) {Lagrangian};
  \node[fillbox=cgreen, text width=2.6cm, right=6mm of l] (s) {stationary point};
  \node[fillbox=corange, text width=2.6cm, right=6mm of s] (e) {$Cw = \lambda w$};
  \draw[->] (l) -- (s);
  \draw[->] (s) -- (e);
\end{tikzpicture}
\end{center}
```

<!-- derivation: pca-eig -->
$$
\begin{aligned}
\mathcal{L} &= w^\top C w - \lambda (w^\top w - 1) && \text{(definition, Lagrangian)} \\
\nabla_w \mathcal{L} &= 2 C w - 2 \lambda w && \text{(chain rule)} \\
2 C w - 2 \lambda w &= 0 && \text{(set the gradient to zero)} \\
C w &= \lambda w && \text{(at the optimum)}
\end{aligned}
$$

(assumption) unit $w$; (result) the best $w$ is an **eigenvector** of $C$

## Scores and Loadings

* At the optimum: $w^\top C w = \lambda_1$, error $(n-1)\,[\text{tr}(C) - \lambda_1]$
* **Loadings**: the eigenvectors $w_j$; sklearn keeps them as rows of `components_`
* **Scores**: coordinates of every example, $Z = W^\top X$ ($c \times n$; `transform`)
* With $c$ components: $X \approx W_c Z_c$, the rank-$c$ reconstruction
* Demo 01: rank-1 reconstruction of the mixture cloud, relative error 0.280
* Scores are new coordinates; loadings say *how* the old ones mix into them

## SVD: The Basis of PCA

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cgray, text width=1.5cm] (x) {$X$\\ $d \times n$};
  \node at ($(x.east)+(4mm,0)$) {$=$};
  \node[fillbox=cblue, text width=1.5cm, right=8mm of x] (u) {$U$\\ $d \times c$\\ loadings};
  \node[fillbox=corange, text width=1.2cm, right=3mm of u] (s) {$D_\sigma$\\ $c \times c$\\ sizes};
  \node[fillbox=cgreen, text width=1.9cm, right=3mm of s] (v) {$B^\top$\\ $c \times n$\\ unit scale};
\end{tikzpicture}
\end{center}
```

<!-- derivation: svd-pca -->
$$
\begin{aligned}
C &= \tfrac{1}{n-1} X X^\top && \text{(definition of } C\text{)} \\
&= \tfrac{1}{n-1}\, U D_\sigma B^\top B D_\sigma U^\top && \text{(definition, } X = U D_\sigma B^\top\text{)} \\
&= U \,\tfrac{D_\sigma^2}{n-1}\, U^\top && \text{(identity, } B^\top B = I\text{)}
\end{aligned}
$$

(assumption) centred $X$; (result) loadings = columns of $U$, $\lambda_j = \sigma_j^2/(n-1)$

* Coordinates $Z = D_\sigma B^\top$; whitened: $\sqrt{n-1}\,B^\top$; no covariance matrix needed

## The Decomposition: Rank-One Terms

![](plots/09/p1_svd_terms.pdf)

* $X = \sum_j \sigma_j u_j b_j^\top$: each digit is a sum of loading images $u_j$ weighted by $\sigma_j b_j[i]$
* Demo 01 (1797 digits): $\sigma_1,\dots,\sigma_4 = 35.4, 33.9, 31.5, 26.6$; $\lambda_j = \sigma_j^2/(n-1)$ matches sklearn

## Fewer Singular Values: the Price

![](plots/09/p1_svd_recon.pdf)

* $\|X - X_k\|_F^2 = \sum_{j>k} \sigma_j^2$: $X_k$ is the best rank-$k$ matrix (Eckart–Young)
* $k=16$: error 0.388, variance kept 0.849
* $k=1$: error 0.923, variance kept 0.149

## A Century Old, Still Everywhere

* Karl Pearson (1901): fit planes to points by least squares
* Harold Hotelling (1933): the same eigenvectors from maximising variance
* One idea, two entrances: **minimum error** = **maximum variance**
* Names you will meet: Karhunen–Loève transform (signal processing), singular value decomposition (numerical linear algebra)

## PCA in Five Steps

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cgray, text width=1.7cm] (a) {1. centre};
  \node[fillbox=cblue, text width=2.0cm, right=4mm of a] (b) {2. covariance};
  \node[fillbox=corange, text width=1.7cm, right=4mm of b] (c) {3. eigen\\ $w_j$};
  \node[fillbox=cgreen, text width=1.7cm, right=4mm of c] (d) {4. project};
  \node[fillbox=cred, text width=1.7cm, right=4mm of d] (e) {5. rebuild};
  \draw[->] (a) -- (b);
  \draw[->] (b) -- (c);
  \draw[->] (c) -- (d);
  \draw[->] (d) -- (e);
\end{tikzpicture}
\end{center}
```

* sklearn: `fit`, then `transform` (step 4), `inverse_transform` (step 5)
* Step 3 by SVD gives the same $w_j$ without a covariance matrix
* The cut $c$ enters only at steps 4 and 5

## PCA by Hand: Ten Points

![](plots/09/p1_pca_hand.pdf)

* Mean (1.81, 1.91); $C$: 0.617, 0.717, covariance 0.615
* Eigenvalues 1.284, 0.049; $w_1$ = (0.678, 0.735); first-component share 0.963
* Rank-1 error 0.192; error$^2$ = 0.037 = second share

## Whitening: the Shared First Step

* PCA scores $z = W^\top x$ already have zero covariance: $\operatorname{Cov}(z) = D$
* **Whitening** goes further: $z \to D^{-1/2} z$ gives covariance $I$
* PCA = decorrelate; whiten = decorrelate **and** rescale
* ICA needs whitened input: the free rotation is where independence lives

## PCA in Action

![](plots/09/p1_pca_axes.pdf)

* Explained variance ratio (EVR) of the cloud: **0.922 / 0.078**
* Correlations 0.712, signal-to-noise ratio (SNR) 0.1 dB

## DEMO B1: `01_mixtures_pca.ipynb`

## The Eigenvalues Read the Spread

* $\lambda_1 / \sum_j \lambda_j$: the fraction of variance the first component holds
* Our cloud: $0.922 / 0.078$ — one direction carries almost everything
* Spambase: 57 eigenvalues, a slow decay — many directions, each small
* The eigenvalues are the variance budget: every rank-$c$ reconstruction spends only $\lambda_1 + \dots + \lambda_c$ of it

## How Many Components? The Scree Plot

![](plots/09/p1_scree.pdf)

* Spambase: 90 % of the variance at **43** of 57 components
* Flat scree (Gaussian data): no component is special

## PCA on Images

![](plots/09/p1_digits.pdf)

* Rank 16 already reads as a digit; rank 4 is a blur
* Raw first component (PC1) correlates 0.927 with pixel intensity, standardised 0.116

## PCA Limits

* Principal component directions are **orthogonal**: fine for spread, wrong for separation
* Scale-dependent: rescale first, or PC1 chases the big units
* A rare but telling feature has small variance: principal component analysis drops it
* Cannot separate sources that share covariance structure

## Variance Sees Only Second Orders

* Covariance compares variables two at a time: $\operatorname{Cov}(x_i, x_j)$
* Structure in third or fourth moments is invisible to it
* PCA cannot untangle the two sines: whitened scores uncorrelated ($\rho = 0.00$), truth needs an oblique cut
* Higher orders are ICA's job — the next block

## Preprocessing Changes the Answer

* Centring is mandatory: on raw spambase, PC1 is document length
* Different units? The largest-scale feature wins
* Standardised spambase: 90 % variance needs 43 components; raw features: 1
* Whitening before ICA is not optional: FastICA requires it

## PCA, Regression and Autoencoders

![](plots/09/p1_pca_vs_reg.pdf)

* Regression: vertical errors; PCA: orthogonal ones
* Angles 47.6°, 57.9° (regressions), 54.4° (PC1)
* Orthogonal error 71.6, 64.8, **62.3** (PC1)
* Linear autoencoder, 8 units: mean squared error 0.02410 vs PCA 0.02391

# ICA

## The Blind Separation Promise

![](plots/09/p2_ica_separation.pdf)

* Correlations **0.996**, SNR **20.6 dB**
* PCA gave 0.1 dB on the same data

## Why PCA Fails Here

![](plots/09/p5_three_views.pdf)

* PCA axes are orthogonal; the sources are **not**
* ICA axes cross at the mixing angle; NMF: columns of $W$

## The ICA Model

<!-- derivation: ica-model -->
$$
\begin{aligned}
x &= As && \text{(definition)} \\
z &= V^\top x && \text{(definition, whitened)} \\
s &= Uz && \text{(definition, } U \text{ orthogonal)}
\end{aligned}
$$

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cblue, text width=2.4cm] (m) {mix\\ $x = As$};
  \node[fillbox=cgreen, text width=2.4cm, right=6mm of m] (w) {whiten\\ $\operatorname{Cov}(z) = I$};
  \node[fillbox=corange, text width=2.4cm, right=6mm of w] (r) {rotate\\ $s = Uz$};
  \draw[->] (m) -- (w);
  \draw[->] (w) -- (r);
\end{tikzpicture}
\end{center}
```

<!-- derivation: ica-whiten -->
(definition) $A = Q D^{1/2} U^\top$ (its SVD, $\operatorname{Cov}(s) = I$); (substitute) whiten with $V^\top = D^{-1/2} Q^\top$; (result) $\operatorname{Cov}(z) = I$, $s = Uz$, $U$ orthogonal

(assumption) sources independent, at most one Gaussian; (result) the search is a **rotation** $U$

* Two assumptions: **independence**; at most one source **Gaussian**


## ICA Is Also a Sum of Rank-One Terms

![](plots/09/p2_ica_terms.pdf)

* $X = AS = \sum_{j=1}^{c} a_j s_j$: one rank-one term per source
* No order: sort by energy (0.37, 0.32, 0.31); correlations 0.998, 0.997, 0.996
* $A = Q D^{1/2} U^\top$: the SVD plus a rotation

## Fewer ICA Terms: the Price

![](plots/09/p2_ica_recon.pdf)

* ICA 0.795, 0.559, 0.019; PCA 0.414, 0.263, 0.019
* PCA is best at every $k$; ICA terms are sources
* Dropping source 2 (0.321 of energy): error 0.567

## Non-Gaussianity Is the Signal

![](plots/09/p2_kurtosis.pdf)

* Mixtures lean Gaussian: kurtosis $-1.25$ and $-0.945$
* Sources: kurtosis $-1.5$; more non-Gaussian means purer
* **Kurtosis** (4th order) sees what the covariance cannot
* Two goals: maximise non-Gaussianity or minimise mutual information

<!-- defines: negentropy, mutual information -->

## FastICA in Five Lines

1. Centre $X$ and **whiten**: covariance $I$, decorrelated
2. Pick a unit direction $w$; project: $s = w^\top z$
3. Fixed-point step: $w \leftarrow \operatorname{E}\!\left[z\, (w^\top z)^3\right] - 3 w$ (the first term is the kurtosis gradient)
4. Normalise $w$; re-orthogonalise against the directions already found
5. Repeat for each component

* The $-3w$ term comes from the Newton step on the constraint $\|w\|=1$; iterate to convergence
* Demo 02: five hand fixed-point steps reach direction $[-0.972, -0.234]$

## How FastICA Uses the Rotation

* Whitening already fixed the scales: only the rotation $U$ is free
* Find $w_1$ by kurtosis, project it out, find $w_2$ in the orthogonal complement
* Deflation vs symmetry: one component at a time, or all updated jointly
* Two sensors need one angle: the demo's walk converges in a handful of steps

## The Kurtosis Gradient, From the Objective

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \draw[cblue, thick] (0,0) ellipse (2.2 and 1.0);
  \draw[->, corange, thick] (0,0) -- (28:2.6) node[right]{$w$: climb $J_4$};
  \draw[->, cgray!70] (0,0) -- (118:1.4) node[above]{flat};
\end{tikzpicture}
\end{center}
```

<!-- derivation: ica-kurt -->
$$
\begin{aligned}
J_4(w) &= \tfrac{1}{4}\,\operatorname{E}\!\left[(w^\top z)^4\right] - \tfrac{3}{4} && \text{(definition, kurtosis proxy at unit } w\text{)} \\
\nabla_w J_4 &= \operatorname{E}\!\left[z\,(w^\top z)^3\right] && \text{(chain rule through the scalar } w^\top z\text{)}
\end{aligned}
$$

(assumption) $z$ whitened, unit $w$; (result) the extremes of $J_4$ are the most non-Gaussian projections

Climb $J_4$ for super-Gaussian sources. Descend for sub-Gaussian ones, like our sines.

## DEMO C1: `02_ica.ipynb`

## The Objective Families, Precisely

* **Kurtosis**: cheap, one statistic — but one outlier can drag $w$ (4th moment)
* **Negentropy** $J(y) = \big(\operatorname{E}[g(y)] - \operatorname{E}[g(\nu)]\big)^2$, $\nu$ Gaussian: robust, the FastICA default
* **Mutual information** $\operatorname{MI}(z_1, z_2) = \sum_j H(z_j) - H(z)$: minimise dependence directly
* All three pick the same rotation when the sources are far from Gaussian

## Why Independence and Not Decorrelation

* Decorrelated is weak: variables can share structure and stay uncorrelated
* Independence is strong: **no** function of one source informs the other
* Diagonal covariance: $d(d-1)/2$ rules; independence: infinitely many
* ICA in practice = whitening (order two) + rotation (all higher orders)

## The Entropy View: Why It Works

* Gaussian has the most entropy at fixed variance: the *least* shaped density
* Negentropy $J(y) = H(\nu) - H(y)$ measures gap from Gaussian
* Maximising $J(w^\top z)$ pushes toward the least Gaussian direction
* Estimating $J$ directly is hard; approximations $g$ make it practical

## Ambiguities: Order, Scale, Sign

![](plots/09/p2_ambiguity.pdf)

* ICA cannot order the sources: which output is speaker 1?
* Nor fix scale or sign: amplitudes 0.43, 0.44 instead of 1; sign flipped
* Only the **shape** of each source is identified

## ICA Limits

* Needs at least as many sensors as sources ($d \geq c$)
* Gaussian source: any rotation works; the demo returns nonsense
* Noise hurts fast: 0.05 gives 20.4 dB, 0.5 gives 2.3 dB
* Used in electroencephalography (EEG) artefact removal, audio, fluorescence unmixing
* More sensors than sources ($d > r$) helps: Demo 05, a third sensor adds 21.2 dB

## ICA Families: Algorithms in Practice

| algorithm | objective | strength |
|:---------------------|:-------------------------------------|:--------------------------------|
| FastICA | negentropy | fast, default |
| Infomax | likelihood | EEG toolboxes |
| JADE | 4th-order cumulants | no step size |
| SOBI | lagged covariances | distinct spectra |

* All whiten first; they differ in the rotation search
* Reduce first: PCA to $c$, then ICA; outputs $\leq$ sensors
* No order, scale or sign: sort by energy, fix the sign

# NMF

## Non-Negativity Is a Data Fact

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cblue, text width=2.2cm] (a) {pixels};
  \node[fillbox=cgreen, text width=2.2cm, right=6mm of a] (b) {counts};
  \node[fillbox=corange, text width=2.2cm, right=6mm of b] (c) {ratings};
  \node[fillbox=cgray, text width=6.4cm, below=5mm of b] (d) {all $\geq 0$: build the constraint into the model};
\end{tikzpicture}
\end{center}
```

* Factor $X \approx WH$ with $W \geq 0$ and $H \geq 0$
* Adding is natural, subtracting is not: **parts-based** representations

## Parts, Not Directions

![](plots/09/p3_parts_vs_directions.pdf)

* PCA components span $[-0.33, 0.47]$: positive and negative strokes
* Non-negative matrix factorisation (NMF) parts span $[0, 4.08]$: each a piece of a digit
* Reconstruction: PCA 0.319, NMF 0.355 relative error: parts cost a little

## The Order of the Rebuild

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cgray, text width=1.9cm] (a) {1. $X \geq 0$\\ shift if needed};
  \node[fillbox=cblue, text width=1.9cm, right=4mm of a] (b) {2. start\\ $W_0, H_0 \geq 0$};
  \node[fillbox=corange, text width=1.9cm, right=4mm of b] (c) {3. sweeps\\ $H$, then $W$};
  \node[fillbox=cgreen, text width=1.9cm, right=4mm of c] (d) {4. rebuild\\ $X \approx WH$};
  \draw[->] (a) -- (b);
  \draw[->] (b) -- (c);
  \draw[->] (c) -- (d);
\end{tikzpicture}
\end{center}
```

<!-- derivation: nmf-rebuild -->
$$
\begin{aligned}
X &\approx W H && \text{(definition, } d \times n = (d \times c)(c \times n)\text{)} \\
&= \sum_{j=1}^{c} w_j h_j && \text{(definition, columns of } W \text{ times rows of } H\text{)}
\end{aligned}
$$

(result) the first $k$ terms $\sum_{j \leq k} w_j h_j$ rebuild part of $X$

* Terms sorted by energy $\|w_j\|\,\|h_j\|$; scale is free: $(WD)(D^{-1}H)$ is the same $\hat X$
* Libraries that store examples as rows return $X^\top \approx H^\top W^\top$: the parts are then the rows of $H$

## Adding Parts Back

![](plots/09/p3_nmf_order.pdf)

* Demo 03, digit 33: part $w_j$, term $h_j[i]\, w_j$, running sum
* All entries $\geq 0$: a new part only adds ink, it never erases earlier ones
* PCA terms are signed and cancel; NMF terms only stack

## Error Against Terms Kept

![](plots/09/p3_nmf_error.pdf)

* Relative error of $\hat X_k$ for $c=8$: 0.894, 0.640, 0.360 at $k = 1, 4, 8$
* SVD terms (centred) reach 0.516, 0.400, 0.319: optimal at every $k$
* NMF has no best prefix: all eight parts are needed together

## The Multiplicative Update

Minimise $\|X - WH\|_F^2$; gradient descent would go negative.

<!-- derivation: nmf-mu -->
$$
\begin{aligned}
W \leftarrow W \,\odot\, \frac{X H^\top}{W H H^\top}, \quad H \leftarrow H \,\odot\, \frac{W^\top X}{W^\top W H} && \text{(definition)}
\end{aligned}
$$

* Numerator: the pull of the data; denominator: the pull of the model — their ratio is the step
* Demo 03: cost 77.1 $\to$ 63.2 in 100 iterations, never rises

![](plots/09/p3_mu_curve.pdf)

* $\odot$: entrywise product; the ratio is always $\geq 0$

## DEMO D1: `03_nmf.ipynb`

## The MU Derivation, Both Halves

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cblue, text width=2.2cm] (g) {gradient};
  \node[fillbox=cgreen, text width=2.2cm, right=6mm of g] (p) {$G^+ - G^-$};
  \node[fillbox=corange, text width=2.2cm, right=6mm of p] (u) {ratio update};
  \draw[->] (g) -- (p);
  \draw[->] (p) -- (u);
\end{tikzpicture}
\end{center}
```

<!-- derivation: nmf-grad -->
$$
\begin{aligned}
\nabla_W \|X - WH\|_F^2 &= 2\,(W H H^\top - X H^\top) && \text{(chain rule)} \\
\nabla_H \|X - WH\|_F^2 &= 2\,(W^\top W H - W^\top X) && \text{(chain rule)}
\end{aligned}
$$

* Split each gradient at the minus sign: $\nabla_W = G_W^+ - G_W^-$, both parts $\geq 0$
* The update $W \leftarrow W \odot G_W^+ / G_W^-$ keeps every entry $\geq 0$
* The same recipe gives the Kullback–Leibler (KL) updates $W \leftarrow W \odot \frac{(X \oslash WH)\,H^\top}{\mathbf{1}H^\top}$ and $H \leftarrow H \odot \frac{W^\top (X \oslash WH)}{W^\top \mathbf{1}}$: $\oslash$ entrywise division, $\mathbf{1}$ all ones

## NMF Clusters Without Being Told

* $H$ is non-negative: each example's weights are a soft cluster membership
* Columns of $W$ are the cluster prototypes; the argmax of each column of $H$ is a hard assignment
* Unlike $k$-means, the prototypes are *parts*, and the assignment is fractional
* Demo 03: the digit parts already sort the pixels into strokes

## Which Cost? Frobenius or KL

* Kullback–Leibler (KL) reweights by $X$: large counts matter more
* Spambase $c=8$: Frobenius relative error 0.005, Kullback–Leibler 0.052
* The cost is a modelling choice, not a detail
* Itakura–Saito (IS) divergence for audio: scale-invariant

## One Zoo, Three Algorithms

| algorithm | update | strength |
|:----------------|:-----------------------------------|:----------------------------------------|
| `nmf_mu` | multiplicative (Lee–Seung) | simple, provably non-increasing |
| `nmf_hals` | alternating least squares | fastest, best rel. error |
| `rwnmf` | weighted + regularised | missing data, robust term |

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cblue, text width=2.6cm] (i) {SVD-based init};
  \node[fillbox=cgreen, text width=2.6cm, right=6mm of i] (s) {MU: safe, slow};
  \node[fillbox=corange, text width=2.6cm, right=6mm of s] (h) {HALS: fast};
  \draw[->] (i) -- (s);
  \draw[->] (i) -- (h);
\end{tikzpicture}
\end{center}
```

* Init matters: non-negative double SVD (NNDSVD) starts from the SVD of $X$
* Solvers: hierarchical alternating least squares (HALS) and multiplicative update (MU)
* Demo 03, $c=8$: HALS 0.355, MU 0.364; $c=16$: 0.257, 0.282
* The zoo differs in speed and constraints, not in the goal

## Missing Values: rwnmf and HALS

* Real matrices have holes: unrated films, sensor dropouts
* Weighted cost: only observed entries count, $\Omega$ the mask
* rwnmf adds an L2 penalty and a robust term; hierarchical alternating least squares (HALS) imputes NaNs
* Demo 03: 25 % missing pixels, root mean square error (RMSE) 0.215 on missing, 0.168 on observed

## Choosing $c$: No Free Knee

* The multiplicative update (MU) cost curve (earlier) shows the fit, not the rank choice
* Digits $c$ sweep: relative error 0.509, 0.438, 0.367, 0.287, 0.196 for $c = 2, 4, 8, 16, 32$
* The curve keeps falling: $c$ is a trade-off, not a discovery
* Pick by downstream use or cross-validated error

## KL and Topic Models: One Cost, a Family

* KL with $WH$ as rates equals, up to scale, probabilistic latent semantic analysis (PLSA)
* Same updates, different story: the cost is a likelihood of a generative model
* That is why KL suits counts: it is the Poisson/multinomial likelihood
* One line of code, three literatures: factorisation, topic models, density ratios

## NMF Is Not Unique

![](plots/09/p3_nmf_unique.pdf)

* Six random starts, same rank: relative error 0.355 to 0.357
* Parts differ: mean matched cosine with start 0 is 0.99, 0.85, 0.91, 0.83, 0.85
* Scale is free too: $(WD)(D^{-1}H)$ equals $WH$ to $10^{-16}$
* Sparse NMF adds an L1 penalty on $H$: fewer, cleaner parts

## NMF Limits

* Needs $X \geq 0$: centred data forces a distorted fit (relative error 0.737)
* Local minima: HALS with NNDSVD init is the best of pyNNMF's solvers
* Scale of parts is arbitrary: normalise before comparing
* Rank matters more than the exact algorithm on small data

# Applications

## Cocktail Party: Three Views of One Problem

* Same two mixtures, three separations
* ICA wins on non-Gaussian, zero-centred signals: corr **0.995 / 0.996**, SNR **19.9 / 20.6 dB**
* PCA: corr 0.718 / 0.712, SNR 0.3 / 0.1 dB
* NMF (shifted to be non-negative; Demo 05): corr 0.928 / 0.961, SNR 7.9 / 10.8 dB; amplitude shrinks
* One sensor is ill-posed; a third adds little (21.2 dB)

## NMF on the Mixtures: the Order of Steps

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cgray, text width=2.0cm] (a) {$X' = X - \min X + 0.05$};
  \node[fillbox=cblue, text width=2.0cm, right=4mm of a] (b) {factor\\ $X' \approx WH$, $c=2$};
  \node[fillbox=corange, text width=2.0cm, right=4mm of b] (c) {read\\ rows of $H$};
  \node[fillbox=cgreen, text width=2.0cm, right=4mm of c] (d) {match order, sign, scale};
  \draw[->] (a) -- (b);
  \draw[->] (b) -- (c);
  \draw[->] (c) -- (d);
\end{tikzpicture}
\end{center}
```

* The shift makes $X' \geq 0$ and adds a constant to every source
* Sources: rows of $H$; mixing weights: columns of $W$
* Demo 05 reads corr 0.928 / 0.961: usable, below ICA, and only after the shift

## Three Real Voices, Three Methods

![](plots/09/p5_clips.pdf)

* Three clips, three microphones; Demo 05 plays every row
* ICA $\geq$ 0.999; PCA $\leq$ 0.762; NMF $\leq$ 0.638

## Three Real Pictures, Three Methods

![](plots/09/p5_pictures.pdf)

* Three 200 $\times$ 200 pictures, one per row of $S$; pixels $\geq 0$
* ICA $\geq$ 0.96; NMF $\geq$ 0.922; PCA $\leq$ 0.876

## Three Synthetic Voices, Three Methods

![](plots/09/p5_voices.pdf)

* Correlation: ICA $\geq$ 0.998, NMF $\geq$ 0.866, PCA $\leq$ 0.723
* ICA keeps the shape; NMF blends; PCA decorrelates

## Three Synthetic Images, Three Methods

![](plots/09/p5_images.pdf)

* One image per row of $S$; mixtures $X = AS$; pixels $\geq 0$
* Correlation: ICA $\geq$ 0.993, NMF $\geq$ 0.993, PCA $\leq$ 0.818

## The Same Skeleton, Check by Check

* PCA: $W$ orthonormal, $H$ uncorrelated — second orders only
* ICA: $W$ orthonormal, $H$ independent — higher orders
* NMF: $W, H \geq 0$ — additive parts
* Every frame of this class was one row of this table

## DEMO E1: `05_audio.ipynb`

## Latent Semantic Indexing (LSI)

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cblue, text width=2.6cm] (a) {documents $\times$ terms};
  \node[fillbox=cgreen, text width=2.6cm, right=6mm of a] (b) {SVD on tf-idf};
  \node[fillbox=corange, text width=2.6cm, right=6mm of b] (c) {topics in $c$ dims};
\end{tikzpicture}
\end{center}
```

* Toy: two topics separate football and politics in 2 D (sv 14.24, 5.04)
* Spambase: tf-idf + TruncatedSVD rank 2, EVR 0.234 and 0.179
* Topics are coarse: ham/spam structure, not named concepts
* Latent semantic indexing (LSI) is PCA on tf-idf: same eigenvectors, sparse input, a different story

## DEMO E2: `04_lsi_cf.ipynb`

![](plots/09/p4_lsi_spambase.pdf)

## Pitfalls of Every Factorisation

* Fit components on train only: the leakage trap applies to $W$ and $H$ too
* Components are not features with meaning: interpret only with the plot in hand
* Sign, scale and order are arbitrary: fix them before comparing runs
* Rank chosen on the reconstruction error is not rank chosen on the task

# Use Cases: One Task, Three Methods

## One Task, Three Methods

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cblue, text width=3.0cm] (a) {\textbf{PCA}\\ compression, denoising, anomaly scores};
  \node[fillbox=cgreen, text width=3.0cm, right=4mm of a] (b) {\textbf{ICA}\\ sources, artifact removal};
  \node[fillbox=corange, text width=3.0cm, right=4mm of b] (c) {\textbf{NMF}\\ topics, soft clusters, strokes, spectrograms};
  \node[fillbox=cgray, text width=9.8cm, below=5mm of b] (d) {shared: features, sound, imputation};
\end{tikzpicture}
\end{center}
```

* One score per task, the same data for all three methods (Demo 06)
* Each method wins where its constraint matches the data

## Sound: Notes on a Spectrogram

![](plots/09/p6_spectro.pdf)

* Three tones, each on and off on its own
* Spectrum correlation: PCA 0.782, ICA 0.949, NMF **0.995**
* Activations: 0.736, 0.899, 0.963
* Negative mass of the "spectra": PCA 0.259, ICA 0.741, NMF 0

## Image Segmentation: Strokes of a Digit

![](plots/09/p6_segments.pdf)

* Each ink pixel takes its largest part $W_{ij}H_{jp}$
* Sparseness of the components: PCA 0.377, ICA 0.377, NMF **0.608**
* The six digits use 3 to 6 of 8 parts

## Compression and Denoising

![](plots/09/p6_denoise.pdf)

* PCA is the minimum-error subspace; fit on clean digits, project noisy ones
* At $c=16$ the relative error is PCA 0.363, NMF 0.374, noisy input 0.514
* NMF pays for non-negativity, a little

## ICA Is PCA in a Rotated Basis

* Whiten with $c$ PCA axes, then rotate: the subspace does not change
* Compression, denoising, anomaly scores: ICA error equals PCA error to $10^{-8}$
* ICA earns its place only when the *basis* matters: sources, not subspaces
* Classifier features at $c=16$, Matthews correlation coefficient (MCC): PCA 0.958, ICA 0.956, NMF 0.873

## Topics in Documents

* Spambase word frequencies, 5 topics; each row of $H$ is a word list
* Topic 1: george, free, re, 415, 857, 650 (spam share 0.03)
* Topic 4: 3d, font, your, money, free, email (spam share **0.96**)
* PCA gives contrasts: george, 857, 415 against you, your, will

## Soft Clustering: Blended Customers

![](plots/09/p6_membership.pdf)

* Customers blend three profiles; $W$ rescaled is the membership
* Correlation with the truth: NMF **1.000**, soft $k$-means 0.923
* Failure: digit classes are not parts
* Adjusted Rand index (ARI): NMF arg-max 0.282, $k$-means 0.666

## Artifact Removal: an Eye Blink

![](plots/09/p6_blink.pdf)

* Five channels: three rhythms and one large blink; ICA finds the blink (corr **1.000**)
* Zero that source and remix: RMSE 1.581 $\to$ **0.060**
* PCA's blink component mixes in the rhythms (corr 0.953): RMSE 0.535

## Imputation: Missing Pixels

![](plots/09/p6_impute.pdf)

* RMSE at 25 % lost: mean 0.271, iterative PCA **0.197**, HALS 0.213
* MU 0.222, rwnmf 0.226, alternating least squares (ALS) 0.231
* Dense data: iterative PCA wins, even at 50 % lost

## DEMO E3: `06_usecases.ipynb`

## Scoreboard: Who Wins Where

![](plots/09/p6_scoreboard.pdf)

* Score relative to the best method on each task (1.0 wins)
* NMF: sound, strokes, sparse imputation; ICA: blink removal; PCA: the rest
* A blank cell: the method does not apply

# Imputation: Ratings, Topics and Time Series

## Collaborative Filtering Is Imputation

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cgray, text width=2.4cm] (r) {ratings $R$\\ users $\times$ items\\ most cells unknown};
  \node[fillbox=cblue, text width=1.6cm, right=9mm of r] (w) {$W$\\ users $\times$ $c$};
  \node[fillbox=corange, text width=1.6cm, right=3mm of w] (h) {$H$\\ $c$ $\times$ items};
  \node[fillbox=cgreen, text width=2.4cm, right=9mm of h] (p) {predict the unknown cells with $WH$};
  \draw[->] (r) -- node[above] {fit $\Omega$} (w);
  \draw[->] (h) -- (p);
\end{tikzpicture}
\end{center}
```

* Fit only the observed cells $\Omega$; $WH$ fills the rest
* Same problem as a missing pixel or a lost hour: a matrix with holes
* Demo 07: MovieLens 100K, 943 users, 603 films, 14.7 % observed
* Baselines: global mean 1.081, item mean 0.999, bias model **0.929**: beat that

## Rank and Regularisation

![](plots/09/p7_cf.pdf)

* Training error always falls with $c$; validation falls, then rises
* Rank 8, little regularisation: train 0.730, validation 1.005
* Validation picks $c=2$, $\alpha=3.0$: test RMSE **0.906** against 0.929

## Which Solver? What Do the Factors Mean?

* `rwnmf` (weighted, L2 term): **0.906**; HALS 0.920, ALS 0.925, MU 0.943
* All four land near the bias model 0.929: the gain is 2.5 %, not a revolution
* Taste 1: Independence Day, Air Force One, Conspiracy Theory
* Taste 2: Brazil, Trainspotting, Dr. Strangelove

## Cold Start and Ranking

* 30 cold users: NMF predicts 0.00, RMSE 3.920; bias 1.019
* Warm users: NMF 0.906, bias model 0.929, item mean 0.999
* Ranking is not rating: hit rate at 10 is popularity **0.653**, NMF 0.465, bias 0.368
* Cold users need the item profile as fallback

## Latent Semantic Indexing Meets NMF

![](plots/09/p7_topics.pdf)

* 1000 news articles: LSI axes carry both signs (0.41); NMF topics are word lists
* Topic purity: 0.77 from an NNDSVD start, 0.54 to 0.62 from random starts
* "goal referee": 55 articles share a word; the latent top 200 holds 191 sport ones

## Time Series Have Block Gaps

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cblue, text width=3.0cm] (a) {8 rooms $\times$ 1440 hours};
  \node[fillbox=cgray, text width=3.0cm, right=6mm of a] (b) {blocks of 48 hours lost};
  \node[fillbox=cgreen, text width=3.0cm, right=6mm of b] (c) {fill with $WH$, shared shapes};
  \draw[->] (a) -- (b);
  \draw[->] (b) -- (c);
\end{tikzpicture}
\end{center}
```

* Real data: hourly room temperatures of one house (UCI), 60 days
* Linear interpolation sees one series; factorisation borrows the others
* Score: RMSE on hidden 48-hour blocks, five datasets
* Linear 0.891, iterative PCA 1.077, HALS with $c=1$ **0.838**

## Rank and Start of the Fill

![](plots/09/p7_ts_rank.pdf)

* Rank 1: 0.838; rank 2: 0.844; ranks 3, 4: 0.992, 1.133
* Start matters little: $c=3$ gives 0.942 (SVD), 0.992 (random)

## The Misalignment Problem and Its Fix

![](plots/09/p7_ts_gap.pdf)

* Shape right, level wrong: the ends miss the observed neighbours
* Alignment shifts the block so both ends meet them
* Rank 1: 0.838 $\to$ 0.743, better than linear 0.891 (Fernandes et al. 2021)

## When Does Alignment Help?

* 12 h: linear **0.566** beats raw NMF 0.800; aligned NMF 0.515 beats both
* 24 h: 0.887, 0.804, aligned 0.757
* 48 h: 0.952, 0.852, aligned 0.831
* 96 h: aligned 0.934, raw 0.930: margins too far
* Rank 4: 1.133 $\to$ 1.145; validate on held-out blocks

## DEMO E4: `07_imputation.ipynb`

# Initialising a Non-Convex Problem

## The Start Picks the Minimum

![](plots/09/p7_init.pdf)

* Digits 0.04 and ratings 0.03 forgive; temperature 0.13
* Humidity: **3.77** between the best and the worst start
* The best start differs on each problem

## What the Literature Offers

* Random, SVD based (NNDSVD, Boutsidis and Gallopoulos 2008; low-rank corrections, Atif et al. 2019)
* Clustering (k-means parts), separable (successive projection), rank-one and metaheuristic starts
* Esposito (2021): comparisons are few; no start wins on every data type
* Demo 07 agrees on real data: the best start changes with the problem

## Sequential Search With the Empty Space Algorithm

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cblue, text width=2.4cm] (a) {propose parts $h$ away from the chosen ones};
  \node[fillbox=cgreen, text width=2.4cm, right=6mm of a] (b) {keep the one that removes most residual};
  \node[fillbox=corange, text width=2.4cm, right=6mm of b] (c) {refine, subtract, next component};
  \draw[->] (a) -- (b);
  \draw[->] (b) -- (c);
  \draw[->] (c.south) -- ++(0,-4mm) -| (a.south);
\end{tikzpicture}
\end{center}
```

* Idea: one component at a time, parts proposed by the empty space algorithm (ESS)
* ESS equals uniform: ratings 0.908 and 0.908, temperature 0.818 and 0.818, digits 0.210 and 0.208
* Viable, not better: the rank-one refinement does the work

## A Better Start: What We Keep

* No start wins: the best is greedy, `nndsvd`, `svd_impute` or successive projection
* Worst case over four problems: `nndsvd` 1.03 times the best, portfolio 1.66
* Humidity: rank 1 gives 4.166, rank 2 gives 4.866 at best and 23.189 at worst
* pyNNMF's `nndsvd` stays the default; improvements go to its repository plan

## Beyond This Class: Method Families

| family | relaxes | use it when |
|:-------------------------|:------------------------------------|:-----------------------------|
| kernel PCA | straight $\to$ curved | structure is non-linear |
| sparse PCA | dense $\to$ few loadings | loadings must be read |
| factor analysis | one noise level | sensors differ in noise |
| autoencoders | linear $\to$ network | non-linear codes |
| t-SNE, UMAP | invertible $\to$ local | a 2-D picture only |
| sparse NMF | dense $H$ $\to$ sparse | few parts wanted |
| semi-NMF | $H \geq 0$ $\to$ signed | mixed-sign data |
| dictionary learning | orthogonal $\to$ over-complete | more parts than sensors |
| LDA topics | cost $\to$ generative | topic probabilities |

* Every row keeps $X \approx WH$ and changes one constraint

# Choosing and Using

## Which Method for Which Data?

| data and goal | first choice | loses when |
|:-------------------------------|:----------------------------|:-------------------------------|
| signed mixtures, sources | ICA | Gaussian or dependent sources |
| subspace, compression, scores | PCA | sources must be separated |
| non-negative: parts, topics | NMF | data has signs |
| holes in the matrix | NMF (sparse), PCA (dense) | solver start is poor |
| documents | NMF topics, SVD search | topics must be named |

* Preprocess first: centring for PCA/ICA, no centring for NMF
* Fit on train only: the leakage trap of Classes 06-07 applies to components too

## Summary: One Map

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cblue, text width=2.6cm] (a) {\textbf{PCA}\\ orthogonal, 2nd order};
  \node[fillbox=cgreen, text width=2.6cm, right=5mm of a] (b) {\textbf{ICA}\\ independent, 4th order};
  \node[fillbox=corange, text width=2.6cm, right=5mm of b] (c) {\textbf{NMF}\\ non-negative, parts};
  \node[fillbox=cgray, text width=8.6cm, below=5mm of b] (d) {all fit $X \approx WH$: same skeleton, different constraint};
\end{tikzpicture}
\end{center}
```

* Take-home: choose the constraint that matches the data, not the metric
* Rank: scree, cumulative variance or cross-validated error; no free knee

## Summary: One Page of Equations

| block | equation |
|:----------|:--------------------------------------------------|
| skeleton | $X \approx WH = \sum_j w_j h_j$ |
| PCA | $Cw = \lambda w$, $C = \frac{1}{n-1}XX^\top$ |
| SVD | $X = U D_\sigma B^\top$, $\lambda_j = \sigma_j^2/(n-1)$ |
| error | $\|X - X_k\|_F^2 = \sum_{j>k}\sigma_j^2$ |
| ICA | $x = As$, $z = V^\top x$, $s = Uz$ |
| FastICA | $w \leftarrow \operatorname{E}[z(w^\top z)^3] - 3w$ |
| NMF | $H \leftarrow H \odot (W^\top X) \oslash (W^\top W H)$ |
| score | $\text{SNR} = 10\log_{10}\tfrac{\rho^2}{1-\rho^2}$ |

* The shapes: $X$ is $d \times n$, $W$ is $d \times c$, $H$ is $c \times n$

## Recap: Library Calls

| task | call | note |
|:------------------------|:------------------------------------|:-------------------------------|
| PCA | `PCA(c).fit(X)` | rows are examples |
| sparse input | `TruncatedSVD(c)` | no centring |
| ICA | `FastICA(c)` | `mixing_` is $A$ |
| NMF | `NMF(c, init="nndsvd")` | $X \geq 0$ |
| NMF, holes | `nmf_hals(X, c)` | NaN is imputed |

* sklearn: examples are rows; this deck: examples are columns, so $X_{\text{sk}} = X^\top$
* `components_` holds $W^\top$ (PCA) or the unmixing rows (ICA)

## Lab 09

| part | task |
|:----------------:|:--------------------------------------------------------------------------|
| A | PCA by hand: covariance, eigen, scores |
| B | ICA by hand: whiten, kurtosis gradient |
| C | NMF multiplicative updates by hand |
| D | collaborative filtering on the ratings matrix |
| E, F | home: LSI; the use-case tournament |

## References

Pearson (1901); Hotelling (1933); Jolliffe (2002); Comon (1994); Hyvärinen & Oja (2000); Lee & Seung (1999, 2001); Deerwester et al. (1990); Koren et al. (2009); Cichocki et al. (2009); Boutsidis & Gallopoulos (2008); Atif et al. (2019); Esposito (2021); Fernandes et al. (2021).
