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
|:---------------------|:-------------------------------------|:--------------------------|
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
|:-----------------|:-------------------------------|:-------------------|
| source | hidden signal | one speaker |
| mixture | what is recorded | microphone output |
| mixing matrix | how sources combine | $X = AS$ |
| loading | a learned column of the factorisation; for PCA, one eigenvector | an NMF part |
| component | one direction of PCA | PC1 |
| score | how much of a component an example has | coordinate of PC1 |

<!-- defines: loading, score, eigenvalue -->

## Notation N9: Factors and Mixtures

Each symbol is new in this class; earlier symbols keep their meaning.

| symbol | meaning | shape | example |
|:-------------|:----------------------------------|:---------:|:-------------------|
| $S$ | sources (rows) | $c \times n$ | two sines |
| $A$ | mixing matrix | $d \times c$ | $2 \times 2$ |
| $X$ | data matrix $= AS$ | $d \times n$ | two microphones |
| $c$ | number of components (rank) | scalar | $c = 2$ |
| $W$ | NMF parts (columns) | $d \times c$ | pixel parts |
| $H$ | NMF factors | $c \times n$ | how much of each part |
| $R$ | ratings matrix (users $\times$ items) | $n \times m$ | $6 \times 8$ |
| $w_j$ | PCA direction (unit vector) | $d$ | first PC |
| $C$ | covariance of the centred data | $d \times d$ | $2 \times 2$ |
| $I$ | identity matrix | $d \times d$ | ones on the diagonal |
| $\lambda, \lambda_e$ | eigenvalue $e$ of the covariance | scalar | $3.2$ |

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

* $W$: one column per component (a direction, a part, a topic)
* $H$: how much of each component every example has
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
\|X - ww^\top X\|_F^2 &= \text{tr}\!\big[(I - ww^\top)\, C\big] && \text{(definition)} \\
&= \text{tr}(C) - \text{tr}(w^\top C w) && \text{(definition, trace)} \\
&= \text{tr}(C) - w^\top C w && \text{(definition, } ww^\top w = w\text{)}
\end{aligned}
$$

(assumption) unit $w$; (result) minimising error = maximising $w^\top C w$

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

* At the optimum: $w^\top C w = \lambda_1$, error $\text{tr}(C) - \lambda_1$
* Eckart–Young: rank-$c$ reconstruction is the best of rank $c$ (singular value decomposition (SVD): $X = U D V^\top$)
* **Loadings**: the eigenvectors $w_j$; sklearn keeps them as rows of `components_`
* **Scores**: the coordinates of each example: $z = X w$ (`transform`)
* With $c$ components: $X \approx Z_c W_c^\top$, the rank-$c$ reconstruction
* Demo 01: rank-1 reconstruction of the mixture cloud, relative error 0.280
* Scores are new coordinates; loadings say *how* the old ones mix into them

## A Century Old, Still Everywhere

* Karl Pearson (1901): fit planes to points by least squares
* Harold Hotelling (1933): the same eigenvectors from maximising variance
* One idea, two entrances: **minimum error** = **maximum variance**
* Names you will meet: Karhunen–Loève transform (signal processing), singular value decomposition (numerical linear algebra)

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

# ICA

## The Blind Separation Promise

![](plots/09/p2_ica_separation.pdf)

* Correlations **0.996**, SNR **20.6 dB**
* PCA gave 0.1 dB on the same data

## Why PCA Fails Here

![](plots/09/p5_three_views.pdf)

* PCA axes are orthogonal; the sources are **not**
* ICA axes cross at the true mixing angle

## The ICA Model

<!-- derivation: ica-model
mode: paper
-->
$$
x = As, \qquad z = V^\top x \ \text{whitened,} \qquad s = Uz, \ U \text{orthogonal}
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
(definition) $A = Q D^{1/2} U^\top$ after its SVD; (substitute) whiten with $V^\top = D^{-1/2} Q^\top$; (result) $\operatorname{Cov}(z) = I$, $s = Uz$, $U$ orthogonal

(assumption) sources independent, at most one Gaussian; (result) the search is a **rotation** $U$

* Two assumptions: **independence**; at most one source **non-Gaussian**
* Mixing leans Gaussian: the rotation must undo it


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
3. Climb the kurtosis: $\nabla_w = \operatorname{E}\!\left[z\, (w^\top z)^3\right] - 3 w$ for unit-variance $w^\top z$
4. Normalise $w$; re-orthogonalise against the directions already found
5. Repeat for each component

* The fixed-point cousin: $w \leftarrow \operatorname{E}[z (w^\top z)^3] - 3w$ iterated to convergence, cubic in practice
* Demo 02: a hand kurtosis gradient walk reaches direction $[-0.972, -0.234]$

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

(assumption) $z$ whitened, $\operatorname{E}[zz^\top] = I$; (result) climbing $J_4$ finds the most kurtotic projection

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

* Needs at least as many sensors as sources ($d \geq r$)
* Gaussian source: any rotation works; the demo returns nonsense
* Noise hurts fast: 0.05 gives 20.4 dB, 0.5 gives 2.3 dB
* Used in electroencephalography (EEG) artefact removal, audio, fluorescence unmixing
* More sensors than sources ($d > r$) helps: Demo 05, a third sensor adds 21.2 dB

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

## The Multiplicative Update

Minimise $\|X - WH\|_F^2$; gradient descent would go negative.

$$
W \leftarrow W \,\odot\, \frac{X H^\top}{W H H^\top},
\qquad
H \leftarrow H \,\odot\, \frac{W^\top X}{W^\top W H}
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
* The same recipe gives the Kullback–Leibler (KL) updates $W \leftarrow W \odot \frac{X/(WH)H^\top}{H^\top}$: one derivation, every cost

## NMF Clusters Without Being Told

* $H$ is non-negative: each example's weights are a soft cluster membership
* Rows of $W$ are the cluster prototypes; argmax over $H$ is a hard assignment
* Unlike $k$-means, the prototypes are *parts*, and the assignment is fractional
* Demo 03: the digit parts already sort the pixels into strokes

## Which Cost? Frobenius or KL

* Kullback–Leibler (KL) reweights by $X$: large counts matter more
* Spambase $c=8$: Frobenius relative error 0.005, Kullback–Leibler 0.052
* The cost is a modelling choice, not a detail
* Itakura–Saito (IS) divergence for audio: scale-invariant

## One Zoo, Three Algorithms

| algorithm | update | strength |
|:--|:--|:--|
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

* Minimising KL with $WH$ as rates is exactly the fit of probabilistic latent semantic analysis (PLSA)
* Same updates, different story: the cost is a likelihood of a generative model
* That is why KL suits counts: it is the Poisson/multinomial likelihood
* One line of code, three literatures: factorisation, topic models, density ratios

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

* BBC news, 1000 articles: LSI axes carry both signs (negative mass 0.41), NMF topics are word lists
* Topic purity: 0.77 from an NNDSVD start, 0.54 to 0.62 from random starts
* "goal referee": 55 articles share a word; the latent top 200 holds 191 sport articles

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
* 24 h: 0.887, 0.804, aligned 0.757; 48 h: 0.952, 0.852, aligned 0.831
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

# Choosing and Using

## Which Method for Which Data?

| data and goal | first choice | loses when |
|:-------------------------|:-----------------------|:------------------------|
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

## Lab 09

| part | task |
|:--:|:-----------------------------------|
| A | PCA by hand: covariance, eigen, scores |
| B | ICA by hand: whiten, kurtosis gradient |
| C | NMF multiplicative updates by hand |
| D | collaborative filtering on the ratings matrix |
| E, F | home: LSI; the use-case tournament |

## References

Pearson (1901); Hotelling (1933); Jolliffe (2002); Comon (1994); Hyvärinen & Oja (2000); Lee & Seung (1999, 2001); Deerwester et al. (1990); Koren et al. (2009); Cichocki et al. (2009); Boutsidis & Gallopoulos (2008); Atif et al. (2019); Esposito (2021); Fernandes et al. (2021).
