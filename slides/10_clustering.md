---
title: Fundamentos de Aprendizagem Automática
subtitle: "Class 10 — Soft and Hard Clustering"
bibliography: "references.bib"
suppress-bibliography: true
---

# Two Kinds of Answer

## Where We Left Off

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cblue, text width=2.4cm] (a) {\textbf{supervised}\\ Classes 03--07: labels $y_i$};
  \node[fillbox=cgreen, text width=2.6cm, right=8mm of a] (b) {\textbf{decompose}\\ Class 09: $X^\top \approx WH$};
  \node[fillbox=cred, text width=5.6cm, below=7mm of b] (d) {\textbf{Class 10:} group the examples, no labels};
  \draw[->] (b) -- (d);
\end{tikzpicture}
\end{center}
```

* Class 09 pulled the data apart; today we group examples
* Non-negative matrix factorisation (NMF) weights are already a soft grouping

## Roadmap of the Class

| Part | Question | Methods |
|:----------------|:---------------------------|:-----------------------------------------------|
| **Two answers** | one label or a degree? | hard $z_{ik}$, soft $u_{ik}$, NMF |
| **Taxonomy** | how are methods related? | three axes, hierarchical tree |
| **Methods** | how does each one work? | $k$-means, hierarchical, DBSCAN, GMM, fuzzy $c$-means |
| **Benchmark** | which one on which shape? | six toy sets, seven methods |
| **Choosing $k$** | how many clusters? | elbow, silhouette, gap, BIC |
| **Use cases** | what is it for? | segments, topics, anomalies, taxonomy |

<!-- defines: cluster, centroid, hard clustering, soft clustering, membership, noise point -->

## Terminology

A **cluster**: examples close to each other, far from the rest.

| term | meaning | example |
|:-----------------------|:--------------------------------------------|:----------------------|
| centroid $\mu_k$ | mean of the points of cluster $k$ | centre of a blob |
| hard clustering | each point gets exactly one cluster | $k$-means label |
| soft clustering | each point gets a degree for every cluster | mixture posterior |
| membership $u_{ik}$ | degree to which point $i$ belongs to $k$ | 0.62 of cluster 1 |
| noise point | belongs to no cluster | DBSCAN label $-1$ |
| linkage | distance between two clusters | single, Ward |
| inertia $J$ | sum of squared distances to the centroids | $k$-means cost |

<!-- defines: hard clustering, soft clustering, membership, noise point, linkage, inertia -->

## Notation N10: Clusters and Memberships

Each symbol is new in this class; earlier symbols keep their meaning.

| symbol | meaning | shape | example |
|:--------------------------|:------------------------------------|:-------------:|:----------------|
| $k$ | number of clusters | scalar | $k = 3$ |
| $z_{ik}$ | hard assignment, $\in \{0, 1\}$ | $n \times k$ | one-hot row |
| $u_{ik}$ | soft membership, rows sum to 1 | $n \times k$ | $(0.6, 0.4, 0)$ |
| $\mu_k$ | centroid of cluster $k$ | $d$ | $(1.6, 1.8)$ |
| $\pi_k$ | mixture weight of component $k$ | scalar | 0.3 |
| $C_k$ | covariance of component $k$ | $d \times d$ | ellipse |
| $J$ | inertia, $\sum_i \lVert x_i - \mu_{z_i} \rVert^2$ | scalar | 8.7 |
| $\epsilon$, $\mathrm{minPts}$ | DBSCAN radius, point count | scalar | 0.2, 5 |
| $m$ | fuzzy $c$-means exponent | scalar | $m = 2$ |

* $k$, $J$, $m$: clustering names ($k$ is not the number of folds)

## Notation N10b: Inside the Derivations

| symbol | meaning | shape | example |
|:-------------------------|:----------------------------------|:------------:|:-------------------|
| $W$ | NMF parts of $X^\top$ | $d \times k$ | prototypes |
| $j$ | index of a cluster or part | integer | $1..k$ |
| $W_{\cdot j}$ | column $j$ of $W$ | $d$ | one part |
| $H$ | NMF weights, rank $c = k$ | $k \times n$ | memberships |
| $h_i$ | column $i$ of $H$ | $k$ | weights of $i$ |
| $h_{ji}$ | entry $j$ of $h_i$ | scalar | 0.4 |
| $G_k$ | set of the points of cluster $k$ | set | five points |
| $\mathcal{N}(x \mid \mu_k, C_k)$ | Gaussian density of component $k$ | scalar | 0.176 |
| $q_{ik}$ | any weights over components, $\sum_k q_{ik} = 1$ | $n \times k$ | $q = u$ at the optimum |
| $d_{ik}$ | distance from point $i$ to centre $k$ | scalar | 1.27 |
| $\mathcal{L}$ | Lagrangian of the constrained cost | scalar | |
| $\lambda$ | its multiplier | scalar | |

# Hard and Soft Assignments

## Hard Clustering: One Label per Point

![](plots/10/a1_hard_soft.pdf)

* One cluster per point: $z_{ik} \in \{0, 1\}$
* A partition: no overlap, every point covered
* Demo 10a: three overlapping blobs, std 1.2

## The Adjusted Rand Index: the Expression

Compare a clustering with known groups over all $\binom{n}{2}$ pairs of points.

$$
\mathrm{RI} = \frac{a + b}{\binom{n}{2}}
$$

$$
\mathrm{ARI} = \frac{\mathrm{RI} - \mathbb{E}[\mathrm{RI}]}{1 - \mathbb{E}[\mathrm{RI}]}
$$

* $a$: pairs together in both partitions; $b$: pairs apart in both
* $\mathbb{E}[\mathrm{RI}]$: the RI of a random partition with the same group sizes
* ARI: 1 identical, 0 chance level, below 0 worse than chance

## Scoring a Clustering: the Adjusted Rand Index

Does a clustering match known groups? Count pairs.

![](plots/10/a1_ari.pdf)

* Pair agrees: together in both, or apart in both
* ARI: share agreeing, chance removed; 1 same, 0 chance
* One swap per group: 0.280; all swapped: 1.000

<!-- defines: adjusted Rand index, ARI -->

## Soft Clustering: A Degree per Point

* A degree per cluster: $u_{ik} \in [0, 1]$, $\sum_k u_{ik} = 1$
* A point between two clusters is split
* Hard label $= \arg\max_k u_{ik}$: mixture and $k$-means agree on 0.980 (ARI 0.636, 0.651)

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cgray, text width=2.4cm] (a) {$\arg\max$: hard row\\ $(0,\,1,\,0)$};
  \node[fillbox=cblue, text width=3.2cm, right=8mm of a] (b) {soft row\\ $(0.37,\,0.62,\,0.01)$};
  \draw[->] (b) -- (a);
\end{tikzpicture}
\end{center}
```

## Top Membership and Entropy: the Expressions

Two numbers per membership row $u_i$: how sure, and how split.

$$
t_i = \max_k u_{ik}
$$

$$
H_i = -\sum_{k=1}^{K} u_{ik} \log_2 u_{ik}
$$

* $t_i$: the confidence of the hard label; 1 when the row is one-hot
* $H_i$ in bits: 0 for a one-hot row, $\log_2 K$ for a flat row
* Convention: $0 \log_2 0 = 0$

## Scoring a Soft Row: Top Membership and Entropy

Two numbers per row: how sure, and how split.

![](plots/10/a1_entropy_curve.pdf)

* Top membership $\max_k u_{ik}$: the confidence of the hard label
* Entropy $-\sum_k u_{ik}\log_2 u_{ik}$: 0 bits one-hot, $\log_2 k$ flat
* Two clusters: 1.000 bit at $(0.5, 0.5)$; three: 1.585 bits flat

<!-- defines: entropy, top membership -->

## Three Points, Three Rows

| point | hard label | $u_{i0}$ | $u_{i1}$ | $u_{i2}$ | entropy |
|:----------------------|:--:|----:|----:|----:|----:|
| centre of cluster 0 | 0 | 0.937 | 0.061 | 0.002 | 0.349 |
| between clusters 0, 1 | 0 | 0.371 | 0.624 | 0.005 | 0.993 |
| far outside | 1 | 0.000 | 1.000 | 0.000 | 0.000 |

![](plots/10/a1_rows.pdf)

* The middle point: label 0, yet 62 % cluster 1
* Far outside: certain, but no "none" option

## Entropy Marks the Border

![](plots/10/a1_entropy.pdf)

* Demo 10a: 100 of 300 points exceed 0.5 bit
* They sit between blobs: a coin toss for a hard label

## DEMO A1: `01_hard_soft.ipynb`

Demo 10a: $k$-means against a mixture on blobs, NMF as soft clustering, the overlap sweep.

## More Overlap, Softer Rows

Same three centres, the cluster std grows from 0.4 to 2.5: how sure is the soft model, and how well does it match the truth?

![](plots/10/a1_overlap.pdf)

* Growing std: mean top membership 1.000 $\to$ 0.893
* ARI 1.000 $\to$ 0.638; soft rows say *where*

## NMF Is a Soft Clustering

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cgray, text width=1.4cm] (x) {$X^\top$\\ $d \times n$};
  \node at ($(x.east)+(4mm,0)$) {$\approx$};
  \node[fillbox=cblue, text width=1.9cm, right=8mm of x] (w) {$W$\\ $d \times k$\\ archetypes};
  \node[fillbox=cgreen, text width=1.9cm, right=3mm of w] (h) {$H$\\ $k \times n$\\ weights};
\end{tikzpicture}
\end{center}
```

* $W, H \geq 0$: columns of $W$ prototypes, of $H$ weights
* Normalise a column of $H$ to sum 1: a row $u_i$
* Needs $X \geq 0$: split features into positive and negative part

<!-- defines: archetype -->

## NMF Memberships on the Blobs

![](plots/10/a1_nmf_soft.pdf)

* Demo 10a, ARI: NMF 0.617, mixture 0.636, $k$-means 0.651
* Membership correlation with the mixture posterior: 0.952, 0.953, 0.952

## The Scale of a Part Is Free

The scale of a part is free: $(WD)(D^{-1}H) = WH$ for a positive diagonal $D$.

<!-- derivation: nmf-membership -->
$$
\begin{aligned}
x_i &\approx W h_i && \text{(definition)} \\
&= \sum_{j=1}^{k} (W_{\cdot j}\, \textstyle\sum_r W_{rj})\,\dfrac{h_{ji}}{\sum_r W_{rj}} && \text{(identity)}
\end{aligned}
$$

(result) rescale $H$ by the column sums of $W$, then normalise

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cblue, text width=0.9cm] (w) {$W$};
  \node[fillbox=cgreen, text width=0.9cm, right=3mm of w] (h) {$H$};
  \node[right=4mm of h] (e) {$=$};
  \node[fillbox=cblue, text width=1.4cm, right=4mm of e] (wd) {$W D$};
  \node[fillbox=cgreen, text width=1.4cm, right=3mm of wd] (dh) {$D^{-1} H$};
\end{tikzpicture}
\end{center}
```

* Demo 10g, news, five topics: raw $H$ ARI 0.133, rescaled **0.735**

## Soft Does Not Mean Honest

![](plots/10/a1_uniform.pdf)

* Uniform noise: rows as decided as real blobs
* Demo 10a: mean top membership 0.882, blobs 0.889
* Entropy measures ambiguity, not whether clusters exist

# Taxonomy of Clustering

## Three Axes

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cblue, text width=3.0cm] (a) {\textbf{assignment}\\ hard or soft};
  \node[fillbox=cgreen, text width=3.0cm, right=4mm of a] (b) {\textbf{structure}\\ divisive or agglomerative};
  \node[fillbox=corange, text width=3.0cm, right=4mm of b] (c) {\textbf{criterion}\\ $k$ or distance};
  \node[fillbox=cgray, text width=9.6cm, below=5mm of b] (d) {every method sits at one point of each axis};
\end{tikzpicture}
\end{center}
```

* Assignment: what a point gets; structure: how groups form
* Criterion: when to stop; the axes are independent

<!-- defines: divisive, agglomerative, k-criterion, distance criterion -->

## Axis 1: Hard or Soft

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cblue, text width=3.2cm] (h1) {$k$-means};
  \node[fillbox=cblue, text width=3.2cm, below=2mm of h1] (h2) {agglomerative tree cut};
  \node[fillbox=cblue, text width=3.2cm, below=2mm of h2] (h3) {DBSCAN, plus noise};
  \node[fillbox=cgreen, text width=3.2cm, right=22mm of h1] (s1) {Gaussian mixture (EM)};
  \node[fillbox=cgreen, text width=3.2cm, below=2mm of s1] (s2) {fuzzy $c$-means};
  \node[fillbox=cgreen, text width=3.2cm, below=2mm of s2] (s3) {NMF, normalised $H$};
  \node[above=1mm of h1] {\textbf{hard} $z_{ik}$};
  \node[above=1mm of s1] {\textbf{soft} $u_{ik}$};
  \draw[->, thick] (s2.west) -- (h2.east);
\end{tikzpicture}
\end{center}
```

* Soft to hard: the $\arg\max$
* Degree: probability (GMM), distance share (FCM), part weight (NMF)

## Axis 2: Divisive or Agglomerative

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cgreen, text width=2.4cm] (a) {one cluster\\ all $n$ points};
  \node[fillbox=cblue, text width=2.4cm, right=22mm of a] (b) {$n$ clusters\\ one point each};
  \draw[->, thick] ([yshift=3mm]a.east) -- node[above] {divisive: split} ([yshift=3mm]b.west);
  \draw[<-, thick] ([yshift=-3mm]a.east) -- node[below] {agglomerative: merge} ([yshift=-3mm]b.west);
\end{tikzpicture}
\end{center}
```

* Agglomerative: start from singletons, merge the closest pair
* Divisive: start from one cluster, split (bisecting $k$-means)
* Flat methods return one partition, no tree

## Axis 3: $k$ or Distance

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=corange, text width=3.4cm] (a1) {$k$-means, GMM, fuzzy $c$-means};
  \node[fillbox=corange, text width=3.4cm, below=2mm of a1] (a2) {spectral clustering, tree cut at $k$};
  \node[fillbox=cred, text width=3.4cm, right=18mm of a1] (b1) {DBSCAN: $\epsilon$, $\mathrm{minPts}$};
  \node[fillbox=cred, text width=3.4cm, below=2mm of b1] (b2) {tree cut at a height $h$};
  \node[above=1mm of a1] {\textbf{$k$-criterion}: say how many};
  \node[above=1mm of b1] {\textbf{distance}: say how close};
\end{tikzpicture}
\end{center}
```

* $k$-criterion: the number is an input (part 5)
* Distance: the number is an output; a tree serves both

## The Taxonomy in One Picture

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize, node distance=3mm, every node/.style={align=center}]
  \node[fillbox=cgray, text width=1.6cm] (r) {clustering};
  \node[fillbox=cblue, text width=1.6cm, below left=7mm and 20mm of r] (h) {hard $z_{ik}$};
  \node[fillbox=cgreen, text width=1.6cm, below right=7mm and 20mm of r] (so) {soft $u_{ik}$};
  \node[fillbox=cblue, text width=1.4cm, below=7mm of h, xshift=-17mm] (a) {$k$-means\\ flat, $k$};
  \node[fillbox=cblue, text width=1.4cm, right=2mm of a] (b) {hierarchical\\ tree, $k$ / $h$};
  \node[fillbox=cblue, text width=1.4cm, right=2mm of b] (c) {DBSCAN\\ flat, $\epsilon$};
  \node[fillbox=cgreen, text width=1.4cm, right=6mm of c] (d) {GMM\\ flat, $k$};
  \node[fillbox=cgreen, text width=1.4cm, right=2mm of d] (e) {fuzzy $c$-means\\ flat, $k$};
  \node[fillbox=cgreen, text width=1.4cm, right=2mm of e] (f) {NMF\\ flat, $k$};
  \draw (r) -- (h); \draw (r) -- (so);
  \draw (h) -- (a); \draw (h) -- (b); \draw (h) -- (c);
  \draw (so) -- (d); \draw (so) -- (e); \draw (so) -- (f);
\end{tikzpicture}
\end{center}
```

* First split: assignment; each box: structure and criterion

# $k$-Means and $k$-Means++

## $k$-Means: Lloyd's Algorithm

![](plots/10/b1_lloyd.pdf)

* Assign to the nearest centroid, move centroids to the mean
* Demo 10b: $J$ 24.75 $\to$ 8.70 in 2 sweeps
* Hard, flat, $k$-criterion

## The Objective and the Update

Minimise $J = \sum_i \lVert x_i - \mu_{z_i} \rVert^2$; set the gradient in $\mu_k$ to 0.

<!-- derivation: kmeans-mean -->
$$
\begin{aligned}
\frac{\partial}{\partial \mu_k} \sum_{i \in G_k} \lVert x_i - \mu_k \rVert^2 &= -2 \sum_{i \in G_k} (x_i - \mu_k) && \text{(definition)} \\
\mu_k &= \frac{1}{\lvert G_k \rvert} \sum_{i \in G_k} x_i && \text{(assumption)}
\end{aligned}
$$

(result) the mean; Demo 10b, cluster 0: $\mu_x = 1.6$

![](plots/10/b1_mean.pdf)


## $k$-Means++: A Better Start

![](plots/10/b1_init.pdf)

* Next centroid $\propto$ squared distance to the nearest [@arthur2007]
* 200 starts: random start 5 % above best in 0.145 of runs
* $k$-means++: 0.000

## $k$-Means: Strengths and Limits

| property | $k$-means |
|:------------------------|:------------------------------------------------------------------|
| strengths | simple, fast, scales to millions of points |
| limits | round equal-size clusters; sensitive to scale and outliers |
| complexity | $O(n\,k\,d)$ per sweep |
| hyperparameters | $k$, the start (`init`), number of restarts (`n_init`) |

![](plots/10/e1_strip_kmeans.pdf){width=80%}

* Keep the lowest $J$ of several starts [@lloyd1982]; standardise

## Voronoi Cells and a Wrong $k$

![](plots/10/b1_voronoi.pdf)

* Boundaries are straight lines halfway between two centroids: a Voronoi partition
* $J$ falls with $k$: 1697, 820, 368, 230, 141 at $k = 1, 2, 3, 6, 10$

# Hierarchical Clustering

## Agglomerative Clustering in Four Lines

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cgray, text width=2.1cm] (a) {1. $n$ singletons};
  \node[fillbox=cblue, text width=2.1cm, right=4mm of a] (b) {2. distance matrix of clusters};
  \node[fillbox=corange, text width=2.1cm, right=4mm of b] (c) {3. merge the closest pair};
  \node[fillbox=cgreen, text width=2.1cm, right=4mm of c] (d) {4. repeat until one};
  \draw[->] (a) -- (b);
  \draw[->] (b) -- (c);
  \draw[->] (c) -- (d);
\end{tikzpicture}
\end{center}
```

* A binary tree: $n - 1$ merges, each at a height
* The only choice is the **linkage**
* No $k$, no random start: same data, same tree

## Agglomerative: the Mechanism

![](plots/10/b1_dendrograms.pdf)

* Ten points: all four linkages give the same two groups
* Last merge height: single 4.61, average 6.95, complete 9.22, Ward 15.44
* A dendrogram draws each merge at its height

## Reading the Dendrogram

![](plots/10/b1_cut.pdf)

* Cut at a height $h$: branches below the line
* Long gaps mark groups: last merges 9.7, 30.4, 41.6
* Three-cluster cut: ARI 0.931

## The Four Linkages

| linkage | $D(A, B)$ |
|:------------|:------------------------------------------------------------------------------|
| single | $\min_{a \in A, b \in B} \lVert a - b \rVert$ |
| complete | $\max_{a \in A, b \in B} \lVert a - b \rVert$ |
| average | $\frac{1}{\lvert A \rvert \lvert B \rvert} \sum_{a, b} \lVert a - b \rVert$ |
| Ward | $\frac{\lvert A \rvert \lvert B \rvert}{\lvert A \rvert + \lvert B \rvert} \lVert \mu_A - \mu_B \rVert^2$ |

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize, x=0.8cm, y=0.8cm]
  \foreach \x/\y in {0/0, 0.6/0.9, 1.0/0.2, 0.3/0.5} \fill[cblue] (\x,\y) circle (1.6pt);
  \foreach \x/\y in {3.4/0.4, 4.1/1.0, 3.8/0.1, 4.4/0.6} \fill[corange] (\x,\y) circle (1.6pt);
  \draw[dashed] (1.0,0.2) -- (3.4,0.4);
  \draw[dotted, thick] (0,0) -- (4.4,0.6);
  \draw[thick, cred] (0.5,0.4) -- (3.9,0.5);
  \node at (2.2,-0.7) {dashed: single; dotted: complete; red: average, Ward};
\end{tikzpicture}
\end{center}
```

* Ward's $D$ is the rise of $J$ on merging [@ward1963]
* Single linkage chains through one close pair

## Which Linkage Mirrors Which Method

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cblue, text width=2.2cm] (a) {single\\ nearest pair};
  \node[fillbox=cgreen, text width=2.2cm, below=2mm of a] (b) {complete\\ farthest pair};
  \node[fillbox=corange, text width=2.2cm, below=2mm of b] (c) {average\\ mean of pairs};
  \node[fillbox=cred, text width=2.2cm, below=2mm of c] (d) {Ward\\ growth of $J$};
  \node[fillbox=cgray, text width=4.4cm, right=22mm of a] (a2) {DBSCAN components, spanning tree};
  \node[fillbox=cgray, text width=4.4cm, right=22mm of b] (b2) {compact groups of similar size};
  \node[fillbox=cgray, text width=4.4cm, right=22mm of c] (c2) {a compromise};
  \node[fillbox=cgray, text width=4.4cm, right=22mm of d] (d2) {$k$-means variance};
  \draw[->] (a) -- (a2); \draw[->] (b) -- (b2); \draw[->] (c) -- (c2); \draw[->] (d) -- (d2);
\end{tikzpicture}
\end{center}
```

* ARI($k$-means, Ward): 0.970 blobs, 0.961 varied, 0.827 aniso
* Not an identity: single chains through bridges

## One Tree Mimics Other Methods

![](plots/10/b1_moons.pdf)

* Single linkage: components of the distance-$h$ graph
* Density-based clustering (DBSCAN), $\mathrm{minPts}=1$, gives them: ARI 1.000
* Ward: the pair that raises $J$ least

## Hierarchical Clustering Is Deterministic

![](plots/10/b1_dendrograms.pdf)

* Demo 10b, ten runs on `varied`: Ward gives 1 labelling
* $k$-means, one random start: 6 distinct costs
* Mean ARI between two runs: $k$-means 0.702, Ward 1.000

## Hierarchical: Strengths and Limits

| property | agglomerative clustering |
|:----------------------------|:--------------------------------------------------------------|
| strengths | deterministic, no $k$ up front, a hierarchy |
| limits | merges never undone; single chains |
| complexity | $O(n^2)$ memory, $O(n^2 \log n)$ time |
| hyperparameters | linkage, metric, cut ($k$ or $h$) |

![](plots/10/e1_strip_hier.pdf){width=80%}

* 8 bridge points: ARI single **0.000**, Ward 1.000

## DEMO B1: `02_kmeans_hier.ipynb`

Demo 10b: Lloyd sweeps, $k$-means++ starts, dendrograms, cuts and linkages.

# DBSCAN

## DBSCAN: Core, Border, Noise

![](plots/10/c1_core_border_noise.pdf)

* **Core**: at least $\mathrm{minPts}$ points within $\epsilon$ ($\epsilon = 1.2$, $\mathrm{minPts} = 4$)
* **Border**: near a core point, not core; the rest is **noise**
* Demo 10c: 8 core, 3 border, 1 noise

<!-- defines: core point, border point -->

## DBSCAN: The Algorithm

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cgray, text width=2.1cm] (a) {1. count the neighbours within $\epsilon$};
  \node[fillbox=cblue, text width=2.1cm, right=4mm of a] (b) {2. take an unvisited core point};
  \node[fillbox=corange, text width=2.1cm, right=4mm of b] (c) {3. grow: add every point reachable from it};
  \node[fillbox=cgreen, text width=2.1cm, right=4mm of c] (d) {4. points left over are noise};
  \draw[->] (a) -- (b);
  \draw[->] (b) -- (c);
  \draw[->] (c) -- (d);
\end{tikzpicture}
\end{center}
```

* Clusters are maximal sets of density-connected points [@ester1996]
* The number of clusters is an output; the shape is free

## DBSCAN: Strengths and Limits

| property | DBSCAN |
|:-----------------------|:-------------------------------------------------------------------|
| strengths | any shape, finds noise, no $k$ |
| limits | one density for all clusters, distances weaken in high dimension |
| complexity | $O(n \log n)$ with an index, $O(n^2)$ without |
| hyperparameters | $\epsilon$, $\mathrm{minPts}$ |

![](plots/10/e1_strip_dbscan.pdf){width=80%}

* A border point keeps the first cluster that reaches it [@schubert2017]

## Two Parameters Decide the Answer

![](plots/10/c1_eps_sweep.pdf)

* $\epsilon = 0.05$: 5 clusters, 269 noise points (ARI 0.011)
* $\epsilon = 0.2$: the two moons (1.000); 0.3: one cluster (0.000)
* The usable window is narrow: tune $\epsilon$ on the data

## The $k$-Distance Graph

![](plots/10/c1_kdistance.pdf)

* Sorted $\mathrm{minPts}$-th neighbour distance: the knee is $\epsilon$
* A knee detector (`kneeliverse`, part 5) finds it: moons $\epsilon$ = 0.126, ARI 0.967
* Blobs: $\epsilon$ = 0.501, 3 clusters, 25 noise points, ARI 0.485

## `minPts` and the Noise It Declares

![](plots/10/c1_minpts.pdf)

* A larger $\mathrm{minPts}$ demands more neighbours: small clusters turn into noise
* Rule of thumb: $\mathrm{minPts} \geq d + 1$; $2d$ for noisy data
* Noisy moons: $\mathrm{minPts} = 5$ gives ARI 0.815; 10 gives $-0.001$
* The choice of $\mathrm{minPts}$ changes the knee: tune the two together

## Failure: Varying Densities

![](plots/10/c1_varied.pdf)

* Three clusters with std 0.4, 1.6, 0.8: no single $\epsilon$ serves all
* $\epsilon = 0.4$: 7 clusters, 85 noise points; 1.3: merges
* $k$-means $k = 3$: ARI 0.797; best DBSCAN 0.714

## Failure: High Dimension

![](plots/10/c1_highdim.pdf)

* Farthest over nearest: 51.0 ($d = 2$), 3.2 ($d = 10$), 1.11 ($d = 1000$)
* Digits: knee $\epsilon = 7.28$, one cluster, ARI 0.000
* Reduce dimension first (PCA)

## DEMO C1: `03_dbscan.ipynb`

Demo 10c: core, border and noise points, the $\epsilon$ sweep, the $k$-distance graph, failures.

# Gaussian Mixtures and EM

## GMM: The Model

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cgray, text width=1.6cm] (a) {pick $k$\\ with prob. $\pi_k$};
  \node[fillbox=cblue, text width=2.6cm, right=8mm of a] (b) {draw $x \sim \mathcal{N}(\mu_k, C_k)$};
  \node[fillbox=cgreen, text width=2.6cm, right=8mm of b] (c) {we see $x$, not $k$};
  \draw[->] (a) -- (b);
  \draw[->] (b) -- (c);
\end{tikzpicture}
\end{center}
```

* $p(x) = \sum_{k} \pi_k\,\mathcal{N}(x \mid \mu_k, C_k)$, $\sum_k \pi_k = 1$
* The label is hidden; its posterior is $u_{ik}$
* Soft, flat, $k$-criterion

## The E-Step: Responsibilities

$u_{ik} = \dfrac{\pi_k\,\mathcal{N}(x_i \mid \mu_k, C_k)}{\sum_j \pi_j\,\mathcal{N}(x_i \mid \mu_j, C_j)}$

![](plots/10/d1_em.pdf)

* Bayes' rule of Class 04; Demo 10d: three blobs, 61 iterations

## The M-Step: Weighted Statistics

| quantity | update |
|:-----------------|:-------------------------------------------------------------------------|
| $N_k$ | $\sum_i u_{ik}$ |
| $\pi_k$ | $N_k / n$ |
| $\mu_k$ | $\frac{1}{N_k}\sum_i u_{ik}\, x_i$ |
| $C_k$ | $\frac{1}{N_k}\sum_i u_{ik}\,(x_i - \mu_k)(x_i - \mu_k)^\top$ |

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cblue, text width=1.9cm] (u) {memberships $u_{ik}$};
  \node[fillbox=cgreen, text width=2.1cm, right=8mm of u] (n) {weighted counts, means, covariances};
  \node[fillbox=corange, text width=1.9cm, right=8mm of n] (t) {new $\pi_k, \mu_k, C_k$};
  \draw[->] (u) -- (n); \draw[->] (n) -- (t);
\end{tikzpicture}
\end{center}
```

* Sample mean and covariance, each point counted $u_{ik}$ times
* $k$-means: equal spherical components, $u_{ik} \in \{0, 1\}$

## One E-Step on Five Points

![](plots/10/d1_estep.pdf)

* $\pi = (0.5, 0.5)$, $\mu = (0, 4)$, $\sigma = 1$
* $x = 2$: 0.500 and 0.500; $x = 3$: 0.018 and 0.982

## Why EM Never Lowers the Likelihood

Any weights $q_{ik}$ with $\sum_k q_{ik} = 1$; $\ln$ is concave (Jensen).

<!-- derivation: em-bound -->
$$
\begin{aligned}
\ln p(x_i) &= \ln \sum_k q_{ik}\, \frac{\pi_k \mathcal{N}(x_i \mid \mu_k, C_k)}{q_{ik}} && \text{(identity)} \\
&\geq \sum_k q_{ik} \ln \frac{\pi_k \mathcal{N}(x_i \mid \mu_k, C_k)}{q_{ik}} && \text{(assumption)}
\end{aligned}
$$

(result) tight at $q_{ik} = u_{ik}$ (E-step); M-step raises it

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize, x=0.6cm, y=0.6cm]
  \draw[thick, cblue] (0,0.2) .. controls (1,1.6) and (2.2,1.8) .. (3.2,1.9) node[right] {$\ln p$};
  \draw[thick, cred] (0.6,0.2) .. controls (1.2,1.1) and (1.8,1.2) .. (2.4,1.1) node[right] {bound};
  \fill (1.3,1.1) circle (1.5pt) node[below] {$\theta_{\text{old}}$};
  \fill (2.2,1.15) circle (1.5pt) node[below] {$\theta_{\text{new}}$};
  \draw[->, dashed] (1.3,1.1) -- (2.2,1.15);
\end{tikzpicture}
\end{center}
```

Demo 10d: $-1172.7 \to -1019.7$ [@dempster1977]

## Four Covariance Structures

![](plots/10/d1_covariance.pdf)

* `spherical`: one variance; `diag`: ellipses on the axes; `tied`: shared; `full`: any
* Stretched blobs ARI: spherical 0.509, diag 0.495, tied, full **1.000**
* Bayesian information criterion (BIC) $= P\ln n - 2\ln\hat{L}$: `tied` 1618.7, `full` 1649.1

<!-- defines: BIC -->

## GMM: Strengths and Limits

| property | Gaussian mixture |
|:---------------------------|:---------------------------------------------------------------|
| strengths | soft rows, ellipses, a density, BIC |
| limits | local optimum, Gaussian shapes, singular $C_k$ |
| complexity | $O(n\,k\,d^2)$ per iteration for `full` |
| hyperparameters | $k$, covariance type, start, `reg_covar` |

![](plots/10/e1_strip_gmm.pdf){width=80%}

* A one-point component: density $\to \infty$; `reg_covar` floors it
* 100 random starts: 31 % end clearly worse

## A Component Is Not a Cluster

![](plots/10/d1_components.pdf)

* A mixture models a density, not groups
* Moons: 2, 5, 10 components give ARI 0.470, 0.321, 0.200
* BIC improves (1102.7, 739.0, 685.5); clusters get worse

## DEMO D1: `04_gmm_fcm.ipynb`

Demo 10d: E- and M-steps of a mixture, covariance structures, fuzzy $c$-means, soft rows for any $m$.

# Fuzzy $c$-Means and NMF

## Fuzzy $c$-Means: the Objective

Minimise $J_m = \sum_{i,k} u_{ik}^m \lVert x_i - \mu_k \rVert^2$ [@bezdek1981].

![](plots/10/d1_fuzzifier.pdf)

* **Fuzzifier** $m > 1$: $m \to 1$ is $k$-means, large $m$ gives $1/k$
* $u_{ik} \propto d_{ik}^{-2/(m-1)}$; $\mu_k$: mean weighted by $u_{ik}^m$

<!-- defines: fuzzifier -->

## Where the Membership Update Comes From

One point $i$, $d_{ik}^2 = \lVert x_i - \mu_k \rVert^2$, $\partial \mathcal{L} / \partial u_{ik} = 0$.

<!-- derivation: fcm-u -->
$$
\begin{aligned}
\mathcal{L} &= \sum_k u_{ik}^m d_{ik}^2 - \lambda \Big( \sum_k u_{ik} - 1 \Big) && \text{(definition)} \\
m\, u_{ik}^{m-1} d_{ik}^2 &= \lambda && \text{(assumption)} \\
u_{ik} &\propto d_{ik}^{-2/(m-1)} && \text{(identity)}
\end{aligned}
$$

(result) $u_{ik} = d_{ik}^{-2/(m-1)} / \sum_j d_{ij}^{-2/(m-1)}$

![](plots/10/d1_fcm_u.pdf)

* $m = 2$: inverse squared distance, normalised

## The Fuzzifier Sets How Soft

![](plots/10/d1_mcurve.pdf)

* `varied`, $m = 1.05$ to 10: mean top membership 0.996 $\to$ 0.432
* Wine, 11 features: $m = 1.1$ ARI 0.924; $m = 2$ ARI 0.091
* At $m = 2$ the centres slide together: 0.87 apart, $k$-means 3.49

## Three Ways to Write $u_{ik}$

| method | $u_{ik}$ means | ARI | mean top $u$ |
|:------------------------------|:--------------------------------|-----------:|------------------:|
| fuzzy $c$-means, $m = 2$ | share of inverse distance | 0.660 | 0.773 |
| GMM, full | posterior probability | 0.636 | 0.889 |
| NMF, rescaled $H$ | weight of an additive part | 0.617 | 0.838 |

![](plots/10/d1_three_u.pdf){width=40%}

* Correlation with GMM: FCM 0.97, NMF 0.95; FCM is softest

## FCM and NMF: Strengths and Limits

| property | fuzzy $c$-means | NMF as clustering |
|:---------------------|:---------------------------------|:------------------------------------|
| strengths | soft, simple, $k$-means-like | parts, sparse, interpretable |
| limits | round clusters, needs $m$ | non-negative data, local optimum |
| complexity | $O(n\,k\,d)$ per sweep | $O(n\,d\,k)$ per update |
| hyperparameters | $k$, $m$ | $k$ (rank), cost, start |

![](plots/10/e1_strip_fcmnmf.pdf){width=80%}

* NMF and spectral clustering share an objective [@ding2005]

# Synthetic Benchmark

## Six Shapes, Six Questions

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cblue, text width=1.9cm] (a) {\textbf{blobs}\\ round, equal};
  \node[fillbox=cgreen, text width=1.9cm, right=3mm of a] (b) {\textbf{aniso}\\ stretched};
  \node[fillbox=corange, text width=1.9cm, right=3mm of b] (c) {\textbf{varied}\\ three variances};
  \node[fillbox=cred, text width=1.9cm, below=4mm of a] (d) {\textbf{moons}\\ two crescents};
  \node[fillbox=cpurple, text width=1.9cm, right=3mm of d] (e) {\textbf{circles}\\ nested rings};
  \node[fillbox=cgray, text width=1.9cm, right=3mm of e] (f) {\textbf{uniform}\\ no clusters};
\end{tikzpicture}
\end{center}
```

* 300 points each: blobs, moons, circles, sheared blobs, noise
* Each set breaks one assumption
* Score: adjusted Rand index (ARI): 1 = truth, 0 = chance


## Seven Methods on Six Sets

![](plots/10/e1_grid.pdf)

## Who Wins Where

![](plots/10/e1_scoreboard.pdf){width=85%}

* Blobs: $k$-means, FCM; stretched: GMM; moons, rings: single
* Mean ARI: DBSCAN 0.694, GMM 0.651, single 0.627, $k$-means 0.497
* No method wins everywhere

## Ten Seeds, Not One

![](plots/10/e1_seeds.pdf)

* Ten fresh data sets per shape
* Stretched: GMM 0.91 $\pm$ 0.18, the rest about 0.74
* Moons: single 1.00, DBSCAN 0.61 $\pm$ 0.20, $k$-means 0.24
* Blobs: four methods tie at 0.95 $\pm$ 0.12

## Failure: Noise and Scale

![](plots/10/e1_noise_scale.pdf)

* 30 noise points added: GMM 0.902 $\to$ 0.516, $k$-means 0.941 $\to$ 0.883
* DBSCAN flags 18 of 30; its ARI falls to 0.000
* One feature $\times$ 100: $k$-means 0.561, GMM 0.931

## DEMO E1: `05_benchmark.ipynb`

Demo 10e: seven methods on six synthetic sets, ten seeds, noise and scale.

# Choosing the Number of Clusters

## Why $k$ Is a Question

![](plots/10/f1_elbow.pdf)

* $J_k$ falls with $k$, 0 at $k = n$: no minimum
* Three blobs: 1697, 820, 368, 310 at $k = 1$ to 4
* The **elbow**: more clusters buy little after it

<!-- defines: elbow -->

## Two Knee Detectors: Kneedle and the L-method

Both read the inertia curve; Kneedle first normalises it to the unit square.

![](plots/10/f1_knee_geometry.pdf)

* **Kneedle**: the point farthest above the diagonal; peak at $k = 3$ (0.632) [@satopaa2011]
* **L-method**: two lines left and right of a split; smallest error at $k = 3$ (42.4) [@salvador2004]

<!-- defines: Kneedle, L-method -->

## Finding the Elbow Without Looking

![](plots/10/f1_detectors.pdf)

* One call each: `kneedle.knee(np.c_[k, J])` and `lmethod.knee(np.c_[k, J])`
* Three blobs: L-method $k = 3$; the library's smoothed Kneedle 4
* A detector is an opinion: plot the curve

## The Silhouette of One Point

$a(i)$: mean distance within its cluster; $b(i)$: to the nearest other cluster.

$s(i) = \dfrac{b(i) - a(i)}{\max(a(i),\, b(i))} \in [-1, 1]$ [@rousseeuw1987]

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cblue, text width=2.2cm] (a) {$a = 1.273$\\ own cluster};
  \node[fillbox=corange, text width=2.2cm, right=8mm of a] (b) {$b = 6.114$\\ nearest other};
  \node[fillbox=cgreen, text width=2.8cm, right=8mm of b] (c) {$s = 0.792$};
  \draw[->] (a) -- (b);
  \draw[->] (b) -- (c);
\end{tikzpicture}
\end{center}
```

* Point $(2.5, 2.0)$; mean of the ten points 0.798
* $s \to 1$ inside; $0$ border; $s < 0$ wrong cluster ($-0.792$)

<!-- defines: silhouette -->

## The Silhouette Diagram

![](plots/10/f1_silhouette.pdf)

* One bar per point, by cluster, sorted; dashed line: the mean
* Mean $s$: 0.493, **0.557**, 0.460, 0.334 at $k = 2, 3, 4, 6$

## Calinski–Harabasz and Davies–Bouldin

| index | formula | choose |
|:--------------------|:--------------------------------------|:-----:|
| Calinski–Harabasz | $\dfrac{B_k/(k-1)}{W_k/(n-k)}$ | max |
| Davies–Bouldin | $\frac{1}{k}\sum_j \max_{l \neq j} \dfrac{s_j + s_l}{\lVert \mu_j - \mu_l \rVert}$ | min |

![](plots/10/f1_ch_db.pdf)

* CH: between over within; DB: worst neighbour; both pick $k = 3$

## The Gap Statistic

![](plots/10/f1_gap.pdf)

* $\mathrm{Gap}(k) = \mathbb{E}[\log J_k^{\text{ref}}] - \log J_k$, uniform reference [@tibshirani2001]
* Smallest $k$ with $\mathrm{Gap}(k) \geq \mathrm{Gap}(k+1) - s_{k+1}$
* Can say $k = 1$: uniform data gives 1, the others 3 to 8
* Cost: $B$ reference fits per $k$

## AIC and BIC for Mixtures

![](plots/10/f1_criteria.pdf)

* $P$ free parameters, $\hat{L}$ maximised likelihood of the mixture
* Akaike (AIC) $= 2P - 2\ln\hat{L}$; $\mathrm{BIC} = P\ln n - 2\ln\hat{L}$ [@schwarz1978]
* BIC: $\ln 300 = 5.70$ per parameter, AIC 2
* Only for mixtures: they need a likelihood

<!-- defines: AIC -->

## Which Criterion, on Which Set

![](plots/10/f1_hits.pdf)

* Six sets, true $k$ found: gap 5, BIC 4, silhouette 2
* 20 draws of 3 blobs: gap 0.95, BIC 0.90, silhouette 0.65

## When No Criterion Helps

![](plots/10/f1_nostructure.pdf)

* Moons: every $k$-means criterion asks for $k = 8$
* Uniform data: Kneedle finds $k = 5$; silhouette 0.379, blobs 0.557
* A wrong model gives a wrong $k$: shape first, then $k$

## DEMO F1: `06_choosing_k.ipynb`

Demo 10f: elbow, silhouette, CH, DB, gap statistic and BIC for the number of clusters.

# Use Cases

## One Task, One Method

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cblue, text width=3.5cm] (a) {customer segments, personas};
  \node[fillbox=cgreen, text width=3.5cm, below=2mm of a] (b) {topics, recommender profiles};
  \node[fillbox=corange, text width=3.5cm, below=2mm of b] (c) {anomalies, fraud, geospatial};
  \node[fillbox=cred, text width=3.5cm, below=2mm of c] (d) {taxonomy, phylogeny};
  \node[fillbox=cgray, text width=3.4cm, right=18mm of a] (a2) {$k$-means, GMM};
  \node[fillbox=cgray, text width=3.4cm, right=18mm of b] (b2) {NMF};
  \node[fillbox=cgray, text width=3.4cm, right=18mm of c] (c2) {DBSCAN};
  \node[fillbox=cgray, text width=3.4cm, right=18mm of d] (d2) {hierarchical (Ward, average)};
  \draw[->] (a) -- (a2); \draw[->] (b) -- (b2); \draw[->] (c) -- (c2); \draw[->] (d) -- (d2);
\end{tikzpicture}
\end{center}
```

* Four tasks, four demos (10g), each with a check
* The method follows the data and the question

## Segments in the Wine Data

![](plots/10/g1_wine.pdf)

* 6 497 wines: silhouette $k = 2$ (0.275), BIC 8, Kneedle 5
* $k = 2$ is red and white: ARI 0.942 ($k$-means), 0.768 (GMM)
* 229 uncertain wines: 0.581 agree, the rest 0.954

## Topics in News

![](plots/10/g1_topics.pdf)

* tf-idf of 1 000 BBC articles; categories unseen
* Top words: mobile, phone; labour, election; england, cup
* ARI: NMF 0.735, $k$-means on a projection 0.866
* NMF gives a topic mix; $k$-means wins the hard score

## Anomalies: Noise as the Alarm

![](plots/10/g1_anomaly.pdf)

* 3 290 hours, 16 series, $\mathrm{minPts} = 32$, $\epsilon = 2.26$ (knee)
* One normal cluster, 47 noise hours (1.4 %): red dots
* $\epsilon \times 0.6$: 1 699 flagged; $\times 1.25$: 3

## Taxonomy: the Digit Tree

![](plots/10/g1_taxonomy.pdf)

* Ward on ten mean images: 1 and 8, then 3 and 9, merge first
* Leaf order 5, 3, 9, 7, 2, 1, 8, 0, 4, 6
* Flat, 1 797 digits, $k = 10$: ARI 0.666 ($k$-means), 0.581 (GMM)

## DEMO G1: `07_usecases.ipynb`

Demo 10g: wine segments, news topics, anomalies, the digit tree.

# Choosing and Using

## Which Method for Which Data?

| data and goal | first choice | loses when |
|:---------------------------------|:---------------------|:-----------------------------------|
| round groups, large $n$ | $k$-means | clusters stretched or unequal |
| ellipses, uncertain borders | GMM | non-Gaussian shapes |
| arbitrary shapes, outliers | DBSCAN | densities differ, high $d$ |
| a hierarchy, small $n$ | Ward or average | $n$ large, chaining (single) |
| non-negative parts, topics | NMF | data has signs |

* Standardise first; fit on training data only (Classes 06–07)
* ARI needs labels; otherwise the silhouette, and look

## Summary: One Map

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cblue, text width=2.6cm] (a) {\textbf{hard}\\ $k$-means, tree, DBSCAN};
  \node[fillbox=cgreen, text width=2.6cm, right=5mm of a] (b) {\textbf{soft}\\ GMM, FCM, NMF};
  \node[fillbox=corange, text width=2.6cm, right=5mm of b] (c) {\textbf{choose $k$}\\ elbow, silhouette, BIC};
  \node[fillbox=cgray, text width=8.6cm, below=5mm of b] (d) {one assumption per method: pick the one that matches the shape};
\end{tikzpicture}
\end{center}
```

* Each method encodes a model of a cluster: check it first
* Soft rows are information, not proof of structure

## Summary: One Page of Equations

| block | equation |
|:-------------|:-----------------------------------------------------------------------------|
| hard, soft | $z_{ik} \in \{0, 1\}$; $u_{ik} \in [0, 1]$, $\sum_k u_{ik} = 1$ |
| $k$-means | $J = \sum_i \lVert x_i - \mu_{z_i} \rVert^2$, $\mu_k = \text{mean}(G_k)$ |
| Ward | $D = \frac{\lvert A \rvert \lvert B \rvert}{\lvert A \rvert + \lvert B \rvert} \lVert \mu_A - \mu_B \rVert^2$ |
| DBSCAN | core: $\lvert N_\epsilon(x) \rvert \geq \mathrm{minPts}$ |
| GMM | $u_{ik} \propto \pi_k \mathcal{N}(x_i \mid \mu_k, C_k)$ |
| FCM | $u_{ik} \propto d_{ik}^{-2/(m-1)}$ |
| silhouette | $s = (b - a)/\max(a, b)$ |
| BIC | $P\ln n - 2\ln\hat{L}$ |

* Shapes: $X$ is $n \times d$, $U$ is $n \times k$

## Recap: Library Calls

| task | call | note |
|:------------|:---------------------------------------|:---------------------------------------|
| $k$-means | `KMeans(k, n_init=10)` | `inertia_` is $J$ |
| tree | `AgglomerativeClustering(k, linkage="ward")` | `scipy.cluster.hierarchy` for the dendrogram |
| DBSCAN | `DBSCAN(eps, min_samples)` | label $-1$ is noise |
| GMM | `GaussianMixture(k, covariance_type)` | `predict_proba` is $U$, `bic` |
| knee | `kneedle.knee(np.c_[k, J])` | from `kneeliverse` |
| NMF | `pynnmf.nmf_hals(X.T, k)` | $X \geq 0$; rescale before the $\arg\max$ |

* sklearn has no fuzzy $c$-means: `faa.clustering.fcm` is 15 lines

## Lab 10

| part | task |
|:-------------:|:-----------------------------------------------------------------------------|
| A | NMF to soft memberships; compare with the hard $\arg\max$ |
| B | $k$-means by hand; $k$-means++ against random starts |
| C | $k$ against $\epsilon$: moons and rings, $k$-distance graph, knee |
| D | EM for a Gaussian mixture by hand; BIC |
| E, F | home: choose $k$ on the wine data; the case study |

<!-- refs:begin -->
## References

:::: {.columns}
::: {.column width="50%"}
\fontsize{5.5}{6.4}\selectfont\setlength{\parskip}{2.5pt}

Arthur, David, and Sergei Vassilvitskii. 2007. “K-Means++: The Advantages of Careful Seeding.” ACM-SIAM Symposium on Discrete Algorithms, 1027–35.

Bezdek, James C. 1981. Pattern Recognition with Fuzzy Objective Function Algorithms. Plenum Press.

Dempster, Arthur P., Nan M. Laird, and Donald B. Rubin. 1977. “Maximum Likelihood from Incomplete Data via the EM Algorithm.” Journal of the Royal Statistical Society: Series B 39 (1): 1–38.

Ding, Chris, Xiaofeng He, and Horst D. Simon. 2005. “On the Equivalence of Nonnegative Matrix Factorization and Spectral Clustering.” SIAM International Conference on Data Mining, 606–10.

Ester, Martin, Hans-Peter Kriegel, Jörg Sander, and Xiaowei Xu. 1996. “A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise.” International Conference on Knowledge Discovery and Data Mining, 226–31.

Lloyd, Stuart P. 1982. “Least Squares Quantization in PCM.” IEEE Transactions on Information Theory 28 (2): 129–37.

Rousseeuw, Peter J. 1987. “Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis.” Journal of Computational and Applied Mathematics 20: 53–65.

:::
::: {.column width="50%"}
\fontsize{5.5}{6.4}\selectfont\setlength{\parskip}{2.5pt}

Salvador, Stan, and Philip Chan. 2004. “Determining the Number of Clusters/Segments in Hierarchical Clustering/Segmentation Algorithms.” IEEE International Conference on Tools with Artificial Intelligence, 576–84.

Satopää, Ville, Jeannie Albrecht, David Irwin, and Barath Raghavan. 2011. “Finding a ‘Kneedle’ in a Haystack: Detecting Knee Points in System Behavior.” International Conference on Distributed Computing Systems Workshops, 166–71.

Schubert, Erich, Jörg Sander, Martin Ester, Hans-Peter Kriegel, and Xiaowei Xu. 2017. “DBSCAN Revisited, Revisited: Why and How You Should (Still) Use DBSCAN.” ACM Transactions on Database Systems 42 (3): 1–21.

Schwarz, Gideon. 1978. “Estimating the Dimension of a Model.” The Annals of Statistics 6 (2): 461–64.

Tibshirani, Robert, Guenther Walther, and Trevor Hastie. 2001. “Estimating the Number of Clusters in a Data Set via the Gap Statistic.” Journal of the Royal Statistical Society: Series B 63 (2): 411–23.

Ward, Joe H. 1963. “Hierarchical Grouping to Optimize an Objective Function.” Journal of the American Statistical Association 58 (301): 236–44.

:::
::::
<!-- refs:end -->
