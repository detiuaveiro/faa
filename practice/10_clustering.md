---
title: "Lab 10 — Soft and Hard Clustering, Built from Scratch"
---

# Overview

**Class 10 · Topic 8.** Lecture: `slides/10_clustering.pdf`. Demo notebooks in `notebooks/10-clustering/`: `01_hard_soft.ipynb`, `02_kmeans_hier.ipynb`, `03_dbscan.ipynb`, `04_gmm_fcm.ipynb`, `05_benchmark.ipynb`, `06_choosing_k.ipynb` and `07_usecases.ipynb`.

**Notebook for this lab:** `notebooks/10-clustering/lab10_clustering.ipynb`, with helpers in `lab10_utils.py` (same folder, do not edit). Cells marked **TODO** contain `raise NotImplementedError`: replace it with your code. Every other cell is ready, and it draws the plots and prints the comparisons once your functions work.

**Solution:** `solutions/lab10_clustering_solution.ipynb` is the fully solved notebook. Before each function it explains how the solution was reached, and after each experiment it answers this guide's questions with the numbers the run produces. Try every task yourself first.

**The rule of this lab.** Every clustering method answers with a label (hard) or a row of memberships (soft), and each one assumes a shape: $k$-means round cells, DBSCAN dense regions, a mixture ellipses, NMF additive parts. You write the core loop of four methods with NumPy and compare each with its library twin on the **same data**. The notebook prints the difference; explain it (the start, the scale, the shape).

**In class (60 minutes): Parts A, B, C and D**, soft memberships from NMF, $k$-means with $k$-means++, DBSCAN with the $k$-distance knee, and one EM step of a Gaussian mixture. **At home: Parts E and F**, choosing $k$ and the wine case study; the solution notebook explains them.

By the end of the lab you should be able to:

1. Turn the factors of an NMF into membership rows and compare them with a hard $\arg\max$.
2. Write Lloyd's loop and $k$-means++ and say how the start changes the cost.
3. Tune $\epsilon$ with the $k$-distance graph and the knee, and say why $k$-means fails on moons and rings.
4. Write the E-step and the M-step of a Gaussian mixture and choose $k$ with the BIC.

| Part | Task | twin | Time |
|:--|:--|:--|:-:|
| A | NMF memberships: rescale, normalise, entropy | `pynnmf.nmf_hals` | 12 min |
| B | $k$-means and $k$-means++ by hand | `KMeans` | 15 min |
| C | $k$-distance knee and DBSCAN by hand | `DBSCAN` | 15 min |
| D | One EM step of a Gaussian mixture, BIC | `GaussianMixture` | 15 min |
| E | Choosing $k$ on the wine data | `silhouette_score` | home |
| F | Challenge: the wine case study | `compare` | home |

## Setup

```bash
cd faa
source venv/bin/activate        # see README: make venv, or pip install .
jupyter lab notebooks/10-clustering/lab10_clustering.ipynb
```

The data: 1 000 BBC **news** articles as a tf-idf matrix (`datasets/bbc_news.csv.zst`), the six **toy sets** of Demo 10e (moons, rings, blobs, stretched blobs, unequal blobs, noise), the **ten points** of the lecture, and the 6 497 **wines** (`datasets/wine_quality.csv.zst`, 11 standardised measurements). The helpers in `lab10_utils.py` load, subsample and draw; the clustering code is yours.

# Part A — From NMF to Soft and Hard Assignments

NMF factors the tf-idf matrix, $X^\top \approx WH$. Column $i$ of $H$ is article $i$ over five topics: scaled and normalised, it is a membership row $u_i$.

## A1. Memberships from $W$ and $H$

Implement `nmf_memberships(W, H)` (`W` is `(d, k)`, `H` is `(k, n)`; return `U` of shape `(n, k)`, every row summing to 1). The scale of a part is free, $(WD)(D^{-1}H) = WH$: multiply row $j$ of $H$ by the column sum of $W_{\cdot j}$, then normalise each article. Then implement `entropy_bits(U)`, the Shannon entropy in bits of every row. The driver compares the raw and the rescaled $\arg\max$ with the true categories and lists the most ambiguous articles.

1. Why does the raw $\arg\max$ of $H$ score so much worse than the rescaled one?
2. A hard clustering gives an ambiguous article one topic. What does the soft row add, and what does the $\arg\max$ throw away?
3. The rows of $U$ sum to 1 but they are not probabilities of a model. What are they?

# Part B — $k$-Means and $k$-Means++ by Hand

$k$-means alternates two steps: assign each point to the nearest centroid, then move each centroid to the mean of its points. The start decides which local optimum is reached.

## B1. The start and the loop

Implement `kmeanspp_init(X, k, rng)` (first centroid a random point, each next one drawn with probability proportional to the squared distance to the nearest chosen centroid) and `kmeans_by_hand(X, k, init, seed, iters)` (`init` is `"random"` or `"k-means++"`; return the labels, the centroids and the cost $J$ at every iteration). The driver runs your loop on the ten points and on the `varied` set (50 starts of each kind) and compares the best cost with `KMeans`.

1. Why can $J$ never rise from one iteration to the next? Say what each of the two steps does to $J$.
2. How often does a random start end more than 5 % above the best cost, and what does $k$-means++ change?
3. Your best cost and `KMeans(n_init=10)` agree. Why is one start not enough in general?

# Part C — $k$ Against $\epsilon$: DBSCAN on the Moons

$k$-means needs $k$ and cuts convex cells; DBSCAN needs a radius $\epsilon$ and $\mathrm{minPts}$ and follows density.

## C1. The $k$-distance graph and its knee

Implement `k_distance(X, m)` (sorted distance of each point to its $m$-th neighbour, itself counting as the first) and `eps_from_knee(d)` (the value of the knee, found with `kneedle.knee` from `kneeliverse`).

## C2. DBSCAN by hand

Implement `dbscan_by_hand(X, eps, min_pts)` (labels, `-1` for noise). Core points have at least `min_pts` points within `eps` (itself included); grow a cluster from each unvisited core point through its core neighbours; a border point joins the first cluster that reaches it. The driver tunes $\epsilon$ on the moons and the rings and compares your labels with `DBSCAN`.

1. $k$-means has ARI near 0 on the rings. Which assumption of $k$-means do the rings break?
2. Your labels and `DBSCAN` should agree except on border points. Why can border points differ?
3. On the rings the knee gives 11 clusters and $1.5\epsilon$ gives 2. What does the table say about $2.5\epsilon$, and what does that say about the knee as a rule?

# Part D — Gaussian Mixtures by Hand: the EM Step

EM alternates the **E-step** (the membership of every point in every component) and the **M-step** (weights, means and covariances re-estimated with the memberships as weights).

## D1. One EM iteration

Implement `em_step(X, pi, mu, Sigma)` (return the new `pi`, `mu`, `Sigma`, the memberships `U` and the log-likelihood of the parameters you were given; use `scipy.stats.multivariate_normal.pdf` and add `1e-6 * I` to each covariance), then `n_params(k, d)` for a full-covariance mixture and `bic(loglik, k, d, n)`. The driver runs 60 of your iterations on the stretched blobs, compares with `GaussianMixture` and scores $k = 1..6$ by BIC.

1. The log-likelihood never falls. Which step makes the bound tight, and which one raises it?
2. $k$-means has ARI about 0.5 on this set and your mixture 1.0. Which parameters of the mixture give it the stretch?
3. Where is the minimum of the BIC, and why does it not keep falling as $k$ grows?

# Part E — Choosing $k$ on the Wine Data, at Home

The first 1 599 wines are red; `is_red` is only used at the end to check an answer, never to fit.

## E1. Three criteria for $k$

Implement `choose_k(X, ks, seed)` on a random sample of 2 000 wines: return the $k$ picked by the **elbow** (Kneedle on the $k$-means inertia), by the **silhouette** (largest mean over $k \geq 2$) and by the **BIC** of a full-covariance mixture (smallest).

1. Do the three criteria agree? Which one would you trust for a customer-segmentation task, and why?
2. The silhouette prefers $k = 2$. What do the two clusters turn out to be?

# Part F — Challenge: the Wine Case Study

Six methods, one data set, no label in the fit. Implement `compare(X, red, k, seed)`: fit $k$-means, Ward, DBSCAN (`min_samples=22`, $\epsilon$ from your knee, on a sample of 3 000 wines), a Gaussian mixture, fuzzy $c$-means (`fcm`) and NMF (positive and negative parts of the standardised wines, your `nmf_memberships`) with $k = 2$. Return a table with the ARI to `is_red`, the silhouette and the number of noise points. Then write three sentences: which method you would give to a winemaker, which soft rows you would look at first, and which method fails and why.

**Going further.** Hold out a third of the wines, fit the mixture on the rest and count the held-out wines with an entropy above 0.5 bit. Then repeat on 2 000 wines of one colour only: what does the BIC pick?
