---
title: "Lab 09 — Blind Signal Separation, Built from Scratch"
---

# Overview

**Class 09 · Topic 7.** Lecture: `slides/09_blind_signal_separation.pdf`. Demo notebooks in `notebooks/09-bss/`: `01_mixtures_pca.ipynb`, `02_ica.ipynb`, `03_nmf.ipynb`, `04_lsi_cf.ipynb`, `05_audio.ipynb`, `06_usecases.ipynb` and `07_imputation.ipynb`.

**Notebook for this lab:** `notebooks/09-bss/lab09_bss.ipynb`, with helpers in `lab09_utils.py` (same folder, do not edit). Cells marked **TODO** contain `raise NotImplementedError`: replace it with your code. Every other cell is ready, and it draws the plots and prints the comparisons once your functions work.

**Solution:** `solutions/lab09_bss_solution.ipynb` is the fully solved notebook. Before each function it explains how the solution was reached, and after each experiment it answers this guide's questions with the numbers the run produces. Try every task yourself first.

**The rule of this lab.** Every separation method is a matrix factorisation $X \approx WH$ with a different constraint: PCA constrains the directions to be orthogonal, ICA constrains them to be least Gaussian, NMF constrains the parts to be non-negative. You write the inner loops with NumPy and compare each with its library twin on the **same data**. The notebook prints the difference; explain it (centring, whitening, the permutation and the sign).

**In class (60 minutes): Parts A, B, C and D**, PCA by hand, ICA by kurtosis ascent, NMF by multiplicative updates and collaborative filtering with missing values. **At home: Parts E and F**, LSI and the challenge; the solution notebook explains them.

By the end of the lab you should be able to:

1. Compute the PCA of a cloud from the covariance eigendecomposition and reconstruct with $c$ components.
2. Whiten the mixtures and recover the sources by one kurtosis gradient ascent per direction.
3. Write the NMF multiplicative updates and read the learned parts of a digit.
4. Fill a ratings matrix with weighted NMF and choose the rank by held-out error.

| Part | Task | twin | Time |
|:--|:--|:--|:-:|
| A | PCA by hand: centre, covariance, eigen, reconstruct | `PCA` | 12 min |
| B | ICA by hand: whiten, kurtosis gradient ascent | `FastICA` | 15 min |
| C | NMF multiplicative updates and the learned parts | `pynnmf.nmf_mu` | 15 min |
| D | Collaborative filtering with missing values | `pynnmf.rwnmf` | 12 min |
| E | LSI: tf-idf + SVD, the latent topics | `TruncatedSVD` | home |
| F | Challenge: the use-case tournament | `tournament` | home |

## Setup

```bash
cd faa
source venv/bin/activate        # see README: make venv, or pip install .
jupyter lab notebooks/09-bss/lab09_bss.ipynb
```

The data: the **two-sine mixtures** (300 samples, two microphones, the running example of the lecture), 400 **digit** images from `datasets/digits.csv.zst` (64 pixels each), a small 6 x 8 **ratings** matrix with held-out cells, and the toy matrices of the demos. The helpers in `lab09_utils.py` build the data and draw the scatter panels; the factorisations are yours.

# Part A — PCA by Hand

PCA is the eigendecomposition of the covariance of the **centred** data: the eigenvector of the largest eigenvalue is the direction of most spread, not the direction of either source.

## A1. Centre, covariance, eigen

Implement `pca_by_hand(X)` (`X` is `(n, d)`; return the scores `T`, the components `V` as rows, unit length, ordered by decreasing eigenvalue, and the eigenvalues `vals`). Centre inside; use `np.linalg.eigh` on the covariance. Then implement `reconstruct(T, V, c)` (keep the first $c$ components and map back to the original space). The driver compares with `PCA` on the mixture cloud and measures the relative reconstruction error of the digits for $c = 1, 2, 8, 32$.

1. Why must the data be centred before the covariance, and what does PC1 chase if you forget it?
2. The two eigenvalues are very different (EVR about 0.92/0.08). What does that say about how the two microphones are related?
3. On digits, what does the reconstruction look like at $c = 4$, and why is even $c = 32$ still a blur?

# Part B — ICA by Hand

ICA whitens, then finds the rotation that makes the two components **least Gaussian**: here, one kurtosis gradient ascent per source direction.

## B1. Whiten and one kurtosis ascent

Implement `kurtosis_gradient(X, w, step=0.05, iters=200)` (`X` is `(2, n)`, rows are signals; return the final unit vector `w` after `iters` steps of gradient ascent on $|\operatorname{kurt}(w^\top X)|$; renormalise `w` each step). After whitening, $\mathbb{E}z^2 = 1$, so the kurtosis of the projected signal is $\mathbb{E}z^4 - 3$ and its gradient is $4\,\mathbb{E}[z^3 X]$. The driver runs your ascent from three starts and compares the found directions with `FastICA` and with PCA, scored by the correlation with the true sources after `match` (the permutation/sign fix every blind method needs).

1. Why does the ascent start from a whitened `X`? What happens to kurtosis maximisation if the components still have different scales?
2. Your two found directions need not be orthogonal. What forces the mixing matrix of `FastICA` to be invertible?
3. The recovered sources match the truth up to sign and scale. Where in your code did you not need to care?

# Part C — NMF Multiplicative Updates by Hand

NMF minimises $\|X - WH\|_F^2$ with $W, H \geq 0$ by the updates
$W \leftarrow W \odot (XH^\top)/(WHH^\top)$ and $H \leftarrow H \odot (W^\top X)/(W^\top WH)$.

## C1. The update loop

Implement `nmf_by_hand(X, k, iters=200, seed=0)` (`X` is `(n, d)` non-negative; return `W` `(n, k)`, `H` `(k, d)` and the Frobenius cost at every iteration, from a positive random init). The driver runs your updates on the toy matrix, checks the cost never rises, compares with `pynnmf.nmf_mu` and shows the learned parts.

1. Why can the multiplicative update never make an entry negative? What is the cost of that guarantee?
2. Your cost and pyNNMF's cost agree to rounding at rank 2 here, but two minima can still differ. Which two things can differ between two minima?
3. Increase $k$ to 4 on the toy (which has true rank 2). What happens to the error, and why is that not free?

# Part D — Collaborative Filtering with Missing Values

Real rating matrices have holes. Weighted NMF counts only the observed entries: minimise $\|M \odot (R - WH)\|_F^2$ where $M$ is 1 on observed cells. pyNNMF's `rwnmf` does exactly this with NaN entries ignored.

## D1. Fill the holes and recommend

Implement `fill(R_train, k, seed=0)` (`R_train` has NaN on unknown cells; return the reconstruction from `pynnmf.rwnmf` with rank `k`) and `recommend(R_pred, R_train, user, k_top=3)` (the indices of the `k_top` largest predicted ratings of `user` among the cells that are NaN in `R_train`). The driver splits the 6 x 8 ratings (70 % train), fills with a rank-1 model and reports the held-out RMSE, the training RMSE, the top recommendations for user 0 and the rank sweep.

1. The test RMSE at rank 1 is better than at rank 2. Why does the richer model lose on 8 test cells?
2. A user with no ratings at all still gets predictions. From where does the model take their taste?
3. What would overfitting look like here (describe it with the train/test RMSE pair, no run needed)?

# Part E — LSI, at Home

Build the term-document matrix of the demo `04_lsi_cf.ipynb` toy, run tf-idf + `TruncatedSVD` (rank 2) and find the two topics. Then project a query ("goal keeper election") and rank the documents by cosine in the latent space. The demo's section E2 is the model to follow.

# Part F — Challenge: the Use-Case Tournament

The same task goes to PCA, ICA and NMF with one score. `tournament(task)` returns the score of each method for `"sound"` (spectra of three notes, higher is better), `"strokes"` (sparseness of digit parts, higher), `"blink"` (error after removing an eye blink, lower) and `"impute"` (error on 25 % hidden pixels, lower). Run the four, then fill a table (task, winner, the constraint that explains it) and say which task has no NMF score, and why. The demo `06_usecases.ipynb` draws each task, `07_imputation.ipynb` adds ratings, time series and the choice of the start.

**Going further.** Choose the rank of the digit factorisation by **cross-validated reconstruction error**: mask 25 % of the pixels at random, fit `nmf_hals` at ranks 2, 4, 8, 16, 32 and compare the RMSE on the missing pixels.
