# pyright: reportArgumentType=false, reportReturnType=false, reportCallIssue=false
# (scikit-learn stubs type `n_init` as str and `cluster_std` as float: false errors, as in Classes 05-07)
"""Clustering helpers of Class 10: toy benchmarks, fuzzy c-means, soft-membership scores and the gap statistic (no plots)."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_circles, make_moons

BENCH = ("blobs", "aniso", "varied", "moons", "circles", "uniform")


def benchmarks(seed: int = 0, n: int = 300) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The six toy sets of the class (X, y) with y the generating group (all one group for `uniform`)."""
    rng = np.random.default_rng(seed)
    blobs = make_blobs(n, centers=3, cluster_std=0.8, random_state=seed)
    stretched = make_blobs(n, centers=3, random_state=170 + seed)
    aniso = (stretched[0] @ np.array([[0.6083, -0.6367], [-0.4089, 0.8525]]), stretched[1])
    varied = make_blobs(n, centers=3, cluster_std=[0.4, 1.6, 0.8], random_state=seed + 2)
    moons = make_moons(n, noise=0.07, random_state=seed)
    circles = make_circles(n, noise=0.05, factor=0.45, random_state=seed)
    uniform = (rng.uniform(-1, 1, (n, 2)), np.zeros(n, dtype=int))
    return dict(zip(BENCH, (blobs, aniso, varied, moons, circles, uniform), strict=True))


def fcm(X: np.ndarray, k: int, m: float = 2.0, iters: int = 100, seed: int = 0, tol: float = 1e-6):
    """Fuzzy c-means: returns the centres (k, d), the memberships U (n, k) with rows summing to 1 and the cost J_m per iteration.

    U starts from the best of 10 k-means++ runs (hard labels smoothed), then centres and memberships alternate."""
    km = KMeans(k, n_init=10, random_state=seed).fit(X)
    U = np.full((len(X), k), 0.1 / max(k - 1, 1))
    U[np.arange(len(X)), km.labels_] = 0.9
    cost = []
    for _ in range(iters):
        Um = U**m
        centres = Um.T @ X / Um.sum(0)[:, None]
        D2 = ((X[:, None, :] - centres[None]) ** 2).sum(-1) + 1e-12
        cost.append(float((Um * D2).sum()))
        inv = D2 ** (-1.0 / (m - 1))
        U_new = inv / inv.sum(1, keepdims=True)
        if np.abs(U_new - U).max() < tol:
            U = U_new
            break
        U = U_new
    return centres, U, cost


def entropy(U: np.ndarray) -> np.ndarray:
    """Shannon entropy in bits of each row of a membership matrix: 0 = certain, log2(k) = undecided."""
    P = np.clip(U, 1e-12, 1.0)
    return -(P * np.log2(P)).sum(1)


def normalise_rows(U: np.ndarray) -> np.ndarray:
    """Rows of a non-negative matrix scaled to sum to 1: NMF weights become membership degrees u_ik."""
    return U / np.clip(U.sum(1, keepdims=True), 1e-12, None)


def gap_statistic(X: np.ndarray, ks, n_ref: int = 10, seed: int = 0):
    """Gap(k) = mean log W_k of uniform reference sets - log W_k of X (Tibshirani 2001), with the standard error."""
    rng = np.random.default_rng(seed)
    lo, hi = X.min(0), X.max(0)
    gap, se = [], []
    for k in ks:
        w = np.log(KMeans(k, n_init=5, random_state=seed).fit(X).inertia_)
        ref = [
            np.log(KMeans(k, n_init=3, random_state=seed).fit(rng.uniform(lo, hi, X.shape)).inertia_)
            for _ in range(n_ref)
        ]
        gap.append(np.mean(ref) - w)
        se.append(np.std(ref) * np.sqrt(1 + 1 / n_ref))
    return np.array(gap), np.array(se)


def nmf_membership(W: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Soft memberships u_ik (n, k) of an NMF X^T = W H (W: d x k, H: k x n).

    The scale of a part is free, (WD)(D^-1 H) = WH, so the scale of each W column is moved into H before the columns of H are normalised."""
    return normalise_rows(H.T * W.sum(0))
