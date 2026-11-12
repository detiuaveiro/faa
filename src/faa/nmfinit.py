"""Initialisers for NMF (Class 09): the standard families and the sequential candidate search.

Every function returns a starting pair (W, H) with W (n, c) and H (c, m), both strictly positive, for a matrix X with a
boolean mask `obs` (True = observed). The pair goes to pyNNMF as `init=lambda x, k: (W, H)`.

Families (taxonomy of Esposito 2021, Mathematics 9(9):1006): random, SVD based (NNDSVD, Boutsidis and Gallopoulos 2008),
clustering based (k-means centroids as parts), separable (successive projection algorithm, Gillis and Vavasis), and
greedy rank-one deflation. `sequential` is the component-by-component search with candidates from the empty space
algorithm (`ess`) or from a uniform draw, scored by the reduction of the residual.
"""

import numpy as np
from pynnmf.initialization import init_nndsvd
from scipy.optimize import nnls

EPS = 1e-6


def w_from_h(X: np.ndarray, obs: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Non-negative weights W of every row on fixed parts H, using the observed cells of the row only."""
    W = np.zeros((X.shape[0], H.shape[0]))
    for i in range(X.shape[0]):
        o = obs[i]
        if o.any():
            W[i] = nnls(H[:, o].T, X[i, o])[0]
    return np.maximum(W, EPS)


def svd_complete(X: np.ndarray, obs: np.ndarray, c: int, rounds: int = 15) -> np.ndarray:
    """Fill the unobserved cells by iterative rank-c SVD (iterative PCA without centring), observed cells stay."""
    fill = X[obs].mean() if (~obs).any() else 0.0
    Xf = np.where(obs, X, fill)
    for _ in range(rounds):
        u, s, vt = np.linalg.svd(Xf, full_matrices=False)
        Xf = np.where(obs, X, (u[:, :c] * s[:c]) @ vt[:c])
    return np.clip(Xf, 0, None)


def nndsvd_complete(X: np.ndarray, obs: np.ndarray, c: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """SVD based: complete the matrix by iterative SVD, then the real NNDSVD (the pyNNMF svd_impute clips instead)."""
    W, H = init_nndsvd(svd_complete(X, obs, c), c)
    return np.maximum(W, EPS), np.maximum(H, EPS)


def kmeans(X: np.ndarray, obs: np.ndarray, c: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Clustering based: the c centroids of the (completed) rows are the parts."""
    from sklearn.cluster import KMeans

    km = KMeans(c, n_init=3, random_state=seed).fit(svd_complete(X, obs, c))  # pyright: ignore[reportArgumentType]
    H = np.maximum(km.cluster_centers_, EPS)
    return w_from_h(X, obs, H), H


def spa(X: np.ndarray, obs: np.ndarray, c: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Separable: successive projection picks c extreme rows of the completed matrix as the parts."""
    Xf = svd_complete(X, obs, c)
    R, idx = Xf.copy(), []
    for _ in range(c):
        j = int(np.argmax((R**2).sum(1)))
        idx.append(j)
        v = R[j] / np.linalg.norm(R[j])
        R = R - np.outer(R @ v, v)
    H = np.maximum(Xf[idx], EPS)
    return w_from_h(X, obs, H), H


def _rank_one(R: np.ndarray, obs: np.ndarray, cand: np.ndarray, refine: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Best rank-one non-negative term of the residual R among the candidate parts, then alternating refinement."""
    Rz = np.where(obs, R, 0.0)
    best = None
    for h in cand:
        w = np.clip(Rz @ h / np.maximum((obs * h**2).sum(1), 1e-9), 0, None)
        gain = (Rz**2 * obs).sum() - (((Rz - np.outer(w, h)) ** 2) * obs).sum()
        if best is None or gain > best[0]:
            best = (gain, w, h)
    assert best is not None
    _, w, h = best
    for _ in range(refine):
        w = np.clip(Rz @ h / np.maximum((obs * h**2).sum(1), 1e-9), 0, None)
        h = np.clip(Rz.T @ w / np.maximum((obs.T * w**2).sum(1), 1e-9), 0, None)
    return np.maximum(w, EPS), np.maximum(h, EPS)


def greedy(X: np.ndarray, obs: np.ndarray, c: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Greedy rank-one deflation: each part starts from the top singular vector of the residual."""
    R = np.where(obs, X, 0.0)
    Ws, Hs = [], []
    for _ in range(c):
        h0 = np.abs(np.linalg.svd(R, full_matrices=False)[2][0])
        w, h = _rank_one(R, obs, h0[None, :], refine=5)
        Ws.append(w)
        Hs.append(h)
        R = np.clip(R - np.outer(w, h), 0, None) * obs
    return np.array(Ws).T, np.array(Hs)


def sequential(
    X: np.ndarray, obs: np.ndarray, c: int, seed: int = 0, use_ess: bool = True, n_cand: int = 48
) -> tuple[np.ndarray, np.ndarray]:
    """Component by component: propose n_cand parts (empty space algorithm away from the parts already chosen, or a uniform
    draw), keep the one that removes most residual, refine it, deflate, move to the next component."""
    import ess

    rng = np.random.default_rng(seed)
    hi = X[obs].max()
    m = X.shape[1]
    bounds = np.tile([0.0, hi], (m, 1))
    R = np.where(obs, X, 0.0)
    Ws, Hs = [], []
    for _ in range(c):
        if use_ess:
            obst = np.array(Hs) if Hs else rng.uniform(0, hi, (1, m))
            cand = ess.ess(obst, bounds, n=n_cand, seed=int(rng.integers(1 << 30)), epochs=30)[len(obst) :]
        else:
            cand = rng.uniform(0, hi, (n_cand, m))
        w, h = _rank_one(R, obs, cand)
        Ws.append(w)
        Hs.append(h)
        R = np.clip(R - np.outer(w, h), 0, None) * obs
    return np.array(Ws).T, np.array(Hs)


def random_scaled(X: np.ndarray, obs: np.ndarray, c: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Random start with the scale of the data: W, H ~ U(0, 2s) with s = sqrt(mean(X_obs) / c), so that the mean of W H is the mean of X.
    (pyNNMF's `random` divides by c * max, which starts a matrix with mean 20 at a product near 0.)"""
    rng = np.random.default_rng(seed)
    s = np.sqrt(X[obs].mean() / c)
    return rng.uniform(EPS, 2 * s, (X.shape[0], c)), rng.uniform(EPS, 2 * s, (c, X.shape[1]))


def portfolio(
    X: np.ndarray,
    obs: np.ndarray,
    c: int,
    fit,
    seed: int = 0,
    val_frac: float = 0.1,
    rounds: int = 30,
    n_random: int = 4,
    val_mask: np.ndarray | None = None,
):
    """Validated portfolio start. Hide `val_frac` of the observed cells, give every candidate start (scaled random draws, completed
    NNDSVD, greedy rank-one, k-means parts) a short fit on the rest, and keep the start whose short fit predicts the hidden cells
    best. No start family wins on every matrix; the held-out cells say which one to trust on this one.
    The hidden cells should look like the cells to predict: pass `val_mask` (for example block gaps for a time series), by default random cells.
    `fit(obs_fit, W0, H0, rounds)` must return the reconstruction W H. Returns W0, H0, the chosen name and the scores."""
    rng = np.random.default_rng(seed)
    val = obs & (rng.random(obs.shape) < val_frac) if val_mask is None else obs & val_mask
    fit_obs = obs & ~val
    cands = {f"random {i}": random_scaled(X, fit_obs, c, seed * 100 + i) for i in range(n_random)}
    for name, fn in (("completed NNDSVD", nndsvd_complete), ("greedy rank-one", greedy), ("k-means", kmeans)):
        cands[name] = fn(X, fit_obs, c, seed)
    scores = {}
    for name, (W0, H0) in cands.items():
        P = fit(fit_obs, W0, H0, rounds)
        scores[name] = float(np.sqrt(((P - X)[val] ** 2).mean()))
    best = min(scores, key=lambda k: scores[k])
    return cands[best][0], cands[best][1], best, scores
