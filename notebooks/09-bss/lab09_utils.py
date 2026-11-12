"""Lab 09 helpers: the two-sine mixtures, the digits pixels, the toy matrices and the plots.

Nothing here needs to be edited. Import it from the lab notebook:

    from lab09_utils import mixture_cloud, digits_matrix, toy_matrix, split_ratings, plot_mixtures, ...

The heavy lifting (sources, mixtures, matching, whitening, the ratings toy) lives in the course
package `faa.bss` (Class 09); this file only splits, masks and draws.
"""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split

import faa.data
from faa.bss import kurtosis, match, mixtures, ratings, snr_db, sources, whiten  # noqa: F401

BLUE, ORANGE, GREEN, RED = plt.rcParams["axes.prop_cycle"].by_key()["color"][:4]


def mixture_cloud(n: int = 300, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The running example of the lecture: X (2, n) microphone signals, S (2, n) true sources, A the 2 x 2 mixing matrix.

    The same generator as the demos (noise 0.05, A = [[0.9, 0.4], [0.7, 1.1]]).
    """
    return mixtures(n, seed=seed)


def digits_matrix(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """`n` digit images (8 x 8 pixels flattened, 64 features) as X (n, 64) with their labels.

    Rows are the examples: each row is one image, as scikit-learn's `load_digits`.
    """
    d = pl.read_csv(faa.data.data_dir() / "digits.csv.zst").sample(n=n, shuffle=True, seed=seed)
    X = d.select(pl.all().exclude("target")).to_numpy()
    return X, d["target"].to_numpy()


def toy_matrix(seed: int = 0) -> np.ndarray:
    """A 6 x 8 non-negative matrix with two latent parts: the NMF playground of Part C."""
    rng = np.random.default_rng(seed)
    W = rng.uniform(0.2, 1.0, (6, 2)) * np.array(
        [[1.0, 0.2], [0.9, 0.3], [1.1, 0.1], [0.2, 1.0], [0.1, 0.9], [0.3, 1.1]]
    )
    H = np.array([[1.0, 0.9, 0.8, 0.1, 0.2, 0.1, 0.0, 0.1], [0.1, 0.0, 0.2, 0.9, 0.8, 1.0, 0.9, 0.8]])
    return np.round(W @ H + rng.uniform(0, 0.05, (6, 8)), 2)


def split_ratings(frac_test: float = 0.3, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The 6 x 8 ratings matrix with a held-out test split of the observed entries.

    Returns R_train (NaN on held-out cells), R_test (NaN except held-out cells), M_train, M_test.
    """
    R, M = ratings(seed)
    idx = np.argwhere(M)
    tr, te = (np.asarray(a) for a in train_test_split(idx, test_size=frac_test, random_state=seed))
    M_train, M_test = np.zeros_like(M), np.zeros_like(M)
    M_train[tr[:, 0], tr[:, 1]] = True
    M_test[te[:, 0], te[:, 1]] = True
    R_train = np.where(M_train, R, np.nan)
    R_test = np.where(M_test, R, np.nan)
    return R_train, R_test, M_train, M_test


def masked_rmse(R_pred: np.ndarray, R_test: np.ndarray, M_test: np.ndarray) -> float:
    """RMSE of `R_pred` on the held-out cells of `M_test`."""
    err = (R_pred - R_test)[M_test]
    return float(np.sqrt(np.mean(err**2)))


def plot_mixtures(X: np.ndarray, S: np.ndarray, S_hat: np.ndarray | None = None) -> Figure:
    """Panels: the two mixtures, the true sources and (if given) the recovered sources."""
    panels: list[tuple[np.ndarray, str]] = [(X, "mixtures $X$"), (S, "true sources $S$")]
    if S_hat is not None:
        panels.append((S_hat, "recovered"))
    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(7.0, 1.9), sharey=True)
    t = np.arange(X.shape[1]) / X.shape[1]
    for ax, (Z, title) in zip(axes, panels, strict=True):
        for i in range(2):
            ax.plot(t, Z[i] + 2.2 * i, color=BLUE if i == 0 else ORANGE, lw=0.7)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("time (s)", fontsize=7)
        ax.tick_params(labelsize=6)
    axes[0].set_ylabel("signal (offset)", fontsize=7)
    fig.tight_layout()
    return fig


def scatter_cloud(X: np.ndarray, w: np.ndarray | None = None, title: str = "the mixture cloud") -> Figure:
    """The two microphones as a 2-D cloud; `w` (2,) draws a PC direction through the mean."""
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    ax.scatter(X[0], X[1], s=4, color=BLUE, alpha=0.5)
    if w is not None:
        m = X.mean(axis=1, keepdims=True)
        for s in (-1, 1):
            ax.plot([m[0, 0], m[0, 0] + s * 3 * w[0]], [m[1, 0], m[1, 0] + s * 3 * w[1]], color=RED, lw=1.5)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("microphone 1", fontsize=7)
    ax.set_ylabel("microphone 2", fontsize=7)
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    return fig


def tournament(task: str, seed: int = 0) -> dict[str, float]:
    """Part F: run PCA, ICA and NMF on the same task with the same score; returns {method: score}.

    Tasks: "sound" (correlation of the recovered spectra with the true ones, higher is better),
    "strokes" (sparseness of the components, higher), "blink" (RMSE of the blink-free channels, lower),
    "impute" (RMSE on 25 % hidden pixels, lower). The demo `06_usecases.ipynb` shows each as a figure.
    """
    import pynnmf
    from sklearn.decomposition import PCA, FastICA

    from faa import bss

    if task == "sound":
        V, _, T = bss.spectro_mix(seed)
        pc, ic = PCA(3).fit(V.T), FastICA(3, random_state=seed, max_iter=2000).fit(V.T)
        Wn = pynnmf.nmf_hals(V, 3, n=400, seed=seed, init="nndsvd")[1]
        temps = {"PCA": pc.components_, "ICA": np.asarray(ic.mixing_).T, "NMF": Wn.T}
        return {m: float(match(T, t)[1].mean()) for m, t in temps.items()}
    X, _ = digits_matrix(1797, seed)
    X = X / X.max()
    if task == "strokes":
        p, i = bss.ica_in_pca_space(X, 8, seed)
        Hn = pynnmf.nmf_hals(X, 8, n=400, seed=seed, init="nndsvd")[2]
        comps = {"PCA": p.components_, "ICA": bss.ica_components(p, i), "NMF": Hn}
        return {m: float(np.mean([bss.hoyer(v) for v in C])) for m, C in comps.items()}
    if task == "impute":
        mk = np.random.default_rng(seed).random(X.shape) > 0.25
        Xh = np.where(mk, X, np.nan)
        Xi = np.where(mk, X, np.nanmean(Xh, 0))
        for _ in range(60):  # iterative PCA: PCA needs complete data
            p = PCA(8).fit(Xi)
            Xi = np.where(mk, X, p.inverse_transform(p.transform(Xi)))
        W, H = pynnmf.nmf_hals(Xh, 8, n=400, seed=seed, init="nndsvd")[1:3]
        rm = lambda Y: float(np.sqrt(((Y - X)[~mk] ** 2).mean()))  # noqa: E731
        return {"PCA": rm(Xi), "ICA": rm(Xi), "NMF": rm(np.where(mk, X, np.clip(W @ H, 0, 1)))}
    if task == "blink":
        Xe, clean, blink, _ = bss.eeg_like(seed)
        cc = clean - clean.mean(1, keepdims=True)
        out = {}
        pe, ie = bss.ica_in_pca_space(Xe.T, 4, seed)
        Z = ie.transform(pe.transform(Xe.T))
        pp = PCA(4).fit(Xe.T)
        P = pp.transform(Xe.T)
        for name, S_, back in (
            ("PCA", P, lambda S_: pp.inverse_transform(S_)),
            ("ICA", Z, lambda S_: pe.inverse_transform(ie.inverse_transform(S_))),
        ):
            k = int(np.argmax([abs(np.corrcoef(S_[:, j], blink)[0, 1]) for j in range(4)]))
            S2 = S_.copy()
            S2[:, k] = 0
            Y = back(S2).T
            out[name] = float(np.sqrt(((Y - Y.mean(1, keepdims=True) - cc) ** 2).mean()))
        out["NMF"] = float("nan")  # NMF needs non-negative data: the channels are signed
        return out
    raise ValueError(f"unknown task {task!r}: sound, strokes, blink or impute")
