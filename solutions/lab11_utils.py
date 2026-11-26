"""Lab 11 helpers: the digits with few labels, the graph, the Keras autoencoder and two small plots.

Nothing here needs to be edited. Import it from the lab notebook:

    from lab11_utils import split, few_labels, knn_affinity, make_ae, fit_ae, plot_codes

The graph, the autoencoder builder and the plots live in the course package `faa.semisup` (Class 11);
the pseudo-label loop, the label spreading, the linear autoencoder gradient and the detector threshold are yours.
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from faa import semisup

BLUE, ORANGE, RED = "#0072B2", "#E69F00", "#D55E00"


def split(seed: int = 0):
    """The 1 797 digits (64 pixels in [0, 1]): train 70 %, test 30 % (stratified). Returns xtr, ytr, xte, yte."""
    X, y = semisup.digits_flat()
    xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    return xtr, ytr, xte, yte


def few_labels(y, per_class: int, seed: int = 0):
    """Indices of `per_class` labelled rows of every class; all other rows count as unlabelled."""
    return semisup.few_labels(y, per_class, seed)


def knn_affinity(X, k: int = 10):
    """Symmetric Gaussian affinity W on the k nearest neighbours (zero diagonal)."""
    return semisup.knn_affinity(X, k)


def make_ae(d_in: int, code: int, hidden=(32,), seed: int = 0):
    """A Keras autoencoder d_in -> hidden -> code -> hidden -> d_in (ReLU, sigmoid output); returns (autoencoder, encoder)."""
    return semisup.autoencoder(d_in, code, hidden=hidden, seed=seed)


def fit_ae(ae, X, noise: float = 0.0, mask: float = 0.0, epochs: int = 100, seed: int = 0):
    """Train an autoencoder (a new noisy or masked copy of X in every epoch; the target stays X)."""
    return semisup.fit_ae(ae, X, noise=noise, mask=mask, epochs=epochs, seed=seed)


def recon_error(ae, X):
    """Mean squared reconstruction error of every row."""
    return semisup.recon_error(ae, X)


def classifier_from(enc, freeze: bool = False, seed: int = 0):
    """A Keras classifier: a copy of the encoder `enc` plus a softmax layer."""
    return semisup.classifier_from(enc, freeze=freeze, seed=seed)


def plot_curve(x, ys: dict, xlabel: str, ylabel: str = "test accuracy", logx: bool = False):
    """One line per entry of `ys` (name -> values) against x."""
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    for (name, v), c in zip(ys.items(), (BLUE, ORANGE, RED, "#009E73", "#56B4E9"), strict=False):
        ax.plot(x, v, marker="o", ms=3, color=c, label=name)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_digits(rows: dict, n: int = 8):
    """Rows of digits: name -> array (m, 64); the first n images of each."""
    fig, axes = plt.subplots(len(rows), n, figsize=(0.55 * n, 0.6 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (name, imgs) in enumerate(rows.items()):
        for c in range(n):
            axes[r, c].imshow(np.asarray(imgs[c]).reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(name, fontsize=6)
    fig.tight_layout()
    return fig
