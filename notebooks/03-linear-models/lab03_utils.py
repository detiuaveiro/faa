"""Lab 03 helpers: the two datasets, the optimizer, and ready-made plots.

Nothing here needs to be edited. Import it from the lab notebook:

    from lab03_utils import load_housing, load_spam, train, plot_history, ...

`train` is the Adam optimizer of Class 02 with a decaying learning rate. It differentiates your loss with
`jax.value_and_grad`, so every model in this lab is *a loss written with `jax.numpy`* plus a call to `train`.
Parameters are a dictionary `{"w": weights, "b": bias}`.
"""

from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

jax.config.update("jax_enable_x64", True)

COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray"]


def _data_dir() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "datasets" / "spambase.csv.zst").exists():
            return parent / "datasets"
    raise FileNotFoundError("datasets/spambase.csv.zst not found: run the notebook from inside the repository")


def load_housing() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """California housing: 20640 districts, 8 features, target = median house value in units of 100 000 USD.

    `AveRooms`, `AveBedrms`, `Population` and `AveOccup` are extremely skewed (a few districts have 1000+ people per
    house), so they are replaced by their natural logarithm. The target is capped at 5.0 in the original data.
    """
    df = pl.read_csv(_data_dir() / "california_housing.csv.zst")
    X = df.drop("MedHouseVal").to_numpy().astype(float)
    names = df.drop("MedHouseVal").columns
    for j, name in enumerate(names):
        if name in ("AveRooms", "AveBedrms", "Population", "AveOccup"):
            X[:, j] = np.log(X[:, j])
    return X, df["MedHouseVal"].to_numpy().astype(float), names


def load_spam() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Spambase: 4601 e-mails, 57 features (word and character frequencies in %, capital-run lengths), 39% spam.

    Returns the raw features; `y = 1` is spam.
    """
    df = pl.read_csv(_data_dir() / "spambase.csv.zst")
    return df.drop("spam").to_numpy().astype(float), df["spam"].to_numpy().astype(int), df.drop("spam").columns


def train(loss: Callable, params: dict, *data, lr: float = 0.05, steps: int = 2000) -> tuple[dict, np.ndarray]:
    """Minimize `loss(params, *data)` with Adam (cosine-decayed step). Returns the final params and the loss history.

    The gradient comes from `jax.value_and_grad`: `loss` must be written with `jax.numpy`.
    """
    value_and_grad = jax.jit(jax.value_and_grad(loss))
    m = jax.tree.map(jnp.zeros_like, params)
    v = jax.tree.map(jnp.zeros_like, params)
    history = []
    for t in range(1, steps + 1):
        value, grad = value_and_grad(params, *data)
        m = jax.tree.map(lambda a, g: 0.9 * a + 0.1 * g, m, grad)
        v = jax.tree.map(lambda a, g: 0.999 * a + 0.001 * g**2, v, grad)
        step = lr * 0.5 * (1 + np.cos(np.pi * (t - 1) / steps))
        params = jax.tree.map(
            lambda p, a, b: p - step * (a / (1 - 0.9**t)) / (jnp.sqrt(b / (1 - 0.999**t)) + 1e-8), params, m, v
        )
        history.append(float(value))
    return params, np.array(history)


def make_blobs(rng: np.random.Generator, n: int = 100, gap: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Two 2D Gaussian blobs, labels 0/1. A large `gap` gives linearly separable classes."""
    y = rng.integers(0, 2, n)
    centres = np.array([[-gap / 2, -gap / 2], [gap / 2, gap / 2]])
    return centres[y] + rng.normal(0, 1.0, (n, 2)), y


def make_xor(rng: np.random.Generator, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """The XOR problem: label 1 when the two coordinates have different signs. Not linearly separable.

    Points keep a distance of at least 0.15 from both axes, so the classes are separable once x1*x2 is a feature.
    """
    X = rng.uniform(0.15, 1.0, (n, 2)) * rng.choice([-1.0, 1.0], (n, 2))
    return X, (X[:, 0] * X[:, 1] < 0).astype(int)


def compare(name: str, ours, theirs) -> None:
    """Print the largest absolute difference between our result and scikit-learn's."""
    diff = float(np.max(np.abs(np.asarray(ours, dtype=float) - np.asarray(theirs, dtype=float))))
    print(f"{name:<32} max |ours - sklearn| = {diff:.2e}")


def plot_history(histories: dict[str, np.ndarray]) -> None:
    """Training loss versus step, one curve per entry, on a log scale."""
    _, ax = plt.subplots(figsize=(7, 3.5))
    for (label, h), color in zip(histories.items(), COLORS):
        ax.plot(h, color=color, label=label)
    ax.set(xlabel="step", ylabel="training loss (log)", yscale="log")
    ax.legend()
    plt.show()


def plot_degree_curve(degrees, train_err, test_err, ylabel: str = "MSE") -> None:
    """Training and test error versus polynomial degree (log scale: the test error explodes)."""
    _, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(degrees, train_err, "o-", color="tab:blue", label="train")
    ax.plot(degrees, test_err, "o-", color="tab:red", label="test")
    ax.set(xlabel="polynomial degree", ylabel=f"{ylabel} (log)", yscale="log", xticks=list(degrees))
    ax.legend()
    plt.show()


def plot_path(lams, coefs: np.ndarray, title: str = "") -> None:
    """Regularization path: every coefficient versus lambda. `coefs` has shape (len(lams), n_features)."""
    _, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(lams, coefs, lw=1)
    ax.set(xscale="log", xlabel=r"$\lambda$ (log)", ylabel="coefficient", title=title)
    plt.show()


def plot_validation_curve(lams, train_err, val_err, ylabel: str = "MSE") -> None:
    """Training and validation error versus lambda, with the best lambda marked."""
    _, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(lams, train_err, "o-", color="tab:blue", label="train")
    ax.plot(lams, val_err, "o-", color="tab:red", label="validation")
    best = int(np.argmin(val_err))
    ax.axvline(lams[best], color="tab:gray", ls="--", label=rf"best $\lambda$ = {lams[best]:.3g}")
    ax.set(xscale="log", xlabel=r"$\lambda$ (log)", ylabel=ylabel)
    ax.legend()
    plt.show()


def plot_boundary(predict: Callable, X: np.ndarray, y: np.ndarray, title: str = "", ax=None) -> None:
    """Data points and the decision regions of `predict` (which maps an (n, 2) array to labels 0/1)."""
    own = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4))
    lo, hi = X.min(0) - 0.3, X.max(0) + 0.3
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 200), np.linspace(lo[1], hi[1], 200))
    zz = np.asarray(predict(np.c_[gx.ravel(), gy.ravel()])).reshape(gx.shape)
    ax.contourf(gx, gy, zz, levels=[-0.5, 0.5, 1.5], colors=["#cfe3f5", "#fbdcc4"], alpha=0.8)
    ax.scatter(*X[y == 0].T, s=12, color="tab:blue")
    ax.scatter(*X[y == 1].T, s=12, color="tab:red")
    ax.set(title=title, xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]))
    if own:
        plt.show()


def plot_confusion(tn: int, fp: int, fn: int, tp: int, ax=None, title: str = "") -> None:
    """A 2x2 confusion matrix (rows: actual, columns: predicted)."""
    own = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow([[tn, fp], [fn, tp]], cmap="Blues")
    for (i, j), value in np.ndenumerate([[tn, fp], [fn, tp]]):
        ax.text(j, i, f"{value}", ha="center", va="center", fontsize=14)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["ham", "spam"], yticklabels=["ham", "spam"], title=title)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    if own:
        plt.show()


def plot_thresholds(y: np.ndarray, scores: np.ndarray, metrics: Callable) -> None:
    """Precision, recall, F1 and MCC as the decision threshold on a probability moves from 0 to 1."""
    ts = np.linspace(0.02, 0.98, 49)
    rows = [metrics(y, (scores >= t).astype(int)) for t in ts]
    _, ax = plt.subplots(figsize=(7, 3.5))
    for key, color in zip(("precision", "recall", "f1", "mcc"), COLORS):
        ax.plot(ts, [r[key] for r in rows], color=color, label=key)
    ax.set(xlabel="threshold", ylim=(0, 1.02))
    ax.legend(ncol=4, loc="lower center")
    plt.show()


def kfold_indices(n: int, k: int, rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray]]:
    """Shuffle `n` indices and cut them into `k` folds. Returns a list of `(train_idx, validation_idx)` pairs."""
    folds = np.array_split(rng.permutation(n), k)
    return [(np.concatenate([folds[j] for j in range(k) if j != i]), folds[i]) for i in range(k)]


def plot_predictions(y: np.ndarray, y_hat: np.ndarray, title: str = "") -> None:
    """Predicted against actual values (the diagonal is a perfect model) and the histogram of the residuals."""
    _, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].scatter(y, y_hat, s=3, alpha=0.3)
    lo, hi = float(min(y.min(), y_hat.min())), float(max(y.max(), y_hat.max()))
    axes[0].plot([lo, hi], [lo, hi], "k--")
    axes[0].set(xlabel="actual", ylabel="predicted", title=title)
    sns.histplot(y - y_hat, bins=40, ax=axes[1])
    axes[1].set(xlabel="residual (actual - predicted)", title="residuals")
    plt.tight_layout()
    plt.show()


def plot_weights(names: list[str], weights: dict[str, np.ndarray]) -> None:
    """Horizontal bars of several weight vectors side by side (ours against scikit-learn's, for instance)."""
    k = len(weights)
    _, ax = plt.subplots(figsize=(7, 0.45 * len(names) * k / 2 + 1.5))
    pos = np.arange(len(names))
    for i, ((label, w), color) in enumerate(zip(weights.items(), COLORS)):
        ax.barh(pos + (i - (k - 1) / 2) * 0.8 / k, np.asarray(w, dtype=float), height=0.8 / k, color=color, label=label)
    ax.set(yticks=pos, yticklabels=names, xlabel="weight")
    ax.axvline(0, color="black", lw=0.8)
    ax.invert_yaxis()
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_cv_scores(scores: dict[str, np.ndarray], ylabel: str = "MCC") -> None:
    """One box per model with every fold's score as a dot: the spread decides whether a ranking is real."""
    _, ax = plt.subplots(figsize=(7, 3.5))
    labels = list(scores)
    values = [np.asarray(scores[k], dtype=float) for k in labels]
    sns.boxplot(data=values, ax=ax, color="lightgray", fliersize=0)
    sns.stripplot(data=values, ax=ax, color="tab:red", size=6)
    ax.set(xticks=range(len(labels)), xticklabels=labels, ylabel=ylabel)
    plt.tight_layout()
    plt.show()
