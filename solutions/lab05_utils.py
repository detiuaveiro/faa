"""Lab 05 helpers: the datasets, the optimizer, the split and scaling of Class 03, and ready-made plots.

Nothing here needs to be edited. Import it from the lab notebook:

    from lab05_utils import load_breast_cancer, load_digits, load_concrete, train, plot_decision, ...

`train` is the Adam optimizer of Class 02 with a decaying learning rate. It differentiates your loss with
`jax.value_and_grad`, so every model of this lab (SVM, kernel SVM, SVR, MLP) is *a loss written with `jax.numpy`* plus a
call to `train`. Parameters are a pytree (a dictionary or a list of dictionaries).
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

jax.config.update("jax_enable_x64", True)


def _data_dir() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "datasets" / "spambase.csv.zst").exists():
            return parent / "datasets"
    raise FileNotFoundError("datasets/spambase.csv.zst not found: run the notebook from inside the repository")


def load_table(name: str, target: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Read `datasets/<name>.csv.zst` with polars: null -> median (numbers) or "missing" (text), text -> one-hot."""
    df = pl.read_csv(_data_dir() / f"{name}.csv.zst")
    X = df.drop(target)
    X = X.with_columns([pl.col(c).fill_null(pl.col(c).median()) for c, t in X.schema.items() if t.is_numeric()])
    cats = [c for c, t in X.schema.items() if t == pl.String]
    if cats:
        X = X.with_columns([pl.col(c).fill_null("missing") for c in cats]).to_dummies(columns=cats)
    return X.to_numpy().astype(float), df[target].to_numpy(), X.columns


def load_breast_cancer() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """569 tumours, 30 measurements of the cell nuclei; `y = 1` benign (357), `0` malignant (212)."""
    X, y, names = load_table("breast_cancer", "target")
    return X, y.astype(int), names


def load_digits() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """1797 images of 8x8 pixels (grey level 0-16, rescaled to 0-1), 10 classes."""
    X, y, names = load_table("digits", "target")
    return X / 16.0, y.astype(int), names


def load_concrete() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """1030 concrete mixtures, 8 numeric features (kg/m3 and age in days), target = compressive strength in MPa."""
    X, y, names = load_table("concrete", "strength")
    return X, y.astype(float), names


def train(loss: Callable, params, *data, lr: float = 0.05, steps: int = 2000) -> tuple[Any, np.ndarray]:
    """Minimize `loss(params, *data)` with Adam (cosine-decayed step). Returns the final params and the loss history.

    The gradient comes from `jax.value_and_grad`: `loss` must be written with `jax.numpy`. `params` is any pytree.
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


def compare(name: str, ours, theirs) -> None:
    """Print the largest absolute difference between our result and the reference's."""
    diff = float(np.max(np.abs(np.asarray(ours, dtype=float) - np.asarray(theirs, dtype=float))))
    print(f"{name:<36} max |ours - reference| = {diff:.2e}")


def stratified_split(y: np.ndarray, test_frac: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """(train_idx, test_idx) with the class proportions kept in both parts (Lab 03, B1)."""
    test = []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        test.append(idx[: round(len(idx) * test_frac)])
    test_idx = np.concatenate(test)
    train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
    return rng.permutation(train_idx), rng.permutation(test_idx)


def fit_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column mean and standard deviation of the training set (a constant column gets 1) (Lab 03, C1)."""
    std = X.std(axis=0)
    return X.mean(axis=0), np.where(std == 0, 1.0, std)


def standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def plot_decision(score: Callable, X: np.ndarray, y: np.ndarray, title: str = "", ax=None, sv: np.ndarray | None = None,
                  margin: bool = False) -> None:
    """Colour map of `score(points) > 0` over a 2D data set; `margin=True` adds the lines score = -1 and +1.

    `score` maps an array (n, 2) to n values (a decision function). `sv` marks points (support vectors) with a ring.
    """
    ax = ax or plt.gca()
    gx, gy = np.meshgrid(np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 220),
                         np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 220))
    z = np.asarray(score(np.c_[gx.ravel(), gy.ravel()])).reshape(gx.shape)
    ax.contourf(gx, gy, z > 0, alpha=0.12, cmap="coolwarm")
    if margin:
        ax.contour(gx, gy, z, levels=[-1, 0, 1], colors="k", linestyles=["--", "-", "--"], linewidths=1)
    else:
        ax.contour(gx, gy, z, levels=[0], colors="k", linewidths=1)
    ax.scatter(*X.T, c=y, cmap="coolwarm", s=18, edgecolor="k", linewidth=0.3)
    if sv is not None and len(sv):
        ax.scatter(*sv.T, s=90, facecolor="none", edgecolor="k", linewidth=1.2)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_history(histories: dict[str, np.ndarray], ylabel: str = "loss", logy: bool = True) -> None:
    for name, h in histories.items():
        plt.plot(h, label=name)
    plt.xlabel("step or epoch")
    plt.ylabel(ylabel)
    if logy:
        plt.yscale("log")
    plt.legend()
    plt.show()


def plot_scores_grid(rows: np.ndarray, cols: np.ndarray, scores: np.ndarray, xlabel: str, ylabel: str, title: str = "") -> None:
    """Heat map of a score over a (rows x cols) grid of hyperparameters (log-spaced values)."""
    fig, ax = plt.subplots(figsize=(6, 4.2))
    im = ax.imshow(scores, origin="lower", cmap="viridis")
    ax.set_xticks(range(len(cols)), [f"{c:g}" for c in cols])
    ax.set_yticks(range(len(rows)), [f"{r:g}" for r in rows])
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, f"{scores[i, j]:.2f}", ha="center", va="center", color="w", fontsize=7)
    plt.colorbar(im)
    plt.show()


def plot_digits(X: np.ndarray, y: np.ndarray, n: int = 20) -> None:
    fig, axes = plt.subplots(2, n // 2, figsize=(n * 0.6, 2.4))
    for ax, img, lab in zip(axes.ravel(), X[:n], y[:n]):
        ax.imshow(img.reshape(8, 8), cmap="gray_r")
        ax.set_title(str(lab), fontsize=8)
        ax.axis("off")
    plt.show()


def plot_pred_vs_actual(y: np.ndarray, preds: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, len(preds), figsize=(4.2 * len(preds), 3.6), squeeze=False)
    for ax, (name, p) in zip(axes[0], preds.items()):
        ax.scatter(y, p, s=8)
        lim = [min(y.min(), p.min()), max(y.max(), p.max())]
        ax.plot(lim, lim, "k:")
        ax.set(xlabel="actual", ylabel="predicted", title=name)
    plt.tight_layout()
    plt.show()
