"""Lab 04 helpers: the datasets, splits, the Class 03 tools you already wrote, and ready-made plots.

Nothing here needs to be edited. Import it from the lab notebook:

    from lab04_utils import load_housing, load_spam, train, plot_reliability, ...

`train` is the Adam optimizer of Class 03: it differentiates your loss with `jax.value_and_grad`. Parameters are a
dictionary of arrays (a *pytree*). `stratified_split`, `fit_standardizer`, `standardize`, `classification_metrics` and
`regression_metrics` are the functions of Lab 03, given here ready-made.
"""

from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

jax.config.update("jax_enable_x64", True)

COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray"]


def _data_dir() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "data" / "spambase.parquet").exists():
            return parent / "data"
    raise FileNotFoundError("data/spambase.parquet not found: run the notebook from inside the repository")


def load_housing() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """California housing: 20640 districts, 8 features, target = median house value in units of 100 000 USD.

    `AveRooms`, `AveBedrms`, `Population` and `AveOccup` are replaced by their natural logarithm (they are extremely
    skewed). The target is capped at 5.0 in the original data.
    """
    df = pl.read_parquet(_data_dir() / "california_housing.parquet")
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
    df = pl.read_parquet(_data_dir() / "spambase.parquet")
    return df.drop("spam").to_numpy().astype(float), df["spam"].to_numpy().astype(int), df.drop("spam").columns


def train(loss: Callable, params: dict, *data, lr: float = 0.05, steps: int = 2000) -> tuple[dict, np.ndarray]:
    """Minimize `loss(params, *data)` with Adam (cosine-decayed step). Returns the final params and the loss history."""
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
    """Print the largest absolute difference between our result and scikit-learn's."""
    diff = float(np.max(np.abs(np.asarray(ours, dtype=float) - np.asarray(theirs, dtype=float))))
    print(f"{name:<36} max |ours - sklearn| = {diff:.2e}")


def stratified_split(y: np.ndarray, test_frac: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """(train_idx, test_idx) with the class proportions kept in both parts (Lab 03, A1)."""
    test = []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        test.append(idx[: round(len(idx) * test_frac)])
    test_idx = np.concatenate(test)
    train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
    return rng.permutation(train_idx), rng.permutation(test_idx)


def fit_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column mean and standard deviation of the training set (a constant column gets 1) (Lab 03, A2)."""
    std = X.std(axis=0)
    return X.mean(axis=0), np.where(std == 0, 1.0, std)


def standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def fit_bins(X: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile bin edges per column, shape (n_bins - 1, d): the interior cut points of the training data."""
    return np.quantile(X, np.linspace(0, 1, n_bins + 1)[1:-1], axis=0)


def apply_bins(X: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """The bin index (0 .. n_bins - 1) of every value, using the edges of `fit_bins`."""
    return np.stack([np.searchsorted(edges[:, j], X[:, j], side="right") for j in range(X.shape[1])], axis=1)


def regression_metrics(y: np.ndarray, y_hat: np.ndarray) -> dict[str, float]:
    e = y - y_hat
    return {
        "mse": float(np.mean(e**2)),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "mae": float(np.mean(np.abs(e))),
        "smape": float(100 * np.mean(2 * np.abs(e) / (np.abs(y) + np.abs(y_hat)))),
        "r2": float(1 - np.sum(e**2) / np.sum((y - y.mean()) ** 2)),
    }


def classification_metrics(y: np.ndarray, y_hat: np.ndarray) -> dict[str, float]:
    tp = int(np.sum((y == 1) & (y_hat == 1)))
    tn = int(np.sum((y == 0) & (y_hat == 0)))
    fp = int(np.sum((y == 0) & (y_hat == 1)))
    fn = int(np.sum((y == 1) & (y_hat == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0
    return {"accuracy": (tp + tn) / len(y), "precision": precision, "recall": recall, "f1": f1, "mcc": float(mcc)}


def plot_beta_posteriors(k: int, n: int, priors: dict[str, tuple[float, float]]) -> None:
    """Beta posteriors of a coin after `k` heads in `n` tosses, one curve per prior (a, b)."""
    from scipy import stats

    theta = np.linspace(0.001, 0.999, 400)
    _, ax = plt.subplots(figsize=(7, 3.5))
    for (label, (a, b)), color in zip(priors.items(), COLORS):
        ax.plot(theta, stats.beta.pdf(theta, a + k, b + n - k), color=color, label=f"{label}: Beta({a},{b}) prior")
    ax.axvline(k / n if n else 0.5, color="k", ls=":")
    ax.set(xlabel=r"$\theta$", title=f"posterior after {k} heads in {n} tosses")
    ax.legend(fontsize=8)
    plt.show()


def plot_learning_curves(sizes, curves: dict[str, list[float]]) -> None:
    """Test accuracy versus the number of training examples, one curve per model."""
    _, ax = plt.subplots(figsize=(7, 3.5))
    for (label, values), color in zip(curves.items(), COLORS):
        ax.plot(sizes, values, "o-", color=color, label=label)
    ax.set(xscale="log", xlabel="training examples", ylabel="test accuracy")
    ax.legend()
    plt.show()


def plot_reliability(y: np.ndarray, probas: dict[str, np.ndarray], bins: int = 10) -> None:
    """Reliability diagrams: observed frequency against predicted probability, one panel per model."""
    _, axes = plt.subplots(1, len(probas), figsize=(4 * len(probas), 3.6), sharey=True, squeeze=False)
    edges = np.linspace(0, 1, bins + 1)
    for ax, (title, p) in zip(axes[0], probas.items()):
        which = np.digitize(p, edges[1:-1])
        mid = [p[which == b].mean() for b in range(bins) if (which == b).sum() >= 5]
        obs = [y[which == b].mean() for b in range(bins) if (which == b).sum() >= 5]
        ax.plot([0, 1], [0, 1], "k--")
        ax.plot(mid, obs, "o-")
        ax.set(title=title, xlabel="predicted probability")
    axes[0][0].set_ylabel("observed frequency")
    plt.show()


def plot_cost_curve(thresholds: np.ndarray, costs: np.ndarray, t_star: float) -> None:
    """Expected cost against the decision threshold, with the Bayes-optimal threshold marked."""
    _, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(thresholds, costs)
    ax.axvline(t_star, color="tab:red", ls="--", label=rf"$t^\star$ = {t_star:.3f}")
    ax.axvline(0.5, color="gray", ls=":", label="0.5")
    ax.set(xlabel="threshold", ylabel="expected cost per e-mail")
    ax.legend()
    plt.show()


def plot_pred_vs_actual(y: np.ndarray, preds: dict[str, np.ndarray]) -> None:
    """Predicted against actual values, one panel per model."""
    _, axes = plt.subplots(1, len(preds), figsize=(4 * len(preds), 3.6), sharey=True, squeeze=False)
    for ax, (title, p) in zip(axes[0], preds.items()):
        ax.scatter(y, p, s=3, alpha=0.3)
        ax.plot([0, 5], [0, 5], "k--")
        ax.set(title=title, xlabel="actual value")
    axes[0][0].set_ylabel("predicted value")
    plt.show()
