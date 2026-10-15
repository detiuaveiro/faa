"""Lab 06 helpers: the datasets, the split and scaling of Class 03, and ready-made plots.

Nothing here needs to be edited. Import it from the lab notebook:

    from lab06_utils import load_breast_cancer, load_german_credit, load_concrete, print_tree, plot_boundary, ...

Trees are plain dictionaries: an internal node has `feature`, `threshold`, `left` and `right` (the examples with
`x[feature] <= threshold` go left); every node has `counts` (class counts, classification) or `value` (mean target,
regression), `pred` and `n`. `print_tree` writes a tree as rules.
"""

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


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


def load_german_credit() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """1000 credit applications, 20 attributes (text ones one-hot encoded: 61 columns); `y = 1` bad credit (30%)."""
    X, y, names = load_table("german_credit", "class")
    return X, (y == "bad").astype(int), names


def load_concrete() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """1030 concrete mixtures, 8 numeric features (kg/m3 and age in days), target = compressive strength in MPa."""
    X, y, names = load_table("concrete", "strength")
    return X, y.astype(float), names


def load_digits() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """1797 images of 8x8 pixels (grey level 0-16, rescaled to 0-1), 10 classes."""
    X, y, names = load_table("digits", "target")
    return X / 16.0, y.astype(int), names


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


def depth_of(node: dict) -> int:
    return 0 if "feature" not in node else 1 + max(depth_of(node["left"]), depth_of(node["right"]))


def leaves_of(node: dict) -> int:
    return 1 if "feature" not in node else leaves_of(node["left"]) + leaves_of(node["right"])


def print_tree(node: dict, names: list[str], class_names: list[str] | None = None, indent: str = "", max_depth: int = 4) -> None:
    """Print a tree as indented rules (deeper levels are cut at `max_depth`)."""
    if "feature" not in node or max_depth < 0:
        if "counts" in node:
            label = class_names[node["pred"]] if class_names else node["pred"]
            print(f"{indent}-> {label}  (n = {node['n']}, counts = {node['counts'].tolist()})")
        else:
            print(f"{indent}-> {node['pred']:.2f}  (n = {node['n']})")
        return
    print(f"{indent}if {names[node['feature']]} <= {node['threshold']:.3g}:")
    print_tree(node["left"], names, class_names, indent + "    ", max_depth - 1)
    print(f"{indent}else:")
    print_tree(node["right"], names, class_names, indent + "    ", max_depth - 1)


def plot_boundary(predict: Callable, X: np.ndarray, y: np.ndarray, title: str = "", ax=None) -> None:
    """Colour map of `predict(points)` over a 2D data set, with the points."""
    ax = ax or plt.gca()
    gx, gy = np.meshgrid(np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 220),
                         np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 220))
    z = np.asarray(predict(np.c_[gx.ravel(), gy.ravel()])).reshape(gx.shape)
    ax.contourf(gx, gy, z, alpha=0.15, cmap="coolwarm")
    ax.scatter(*X.T, c=y, cmap="coolwarm", s=14, edgecolor="k", linewidth=0.3)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_curves(x, curves: dict[str, np.ndarray], xlabel: str, ylabel: str = "accuracy", logx: bool = False) -> None:
    for name, c in curves.items():
        plt.plot(x, c, marker="o", ms=3, label=name)
    if logx:
        plt.xscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
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
