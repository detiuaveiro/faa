"""Lab 01 helpers: the data sets, the two models and the ready-made plots.

Nothing here needs to be edited. Import it from the lab notebook:

    from lab01_utils import load_breast_cancer, load_heart_disease, make_tree, make_logreg, compare, ...

In every data set the label `y = 1` is the *positive* class, the event to detect: a malignant tumour (the file codes benign as 1:
`load_breast_cancer` recodes it), a heart disease. `make_tree` and `make_logreg` return a fresh, unfitted scikit-learn model; the logistic regression carries
its own `StandardScaler` (a `Pipeline`), so the scaling is always fitted on the training part of whatever it is fitted on.

Given functions (not for you to write): `majority_predict`, `fold_scores`, `select_top_k` and the two tests of Part F,
`corrected_t` and `mcnemar_exact` (with `mcnemar_counts` and `mcnemar_chi2`). Convention of the two tests: model A is the
first, model B the second; `delta_j = s_j(A) - s_j(B)`; `J` is the number of scores (5 x 10-fold CV: J = 50, `k = 10` is the
number of folds); `n10` counts the examples only A gets right, `n01` those only B gets right, `N = n10 + n01`.
"""

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def _data_dir() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "datasets" / "breast_cancer.csv.zst").exists():
            return parent / "datasets"
    raise FileNotFoundError("datasets/breast_cancer.csv.zst not found: run the notebook from inside the repository")


def load_breast_cancer() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """569 tumours, 30 measurements of the cell nuclei; `y = 1` malignant (212), `0` benign (357)."""
    df = pl.read_csv(_data_dir() / "breast_cancer.csv.zst")
    X = df.drop("target")
    y = 1 - df["target"].to_numpy()  # the file codes benign as 1: here the positive class is the malignant one
    return X.to_numpy().astype(float), y.astype(int), X.columns


def load_heart_disease() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """303 patients, 13 attributes (text ones one-hot encoded, 7 missing values filled with the median); `y = 1` disease (138)."""
    df = pl.read_csv(_data_dir() / "heart_disease.csv.zst")
    X = df.drop("num")
    X = X.with_columns([pl.col(c).fill_null(pl.col(c).median()) for c, t in X.schema.items() if t.is_numeric()])
    cats = [c for c, t in X.schema.items() if t == pl.String]
    X = X.to_dummies(columns=cats)
    return X.to_numpy().astype(float), (df["num"] == ">50_1").to_numpy().astype(int), X.columns


def make_tree() -> DecisionTreeClassifier:
    """A decision tree of depth 3 (needs no scaling). The demo of the lecture uses an unpruned tree to show memorization: the numbers differ on purpose."""
    return DecisionTreeClassifier(max_depth=3, random_state=0)


def make_logreg() -> Pipeline:
    """Standardization (fitted on the part it is fitted on) followed by a logistic regression."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def majority_predict(y_train: np.ndarray, n: int) -> np.ndarray:
    """`n` copies of the most frequent class of `y_train`: the baseline that ignores the features."""
    return np.full(n, np.bincount(y_train).argmax())


def fold_scores(y: np.ndarray, pred: np.ndarray, folds: Sequence[np.ndarray]) -> np.ndarray:
    """The k scores of a cross-validation: the accuracy of the out-of-fold predictions `pred` inside each test fold."""
    return np.array([np.mean(y[test] == pred[test]) for test in folds])


def select_top_k(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Indices of the `k` columns of `X` with the largest absolute Pearson correlation with `y` (for two classes the order of `f_classif`)."""
    Xc, yc = X - X.mean(axis=0), y - y.mean()
    r = (Xc * yc[:, None]).sum(axis=0) / np.sqrt((Xc**2).sum(axis=0) * (yc**2).sum())
    return np.argsort(-np.abs(r))[:k]


def t_quantile(k: int, level: float = 0.95) -> float:
    """Two-sided Student-t quantile with `k - 1` degrees of freedom (2.262 for k = 10 and 95%)."""
    return float(stats.t.ppf(0.5 + level / 2, k - 1))


def naive_ttest(diffs: np.ndarray) -> tuple[float, float]:
    """The ordinary one-sample t-test of the mean difference against 0 (SciPy): `(T, p_val)`. Part F shows why it is too optimistic."""
    res: Any = stats.ttest_1samp(diffs, 0.0)
    return float(res.statistic), float(res.pvalue)


def corrected_t(diffs: np.ndarray, n_train: float, n_test: float) -> tuple[float, float]:
    """Corrected resampled t-test (Nadeau and Bengio, 2003) on the J paired differences `delta_j = s_j(A) - s_j(B)`.

    T = mean(delta) / sqrt((1/J + n_test/n_train) * s_delta^2), p_val = 2 P(t_{J-1} > |T|); `(0.0, 1.0)` if all differences are equal.
    """
    J = len(diffs)
    s2 = diffs.var(ddof=1)
    if s2 == 0:
        return 0.0, 1.0
    T = diffs.mean() / np.sqrt((1 / J + n_test / n_train) * s2)
    return float(T), float(2 * stats.t.sf(abs(T), J - 1))


def mcnemar_counts(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> tuple[int, int, int, int]:
    """(n11, n10, n01, n00): both right, only A right, only B right, both wrong."""
    a, b = pred_a == y, pred_b == y
    return int(np.sum(a & b)), int(np.sum(a & ~b)), int(np.sum(~a & b)), int(np.sum(~a & ~b))


def mcnemar_exact(n10: int, n01: int) -> float:
    """Exact two-sided McNemar p-value: 2 P(Binomial(N, 1/2) <= min(n10, n01)), capped at 1. Use it for N = n10 + n01 < 25."""
    N = n10 + n01
    if N == 0:
        return 1.0
    tail = sum(math.comb(N, i) for i in range(min(n10, n01) + 1)) / 2**N
    return min(1.0, 2 * tail)


def mcnemar_chi2(n10: int, n01: int, continuity: bool = False) -> tuple[float, float]:
    """`(chi2, p_val)` of the chi-squared approximation (1 degree of freedom, large N): McNemar (1947) `(n10 - n01)^2 / N`;
    with `continuity=True` Edwards' (1948) correction `(|n10 - n01| - 1)^2 / N`."""
    N = n10 + n01
    if N == 0:
        return 0.0, 1.0
    gap = abs(n10 - n01) - (1 if continuity else 0)
    chi2 = max(gap, 0) ** 2 / N
    return float(chi2), float(stats.chi2.sf(chi2, 1))


def compare(name: str, ours, theirs) -> None:
    """Print the largest absolute difference between our result and the reference's."""
    diff = float(np.max(np.abs(np.asarray(ours, dtype=float) - np.asarray(theirs, dtype=float))))
    print(f"{name:<44} max |ours - reference| = {diff:.2e}")


def as_splits(folds: Sequence[np.ndarray], n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train, test) index pairs from a list of test folds, the format of scikit-learn's `cv=` argument."""
    return [(np.setdiff1d(np.arange(n), test), test) for test in folds]


def thin_positive(X: np.ndarray, y: np.ndarray, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Keep every negative and just enough random positives for them to be `frac` of the result."""
    rng = np.random.default_rng(seed)
    neg = np.flatnonzero(y == 0)
    n_pos = round(frac / (1 - frac) * len(neg))
    pos = rng.permutation(np.flatnonzero(y == 1))[:n_pos]
    keep = np.sort(np.concatenate([neg, pos]))
    return X[keep], y[keep]


def make_noise(n: int, d: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """`d` Gaussian features and a coin-flip label that has nothing to do with them: no model can beat 50%."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)), rng.integers(0, 2, size=n)


def plot_class_hist(X: np.ndarray, y: np.ndarray, names: list[str], features: Sequence[str]) -> None:
    """One histogram per feature, benign and malignant stacked."""
    fig, axes = plt.subplots(1, len(features), figsize=(4.2 * len(features), 3.2), squeeze=False)
    for ax, f in zip(axes[0], features):
        j = names.index(f)
        ax.hist([X[y == 0, j], X[y == 1, j]], bins=25, stacked=True, label=["benign", "malignant"])
        ax.set(xlabel=f, ylabel="tumours")
    axes[0][0].legend()
    plt.tight_layout()
    plt.show()


def plot_spread(values: dict[str, np.ndarray], xlabel: str = "test accuracy") -> None:
    """Histogram of the same score measured on many splits, one colour per model."""
    for name, v in values.items():
        plt.hist(v, bins=12, alpha=0.6, label=f"{name} (min {v.min():.3f}, max {v.max():.3f})")
    plt.xlabel(xlabel)
    plt.ylabel("splits")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_folds(scores: np.ndarray, lo: float, hi: float, hold_out: float | None = None) -> None:
    """Fold scores as points, the mean as a line and the confidence interval as a band."""
    plt.figure(figsize=(7, 2.8))
    plt.scatter(np.arange(1, len(scores) + 1), scores, zorder=3, label="fold score")
    plt.axhline(float(scores.mean()), color="k", lw=1, label="mean")
    plt.axhspan(lo, hi, alpha=0.2, label="95% interval of the mean")
    if hold_out is not None:
        plt.axhline(hold_out, color="C3", ls=":", label="one hold-out (Part B)")
    plt.xlabel("fold")
    plt.ylabel("accuracy")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()


def plot_leak(n_keep: Sequence[int], leaky: Sequence[float], clean: Sequence[float]) -> None:
    """Accuracy on pure noise against the number of selected features, selection outside and inside the folds."""
    plt.figure(figsize=(6, 3))
    plt.plot(n_keep, leaky, marker="o", label="selection before the split (leak)")
    plt.plot(n_keep, clean, marker="o", label="selection inside each fold")
    plt.axhline(0.5, color="k", ls=":", label="chance")
    plt.xlabel("selected features")
    plt.ylabel("cross-validated accuracy")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()
