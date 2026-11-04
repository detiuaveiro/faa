"""Scores of a fitted model and the tests that compare two models: McNemar, permutation, bootstrap, corrected t-test."""

import numpy as np
from scipy.stats import binomtest
from scipy.stats import t as t_dist
from sklearn.metrics import log_loss, matthews_corrcoef, mean_absolute_error

from faa.synthetic import TRAIN_RANGE

LL_CLIP = 1e-3  # probabilities are clipped to [1e-3, 1 - 1e-3] before the log loss: 1-NN has hard 0/1 outputs


def scores(model, X_te: np.ndarray, y_te: np.ndarray) -> tuple[float, float]:
    """Test accuracy and log loss (probabilities clipped at LL_CLIP) of a fitted classifier."""
    p = np.clip(model.predict_proba(X_te)[:, 1], LL_CLIP, 1 - LL_CLIP)
    return float(np.mean((p > 0.5) == y_te)), float(log_loss(y_te, p, labels=[0, 1]))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def range_rmse(model, X_te, f, train_range=TRAIN_RANGE) -> tuple[float, float]:
    """RMSE against the noiseless function inside the training range and outside it (the extrapolation error)."""
    x = X_te[:, 0]
    inside = (x >= train_range[0]) & (x <= train_range[1])
    pred = model.predict(X_te)
    return rmse(pred[inside], f(x[inside])), rmse(pred[~inside], f(x[~inside]))


def mean_pairwise_corr(E: np.ndarray) -> float:
    """Mean correlation between the rows of E (one row per member, one column per example), diagonal excluded."""
    C = np.corrcoef(E)
    C = C[~np.eye(len(C), dtype=bool)]
    return float(np.nanmean(C))


def mcc(y, pred) -> float:
    """Matthews correlation of a binary prediction: the classification metric of the course."""
    return float(matthews_corrcoef(y, np.asarray(pred).astype(int)))


def mae(y, pred) -> float:
    """Mean absolute error: the regression metric of the course."""
    return float(mean_absolute_error(y, pred))


def task_score(task: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MCC (task "c", higher is better) or MAE (task "r", lower is better)."""
    return float(matthews_corrcoef(y_true, y_pred) if task == "c" else mean_absolute_error(y_true, y_pred))


def holm(p: np.ndarray) -> np.ndarray:
    """Holm's step-down adjustment of a family of p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    return np.minimum(adjusted, 1.0)[np.argsort(order)]


def mcnemar(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Exact McNemar test of two classifiers on one test set: rows only A got right, rows only B got right, two-sided p."""
    right_a, right_b = a == y, b == y
    only_a, only_b = int(np.sum(right_a & ~right_b)), int(np.sum(right_b & ~right_a))
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return only_a, only_b, float(p)


def _mcc(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """MCC of binary predictions; `pred` has one row per resample."""
    tp, fp = (pred * y).sum(-1), (pred * (1 - y)).sum(-1)
    fn, tn = ((1 - pred) * y).sum(-1), ((1 - pred) * (1 - y)).sum(-1)
    d = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return np.divide(tp * tn - fp * fn, d, out=np.zeros_like(d, dtype=float), where=d > 0)


def _gain(task: str, y: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """How much A beats B on each resample: MCC of A minus MCC of B, or MAE of B minus MAE of A."""
    return _mcc(y, a) - _mcc(y, b) if task == "c" else np.abs(b - y).mean(-1) - np.abs(a - y).mean(-1)


def permutation_test(task: str, y, a, b, n: int = 9999, seed: int = 0, block: int = 200) -> tuple[float, float]:
    """Paired permutation test of the metric itself: swap the two predictions of random rows. Returns the gain of A and p."""
    y, a, b = np.asarray(y, float), np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    seen = float(_gain(task, y, a, b))
    extreme = 0
    for start in range(0, n, block):
        swap = rng.random((min(block, n - start), len(y))) < 0.5
        extreme += int(np.sum(np.abs(_gain(task, y, np.where(swap, b, a), np.where(swap, a, b))) >= abs(seen) - 1e-12))
    return seen, (extreme + 1) / (n + 1)


def bootstrap_ci(task: str, y, a, b, n: int = 2000, seed: int = 0, block: int = 200) -> tuple[float, float]:
    """95 % percentile interval of the gain of A over B: resample the test rows, the same rows for both models."""
    y, a, b = np.asarray(y, float), np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    gains = []
    for start in range(0, n, block):
        rows = rng.integers(0, len(y), (min(block, n - start), len(y)))
        gains.append(_gain(task, y[rows], a[rows], b[rows]))
    low, high = np.percentile(np.concatenate(gains), [2.5, 97.5])
    return float(low), float(high)


def corrected_ttest(gains: np.ndarray, test_share: float) -> float:
    """Nadeau and Bengio's t-test of CV gains: `test_share` is test rows over training rows; folds share training rows."""
    gains = np.asarray(gains, float)
    k = len(gains)
    se = np.sqrt((1 / k + test_share) * gains.var(ddof=1))
    return float(2 * t_dist.sf(abs(gains.mean()) / se, k - 1)) if se > 0 else 1.0


def bayes_signed_rank(gains, rope: float, n: int = 50000, seed: int = 0) -> tuple[float, float, float]:
    """Bayesian signed-rank test of one gain per table: P(A worse), P(practically equal), P(A better); |gain| < rope is equal."""
    z = np.r_[0.0, np.asarray(gains, float)]  # the Dirichlet prior puts a pseudo-table with gain 0 and weight 1/2 first
    w = np.random.default_rng(seed).dirichlet(np.r_[0.5, np.ones(len(gains))], n)
    pairs = z[:, None] + z[None, :]
    worse = np.einsum("ij,ij->i", w @ (pairs < -2 * rope), w)
    better = np.einsum("ij,ij->i", w @ (pairs > 2 * rope), w)
    equal = 1 - worse - better
    winner = np.argmax(np.c_[worse, equal, better], axis=1)
    return tuple(float(np.mean(winner == i)) for i in range(3))  # type: ignore
