"""Scores of a fitted model: test accuracy and log loss, RMSE inside and outside the training range, error correlation."""

import numpy as np
from sklearn.metrics import log_loss

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
