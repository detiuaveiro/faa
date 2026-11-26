"""Helpers of Class 11: few labels, self-training, label spreading, autoencoders (Keras 3 on the JAX backend)."""

import os
from typing import Any

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras
import numpy as np
import polars as pl
from scipy.sparse import issparse
from scipy.sparse import vstack as svstack
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression

from faa import data

# ----------------------------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------------------------


def digits_flat() -> tuple[np.ndarray, np.ndarray]:
    """The 1797 digits as rows of 64 pixels scaled to [0, 1], and their labels."""
    df = pl.read_csv(data.data_dir() / "digits.csv.zst")
    return df.drop("target").to_numpy().astype("float32") / 16.0, df["target"].to_numpy()


def few_labels(y: np.ndarray, per_class: int, seed: int = 0) -> np.ndarray:
    """Indices of `per_class` labelled rows of every class (the rest of the rows is treated as unlabelled)."""
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.choice(np.flatnonzero(y == c), per_class, replace=False) for c in np.unique(y)])


def flip_labels(y: np.ndarray, frac: float, seed: int = 0) -> np.ndarray:
    """A copy of `y` where a fraction of the rows gets a wrong label (another class, at random)."""
    rng = np.random.default_rng(seed)
    out = y.copy()
    k = np.unique(y)
    for i in rng.choice(len(y), int(frac * len(y)), replace=False):
        out[i] = rng.choice(k[k != y[i]])
    return out


# ----------------------------------------------------------------------------------------------------------------
# Self-training and label spreading
# ----------------------------------------------------------------------------------------------------------------


def self_train(
    Xl, yl, Xu, threshold: float = 0.9, rounds: int = 10, C: float = 10.0, y_true_u: np.ndarray | None = None
):
    """Pseudo-labelling: fit, label the unlabelled rows whose top probability is >= threshold, add them, repeat.
    Returns the final model and a list of dicts per round (labelled size, rows added, accuracy of the added pseudo-labels if `y_true_u` is given)."""
    sparse = issparse(Xl)
    X: Any = Xl.copy()
    y = yl.copy()
    pool = np.arange(Xu.shape[0])
    history = []
    model = LogisticRegression(C=C, max_iter=2000).fit(X, y)
    for r in range(rounds):
        if len(pool) == 0:
            break
        proba = model.predict_proba(Xu[pool])
        sure = proba.max(1) >= threshold
        if not sure.any():
            break
        take = pool[sure]
        pseudo = model.classes_[proba[sure].argmax(1)]
        acc = float((pseudo == y_true_u[take]).mean()) if y_true_u is not None else float("nan")
        X = svstack([X, Xu[take]]) if sparse else np.vstack([X, Xu[take]])
        y = np.concatenate([y, pseudo])
        pool = pool[~sure]
        model = LogisticRegression(C=C, max_iter=2000).fit(X, y)
        history.append({"round": r + 1, "labelled": len(y), "added": int(sure.sum()), "pseudo_acc": acc})
    return model, history


def co_train(
    Xl_a,
    Xl_b,
    yl,
    Xu_a,
    Xu_b,
    threshold: float = 0.9,
    rounds: int = 10,
    C: float = 10.0,
    y_true_u: np.ndarray | None = None,
):
    """Co-training (Blum and Mitchell, 1998) with two views of every row: classifier A sees the columns `a`, classifier B the columns `b`.
    Each round, every classifier labels the unlabelled rows it is sure about (probability >= threshold) and hands them to the OTHER one.
    Returns the two models and a list of dicts per round (rows handed over, accuracy of those pseudo-labels)."""
    XA, XB, yA, yB = Xl_a.copy(), Xl_b.copy(), yl.copy(), yl.copy()
    pool = np.arange(len(Xu_a))
    ma = LogisticRegression(C=C, max_iter=2000).fit(XA, yA)
    mb = LogisticRegression(C=C, max_iter=2000).fit(XB, yB)
    history = []
    for r in range(rounds):
        if len(pool) == 0:
            break
        pa, pb = ma.predict_proba(Xu_a[pool]), mb.predict_proba(Xu_b[pool])
        sa, sb = pa.max(1) >= threshold, pb.max(1) >= threshold
        if not (sa.any() or sb.any()):
            break
        la, lb = ma.classes_[pa.argmax(1)], mb.classes_[pb.argmax(1)]
        ta, tb = pool[sa], pool[sb]
        acc = (
            float(np.mean(np.concatenate([la[sa] == y_true_u[ta], lb[sb] == y_true_u[tb]])))
            if y_true_u is not None
            else float("nan")
        )
        XB, yB = np.vstack([XB, Xu_b[ta]]), np.concatenate([yB, la[sa]])  # A teaches B
        XA, yA = np.vstack([XA, Xu_a[tb]]), np.concatenate([yA, lb[sb]])  # B teaches A
        pool = pool[~(sa | sb)]
        ma = LogisticRegression(C=C, max_iter=2000).fit(XA, yA)
        mb = LogisticRegression(C=C, max_iter=2000).fit(XB, yB)
        history.append({"round": r + 1, "from_a": int(sa.sum()), "from_b": int(sb.sum()), "pseudo_acc": acc})
    return ma, mb, history


def knn_affinity(X: np.ndarray, k: int = 10, sigma: float | None = None) -> np.ndarray:
    """Symmetric Gaussian affinity on the k nearest neighbours (zero diagonal)."""
    D = cdist(X, X)
    sigma = float(np.median(np.sort(D, axis=1)[:, k])) if sigma is None else sigma
    W = np.exp(-(D**2) / (2 * sigma**2))
    np.fill_diagonal(W, 0.0)
    keep = np.zeros_like(W, dtype=bool)
    idx = np.argsort(D, axis=1)[:, 1 : k + 1]
    keep[np.repeat(np.arange(len(X)), k), idx.ravel()] = True
    W = np.where(keep | keep.T, W, 0.0)
    return W


def normalised(W: np.ndarray) -> np.ndarray:
    """S = D^(-1/2) W D^(-1/2)."""
    d = W.sum(1)
    d[d == 0] = 1.0
    return W / np.sqrt(np.outer(d, d))


def label_spreading(
    W: np.ndarray, y: np.ndarray, labelled: np.ndarray, alpha: float = 0.9, n_classes: int | None = None
) -> np.ndarray:
    """Closed form F = (I - alpha S)^-1 Y0 of Zhou et al. (2004); returns the class of every row."""
    n_classes = n_classes or int(y.max()) + 1
    Y0 = np.zeros((len(W), n_classes))
    Y0[labelled, y[labelled]] = 1.0
    S = normalised(W)
    F = np.linalg.solve(np.eye(len(W)) - alpha * S, Y0)
    return F.argmax(1)


def label_spreading_sparse(
    G, y: np.ndarray, labelled: np.ndarray, alpha: float = 0.9, n_classes: int | None = None, iters: int = 60
) -> np.ndarray:
    """The same scores for a large sparse graph `G` (scipy sparse, symmetric): the series F = sum_t (alpha S)^t Y0 by iteration, no matrix inverse."""
    n_classes = n_classes or int(y.max()) + 1
    d = np.asarray(G.sum(1)).ravel()
    d[d == 0] = 1.0
    S = G.multiply(1 / np.sqrt(d)[:, None]).multiply(1 / np.sqrt(d)[None, :]).tocsr()
    Y0 = np.zeros((G.shape[0], n_classes))
    Y0[labelled, y[labelled]] = 1.0
    F = Y0.copy()
    for _ in range(iters):
        F = alpha * (S @ F) + Y0
    return F.argmax(1)


# ----------------------------------------------------------------------------------------------------------------
# Autoencoders
# ----------------------------------------------------------------------------------------------------------------


def autoencoder(
    d_in: int, code: int, hidden: tuple[int, ...] = (), activation: str = "relu", l1: float = 0.0, seed: int = 0
):
    """Dense autoencoder d_in -> hidden -> code -> reversed hidden -> d_in (output sigmoid). Returns (autoencoder, encoder).
    `hidden = ()` and `activation = "linear"` give the linear autoencoder; `l1 > 0` puts an L1 activity penalty on the code (sparse)."""
    keras.utils.set_random_seed(seed)
    inp = keras.layers.Input((d_in,))
    x = inp
    for h in hidden:
        x = keras.layers.Dense(h, activation=activation)(x)
    reg = keras.regularizers.L1(l1) if l1 > 0 else None
    z = keras.layers.Dense(
        code, activation=activation if activation != "linear" else None, activity_regularizer=reg, name="code"
    )(x)
    x = z
    for h in reversed(hidden):
        x = keras.layers.Dense(h, activation=activation)(x)
    out = keras.layers.Dense(d_in, activation=None if activation == "linear" else "sigmoid")(x)
    ae = keras.Model(inp, out)
    ae.compile(optimizer=keras.optimizers.Adam(3e-3), loss="mse")
    return ae, keras.Model(inp, z)


def fit_ae(
    ae, X, noise: float = 0.0, mask: float = 0.0, epochs: int = 60, batch: int = 64, seed: int = 0, validation=None
):
    """Train an autoencoder; with `noise` > 0 the input gets Gaussian noise, with `mask` > 0 that fraction of values is set to 0 (the target stays clean)."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(epochs):
        Xin = X
        if noise > 0:
            Xin = np.clip(Xin + rng.normal(0, noise, X.shape), 0, 1).astype("float32")
        if mask > 0:
            Xin = (Xin * (rng.random(X.shape) >= mask)).astype("float32")
        h = ae.fit(Xin, X, epochs=1, batch_size=batch, verbose=0, validation_data=validation)
        best = h.history
    return best


def recon_error(ae, X) -> np.ndarray:
    """Mean squared reconstruction error of every row."""
    return np.mean((np.asarray(ae.predict(X, verbose=0)) - X) ** 2, axis=1)


def classifier_on(Z_l, y_l, Z_t, y_t, C: float = 1.0) -> float:
    """Accuracy of a logistic regression trained on the codes of the labelled rows."""
    return float(LogisticRegression(C=C, max_iter=3000).fit(Z_l, y_l).score(Z_t, y_t))


def classifier_from(enc, n_out: int = 10, freeze: bool = False, seed: int = 0):
    """A classifier made of a copy of the encoder `enc` (trained or not) and a softmax layer; `freeze` keeps the encoder weights fixed."""
    keras.utils.set_random_seed(seed)
    body = keras.models.clone_model(enc)
    body.set_weights(enc.get_weights())
    body.trainable = not freeze
    model = keras.Sequential([body, keras.layers.Dense(n_out, activation="softmax")])
    model.compile(optimizer=keras.optimizers.Adam(3e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
