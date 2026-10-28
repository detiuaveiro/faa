"""Lab 07 helpers: the data sets, the splits, the three members of Part A, and ready-made plots.

Nothing here needs to be edited. Import it from the lab notebook:

    from lab07_utils import moons_split, sine_split, load_heart, three_members, plot_boundaries, ...

Convention (Class 01): in a two-class problem the positive class is 1 (orange in the plots), class 0 is blue.
"""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification, make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from faa import boundaries
from faa.data import load_heart  # noqa: F401  (re-exported for the notebook)


def moons_split(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The running example of the lecture: 300 moons with noise 0.3, stratified 70/30: X_train, X_test, y_train, y_test."""
    X, y = make_moons(300, noise=0.3, random_state=seed)
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)


def ten_points() -> tuple[np.ndarray, np.ndarray]:
    """The ten moons points of the AdaBoost worked example (`make_moons(10, noise=0.2, random_state=2)`)."""
    return make_moons(10, noise=0.2, random_state=2)


def sine_split(seed: int = 0) -> tuple[np.ndarray, ...]:
    """Noisy sine: 80 training and 40 validation points in [-3, 3], 300 test points in [-4.5, 4.5].

    Returns X_train, y_train, X_val, y_val, X_test, y_test_clean (the test targets are the noiseless sine)."""
    rng = np.random.default_rng(seed)
    x_tr = rng.uniform(-3, 3, 80)
    x_va = rng.uniform(-3, 3, 40)
    x_te = np.linspace(-4.5, 4.5, 300)
    f = lambda x: np.sin(1.5 * x)  # noqa: E731
    return (
        x_tr[:, None], f(x_tr) + rng.normal(0, 0.3, 80),
        x_va[:, None], f(x_va) + rng.normal(0, 0.3, 40),
        x_te[:, None], f(x_te),
    )  # fmt: skip


def noisy_features(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """1000 points with 20 features, 4 of them informative and correlated: the set where `max_features` matters."""
    return make_classification(
        1000, n_features=20, n_informative=4, n_redundant=10, n_clusters_per_class=2, class_sep=0.8, random_state=seed
    )


def three_members(seed: int = 0) -> dict:
    """The three unfitted members of Part A: logistic regression, 15-NN and an RBF SVM (calibrated: it needs `predict_proba`)."""
    return {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression()),
        "15-NN": make_pipeline(StandardScaler(), KNeighborsClassifier(15)),
        "SVM RBF": make_pipeline(
            StandardScaler(), CalibratedClassifierCV(SVC(random_state=seed), method="sigmoid", cv=5, ensemble=False)
        ),
    }


class _Proba:
    """Wraps a function `X -> P(class 1)` as a classifier with `predict_proba`, so the shared boundary panel can draw it."""

    def __init__(self, fn):
        self.fn = fn

    def predict_proba(self, X):
        p = np.asarray(self.fn(X))
        return np.c_[1 - p, p]


def plot_boundaries(models: dict, X_tr, y_tr, X_te, y_te, ncols: int = 4):
    """One panel per fitted model (`faa.boundaries.boundary_panel`): P(class 1) as a blue-white-orange map, the 0.5 contour,
    train points filled, test points open, test accuracy and log loss under the panel. `models` maps a title to a fitted
    classifier with `predict_proba` (or to a function `X -> P(class 1)`). Returns the figure."""
    nrows = -(-len(models) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.3 * nrows), squeeze=False)
    lim = boundaries.plane_limits(np.vstack([X_tr, X_te]))
    for ax, (title, m) in zip(axes.ravel(), models.items(), strict=False):
        boundaries.boundary_panel(
            ax, _Proba(m) if callable(m) else m, X_tr, y_tr, X_te, y_te, title=title, lim=lim, fontsize=8
        )
    for ax in axes.ravel()[len(models) :]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def compare(rows: list[tuple]) -> pl.DataFrame:
    """A polars table `name | ours | library | agreement` from rows `(name, ours, library, agreement)`."""
    return pl.DataFrame(rows, schema=["quantity", "ours", "library", "agreement"], orient="row")
