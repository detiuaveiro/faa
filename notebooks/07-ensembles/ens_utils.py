"""Class 07 helpers for the demo notebooks: data, model galleries, and the boundary panel.

The notebooks import this file (`import ens_utils as eu`). It holds what several demos share:

* synthetic 2-D classification sets with fixed seeds (`class_sets`), the 70/30 stratified split (`split`);
* the ten classifiers of the gallery (`gallery_models`) and the nine regressors of the regression twin (`reg_models`);
* `boundary_panel`: one axis with the probability map of a fitted model, the 0.5 contour, the training points (filled),
  the test points (open) and the test accuracy and log loss under the panel;
* `curve_panel`: the same for a regression curve; polars loaders of the datasets of `datasets/`.

Convention (Class 01): in a 2-class problem the positive class is 1 (orange in the plots), class 0 is blue.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_blobs, make_circles, make_classification, make_moons
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

sys.path.append(str(Path(__file__).resolve().parents[1]))
import plotlib  # noqa: E402

BLUE, ORANGE, GREEN, RED, SKY, PURPLE, GREY = (
    plotlib.BLUE,
    plotlib.ORANGE,
    plotlib.GREEN,
    plotlib.RED,
    plotlib.SKY,
    plotlib.PURPLE,
    plotlib.GREY,
)
CLASS_COLOURS = (BLUE, ORANGE)
GRID_N = 300  # 300 x 300 = 90 000 predictions per panel
LL_CLIP = 1e-3  # probabilities are clipped to [1e-3, 1 - 1e-3] before the log loss: 1-NN has hard 0/1 outputs
PROB_CMAP = LinearSegmentedColormap.from_list("blue_white_orange", [BLUE, "#FFFFFF", ORANGE])
LINE = "#333333"


def data_dir() -> Path:
    """The `datasets/` folder of the repository, found from this file."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "datasets" / "spambase.csv.zst").exists():
            return parent / "datasets"
    raise FileNotFoundError("datasets/spambase.csv.zst not found: run the notebook from inside the repository")


def load_table(name: str, target: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Read `datasets/<name>.csv.zst` with polars: null -> median (numbers) or "missing" (text), text -> one-hot."""
    df = pl.read_csv(data_dir() / f"{name}.csv.zst")
    X = df.drop(target)
    X = X.with_columns([pl.col(c).fill_null(pl.col(c).median()) for c, t in X.schema.items() if t.is_numeric()])
    cats = [c for c, t in X.schema.items() if t == pl.String]
    if cats:
        X = X.with_columns([pl.col(c).fill_null("missing") for c in cats]).to_dummies(columns=cats)
    return X.to_numpy().astype(float), df[target].to_numpy(), X.columns


def load_breast() -> tuple[np.ndarray, np.ndarray]:
    """Breast cancer, 569 x 30; y = 1 means malignant (the file codes benign as 1: recoded, as in Class 01)."""
    X, t, _ = load_table("breast_cancer", "target")
    return X, (1 - t).astype(int)


def load_heart() -> tuple[np.ndarray, np.ndarray]:
    """Heart disease (Cleveland), 303 patients, categorical features one-hot; y = 1 means disease (`>50_1`)."""
    X, num, _ = load_table("heart_disease", "num")
    return X, (num == ">50_1").astype(int)


def load_spam() -> tuple[np.ndarray, np.ndarray]:
    """Spambase, 4601 e-mails x 57 features; y = 1 means spam."""
    X, y, _ = load_table("spambase", "spam")
    return X, y.astype(int)


def load_california(n: int | None = None, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """California housing with the logarithm of the four skewed features (Class 03); `n` rows drawn at random (all if None)."""
    X, y, names = load_table("california_housing", "MedHouseVal")
    for j, c in enumerate(names):
        if c in ("AveRooms", "AveBedrms", "Population", "AveOccup"):
            X[:, j] = np.log(X[:, j])
    if n is not None and n < len(y):
        idx = np.random.default_rng(seed).choice(len(y), n, replace=False)
        X, y = X[idx], y[idx]
    return X, y.astype(float)


def class_sets(seed: int = 0, n: int = 300) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Five 2-D classification sets of n points, generated (no download): moons, overlapping blobs, circles, a set with
    two clusters per class, and XOR. The seed changes the draw, not the recipe."""
    rng = np.random.default_rng(seed)
    xor = rng.uniform(-1.5, 1.5, size=(n, 2))
    y_xor = ((xor[:, 0] * xor[:, 1]) > 0).astype(int)
    xor = xor + rng.normal(0, 0.25, size=xor.shape)  # noise moves points across the axes
    return {
        "moons": make_moons(n, noise=0.3, random_state=seed),
        "blobs": make_blobs(n, centers=[(-1.2, -1.0), (1.2, 1.0)], cluster_std=1.4, random_state=seed),
        "circles": make_circles(n, noise=0.15, factor=0.5, random_state=seed),
        "clusters": make_classification(
            n, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=2, class_sep=1.1, flip_y=0.03,
            random_state=seed,
        ),
        "xor": (xor, y_xor),
    }  # fmt: skip


def split(X: np.ndarray, y: np.ndarray, seed: int = 0):
    """The 70/30 stratified split of every demo: X_train, X_test, y_train, y_test."""
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)


def gallery_models(seed: int = 0) -> dict:
    """The ten classifiers of the gallery, in the order of the slide: five single families of Classes 03-05, k-NN twice,
    a tree, then two ensembles (forest, boosting) and the soft vote of the six single models."""
    scaled = lambda m: make_pipeline(StandardScaler(), m)  # noqa: E731
    singles = {
        "logistic": scaled(LogisticRegression()),
        "naive Bayes": GaussianNB(),
        "SVM RBF": scaled(CalibratedClassifierCV(SVC(C=1.0), method="sigmoid", cv=5, ensemble=False)),
        "MLP": scaled(MLPClassifier(hidden_layer_sizes=(32, 32), alpha=1e-3, max_iter=3000, random_state=seed)),
        "1-NN": scaled(KNeighborsClassifier(n_neighbors=1)),
        "15-NN": scaled(KNeighborsClassifier(n_neighbors=15)),
        "tree": DecisionTreeClassifier(random_state=seed),
    }
    vote_members = [(k, singles[k]) for k in ("logistic", "naive Bayes", "SVM RBF", "MLP", "15-NN", "tree")]
    return {
        **singles,
        "forest": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1),
        "boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=seed),
        "vote": VotingClassifier(vote_members, voting="soft"),
    }


def scores(model, X_te: np.ndarray, y_te: np.ndarray) -> tuple[float, float]:
    """Test accuracy and log loss (probabilities clipped at LL_CLIP) of a fitted classifier."""
    p = np.clip(model.predict_proba(X_te)[:, 1], LL_CLIP, 1 - LL_CLIP)
    return float(np.mean((p > 0.5) == y_te)), float(log_loss(y_te, p, labels=[0, 1]))


def plane_limits(X: np.ndarray, pad: float = 0.35) -> tuple[tuple[float, float], tuple[float, float]]:
    """x and y limits of a point cloud with a margin."""
    lo, hi = X.min(axis=0) - pad, X.max(axis=0) + pad
    return (float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1]))


def prob_map(model, lim, n: int = GRID_N) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """P(class 1) of a fitted classifier on an n x n grid of the limits `lim`: returns gx, gy, p (all n x n)."""
    gx, gy = np.meshgrid(np.linspace(*lim[0], n), np.linspace(*lim[1], n))
    p = model.predict_proba(np.c_[gx.ravel(), gy.ravel()])[:, 1].reshape(gx.shape)
    return gx, gy, p


def boundary_panel(
    ax,
    model,
    X_tr,
    y_tr,
    X_te,
    y_te,
    title: str = "",
    lim=None,
    fontsize: float = 6.0,
    metrics: bool = True,
    points: bool = True,
    sizes: np.ndarray | None = None,
) -> tuple[float, float]:
    """Draw a fitted 2-D classifier on `ax`: the probability of class 1 as a blue-white-orange map, the 0.5 contour as
    the decision boundary, training points filled, test points open. Under the panel: test accuracy and log loss.
    `sizes` (one number per training point) scales the training markers, e.g. the weight of each point in boosting.
    No ticks (the axes are the two features, the same in every panel). Returns (accuracy, log loss)."""
    lim = lim or plane_limits(np.vstack([X_tr, X_te]))
    gx, gy, p = prob_map(model, lim)
    ax.imshow(p, extent=(*lim[0], *lim[1]), origin="lower", cmap=PROB_CMAP, vmin=0, vmax=1, alpha=0.6, aspect="auto",
              interpolation="bilinear", zorder=0)  # fmt: skip
    ax.contour(gx, gy, p, levels=[0.5], colors=LINE, linewidths=0.7, zorder=2)
    if points:
        for c in (0, 1):
            s_c = 6.5 if sizes is None else sizes[y_tr == c]
            ax.scatter(*X_tr[y_tr == c].T, s=s_c, color=CLASS_COLOURS[c], edgecolor="white", linewidth=0.3, zorder=3)
            ax.scatter(
                *X_te[y_te == c].T, s=7.5, facecolor="none", edgecolor=CLASS_COLOURS[c], linewidth=0.65, zorder=3
            )
    acc, ll = scores(model, X_te, y_te)
    ax.set(xlim=lim[0], ylim=lim[1], xticks=[], yticks=[])
    ax.grid(False)
    for s in ax.spines.values():
        s.set(visible=True, linewidth=0.4, color=LINE)
    if title:
        ax.set_title(title, fontsize=fontsize + 0.5, pad=2)
    if metrics:
        ax.set_xlabel(f"acc {acc:.2f}  ll {ll:.2f}", fontsize=fontsize, labelpad=1.5)
    return acc, ll


def contour_line(ax, model, lim, colour: str, ls: str = "-", lw: float = 0.9, label: str | None = None) -> None:
    """Only the 0.5 contour of a fitted classifier (to overlay several models on one panel)."""
    gx, gy, p = prob_map(model, lim)
    cs = ax.contour(gx, gy, p, levels=[0.5], colors=colour, linewidths=lw, linestyles=ls, zorder=2)
    if label:
        cs.set_label(label)


def clean_axes(ax, lim=None) -> None:
    """Box with no ticks and no grid (the feature plane of a gallery panel)."""
    ax.set(xticks=[], yticks=[])
    if lim:
        ax.set(xlim=lim[0], ylim=lim[1])
    ax.grid(False)
    for s in ax.spines.values():
        s.set(visible=True, linewidth=0.4, color=LINE)


# ---- regression twin -------------------------------------------------------------------------------------------------
TRAIN_RANGE, TEST_RANGE = (-3.0, 3.0), (-4.5, 4.5)  # the models are queried on a longer range than they were trained on
TARGETS = {  # name: (function, noise sd)
    "sine": (lambda x: np.sin(1.5 * x), 0.3),
    "cubic": (lambda x: (x**3 - 4 * x) / 6, 0.5),
    "step": (lambda x: np.where(x < -1, -1.0, np.where(x < 1.5, 0.0, 1.5)), 0.3),
}


def reg_data(name: str, seed: int = 0, n_train: int = 80, n_test: int = 300):
    """Noisy target `name` (sine, cubic, step): n_train points in the training range, n_test in the longer test range.
    Returns X_train (n, 1), y_train, X_test, y_test (noisy), f (the noiseless function)."""
    f, sd = TARGETS[name]
    rng = np.random.default_rng(seed)
    x_tr = rng.uniform(*TRAIN_RANGE, n_train)
    x_te = np.sort(rng.uniform(*TEST_RANGE, n_test))
    return x_tr[:, None], f(x_tr) + rng.normal(0, sd, n_train), x_te[:, None], f(x_te) + rng.normal(0, sd, n_test), f


def reg_models(seed: int = 0) -> dict:
    """The nine regressors of the twin gallery; the last one is the mean of the five before the forest."""
    scaled = lambda m: make_pipeline(StandardScaler(), m)  # noqa: E731
    singles = {
        "linear": LinearRegression(),
        "polynomial 5": make_pipeline(StandardScaler(), PolynomialFeatures(degree=5), Ridge(alpha=1e-3)),
        "SVR RBF": scaled(SVR(C=10.0, epsilon=0.1)),
        "MLP": scaled(MLPRegressor(hidden_layer_sizes=(32, 32), activation="tanh", solver="lbfgs", alpha=1e-2, max_iter=3000, random_state=seed)),
        "5-NN": scaled(KNeighborsRegressor(n_neighbors=5)),
        "tree": DecisionTreeRegressor(random_state=seed),
    }  # fmt: skip
    members = [(k, singles[k]) for k in ("polynomial 5", "SVR RBF", "MLP", "5-NN", "tree")]
    return {
        **singles,
        "forest": RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=1),
        "boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=seed),
        "vote": VotingRegressor(members),
    }


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def range_rmse(model, X_te, f) -> tuple[float, float]:
    """RMSE against the noiseless function inside the training range and outside it (the extrapolation error)."""
    x = X_te[:, 0]
    inside = (x >= TRAIN_RANGE[0]) & (x <= TRAIN_RANGE[1])
    pred = model.predict(X_te)
    return rmse(pred[inside], f(x[inside])), rmse(pred[~inside], f(x[~inside]))


def curve_panel(
    ax, model, X_tr, y_tr, X_te, y_te, f, title: str = "", fontsize: float = 6.0, ylim=None, colour=BLUE
) -> tuple[float, float]:
    """Draw a fitted 1-D regressor: the noiseless target (grey), training points (filled), test points (open), the
    prediction (line); the training range is the white band. Under the panel: RMSE against the target inside / outside it."""
    xs = np.linspace(*TEST_RANGE, 400)
    ax.axvspan(*TRAIN_RANGE, color="#EEF1F4", lw=0, zorder=0)
    ax.scatter(X_te[:, 0], y_te, s=3, facecolor="none", edgecolor=GREY, linewidth=0.35, zorder=1)
    ax.scatter(X_tr[:, 0], y_tr, s=4, color=ORANGE, edgecolor="white", linewidth=0.2, zorder=3)
    ax.plot(xs, f(xs), color=LINE, lw=0.6, ls=":", zorder=2)
    ax.plot(xs, model.predict(xs[:, None]), color=colour, lw=1.0, zorder=4)
    r_in, r_out = range_rmse(model, X_te, f)
    ax.set(xlim=TEST_RANGE, xticks=[], yticks=[])
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(False)
    for s in ax.spines.values():
        s.set(visible=True, linewidth=0.4, color=LINE)
    if title:
        ax.set_title(title, fontsize=fontsize + 0.5, pad=2)
    ax.set_xlabel(f"in {r_in:.2f}  out {r_out:.2f}", fontsize=fontsize, labelpad=1.5)
    return r_in, r_out


def mean_pairwise_corr(E: np.ndarray) -> float:
    """Mean correlation between the rows of E (one row per member, one column per example), diagonal excluded."""
    C = np.corrcoef(E)
    C = C[~np.eye(len(C), dtype=bool)]
    return float(np.nanmean(C))


def new_fig(rows: int, cols: int, width: float = plotlib.FULL, panel_h: float = 0.95, **kw):
    """A gallery figure at the real size on the slide: `cols` panels over `width` inches."""
    fig, axs = plt.subplots(rows, cols, figsize=(width, panel_h * rows), **kw)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.93, bottom=0.07)
    return fig, axs
