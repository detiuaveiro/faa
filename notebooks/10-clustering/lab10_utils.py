"""Lab 10 helpers: the toy sets, the news tf-idf matrix, the wine matrix and the cluster plots.

Nothing here needs to be edited. Import it from the lab notebook:

    from lab10_utils import ten_points, benchmark, news_matrix, wine_matrix, plot_labels, ...

The generators and the soft-membership scores live in the course package `faa.clustering` (Class 10); this file only
loads, subsamples and draws.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from sklearn.feature_extraction.text import TfidfVectorizer

import faa.data
from faa.clustering import benchmarks, entropy, fcm  # noqa: F401

BLUE, ORANGE, GREEN, RED, SKY, PURPLE, GREY = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#56B4E9",
    "#CC79A7",
    "#8C8C8C",
)
COLOURS = [BLUE, ORANGE, GREEN, RED, SKY, PURPLE]


def ten_points() -> np.ndarray:
    """The ten points of the lecture (two groups of five): the by-hand example of Parts B and D."""
    return np.array([[1.0, 1.0], [1.5, 2.0], [2.0, 1.5], [1.0, 2.5], [2.5, 2.0],
                     [6.0, 5.0], [7.0, 6.0], [6.5, 7.0], [7.5, 5.5], [8.0, 7.0]])  # fmt: skip


def benchmark(name: str, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """One of the toy sets of Demo 10e: `blobs`, `aniso`, `varied`, `moons`, `circles`, `uniform`; returns X (300, 2) and y."""
    return benchmarks(seed)[name]


def news_matrix(n_words: int = 2000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The tf-idf matrix of 1000 BBC news articles: X (1000, n_words) >= 0, the categories (never shown to a model) and the words."""
    texts, cat = faa.data.load_news()
    vec = TfidfVectorizer(max_features=n_words, stop_words="english", min_df=3)
    X = vec.fit_transform(texts).toarray()
    return X, cat, np.array(vec.get_feature_names_out())


def wine_matrix() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """The 6 497 wines, 11 standardised features, and `is_red` (the first 1 599 rows are the red wines; used only to check)."""
    from sklearn.preprocessing import StandardScaler

    X, _, names = faa.data.load_table("wine_quality", "quality")
    red = np.r_[np.ones(1599), np.zeros(len(X) - 1599)].astype(int)
    return StandardScaler().fit_transform(X), red, names


def plot_labels(
    X: np.ndarray, labels: np.ndarray, ax: Axes | None = None, title: str = "", size: np.ndarray | float = 8
) -> Axes:
    """Scatter of a 2-D set coloured by label; label -1 (noise) is grey."""
    if ax is None:
        _, ax = plt.subplots(figsize=(2.6, 2.2))
    ax.scatter(*X.T, c=[GREY if v < 0 else COLOURS[int(v) % 6] for v in labels], s=size, lw=0)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def plot_cost(costs: list[float], ax: Axes | None = None, label: str = "", ylabel: str = "cost") -> Axes:
    """A cost or log-likelihood curve against the iteration."""
    if ax is None:
        _, ax = plt.subplots(figsize=(3.2, 2.0))
    ax.plot(costs, lw=1.2, label=label)
    ax.set_xlabel("iteration", fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(labelsize=6)
    return ax


def plot_kdistance(d: np.ndarray, eps: float | None = None, ax: Axes | None = None) -> Axes:
    """The sorted k-distance curve and, if given, the chosen eps as a dashed line."""
    if ax is None:
        _, ax = plt.subplots(figsize=(3.2, 2.0))
    ax.plot(d, color=BLUE, lw=1.2)
    if eps is not None:
        ax.axhline(eps, color=RED, ls="--", lw=0.8)
    ax.set_xlabel("points sorted by distance", fontsize=7)
    ax.set_ylabel("distance to the m-th neighbour", fontsize=7)
    ax.tick_params(labelsize=6)
    return ax
