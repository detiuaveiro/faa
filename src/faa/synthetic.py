"""Generated data sets for the demos: 2-D classification sets and noisy 1-D regression targets (fixed seeds, no download)."""

import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_classification, make_friedman1, make_hastie_10_2, make_moons


def class_sets(seed: int = 0, n: int = 300) -> dict:
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


def spiral(n: int, seed: int = 0, noise: float = 0.05, turns: float = 1.75) -> tuple[np.ndarray, np.ndarray]:
    """Two interleaved spirals in the plane: the boundary winds around the origin, so only a local model follows it."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    t = np.sqrt(rng.uniform(0.02, 1, n)) * turns * 2 * np.pi
    X = 3 * np.c_[t * np.cos(t + np.pi * y), t * np.sin(t + np.pi * y)] / (turns * 2 * np.pi)
    return X + rng.normal(0, noise, (n, 2)), y


def checkerboard(n: int, seed: int = 0, cells: int = 4, noise: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """`cells` x `cells` squares of alternating class: many axis-aligned pieces, no global shape."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, 2))
    y = ((np.floor(X[:, 0] * cells) + np.floor(X[:, 1] * cells)) % 2).astype(int)
    return 3 * X + rng.normal(0, noise, (n, 2)), y


def flip_labels(y: np.ndarray, share: float, seed: int = 0) -> np.ndarray:
    """Labels with a random `share` flipped: the noise no model can remove; it caps the MCC at about 1 - 2 share."""
    flipped = np.random.default_rng(seed + 999).random(len(y)) < share
    return np.where(flipped, 1 - y, y)


N_TRAIN = {"plane": 600, "curve": 120, "table": 1000}  # a curve has one input: few points are already a hard task
N_TEST = 3000
FLIP = 0.10  # share of flipped labels in the noisy planes: the MCC cannot pass 1 - 2 FLIP = 0.80 there
NOISE = 0.10  # sd of the noise of the curves: the MAE cannot go below 0.8 NOISE = 0.08 there


def _xor(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.5, 1.5, (n, 2))
    return X + rng.normal(0, 0.08, X.shape), (X[:, 0] * X[:, 1] > 0).astype(int)


def _flipped(make):
    def noisy(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        X, y = make(n, seed)
        return X, flip_labels(y, FLIP, seed)

    return noisy


def _skewed(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X, y = make_blobs(n, centers=[(0, 0), (1.5, 1.5), (-1.5, 1.5)], cluster_std=[1.0, 0.45, 0.45], random_state=seed)  # type: ignore
    keep = np.where(y == 0, 1.0, 0.25) > rng.random(n)  # about one positive for four negatives
    X, y = X[keep], (y[keep] > 0).astype(int)  # type: ignore
    return np.resize(X, (n, 2)), np.resize(y, n)


PLANES = {
    "moons": _flipped(lambda n, s: make_moons(n, noise=0.15, random_state=s)),
    "circles": _flipped(lambda n, s: make_circles(n, noise=0.08, factor=0.6, random_state=s)),
    "XOR": _flipped(_xor),
    "skewed blobs": _skewed,
    "spiral": _flipped(lambda n, s: spiral(n, s)),
    "checkerboard": lambda n, s: checkerboard(n, s),
}
CLASS_TABLES = {
    "20 features": lambda n, s: make_classification(
        n, n_features=20, n_informative=10, n_redundant=5, flip_y=0.05, class_sep=0.8, n_clusters_per_class=3, random_state=s
    ),
    "Hastie": lambda n, s: (lambda d: (d[0], (d[1] > 0).astype(int)))(make_hastie_10_2(n, random_state=s)),
    "kinked border": lambda n, s: (lambda d: (d[0], (d[1] > np.median(d[1])).astype(int)))(_kinks(n, s)),
}  # fmt: skip
CURVES = {
    "cubic": lambda x: (x**3 - 4 * x) / 4,
    "sine": lambda x: np.sin(2 * x),
    "exponential": lambda x: np.exp(x / 1.2) / 3,
    "chirp": lambda x: np.sin(x * x),
    "step": lambda x: np.select([x < -1.5, x < 0, x < 1.5], [-1.0, 0.5, -0.5], 1.0),
    "sawtooth": lambda x: (x % 1.5) / 0.75 - 1,
}


def _kinks(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_friedman1(n, n_features=8, noise=0.3, random_state=seed)
    return X, y + 6 * np.abs(X[:, 5] - 0.5) * (X[:, 6] > 0.5) + 3 * np.sin(8 * X[:, 7])


REG_TABLES = {"Friedman 1": lambda n, s: make_friedman1(n, noise=0.3, random_state=s), "Friedman kinks": _kinks}


def task_of(name: str) -> str:
    """The task of a data set: "c" classification, "r" regression."""
    return "c" if name in PLANES or name in CLASS_TABLES else "r"


def kind_of(name: str) -> str:
    """"plane" (two inputs), "curve" (one input) or "table" (many inputs)."""
    return "plane" if name in PLANES else "curve" if name in CURVES else "table"


def load(name: str, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """X_train, y_train, X_test, y_test of a data set: N_TRAIN[kind] points to learn from and N_TEST to score on."""
    n_train = N_TRAIN[kind_of(name)]
    n = n_train + N_TEST
    if name in CURVES:
        rng = np.random.default_rng(seed)
        x = rng.uniform(-3, 3, n)
        X, y = x[:, None], CURVES[name](x) + rng.normal(0, NOISE, n)
    else:
        X, y = {**PLANES, **CLASS_TABLES, **REG_TABLES}[name](n, seed)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]
