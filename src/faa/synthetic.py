"""Generated data sets for the demos: 2-D classification sets and noisy 1-D regression targets (fixed seeds, no download)."""

import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_classification, make_moons


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
