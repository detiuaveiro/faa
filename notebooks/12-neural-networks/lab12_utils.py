"""Lab 12 helpers: the data of the class (digits as images, the memory task, the climate windows) and two small plots.

Nothing here needs to be edited. Import it from the lab notebook:

    from lab12_utils import digits_split, memory, climate, random_shift, transformer, plot_maps, plot_attention

The data generators live in the course package `faa.neural` (Class 12); the convolution, the LSTM step and the attention are yours.
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from faa import neural

BLUE, ORANGE, RED = "#0072B2", "#E69F00", "#D55E00"


def digits_split(seed: int = 0):
    """The 1 797 digits as (n, 8, 8, 1) images in [0, 1]: train 60 %, validation 20 %, test 20 % (stratified)."""
    X, y = neural.load_digits_images()
    xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    xva, xte, yva, yte = train_test_split(xte, yte, test_size=0.5, random_state=seed, stratify=yte)
    return xtr, ytr, xva, yva, xte, yte


def random_shift(X, max_shift: int, seed: int = 0):
    """Every image moved by its own random (dx, dy) in [-max_shift, max_shift], zeros at the border."""
    return neural.random_shift(X, max_shift, seed)


def memory(lag: int, n: int, seed: int = 0):
    """The memory task: the first step holds one of four symbols, `lag` steps of noise follow; returns x (n, lag + 1, 5) and y (n,)."""
    return neural.memory_task(n, lag, seed=seed)


def climate(horizon: int = 6):
    """Windows of the indoor climate (24 hours of five temperatures). The target is the change of room 1 from its last value,
    `horizon` hours after the window. Returns xtr, ytr, xva, yva, xte, yte and sd (the standard deviation of room 1, to turn a standardised error into degrees)."""
    xtr, ytr, xva, yva, xte, yte, _, sd = neural.climate_windows(24, horizon, 5)
    return xtr, ytr - xtr[:, -1, 0], xva, yva - xva[:, -1, 0], xte, yte - xte[:, -1, 0], sd


def lorenz(lookback: int = 24, horizon: int = 10):
    """Windows of the noisy Lorenz series (first coordinate, standardised): xtr, ytr, xva, yva, xte, yte, chronological 70/15/15.
    The target is the value itself, `horizon` steps after the window; the error is a root mean square error in units of the series."""
    return neural.lorenz_windows(20000, lookback, horizon)


def ridge_rmse(xtr, ytr, xte, yte) -> float:
    """Test RMSE of a ridge regression on the flattened window."""
    model = Ridge(1.0).fit(xtr.reshape(len(xtr), -1), ytr)
    return float(np.sqrt(np.mean((model.predict(xte.reshape(len(xte), -1)) - yte) ** 2)))


def ridge_mae(xtr, ytr, xte, yte, sd) -> float:
    """Test MAE in degrees of a ridge regression on the flattened window."""
    model = Ridge(1.0).fit(xtr.reshape(len(xtr), -1), ytr)
    return float(np.abs(model.predict(xte.reshape(len(xte), -1)) - yte).mean() * sd)


def transformer(input_shape):
    """One encoder block (32 units, 2 heads, sinusoidal positions), the last position, a softmax over 4 classes."""
    return neural.transformer_classifier(input_shape, n_out=4)


def plot_maps(img, maps):
    """The digit and its feature maps side by side."""
    f, ax = plt.subplots(1, len(maps) + 1, figsize=(1.6 * (len(maps) + 1), 1.8))
    ax[0].imshow(img[..., 0], cmap="gray_r")
    ax[0].set_title("digit")
    for k, (a, m) in enumerate(zip(ax[1:], maps)):
        a.imshow(m, cmap="RdBu_r", vmin=-abs(m).max(), vmax=abs(m).max())
        a.set_title(f"filter {k}")
    for a in ax:
        a.axis("off")
    plt.show()


def plot_attention(A, A_masked):
    """Attention weights without and with the causal mask."""
    f, ax = plt.subplots(1, 2, figsize=(5, 2.2))
    for a, m, t in zip(ax, (A, A_masked), ("weights", "weights, causal mask")):
        a.imshow(m, cmap="Blues", vmin=0, vmax=1)
        a.set_title(t)
        a.set_xlabel("key position")
        a.set_ylabel("query position")
    plt.tight_layout()
    plt.show()
