"""Lab 02 helpers: the three problems, a challenge dataset, and ready-made plots.

Nothing here needs to be edited. Import it from the lab notebook:

    from lab02_utils import PROBLEMS, plot_paths, ...

Every loss takes parameters `theta` with the two coordinates in the last axis, so the same function evaluates one
point, shape (2,), or a whole population, shape (n, 2). Losses are written with `jax.numpy`, so `jax.grad` and
`jax.hessian` work on them directly; `Problem.numpy_loss` wraps them for pyBlindOpt and your GA.
"""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray"]


@dataclass(frozen=True)
class Problem:
    name: str
    loss: Callable  # theta[..., 2] -> loss[...], JAX-differentiable
    bounds: np.ndarray  # shape (2, 2): [[low, high], [low, high]]
    start: np.ndarray  # starting point for the gradient-based methods
    optimum: np.ndarray  # best known parameters
    labels: tuple[str, str]
    data: tuple[np.ndarray, np.ndarray] | None = None  # (x, y) for the fitting problems
    model: Callable | None = None  # model(theta, x) -> y_hat

    def numpy_loss(self, theta) -> np.ndarray:
        """The loss on NumPy input, returning NumPy output (for pyBlindOpt and the GA)."""
        return np.asarray(self.loss(jnp.asarray(theta, dtype=float)))


# P1: fit a line y = w x + b by mean squared error. Convex, but elongated (condition number ~23).
_rng = np.random.default_rng(42)
_x1 = np.sort(_rng.uniform(0, 3, 40))
_y1 = 1.5 * _x1 - 2 + _rng.normal(0, 0.3, _x1.size)


def _line(theta, x):
    return theta[..., 0, None] * x + theta[..., 1, None]


def _p1_loss(theta):
    return jnp.mean((_line(theta, _x1) - _y1) ** 2, axis=-1)


# P2: the Rosenbrock valley. Non-convex, curved, badly conditioned; minimum at (1, 1).
def _rosenbrock(theta):
    x, y = theta[..., 0], theta[..., 1]
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


# P3: fit a sinusoid y = sin(omega x + phi). Multimodal: many local minima along omega.
_x3 = np.sort(_rng.uniform(0, 10, 80))
_y3 = np.sin(1.7 * _x3 + 0.8) + _rng.normal(0, 0.2, _x3.size)


def _sinusoid(theta, x):
    return jnp.sin(theta[..., 0, None] * x + theta[..., 1, None])


def _p3_loss(theta):
    return jnp.mean((_sinusoid(theta, _x3) - _y3) ** 2, axis=-1)


PROBLEMS = {
    "P1": Problem(
        "P1: line fit (convex)",
        _p1_loss,
        np.array([[-5.0, 5.0], [-5.0, 5.0]]),
        np.array([-4.0, 4.0]),
        np.linalg.lstsq(np.stack([_x1, np.ones_like(_x1)], 1), _y1, rcond=None)[0],
        ("w (slope)", "b (intercept)"),
        (_x1, _y1),
        _line,
    ),
    "P2": Problem(
        "P2: Rosenbrock valley",
        _rosenbrock,
        np.array([[-2.0, 2.0], [-1.0, 3.0]]),
        np.array([-1.5, 2.5]),
        np.array([1.0, 1.0]),
        ("x", "y"),
    ),
    "P3": Problem(
        "P3: sinusoid fit (multimodal)",
        _p3_loss,
        np.array([[0.1, 4.0], [-np.pi, np.pi]]),
        np.array([1.0, 0.0]),
        np.array([1.70517943, 0.81301265]),  # refined with Newton
        ("omega (frequency)", "phi (phase)"),
        (_x3, _y3),
        _sinusoid,
    ),
}


# ---------------------------------------------------------------------------------------------- pyBlindOpt helpers
class Recorder:
    """pyBlindOpt callback that stores a copy of the population at every epoch.

    `Recorder.attach(opt)` stores the initial population and registers the recorder as a callback of `opt`.
    """

    def __init__(self, optimizer=None):
        self.pops = []
        if optimizer is not None:
            self.pops.append(optimizer.pop.copy())

    def __call__(self, _epoch, _scores, pop):
        self.pops.append(pop.copy())

    @classmethod
    def attach(cls, optimizer):
        rec = cls(optimizer)
        optimizer.callbacks = [*optimizer.callbacks, rec]
        return rec


class Counted:
    """Wraps an objective and counts the evaluated points in `.n`."""

    def __init__(self, f):
        self.f, self.n = f, 0

    def __call__(self, theta):
        theta = np.atleast_2d(theta)
        self.n += theta.shape[0]
        return self.f(theta)


# ---------------------------------------------------------------------------------------------- challenge data
def make_classification(n_train=300, n_test=1000, d=6, label_noise=0.1, seed=0):
    """Two classes separated by a hidden hyperplane, with a fraction of flipped labels.

    Returns (X_train, y_train, X_test, y_test) with labels in {0, 1}.
    """
    rng = np.random.default_rng(seed)
    w_true = rng.normal(size=d)
    X = rng.normal(size=(n_train + n_test, d)) * rng.uniform(0.5, 2.0, d)
    y = (X @ w_true + 0.5 > 0).astype(int)
    flip = rng.random(y.size) < label_noise
    y[flip] = 1 - y[flip]
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


# ---------------------------------------------------------------------------------------------- plots
def _grid(problem, n=300):
    (x0, x1), (y0, y1) = problem.bounds
    X, Y = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    Z = problem.numpy_loss(np.stack([X, Y], axis=-1))
    return X, Y, Z


def plot_landscape(problem, ax=None):
    """Filled contours of log(1 + loss - min) over the search box, with the best known optimum."""
    ax = ax or plt.gca()
    X, Y, Z = _grid(problem)
    ax.contourf(X, Y, np.log1p(Z - Z.min()), levels=30, cmap="viridis")
    ax.plot(*problem.optimum, "r*", ms=14, mec="w")
    ax.set(xlim=problem.bounds[0], ylim=problem.bounds[1], xlabel=problem.labels[0], ylabel=problem.labels[1])
    return ax


def plot_paths(problem, paths, title=""):
    """Paths on the landscape (left) and loss per iteration (right). `paths` maps a label to an (n, 2) array."""
    _, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    plot_landscape(problem, axes[0])
    for (label, path), color in zip(paths.items(), COLORS):
        path = np.asarray(path)
        loss = problem.numpy_loss(path)
        axes[0].plot(path[:, 0], path[:, 1], ".-", color=color, ms=3, lw=1, label=label)
        axes[0].plot(*path[0], "o", color=color, mec="k")
        axes[1].semilogy(np.maximum(loss - problem.numpy_loss(problem.optimum), 1e-16), color=color,
                         label=f"{label}: final loss {loss[-1]:.4g}")
    axes[0].set_title(f"{problem.name} {title}")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[1].set(xlabel="iteration", ylabel="loss - best known loss (log)", title="convergence")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_population(problem, pops, epochs=(0, 5, 20, -1), title=""):
    """Snapshots of a population history (list of (n_pop, 2) arrays) at the given epochs."""
    fig, axes = plt.subplots(1, len(epochs), figsize=(4 * len(epochs), 3.8))
    for ax, epoch in zip(axes, epochs):
        pop = np.asarray(pops[epoch])
        plot_landscape(problem, ax)
        ax.scatter(pop[:, 0], pop[:, 1], s=14, color="w", edgecolor="k", zorder=3)
        best = problem.numpy_loss(pop).min()
        ax.set_title(f"epoch {epoch if epoch >= 0 else len(pops) + epoch}: best {best:.3g}", fontsize=10)
    fig.suptitle(f"{problem.name} {title}")
    plt.tight_layout()
    plt.show()


def best_so_far(pops, problem):
    """Best loss seen up to each epoch of a population history."""
    return np.minimum.accumulate([problem.numpy_loss(np.asarray(p)).min() for p in pops])


def plot_convergence(curves, title="", xlabel="epoch"):
    """Best-so-far curves: `curves` maps a label to a 1D array."""
    _, ax = plt.subplots()
    for (label, curve), color in zip(curves.items(), COLORS):
        ax.semilogy(np.maximum(np.asarray(curve), 1e-16), color=color, label=label)
    ax.set(xlabel=xlabel, ylabel="best loss so far (log)", title=title)
    ax.legend(fontsize=8)
    plt.show()


def plot_fit(problem, thetas, title=""):
    """Data and fitted models for P1 and P3. `thetas` maps a label to a parameter vector."""
    if problem.data is None or problem.model is None:
        print(f"{problem.name} is not a fitting problem: nothing to plot.")
        return
    x, y = problem.data
    grid = np.linspace(x.min(), x.max(), 500)
    _, ax = plt.subplots()
    ax.plot(x, y, "k.", label="data")
    for (label, theta), color in zip(thetas.items(), COLORS):
        theta = np.asarray(theta)
        y_hat = np.asarray(problem.model(jnp.asarray(theta, dtype=float), grid))
        ax.plot(grid, y_hat, color=color, lw=2, label=f"{label}: loss {float(problem.numpy_loss(theta)):.4f}")
    ax.set(xlabel="x", ylabel="y", title=f"{problem.name} {title}")
    ax.legend(fontsize=8)
    plt.show()


def plot_boxes(results, title="", ylabel="final loss", log=True):
    """Box plots of repeated runs: `results` maps a label to a list of final values (one per seed)."""
    _, ax = plt.subplots(figsize=(max(6, 1.3 * len(results)), 4))
    ax.boxplot([np.maximum(np.asarray(v), 1e-16) for v in results.values()], tick_labels=list(results))
    if log:
        ax.set_yscale("log")
    ax.set(ylabel=ylabel, title=title)
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.show()
