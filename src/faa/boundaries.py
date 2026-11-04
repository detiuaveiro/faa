"""Decision-boundary panels (classification) and prediction-curve panels (regression) of a fitted model.

A panel is one matplotlib axis with no ticks: draw a gallery with `new_fig`, one panel per model, and save it with
`plotlib.save` (it shows the figure in the notebook and writes the transparent PDF of the slide).
Convention (Class 01): in a 2-class problem the positive class is 1 (orange), class 0 is blue.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from faa import plotlib
from faa.metrics import range_rmse, task_score
from faa.synthetic import TEST_RANGE, TRAIN_RANGE

BLUE, ORANGE, GREY = plotlib.BLUE, plotlib.ORANGE, plotlib.GREY
CLASS_COLOURS = (BLUE, ORANGE)
GRID_N = 300  # 300 x 300 = 90 000 predictions per panel
PROB_CMAP = LinearSegmentedColormap.from_list("blue_white_orange", [BLUE, "#FFFFFF", ORANGE])
LINE = "#333333"


def plane_limits(X: np.ndarray, pad: float = 0.35) -> tuple[tuple[float, float], tuple[float, float]]:
    """x and y limits of a point cloud with a margin."""
    lo, hi = X.min(axis=0) - pad, X.max(axis=0) + pad
    return (float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1]))


def prob_map(model, lim, n: int = GRID_N) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """P(class 1) of a fitted classifier on an n x n grid of the limits `lim`: returns gx, gy, p (all n x n)."""
    gx, gy = np.meshgrid(np.linspace(lim[0][0], lim[0][1], n), np.linspace(lim[1][0], lim[1][1], n))
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
) -> float:
    """Draw a fitted 2-D classifier on `ax`: the probability of class 1 as a blue-white-orange map, the 0.5 contour as
    the decision boundary, training points filled, test points open. Under the panel: test MCC.
    `sizes` (one number per training point) scales the training markers, e.g. the weight of each point in boosting.
    No ticks (the axes are the two features, the same in every panel). Returns the MCC."""
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
    score = task_score("c", y_te, model.predict_proba(X_te)[:, 1] > 0.5)
    ax.set(xlim=lim[0], ylim=lim[1], xticks=[], yticks=[])
    ax.grid(False)
    for s in ax.spines.values():
        s.set(visible=True, linewidth=0.4, color=LINE)
    if title:
        ax.set_title(title, fontsize=fontsize + 0.5, pad=2)
    if metrics:
        ax.set_xlabel(f"MCC {score:.2f}", fontsize=fontsize, labelpad=1.5)
    return score


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


def curve_panel(
    ax,
    model,
    X_tr,
    y_tr,
    X_te,
    y_te,
    f=None,
    title: str = "",
    fontsize: float = 6.0,
    ylim=None,
    colour=BLUE,
    train_range=TRAIN_RANGE,
    test_range=TEST_RANGE,
) -> tuple[float, float]:
    """Draw a fitted 1-D regressor: training points (filled), test points (open), the prediction (line); the training
    range is the shaded band. With the noiseless target `f` (generated data): `f` is the dotted line and the label under
    the panel is the RMSE against it inside / outside the training range (the extrapolation error). Without `f` (real
    data): the label is the test RMSE against the noisy targets."""
    xs = np.linspace(test_range[0], test_range[1], 400)
    ax.axvspan(*train_range, color="#EEF1F4", lw=0, zorder=0)
    ax.scatter(X_te[:, 0], y_te, s=3, facecolor="none", edgecolor=GREY, linewidth=0.35, zorder=1)
    ax.scatter(X_tr[:, 0], y_tr, s=4, color=ORANGE, edgecolor="white", linewidth=0.2, zorder=3)
    if f is not None:
        ax.plot(xs, f(xs), color=LINE, lw=0.6, ls=":", zorder=2)
    ax.plot(xs, model.predict(xs[:, None]), color=colour, lw=1.0, zorder=4)
    ax.set(xlim=test_range, xticks=[], yticks=[])
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(False)
    for s in ax.spines.values():
        s.set(visible=True, linewidth=0.4, color=LINE)
    if title:
        ax.set_title(title, fontsize=fontsize + 0.5, pad=2)
    if f is None:
        r_in = r_out = float(np.sqrt(np.mean((model.predict(X_te) - y_te) ** 2)))
        ax.set_xlabel(f"test RMSE {r_in:.2f}", fontsize=fontsize, labelpad=1.5)
    else:
        r_in, r_out = range_rmse(model, X_te, f, train_range)
        ax.set_xlabel(f"in {r_in:.2f}  out {r_out:.2f}", fontsize=fontsize, labelpad=1.5)
    return r_in, r_out


def new_fig(rows: int, cols: int, width: float = plotlib.FULL, panel_h: float = 0.95, hspace: float = 0.42, **kw):
    """A gallery figure at the real size on the slide: `cols` panels over `width` inches; `hspace` leaves room for the
    label under a panel and the title of the panel below it."""
    fig, axs = plt.subplots(rows, cols, figsize=(width, panel_h * rows), **kw)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.93, bottom=0.07, hspace=hspace)
    return fig, axs


def frontier_panel(ax, model, X_tr, y_tr, lim, title: str = "", label: str = "", fontsize: float = 5.6, frame=None) -> None:
    """A fitted 2-D classifier on `ax`: class-1 probability (or its 0/1 prediction), the frontier and at most 120
    training points. `title` goes above the panel, `label` below (usually the test score), `frame` colours the box."""
    gx, gy = np.meshgrid(np.linspace(*lim[0], 150), np.linspace(*lim[1], 150))  # type: ignore
    grid = np.c_[gx.ravel(), gy.ravel()]
    p = (model.predict_proba(grid)[:, 1] if hasattr(model, "predict_proba") else model.predict(grid)).reshape(gx.shape)
    ax.imshow(p, extent=(*lim[0], *lim[1]), origin="lower", cmap=PROB_CMAP, vmin=0, vmax=1, alpha=0.7, aspect="auto",
              interpolation="nearest", zorder=0)  # fmt: skip
    if p.min() < 0.5 < p.max():
        ax.contour(gx, gy, p, levels=[0.5], colors=LINE, linewidths=0.45, zorder=2)
    ax.scatter(*X_tr[:120].T, s=0.7, c=np.array(CLASS_COLOURS)[y_tr[:120]], linewidths=0, zorder=3)
    _score_box(ax, title, label, fontsize, frame, lim)


def fit_panel(ax, model, X_tr, y_tr, f, ylim, title: str = "", label: str = "", fontsize: float = 5.6, frame=None) -> None:
    """A fitted 1-D regressor on `ax`: training points, the true function `f` (dotted) and the fit (blue line)."""
    xs = np.linspace(X_tr.min(), X_tr.max(), 400)
    ax.scatter(X_tr[:, 0], y_tr, s=0.9, color=ORANGE, linewidths=0, zorder=2)
    ax.plot(xs, f(xs), color=LINE, lw=0.45, ls=":", zorder=3)
    ax.plot(xs, model.predict(xs[:, None]), color=BLUE, lw=0.8, zorder=4)
    _score_box(ax, title, label, fontsize, frame, ((xs[0], xs[-1]), ylim))


def _score_box(ax, title, label, fontsize, frame, lim) -> None:
    clean_axes(ax, lim)
    if frame:
        for s in ax.spines.values():
            s.set(linewidth=1.1, color=frame)
    if title:
        ax.set_title(title, fontsize=fontsize, pad=1.2)
    if label:
        ax.set_xlabel(label, fontsize=fontsize, labelpad=0.8)


def family_gallery(rows: dict, fitted: dict, scores_: dict, X_tr, y_tr, f=None, notes=None, metric="MCC", best=None, columns=6, panel_h=0.62):
    """One row of panels per family (`rows`: {family: names}), frontiers when `f` is None, curves otherwise. `notes`
    ({family: text}) fills the free cell of a row, e.g. the members of a vote; `best` gets a red frame."""
    lim = plane_limits(X_tr, pad=0.25) if f is None else None
    ylim = (y_tr.min() - 0.25 * np.ptp(y_tr), y_tr.max() + 0.25 * np.ptp(y_tr))
    fig, axs = plt.subplots(len(rows), columns, figsize=(plotlib.FULL, panel_h * len(rows) + 0.06), squeeze=False)
    fig.subplots_adjust(left=0.035, right=0.998, top=0.96, bottom=0.05, hspace=0.62, wspace=0.05)
    for r, (family, names) in enumerate(rows.items()):
        for c in range(columns):
            axs[r, c].axis("on" if c < len(names) else "off")
        for c, m in enumerate(names):
            kw = {"title": m, "label": f"{metric} {scores_[m]:.2f}", "frame": plotlib.RED if m == best else None}
            if f is None:
                frontier_panel(axs[r, c], fitted[m], X_tr, y_tr, lim, **kw)
            else:
                fit_panel(axs[r, c], fitted[m], X_tr, y_tr, f, ylim, **kw)
        axs[r, 0].set_ylabel(family, fontsize=6, labelpad=1.5)
        if notes and family in notes and len(names) < columns:
            axs[r, len(names)].text(0.03, 0.5, notes[family], fontsize=5.3, va="center", transform=axs[r, len(names)].transAxes)
    return fig
