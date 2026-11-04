"""Plot style and save helper (matplotlib for geometry and custom charts, seaborn for distributions, categories and heatmaps) for the slide figures of all FAA classes.
Lives in src/faa/plotlib.py (package `faa`, installed by `make venv`): a demo runs `from faa import plotlib` and calls `plotlib.setup(module="06")` once (the class number).
Colours are the Okabe-Ito palette of slides/header.tex (cblue ... cgray).

Every figure is a PDF with a transparent background in `assets/figures/plots/<MODULE>/`; the slides include it as
`![](plots/<MODULE>/<name>.pdf)`. Sizes are in inches at the real size on the slide (text width 4.25 in, a 48% column
2.04 in), so the text of a figure is as large on the slide as it is here (6.5 to 7.5 pt). Text is placed by
matplotlib inside the figure bounds (`bbox_inches="tight"`), it cannot spill over the slide.

The PDF and the notebook are two separate outputs of one drawing. In a notebook `setup` selects the interactive backend
(ipympl: zoom, pan, cursor readout) and `save` leaves the live figure open: it fills the width of the cell output and is
redrawn at that size, never a picture of the slide-size PDF.
"""

import functools
from pathlib import Path

import anywidget
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import traitlets
from IPython import get_ipython
from IPython.display import display

ROOT = Path(__file__).resolve().parents[2]  # repo root: this file lives in src/faa/
MODULE = "00"  # class number, set by setup(module=...)
BLUE, ORANGE, GREEN, RED, SKY, PURPLE, GREY = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#56B4E9",
    "#CC79A7",
    "#8C8C8C",
)
LIGHT = "#C8C8C8"
FULL, COL = 4.25, 2.04  # \textwidth and a 48% column of a faa frame (measured: 307.3 pt and 147.5 pt), inches


class _OutputWidth(anywidget.AnyWidget):
    """Invisible widget that reports the width in pixels of the notebook output it sits in, and every later change."""

    _esm = """
    export default { render({ model, el }) {
      el.style.height = "0";
      const observer = new ResizeObserver(([entry]) => {
        const width = Math.round(entry.contentRect.width);
        if (width > 0 && Math.abs(width - model.get("width")) > 8) { model.set("width", width); model.save_changes(); }
      });
      observer.observe(el);
      return () => observer.disconnect();
    } };
    """
    width = traitlets.Int(0).tag(sync=True)


def setup(module: str | None = None) -> None:
    """Common matplotlib style: Noto Sans, small text, no top/right spines, transparent PDF. `module` is the class number (\"06\"): plots go to assets/figures/plots/<module>/.
    In a notebook it also selects the interactive backend."""
    global MODULE
    if module is not None:
        MODULE = module
    if (ip := get_ipython()) is not None and mpl.get_backend() != "widget":
        ip.run_line_magic("matplotlib", "widget")
    mpl.rcParams.update({
        "font.family": ["Noto Sans", "DejaVu Sans"], "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.3, "legend.frameon": False,
        "axes.grid": True, "grid.color": "#D5D8DC", "grid.linewidth": 0.4, "axes.axisbelow": True,
        "figure.dpi": 130, "figure.max_open_warning": 0, "pdf.fonttype": 42, "mathtext.fontset": "dejavusans",
    })  # fmt: skip
    sns.set_palette(
        [BLUE, ORANGE, GREEN, RED, SKY, PURPLE, GREY]
    )  # seaborn functions (histplot, kdeplot, heatmap, barplot) take the deck colours


def styled(func):
    """Decorator for a plot function called from a demo notebook: the deck style applies inside the call only, so the
    notebook's own plots keep their look."""

    @functools.wraps(func)
    def wrapper(*args, **kw):
        with mpl.rc_context():
            setup()
            return func(*args, **kw)

    return wrapper


def fig(kind: str = "full", height: float = 1.6, **kw):
    """Figure at the size it has on the slide: kind "full" (text width) or "col" (a column of a two-column frame).
    Use in a demo notebook: `f, ax = plotlib.fig("col", 2.1)`; extra keywords go to plt.subplots (ncols, gridspec_kw...)."""
    return plt.subplots(figsize=(FULL if kind == "full" else COL, height), **kw)


def _fit_inside(fig, positions: list) -> None:
    """Shrink the axes of a live figure from `positions` until every label is inside it. The PDF needs no such step: its
    `bbox_inches="tight"` grows the page instead, which a live canvas cannot do."""
    for ax, position in zip(fig.axes, positions):
        ax.set_position(position)
    width, height = fig.get_size_inches()
    for _ in range(4):  # text keeps its size while the axes shrink, so one step is not enough
        box = fig.get_tightbbox(fig.canvas.get_renderer())
        left, right = min(0.0, box.x0 - 0.04), max(width, box.x1 + 0.04)
        bottom, top = min(0.0, box.y0 - 0.04), max(height, box.y1 + 0.04)
        if (left, right, bottom, top) == (0.0, width, 0.0, height):
            break
        for ax in fig.axes:
            p = ax.get_position()
            ax.set_position([(p.x0 * width - left) / (right - left), (p.y0 * height - bottom) / (top - bottom),
                             p.width * width / (right - left), p.height * height / (top - bottom)])  # fmt: skip


def _fill_output(fig) -> None:
    """Keep a live figure as wide as the cell output: same proportions, text in points, so the panels gain pixels."""
    canvas, aspect = fig.canvas, fig.get_figheight() / fig.get_figwidth()
    canvas.header_visible, canvas.toolbar_position = False, "top"
    positions = [ax.get_position() for ax in fig.axes]

    def resize(change) -> None:
        canvas.handle_resize({"width": change["new"] - 12, "height": round((change["new"] - 12) * aspect)})
        _fit_inside(fig, positions)
        canvas.draw_idle()

    _fit_inside(fig, positions)
    width = _OutputWidth()
    width.observe(resize, "width")
    display(width)


def save(fig, name: str) -> None:
    """Write assets/figures/plots/<MODULE>/<name>.pdf (transparent, the size of the slide), silently.

    In a notebook the figure stays open as an interactive plot that fills the width of the output; the PDF is never shown."""
    out = ROOT / "assets" / "figures" / "plots" / MODULE
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", transparent=True, bbox_inches="tight", pad_inches=0.03)
    if get_ipython() is None:
        plt.close(fig)
    else:
        _fill_output(fig)


def save_png(fig, name: str, dpi: int = 200) -> Path:
    """Write assets/figures/quiz/<name>.png (white background, 200 dpi) for a quiz question and close the figure.

    Moodle shows the image through an <img> link to raw.githubusercontent.com, so the file must be a PNG that is committed."""
    out = ROOT / "assets" / "figures" / "quiz"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out / f"{name}.png"


def score_board(values: dict, rows, columns, colours: dict, higher: bool = True, left: float = 0.125):
    """Grid of scores: `values[(row, column)]`; a missing cell stays grey, so one layout can fill up from figure to
    figure. Bold marks the best score of a row; the colour of a cell is its distance to that best score."""
    f, ax = plt.subplots(figsize=(FULL, 0.135 * len(rows) + 0.62))
    f.subplots_adjust(left=left, right=0.998, top=1 - 0.6 / (0.135 * len(rows) + 0.62), bottom=0.01)
    cmap = plt.get_cmap("RdYlGn")
    for i, r in enumerate(rows):
        row = [values[(r, c)] for c in columns if (r, c) in values]
        best = max(row) if higher else min(row)
        for j, c in enumerate(columns):
            if (r, c) not in values:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color="#F2F3F5", lw=0))  # type: ignore
                continue
            v = values[(r, c)]
            shade = max(0.0, min(1.0, v)) if higher else max(0.0, 1 - (v / best - 1) / 1.5)
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=cmap(shade), alpha=0.75, lw=0))  # type: ignore
            ax.text(j + 0.5, i + 0.5, f"{v:.2f}".replace("0.", ".", 1) if abs(v) < 1 else f"{v:.1f}", ha="center",
                    va="center", fontsize=5.3, fontweight="bold" if v == best else "normal")  # fmt: skip
    ax.set(xlim=(0, len(columns)), ylim=(len(rows), 0), xticks=[j + 0.5 for j in range(len(columns))],
           yticks=[i + 0.5 for i in range(len(rows))])  # fmt: skip
    ax.set_xticklabels(columns, rotation=90, ha="center", va="bottom", fontsize=5.4)
    ax.set_yticklabels(rows, fontsize=5.8)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=1)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for t, c in zip(ax.get_xticklabels(), columns):
        t.set_color(colours[c])
    return f
