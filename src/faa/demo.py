"""Setup and printing of the demo notebooks: one call replaces the header cell, and each cell prints with `slide` and `show`."""

import warnings

import polars as pl

from faa import plotlib


def setup(module: str) -> None:
    """Silence warnings, set the plain table style of the course and the plot style of `module` (its folder of PDFs)."""
    warnings.filterwarnings("ignore")
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_cols(24)
    pl.Config.set_float_precision(3)
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
    pl.Config.set_tbl_width_chars(170)
    pl.Config.set_fmt_str_lengths(40)
    plotlib.setup(module=module)


def slide(title: str) -> None:
    """Every cell that prints a number of a slide starts with this line: the frame title of the deck."""
    print(f"--- slide: {title} ---")


def show(frame: pl.DataFrame, digits: int = 3) -> None:
    """Print a polars table with rounded floats."""
    print(frame.with_columns(pl.col(pl.Float64).round(digits)))
