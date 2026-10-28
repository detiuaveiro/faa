"""Loaders of the real data sets of `datasets/` (polars) and the stratified split used by the demos."""

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split


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


def split(X: np.ndarray, y: np.ndarray, seed: int = 0):
    """The 70/30 stratified split of every demo: X_train, X_test, y_train, y_test."""
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)


def real_2d(name: str, n: int = 1000, seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Two real features and a class, standardised, for a decision-boundary gallery (`n` rows drawn at random).

    * `breast`: `worst_concave_points` and `worst_texture` of the nuclei, `y = 1` malignant: two informative, almost
      separable features (the boundary is close to a line).
    * `california_map`: `Longitude` and `Latitude` of a district, `y = 1` when its median house value is above the
      median: the classes follow the coast, a boundary no line can draw.
    """
    if name == "breast":
        df = pl.read_csv(data_dir() / "breast_cancer.csv.zst")
        cols, y = ["worst_concave_points", "worst_texture"], (1 - df["target"].to_numpy()).astype(int)
    elif name == "california_map":
        df = pl.read_csv(data_dir() / "california_housing.csv.zst")
        cols, y = ["Longitude", "Latitude"], (df["MedHouseVal"].to_numpy() > df["MedHouseVal"].median()).astype(int)
    else:
        raise ValueError(f"unknown real 2-D set {name!r}")
    X = df.select(cols).to_numpy().astype(float)
    if n < len(y):
        idx = np.random.default_rng(seed).choice(len(y), n, replace=False)
        X, y = X[idx], y[idx]
    return (X - X.mean(axis=0)) / X.std(axis=0), y, cols


def real_1d_regression(seed: int = 0, n: int = 300, q: float = 0.75):
    """California housing: the median house value (100 000 USD) against `MedInc` (10 000 USD), `n` districts drawn at random.

    The training part is the districts with `MedInc` up to its quantile `q`; the test part is all of them, so the plot
    shows what each regressor does beyond the incomes it has seen. Returns X_train (n, 1), y_train, X_test, y_test,
    train_range, test_range."""
    df = pl.read_csv(data_dir() / "california_housing.csv.zst")
    x, y = df["MedInc"].to_numpy().astype(float), df["MedHouseVal"].to_numpy().astype(float)
    idx = np.random.default_rng(seed).choice(len(y), n, replace=False)
    x, y = x[idx], y[idx]
    cut = float(np.quantile(x, q))
    tr = x <= cut
    order = np.argsort(x)
    return (
        x[tr][:, None], y[tr], x[order][:, None], y[order],
        (float(x.min()), cut), (float(x.min()), float(x.max())),
    )  # fmt: skip
