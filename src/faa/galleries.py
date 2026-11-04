"""Galleries of models that draw a boundary (classification) or a curve (regression) on the same task."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier, XGBRegressor

from faa import boundaries, plotlib
from faa.metrics import task_score
from faa.synthetic import CLASS_TABLES, CURVES, PLANES, REG_TABLES, load, task_of

SEEDS = (0, 1, 2, 3, 4)
SHALLOW_C = ("logistic", "naive Bayes", "tree", "1-NN", "SVM RBF", "MLP")
VOTING_C = ("hard vote", "soft vote", "weighted vote", "15-NN", "15-NN by distance")
BAGGING_C = ("bagged trees", "random forest")
BOOSTING_C = ("AdaBoost", "gradient boosting", "XGBoost")
STACK_C = ("stacking",)
NEURAL = ("NN A", "NN B", "neural stack")

SHALLOW_R = ("linear", "polynomial", "tree", "1-NN", "SVR RBF", "MLP")
VOTING_R = ("median vote", "mean vote", "weighted vote", "5-NN", "5-NN by distance")
BAGGING_R, BOOSTING_R, STACK_R = BAGGING_C, BOOSTING_C, STACK_C

FAMILIES = {
    "c": {"shallow": SHALLOW_C, "voting": VOTING_C, "bagging": BAGGING_C, "boosting": BOOSTING_C, "meta": STACK_C + NEURAL},
    "r": {"shallow": SHALLOW_R, "voting": VOTING_R, "bagging": BAGGING_R, "boosting": BOOSTING_R, "meta": STACK_R + NEURAL},
}  # fmt: skip


CLASS_SETS = [*PLANES, *CLASS_TABLES]
REG_SETS = [*CURVES, *REG_TABLES]
COLOUR = dict(zip(FAMILIES["c"], (plotlib.GREY, plotlib.BLUE, plotlib.GREEN, plotlib.ORANGE, plotlib.PURPLE)))
MEMBERS = {
    "c": "members:\nlogistic, naive\nBayes, tree, 1-NN,\nSVM RBF, MLP",
    "r": "members:\nlinear, polynomial,\ntree, 1-NN,\nSVR RBF, MLP",
}


def gallery_models(seed: int = 0) -> dict:
    """The ten classifiers of the gallery, in the order of the slide: five single families of Classes 03-05, k-NN twice,
    a tree, then two ensembles (forest, boosting) and the soft vote of the six single models."""
    scaled = lambda m: make_pipeline(StandardScaler(), m)  # noqa: E731
    singles = {
        "logistic": scaled(LogisticRegression()),
        "naive Bayes": GaussianNB(),
        "SVM RBF": scaled(CalibratedClassifierCV(SVC(C=1.0), method="sigmoid", cv=5, ensemble=False)),  # pyright: ignore[reportArgumentType]
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


def scaled(model):
    return make_pipeline(StandardScaler(), model)


def shallow_classifiers(seed: int = 0) -> dict:
    """One model per family of Classes 03 to 06. None of them combines anything: k-NN appears only with k = 1."""
    return {
        "logistic": scaled(LogisticRegression()),
        "naive Bayes": GaussianNB(),
        "tree": DecisionTreeClassifier(max_depth=5, random_state=seed),
        "1-NN": scaled(KNeighborsClassifier(n_neighbors=1)),
        "SVM RBF": scaled(CalibratedClassifierCV(SVC(C=1.0), method="sigmoid", cv=3, ensemble=False)),  # pyright: ignore[reportArgumentType]
        "MLP": scaled(MLPClassifier(hidden_layer_sizes=(16,), max_iter=800, random_state=seed)),
    }


def shallow_regressors(seed: int = 0) -> dict:
    """The regression twins of the shallow classifiers."""
    return {
        "linear": scaled(LinearRegression()),
        "polynomial": scaled(make_pipeline(PolynomialFeatures(degree=3), Ridge(alpha=1e-3))),
        "tree": DecisionTreeRegressor(max_depth=5, random_state=seed),
        "1-NN": scaled(KNeighborsRegressor(n_neighbors=1)),
        "SVR RBF": scaled(SVR(C=1.0, epsilon=0.1)),
        "MLP": scaled(
            MLPRegressor(hidden_layer_sizes=(8,), activation="tanh", solver="lbfgs", alpha=1e-2, max_iter=2000, random_state=seed)
        ),
    }  # fmt: skip


class WeightedVote(BaseEstimator):
    """A vote whose weights come from 5-fold cross-validation on the training set: the MCC of each member, floored at 0
    (a member no better than chance has no say), or the inverse of its MAE."""

    def __init__(self, members: dict | None = None, task: str = "c"):
        self.members = members
        self.task = task

    def fit(self, X, y):
        items = list((self.members or {}).items())
        cv = [cross_val_predict(m, X, y, cv=5) for _, m in items]
        if self.task == "c":
            weights = [max(0.0, task_score("c", y, p)) for p in cv]  # type: ignore
            self.model_ = VotingClassifier(items, voting="soft", weights=weights if sum(weights) else None)
        else:
            self.model_ = VotingRegressor(items, weights=[1 / task_score("r", y, p) for p in cv])  # type: ignore
        self.weights_ = dict(zip((k for k, _ in items), self.model_.weights or [1.0] * len(items)))
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)  # type: ignore


class MedianVote(BaseEstimator):
    """The regression analogue of a hard vote: the median ignores one member far from the others."""

    def __init__(self, members: dict | None = None):
        self.members = members

    def fit(self, X, y):
        self.fitted_ = [m.fit(X, y) for m in (self.members or {}).values()]
        return self

    def predict(self, X):
        return np.median([m.predict(X) for m in self.fitted_], axis=0)


def classifiers(seed: int = 0) -> dict:
    """Every scikit-learn and XGBoost classifier of the class, by name. The votes use all six shallow models."""
    shallow = shallow_classifiers(seed)
    members = list(shallow.items())
    tree = DecisionTreeClassifier(random_state=seed)
    return {
        **shallow,
        "hard vote": VotingClassifier(members, voting="hard"),
        "soft vote": VotingClassifier(members, voting="soft"),
        "weighted vote": WeightedVote(shallow, "c"),
        "15-NN": scaled(KNeighborsClassifier(n_neighbors=15)),
        "15-NN by distance": scaled(KNeighborsClassifier(n_neighbors=15, weights="distance")),
        "bagged trees": BaggingClassifier(tree, n_estimators=100, random_state=seed),
        "random forest": RandomForestClassifier(n_estimators=200, random_state=seed),
        "AdaBoost": AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=100, random_state=seed),
        "gradient boosting": GradientBoostingClassifier(random_state=seed),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=seed, n_jobs=1),
        "stacking": StackingClassifier(members, final_estimator=LogisticRegression(), cv=5),
    }


def regressors(seed: int = 0) -> dict:
    """Every scikit-learn and XGBoost regressor of the class, by name. The votes use all six shallow models."""
    shallow = shallow_regressors(seed)
    members = list(shallow.items())
    tree = DecisionTreeRegressor(random_state=seed)
    return {
        **shallow,
        "median vote": MedianVote(shallow),
        "mean vote": VotingRegressor(members),
        "weighted vote": WeightedVote(shallow, "r"),
        "5-NN": scaled(KNeighborsRegressor(n_neighbors=5)),
        "5-NN by distance": scaled(KNeighborsRegressor(n_neighbors=5, weights="distance")),
        "bagged trees": BaggingRegressor(tree, n_estimators=100, random_state=seed),
        "random forest": RandomForestRegressor(n_estimators=200, random_state=seed),
        "AdaBoost": AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=100, random_state=seed),
        "gradient boosting": GradientBoostingRegressor(random_state=seed),
        "XGBoost": XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=seed, n_jobs=1),
        "stacking": StackingRegressor(members, final_estimator=Ridge(alpha=1.0), cv=5),
    }


def build(task: str, name: str, seed: int = 0):
    """An unfitted model by name."""
    return (classifiers if task == "c" else regressors)(seed)[name]


def _fit_score(dataset: str, name: str, seed: int) -> tuple[str, str, int, float]:
    with threadpool_limits(1), warnings.catch_warnings():
        warnings.simplefilter("ignore")  # worker processes do not inherit the notebook's filter
        task = task_of(dataset)
        X_tr, y_tr, X_te, y_te = load(dataset, seed)
        return dataset, name, seed, task_score(task, y_te, build(task, name, seed).fit(X_tr, y_tr).predict(X_te))


def evaluate(datasets, names, seeds=SEEDS) -> pl.DataFrame:
    """Test score of each model on each data set and seed, in parallel: columns dataset, model, seed, score."""
    rows = Parallel(n_jobs=-1)(delayed(_fit_score)(d, m, s) for d in datasets for m in names for s in seeds)
    return pl.DataFrame(rows, schema=["dataset", "model", "seed", "score"], orient="row")


def wide(results: pl.DataFrame, names, datasets) -> pl.DataFrame:
    """Mean score over the seeds: one row per model, one column per data set."""
    table = results.group_by("dataset", "model").agg(pl.col("score").mean()).pivot(on="dataset", index="model", values="score")
    return table.join(pl.DataFrame({"model": list(names)}).with_row_index("i"), on="model").sort("i").select("model", *datasets)


def gain(results: pl.DataFrame, datasets, group, reference, task: str) -> pl.DataFrame:
    """For each data set: the best model of `group` against the best model of `reference`, seed by seed. `gain` is
    positive when `group` is better, whatever the metric; `seeds won` counts the seeds where it is."""
    means = results.group_by("dataset", "model").agg(pl.col("score").mean())
    rows = []
    for d in datasets:
        here = means.filter(pl.col("dataset") == d).sort("score", descending=task == "c")
        a = here.filter(pl.col("model").is_in(list(group)))["model"][0]
        b = here.filter(pl.col("model").is_in(list(reference)))["model"][0]
        seeds = results.filter(pl.col("dataset") == d)
        va, vb = (seeds.filter(pl.col("model") == m).sort("seed")["score"].to_numpy() for m in (a, b))
        diff = (va - vb) if task == "c" else (vb - va)
        rows.append((d, a, b, float(va.mean()), float(vb.mean()), float(diff.mean()), float(diff.std(ddof=1)), int((diff > 0).sum())))
    return pl.DataFrame(rows, schema=["dataset", "best", "reference", "best score", "reference score", "gain", "sd", "seeds won"], orient="row")  # fmt: skip


def names_of(task: str, families) -> list[str]:
    """The scikit-learn and XGBoost models of some families; the networks are trained in the last demo."""
    return [m for f in families for m in FAMILIES[task][f] if m not in NEURAL]


def board(results: pl.DataFrame, task: str, sets):
    """The score board of the class: every model has its column from the first figure on, later families stay grey."""
    cols = [m for ms in FAMILIES[task].values() for m in ms]
    colours = {m: COLOUR[f] for f, ms in FAMILIES[task].items() for m in ms}
    mean = results.group_by("dataset", "model").agg(pl.col("score").mean())
    values = {(d, m): s for d, m, s in mean.iter_rows()}
    return plotlib.score_board(values, sets, cols, colours, higher=task == "c")


def family_figure(dataset: str, rows: dict, notes=None, extra=None, seed: int = 0):
    """Fit every model of `rows` on one seed and draw one row of panels per family; `extra` holds fitted networks."""
    task = task_of(dataset)
    X_tr, y_tr, X_te, y_te = load(dataset, seed)
    extra = extra or {}
    names = [m for ms in rows.values() for m in ms if m not in extra]
    fitted = dict(zip(names, Parallel(n_jobs=-1)(delayed(build(task, m, seed).fit)(X_tr, y_tr) for m in names))) | extra
    scores = {m: task_score(task, y_te, f.predict(X_te)) for m, f in fitted.items()}  # type: ignore
    best = (max if task == "c" else min)(scores, key=scores.get)  # type: ignore
    f = CURVES.get(dataset)
    return boundaries.family_gallery(rows, fitted, scores, X_tr, y_tr, f, notes, "MCC" if task == "c" else "MAE", best)


def gain_figure(table: pl.DataFrame, xlabel: str):
    """Gain of the best model of a family over the best earlier model, one bar per data set, sd over the seeds as whisker."""
    fg, ax = plt.subplots(figsize=(plotlib.FULL, 0.16 * len(table) + 0.42))
    fg.subplots_adjust(left=0.16, right=0.66, top=0.98, bottom=0.2)
    y = np.arange(len(table))[::-1]
    wins = [g > s and n == len(SEEDS) for g, s, n in zip(table["gain"], table["sd"], table["seeds won"])]
    ax.barh(
        y,
        table["gain"],
        xerr=table["sd"],
        color=[plotlib.GREEN if w else plotlib.GREY for w in wins],
        height=0.62,
        error_kw={"lw": 0.6, "capsize": 1.2},
    )
    ax.axvline(0, color=boundaries.LINE, lw=0.6)
    ax.set_yticks(y, table["dataset"].to_list(), fontsize=6.3)
    ax.set_xlabel(xlabel, fontsize=6.6)
    ax.tick_params(axis="x", labelsize=6)
    ax.grid(axis="y", visible=False)
    right = ax.get_xlim()[1]
    for yi, a, b in zip(y, table["best"], table["reference"]):
        ax.text(right * 1.03, yi, f"{a} vs {b}", fontsize=5.3, va="center", clip_on=False)
    return fg
