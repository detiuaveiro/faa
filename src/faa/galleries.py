"""Galleries of models that draw a boundary (classification) or a curve (regression) on the same task."""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


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
