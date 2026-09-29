# ![FAA Logo](assets/logo.svg) Fundamentos de Aprendizagem Automática (FAA)

Curricular Unit Code: [**41010**](https://www.ua.pt/pt/uc/15262) | Universidade de Aveiro

Professor: [**Mário Antunes**](https://www.ua.pt/en/p/80336171) ([`mario.antunes@ua.pt`](mailto:mario.antunes@ua.pt))

Academic Year: [**2026/2027 – 1.º Semestre**](https://www.ua.pt/file/89688)

---

## 1. Overview

**FAA** is a *fundamentals* course in machine learning. Before any modern architecture, it builds a solid understanding of three questions:

* **Why is learning possible?** A model must *compress* noisy data into a reliable, computable pattern.
* **How is a model fitted?** Finding that pattern is an *optimization* problem.
* **How do we know it worked?** Every result must be *evaluated* with statistical care, knowing the pitfalls.

The classic model families (supervised, unsupervised, semi- and self-supervised, reinforcement learning) are then studied on these foundations.

**Class format:** each week has one 3-hour class: **1 hour of lecture** followed by **2 hours of lab work**. Class 01 is the exception: a ~2-hour lecture with live Jupyter demos.

---

## 2. Topics

A topic may span more than one class.

| # | Topic | Classes | Main contents |
|:-:|:--|:-:|:--|
| 1 | Learning as Compression and Optimization | 01, 02 | Entropy, Kolmogorov complexity, MDL; evaluation and pitfalls; blind (population-based) and gradient-based optimization, JAX |
| 2 | Learning Taxonomies | 01 | Supervised, unsupervised, semi-supervised, self-supervised, and reinforcement learning |
| 3 | Supervised: Linear and Probabilistic Models | 03, 04 | Perceptron, linear and logistic regression, polynomial features, regularization, splits and metrics; MLE/MAP, Naive Bayes for classification and regression (discrete and continuous features), calibration, cost-based decisions |
| 4 | Supervised: SVM and Neural Networks | 05 | Maximum margin, kernels, SVR; perceptron, MLP, backpropagation |
| 5 | Supervised: Decision Trees and KNN | 06 | Impurity, pruning, regression trees; distances, scaling, choosing $k$ |
| 6 | Supervised: Ensemble Models | 07 | Bagging, random forests, boosting, stacking |
| 7 | Unsupervised: Blind Signal Separation | 09 | PCA, ICA, NMF |
| 8 | Unsupervised: Soft/Hard Clustering | 10 | $k$-means, hierarchical, DBSCAN; Gaussian mixtures (EM), fuzzy c-means |
| 9 | Semi-Supervised and Self-Supervised Learning | 11, 12 | Pseudo-labelling, AutoEncoders, Transformers, JEPA |
| 10 | Reinforcement Learning | 13 | MDPs, exploration vs. exploitation, Q-learning, policy-based methods |

---

## 3. Schedule

Classes run from **14 September** to **22 December 2026**. **TP1** meets on Thursdays and **TP2** on Fridays.

| Class | TP1 | TP2 | Topic | Milestone |
|:-:|:--|:--|:--|:--|
| 01 | 17 Sep | 18 Sep | T1 + T2: Learning as Compression, Pitfalls, and Taxonomies | Course presentation |
| 02 | 24 Sep | 25 Sep | T1: Blind vs. Gradient-Based Optimization | |
| 03 | 01 Oct | 02 Oct | T3: Linear Models | **Project 1 released** |
| 04 | 08 Oct | 09 Oct | T3: Probabilistic Models | |
| 05 | 15 Oct | 16 Oct | T4: SVM and Neural Networks | |
| 06 | 22 Oct | 23 Oct | T5: Decision Trees and KNN | Project 1 checkpoint |
| 07 | 29 Oct | 30 Oct | T6: Ensemble Models | |
| 08 | 05 Nov | 06 Nov | **Test 1** (Topics 1–6) + Project 1 synthesis | **Project 1 due** |
| 09 | 12 Nov | 13 Nov | T7: Blind Signal Separation | **Project 2 released** |
| 10 | 19 Nov | 20 Nov | T8: Soft/Hard Clustering | |
| 11 | 26 Nov | 27 Nov | T9: AutoEncoders and Semi-Supervised Learning | |
| 12 | 03 Dec | 04 Dec | T9: Transformers and JEPA | |
| 13 | 10 Dec | 11 Dec | T10: Reinforcement Learning | Project 2 clinic |
| 14 | 17 Dec | 18 Dec | **Test 2** (Topics 7–10) + Project 2 presentations | **Project 2 due** |

---

## 4. Class Materials

### Class 01 — Learning as Compression, Pitfalls, and Taxonomies

**Slides:** [`slides/01_foundations.pdf`](slides/01_foundations.pdf)

| Block | Contents | Demo notebook |
|:--|:--|:--|
| Course introduction | Professor, topics, schedule, evaluation, environment setup | — |
| What is learning? | Task/experience/performance; *The Signal and the Noise*; pattern vs. noise; memorization vs. generalization | [`01_learning_as_compression.ipynb`](notebooks/01-foundations/01_learning_as_compression.ipynb) — **D1** |
| Math toolbox | Summation, mean, MSE, probability, logarithms, argmin, variance | — |
| Learning as compression | Entropy, Kolmogorov complexity, MDL/BIC, likelihood and regularization as code length, the link to optimization | [`01_learning_as_compression.ipynb`](notebooks/01-foundations/01_learning_as_compression.ipynb) — **D2, D3** |
| Evaluation and statistical validation | Why evaluate; train/validation/test; cross-validation; confidence intervals; baselines and metrics; corrected t-test; McNemar's test | [`02_evaluation_validation.ipynb`](notebooks/01-foundations/02_evaluation_validation.ipynb) — **D4a–D4f** |
| Limitations and pitfalls | Under/overfitting and bias–variance; curse of dimensionality; No Free Lunch; data leakage; correlation vs. causation; shortcuts, distribution shift, sampling bias | [`03_pitfalls.ipynb`](notebooks/01-foundations/03_pitfalls.ipynb) — **D5–D10** |
| Learning taxonomies | Supervised, unsupervised, semi-supervised, self-supervised, and reinforcement learning | [`04_taxonomies.ipynb`](notebooks/01-foundations/04_taxonomies.ipynb) — **D11** |


### Class 02 — Blind vs. Gradient-Based Optimization

**Slides:** [`slides/02_optimization.pdf`](slides/02_optimization.pdf) · **Lab guide:** [`practice/02_optimization.pdf`](practice/02_optimization.pdf)

| Block | Contents | Demo notebook |
|:--|:--|:--|
| What is optimization? | Formal problem, numerical example, local/global minima, optimality conditions, curvature, eigenvalues and conditioning, convexity, common landscapes | — |
| Optimization in ML | Training as empirical risk minimization with regularization, common losses, parameters vs. hyperparameters | — |
| Gradient-based | Gradient descent and the learning rate, local minima, Newton's method, momentum, RMSProp, Adam, SGD | [`01_gradient_based.ipynb`](notebooks/02-optimization/01_gradient_based.ipynb) — **G1–G5** |
| Automatic differentiation | JAX `grad`/`jit`/`vmap`/`hessian`, custom models and losses, differentiating through an ODE solver | [`01_gradient_based.ipynb`](notebooks/02-optimization/01_gradient_based.ipynb) — **G6** |
| Blind optimization | Random search, hill climbing, simulated annealing; initialization (LHS, Sobol, OBL, QOBL, OBLESA); GA, DE, GWO/EGWO with pyBlindOpt | [`02_blind_optimization.ipynb`](notebooks/02-optimization/02_blind_optimization.ipynb) — **B1–B3** |
| Comparing optimizers | Equal budgets, many seeds, noise, No Free Lunch, hybrid (memetic) search | [`02_blind_optimization.ipynb`](notebooks/02-optimization/02_blind_optimization.ipynb) — **B4–B5** |

**Lab:** [`lab02_optimization.ipynb`](notebooks/02-optimization/lab02_optimization.ipynb) (helpers in `lab02_utils.py`); solved and explained in [`solutions/lab02_optimization_solution.ipynb`](solutions/lab02_optimization_solution.ipynb). Three problems (line fit, Rosenbrock, sinusoid fit): gradient descent by hand and with JAX, a genetic algorithm from scratch, pyBlindOpt initializations and optimizers, Newton's method, and a challenge that trains a classifier on the 0/1 loss.
**Tools:** [pyOptViewer](https://github.com/mariolpantunes/pyOptViewer) animates every pyBlindOpt algorithm; *blindgame* lets students play the black-box optimizer (GECCO 2025 Fun Competition).

### Class 03 — Linear Models

**Slides:** [`slides/03_linear_models.pdf`](slides/03_linear_models.pdf) · **Lab guide:** [`practice/03_linear_models.pdf`](practice/03_linear_models.pdf)

| Block | Contents | Demo notebook |
|:--|:--|:--|
| Splitting and preparing data | Hold-out, stratified, k-fold, grouped, temporal; scaling on the training set only; skewed features | [`01_linear_regression.ipynb`](notebooks/03-linear-models/01_linear_regression.ipynb) — **R1** |
| Linear regression | Least squares, normal equation, gradient descent with `jax.grad`, curvature and the learning rate, the probabilistic view | [`01_linear_regression.ipynb`](notebooks/03-linear-models/01_linear_regression.ipynb) — **R2** |
| Regression metrics | MSE, RMSE, MAE, sMAPE, $R^2$; outliers, asymmetry, baselines | [`01_linear_regression.ipynb`](notebooks/03-linear-models/01_linear_regression.ipynb) — **R3** |
| Polynomial features and regularization | Feature expansion and overfitting; ridge, lasso, elastic net; geometry, MAP view, paths, choosing $\lambda$ | [`01_linear_regression.ipynb`](notebooks/03-linear-models/01_linear_regression.ipynb) — **R4–R5** |
| Perceptron | Update rule as a subgradient, convergence, XOR | [`02_perceptron_logistic.ipynb`](notebooks/03-linear-models/02_perceptron_logistic.ipynb) — **C1** |
| Logistic regression | Sigmoid and log-odds, cross-entropy, regularization, non-linear boundaries, softmax | [`02_perceptron_logistic.ipynb`](notebooks/03-linear-models/02_perceptron_logistic.ipynb) — **C2, C4** |
| Classification metrics | Confusion matrix, accuracy, precision, recall, $F_1$, MCC, thresholds | [`02_perceptron_logistic.ipynb`](notebooks/03-linear-models/02_perceptron_logistic.ipynb) — **C3** |

**Data:** [Spambase](https://archive.ics.uci.edu/dataset/94/spambase) (spam detection) and California housing (house prices), stored as parquet in `data/`.
**Lab:** [`lab03_linear_models.ipynb`](notebooks/03-linear-models/lab03_linear_models.ipynb) (helpers in `lab03_utils.py`): every model is a loss written with `jax.numpy`, trained with `jax.grad` and compared with its scikit-learn twin; solved and explained in [`solutions/lab03_linear_models_solution.ipynb`](solutions/lab03_linear_models_solution.ipynb).

### Class 04 — Probabilistic Models

**Slides:** [`slides/04_probabilistic_models.pdf`](slides/04_probabilistic_models.pdf) · **Lab guide:** [`practice/04_probabilistic_models.pdf`](practice/04_probabilistic_models.pdf)

| Block | Contents | Demo notebook |
|:--|:--|:--|
| Maximum likelihood | Likelihood and NLL, Bernoulli and Gaussian MLE, losses as negative log-likelihoods | [`01_mle_map.ipynb`](notebooks/04-probabilistic-models/01_mle_map.ipynb) — **M1–M3** |
| MAP | Beta prior and Laplace smoothing, Gaussian prior and ridge, the posterior of the weights | [`01_mle_map.ipynb`](notebooks/04-probabilistic-models/01_mle_map.ipynb) — **M4–M5** |
| Naive Bayes, classification | Bayes' rule, independence, Bernoulli/categorical (discrete) and Gaussian (continuous) features, generative vs. discriminative | [`02_naive_bayes_classification.ipynb`](notebooks/04-probabilistic-models/02_naive_bayes_classification.ipynb) — **N1–N3** |
| Naive Bayes, regression | Binned target with expected value; continuous target (linear-Gaussian, closed-form posterior); discrete features; correlated evidence | [`03_naive_bayes_regression.ipynb`](notebooks/04-probabilistic-models/03_naive_bayes_regression.ipynb) — **R1–R4** |
| Probabilities and decisions | Log-loss, Brier score, reliability; Platt and isotonic calibration; cost-based thresholds | [`02_naive_bayes_classification.ipynb`](notebooks/04-probabilistic-models/02_naive_bayes_classification.ipynb) — **N4–N5** |

**Lab:** [`lab04_probabilistic_models.ipynb`](notebooks/04-probabilistic-models/lab04_probabilistic_models.ipynb) (helpers in `lab04_utils.py`): MLE/MAP with `jax.grad`, Naive Bayes by counting, Naive Bayes regression, calibration and costs, each compared with its scikit-learn twin; solved and explained in [`solutions/lab04_probabilistic_models_solution.ipynb`](solutions/lab04_probabilistic_models_solution.ipynb).

---

## 5. Evaluation

Students choose **one** of two modes.

### Discrete Evaluation (recommended)

| Component | Weight | When |
|:--|:-:|:--|
| Theoretical Test 1 | 25% | Class 08 (05/06 Nov 2026), Topics 1–6 |
| Theoretical Test 2 | 25% | Class 14 (17/18 Dec 2026), Topics 7–10 |
| Project 1 | 25% | Released Class 03 (01/02 Oct), due Class 08 (05/06 Nov) |
| Project 2 | 25% | Released Class 09 (12/13 Nov), due Class 14 (17/18 Dec) |

### Final Evaluation

| Component | Weight | When |
|:--|:-:|:--|
| Final Exam | 50% | Regular exam season (*Época Normal*) |
| Comprehensive Project | 50% | Project delivery and defense |

---

## 6. Getting Started

The notebooks run in a local virtual environment named `venv/` at the repository root, **created with `make venv`**. Use the Makefile rather than installing packages by hand, so that everyone has the same environment. The dependencies are declared in `pyproject.toml`; the repository is not a Python package, so the install step only pulls the dependencies.

```bash
git clone https://github.com/detiuaveiro/faa.git && cd faa
export KERAS_BACKEND=jax          # add this line to ~/.bashrc
make venv                         # NumPy against the system OpenBLAS if available, else the wheel
source venv/bin/activate
jupyter lab notebooks/
```

`make venv` works on any system. If `pkg-config` finds OpenBLAS and a C/C++ compiler is installed, it builds NumPy from source against the system OpenBLAS, tuned to the local CPU (`cpu-baseline=native`); otherwise, or if that build fails, it falls back to the prebuilt NumPy wheel (bundled OpenBLAS, slightly slower on some CPUs). It then installs everything else. To enable the source build:

| System | Prerequisites |
|:--|:--|
| Debian/Ubuntu/WSL | `sudo apt install python3-venv build-essential pkg-config libopenblas-dev` |
| macOS | `brew install openblas pkg-config` (Xcode command-line tools for the compiler) |
| Slackware | `openblas` package |

`make venv NUMPY_SOURCE=no` forces the wheel; `NUMPY_SOURCE=yes` forces the source build and fails if it does not work. Plain `python3 -m venv venv && source venv/bin/activate && pip install .` is equivalent to the wheel path.

**If `make venv` fails or the notebooks do not run, notify the professor** ([`mario.antunes@ua.pt`](mailto:mario.antunes@ua.pt)) with your operating system, the output of `python3 --version` and the last lines of the error.

**Software stack**

| Role | Libraries |
|:--|:--|
| Acceleration | NumPy (OpenBLAS; compiled locally by `make venv` when possible), JAX |
| Machine learning | scikit-learn; Keras 3 on the JAX backend (no TensorFlow or PyTorch) |
| Parallelization | joblib |
| Data handling | polars |
| Plotting | matplotlib, seaborn |
| Group libraries | pyBlindOpt, EmptySpaceSearch, kneeliverse, pyUTSAlgorithms, pyNNMF |
| Presentation | JupyterLab |

Linters, type checkers and hook tools (`ruff`, `basedpyright`, `vulture`, `pre-commit`, `nbstripout`) are expected to be installed system-wide (e.g. `uv tool install nbstripout`) and are not listed in `pyproject.toml`.

---

## 7. Repository Structure

| Path | Contents |
|:--|:--|
| `slides/` | Lectures in Pandoc Markdown with inline TikZ figures, compiled to Beamer PDFs (`moloch` theme) |
| `notebooks/` | Jupyter notebooks, one folder per class |
| `data/` | Datasets used by the notebooks (parquet) |
| `practice/` | Lab guides in Pandoc Markdown, compiled to A4 PDFs |
| `solutions/` | Solved lab notebooks with step-by-step explanations (try the lab first) |
| `projects/` | Project specifications |
| `assets/` | Logo, banner, and figures |
| `Makefile`, `Makefile.inc` | Build system: `make all` compiles every PDF (intermediate files cached in `/dev/shm`); `make venv` creates the Python environment |
| `.pre-commit-config.yaml` | Quality gate: formatting checks, notebook outputs stripped (`nbstripout`), `ruff` on notebooks, and a full build |
| `pyproject.toml` | Python dependencies (`pip install .`) and ruff/basedpyright configuration |

---

## 8. Bibliography

* **David J. C. MacKay**, *Information Theory, Inference, and Learning Algorithms*, Cambridge University Press, 2003. [Online edition](https://www.inference.org.uk/itila/book.html)
* **Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, and Jonathan Taylor**, *An Introduction to Statistical Learning with Applications in Python*, Springer, 2023. [statlearning.com](https://www.statlearning.com/)
* **Christopher M. Bishop**, *Pattern Recognition and Machine Learning*, Springer, 2006. [Online edition](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/)
* **David Barber**, *Bayesian Reasoning and Machine Learning*, Cambridge University Press, 2012. [Online edition](http://web4.cs.ucl.ac.uk/staff/D.Barber/textbook/091117.pdf)
* **Ian Goodfellow, Yoshua Bengio, and Aaron Courville**, *Deep Learning*, MIT Press, 2016. [deeplearningbook.org](https://www.deeplearningbook.org/)
* **Richard S. Sutton and Andrew G. Barto**, *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018. [Online edition](http://incompleteideas.net/book/the-book-2nd.html)
* **Nate Silver**, *The Signal and the Noise: Why So Many Predictions Fail — but Some Don't*, Penguin, 2012.

---

## 9. License and Author

* **Author:** Mário Antunes (`mario.antunes@ua.pt`)
* **License:** MIT — see [LICENSE](LICENSE).
