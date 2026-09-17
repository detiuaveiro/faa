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
| 3 | Supervised: Linear and Probabilistic Models | 03, 04 | Linear and logistic regression, regularization; Bayes' rule, MLE/MAP, Naive Bayes |
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

The notebooks run in a local virtual environment named `venv/` at the repository root.

```bash
git clone https://github.com/detiuaveiro/faa.git && cd faa
export KERAS_BACKEND=jax          # add this line to ~/.bashrc
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/
```

**Software stack**

| Role | Libraries |
|:--|:--|
| Acceleration | NumPy, JAX (`--system-site-packages` reuses a system NumPy built against an optimized OpenBLAS) |
| Machine learning | scikit-learn; Keras 3 on the JAX backend (no TensorFlow or PyTorch) |
| Parallelization | joblib |
| Data handling | polars |
| Plotting | matplotlib, seaborn |
| Group libraries | pyBlindOpt, EmptySpaceSearch, kneeliverse, pyUTSAlgorithms, pyNNMF |
| Presentation | Jupyter Notebook |

Linters and type checkers (`ruff`, `basedpyright`, `vulture`) are expected to be installed system-wide and are not listed in `requirements.txt`.

---

## 7. Repository Structure

| Path | Contents |
|:--|:--|
| `slides/` | Lectures in Pandoc Markdown with inline TikZ figures, compiled to Beamer PDFs (`moloch` theme) |
| `notebooks/` | Jupyter notebooks, one folder per class |
| `practice/` | Lab guides in Pandoc Markdown, compiled to A4 PDFs |
| `projects/` | Project specifications |
| `assets/` | Logo, banner, and figures |
| `Makefile`, `Makefile.inc` | Build system: `make all` compiles every PDF (intermediate files cached in `/dev/shm`) |
| `.pre-commit-config.yaml` | Quality gate: formatting checks, `ruff` on notebooks, and a full build |
| `requirements.txt` | Python dependencies |

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
