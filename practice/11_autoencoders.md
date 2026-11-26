---
title: "Lab 11 — Self-Training, Label Spreading and Autoencoders, Built from Scratch"
---

# Overview

**Class 11 · Topic 9.** Lecture: `slides/11_autoencoders.pdf`. Demo notebooks in `notebooks/11-autoencoders/`: `01_autoencoder.ipynb`, `02_variants.ipynb`, `03_text_compression.ipynb`, `04_few_labels.ipynb`, `05_self_training.ipynb`, `06_label_spreading.ipynb` and `07_pretraining.ipynb`.

**Notebook for this lab:** `notebooks/11-autoencoders/lab11_autoencoders.ipynb`, with helpers in `lab11_utils.py` (same folder, do not edit). Cells marked **TODO** contain `raise NotImplementedError`: replace it with your code. Every other cell is ready, and it draws the plots and prints the comparisons once your functions work.

**Solution:** `solutions/lab11_autoencoders_solution.ipynb` is the fully solved notebook. Before each function it explains how the solution was reached, and after each experiment it answers this guide's questions with the numbers the run produces. Try every task yourself first.

**The rule of this lab.** Each idea of the class is one small function: the pseudo-label loop, label spreading, the gradient of a linear autoencoder, the threshold of a detector. You write it with NumPy and compare it with its library twin (scikit-learn, PCA, Keras). The notebook prints the difference; explain it (another graph, another stopping rule, rounding).

**In class (60 minutes): Parts A, B, C and D**, self-training, label spreading, a linear autoencoder against PCA, and the reconstruction error as an anomaly score. **At home: Parts E and F**, pretraining with few labels and denoising; the solution notebook explains them.

By the end of the lab you should be able to:

1. Run self-training with a confidence threshold and say why its pseudo-labels get worse in later rounds.
2. Build a graph of the rows and spread a few labels with the closed form.
3. Write the gradient step of a linear autoencoder and show that it learns the PCA subspace.
4. Turn a reconstruction error into an alarm with a threshold from held-out normal rows.

| Part | Task | twin | Time |
|:--|:--|:--|:-:|
| A | pseudo-label loop and threshold | `SelfTrainingClassifier` | 12 min |
| B | label spreading on a kNN graph | `LabelSpreading` | 14 min |
| C | linear autoencoder by gradient descent | `PCA` | 16 min |
| D | anomaly threshold and report | PCA detector | 12 min |
| E | probe of a pretrained encoder | random encoder | home |
| F | Challenge: denoising | plain autoencoder | home |

## Setup

```bash
cd faa
source venv/bin/activate        # see README: make venv, or pip install .
jupyter lab notebooks/11-autoencoders/lab11_autoencoders.ipynb
```

The data: the 1 797 **digits** with 64 pixels each (`datasets/digits.csv.zst`), 70 % for training and 30 % for the test; a few training digits are labelled (`few_labels`), the others are not. The helpers in `lab11_utils.py` split, build the graph, build the Keras autoencoder and draw; the loops and formulas are yours. Keras runs on the JAX backend (`KERAS_BACKEND=jax`, set in the first cell).

# Part A — Self-Training

The model labels its own data: fit, give the unlabelled rows with a top probability of at least $\tau$ their most probable class, add them, repeat.

## A1. The pseudo-label loop

Implement `self_train(xl, yl, xu, tau, rounds)`: a logistic regression is fitted in every round; return the final model and the rows added per round. The driver compares it with the model on 50 labels and with `SelfTrainingClassifier`.

## A2. The threshold

The driver sweeps $\tau$ and prints the share of unlabelled rows that became pseudo-labelled and the test accuracy.

1. Which $\tau$ is best, and why does a small $\tau$ not help more? Name the failure behind the later rounds.
2. Why do your accuracy and the twin's differ a little?
3. Five of the 50 labels are wrong. What does self-training do with the error?

# Part B — Label Spreading

Rows are nodes; `knn_affinity(X, k)` gives the Gaussian weights $W$ on the $k$ nearest neighbours. The labels flow along the edges: $F = (I - \alpha S)^{-1} Y_0$ with $S = D^{-1/2} W D^{-1/2}$.

## B1. Normalise and solve

Implement `spread(W, y, labelled, alpha)`: build $S$ and $Y_0$, solve the system with `np.linalg.solve` and return the class of every node. The driver puts all 1 797 digits in one graph and scores the unlabelled nodes against `LabelSpreading`.

## B2. The value of $\alpha$

The driver sweeps $\alpha$ on your function.

1. What happens at $\alpha \to 0$?
2. Why does the graph beat the pseudo-label loop on the digits, and on which data would it fail?
3. Your accuracy and the twin's differ. Which choice of the graph explains it?

# Part C — A Linear Autoencoder by Gradient Descent

With $Z = X E^\top$, $\hat{X} = Z D^\top$ and $R = \hat{X} - X$, the loss is $L = \frac{1}{n} \lVert R \rVert_F^2$ and the gradients are $\frac{2}{n} R^\top Z$ for $D$ and $\frac{2}{n} (R D)^\top X$ for $E$.

## C1. The gradient step

Implement `linear_ae(X, k, lr, steps, seed)`: small random $E$ and $D$, `steps` plain gradient steps on the centred digits, return $E$, $D$ and the losses. The driver compares the test error and the principal angle with PCA, then adds the ReLU network of `make_ae`.

1. Are the two test errors equal? Why does a network with no non-linearity learn the PCA subspace?
2. Is the angle near 0? Which property of the singular values makes it large even when the errors agree?
3. The ReLU network against PCA at the same $k$: which rebuilds better, which has more weights, and what do you check before calling it better?

# Part D — The Reconstruction Error as an Anomaly Score

Normal rows are the digits 0 to 7; the anomalies are 8 and 9. The autoencoder sees only normal rows; the score is its reconstruction error.

## D1. The threshold and the report

Implement `report(e_val, e_test, is_anomaly, q)`: $\theta$ is the quantile $q$ of the errors of the held-out normal rows; return $\theta$, the recall, the false-alarm rate and the AUROC. The driver compares the autoencoder ($k = 4$) with a PCA detector.

## D2. The quantile

The driver sweeps $q$ for the autoencoder.

1. What does the AUROC say that recall and false alarms do not? Why does it not change with $q$?
2. Which detector wins at code size 4, and which would you try at code size 64?
3. What would go wrong if the threshold came from the rows the autoencoder was fitted on?

# Part E — Pretraining with Few Labels, at Home

## E1. A probe on the code

Implement `probe(encoder, x_labelled, y_labelled, x_test, y_test)`: encode, fit a logistic regression on the codes of the labelled rows, return the test accuracy. The driver compares pixels, a random encoder, an encoder pretrained on the unlabelled training digits (frozen) and the same encoder fine-tuned, for 1, 5 and 20 labels per class.

1. How much does the pretraining add to the random encoder? Does it beat the pixels? Say why or why not.
2. When would you expect pretraining to pay: which data and how many unlabelled rows?

# Part F — Challenge: Denoising, at Home

## F1. Clean the digits

Implement `noisy(X, sd, seed)`: Gaussian noise of standard deviation `sd` on every pixel, clipped to $[0, 1]$. The driver trains a plain and a denoising autoencoder and shows clean, noisy and cleaned digits.

1. Why does the plain autoencoder not remove the noise, while the denoising one does?
2. Which noise would you use in training, and what happens if it is much larger than in the test?

**Going further.** Replace the Gaussian noise by masking (hide a share of the pixels) and compare the probe accuracy of Part E. Add an L1 penalty on the code and count the active units.
