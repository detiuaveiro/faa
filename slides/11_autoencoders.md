---
title: Fundamentos de Aprendizagem Automática
subtitle: "Class 11 — AutoEncoders and Semi-Supervised Learning"
bibliography: "references.bib"
suppress-bibliography: true
toc: true
toc-title: "Contents"
toc-depth: 1
section-titles: false
themeoptions:
  - sectionpage=none
  - numbering=fraction
  - progressbar=frametitle
---

# Autoencoders

## Where We Left Off

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cblue, text width=2.6cm] (a) {\textbf{Classes 03-07}\\ supervised: every row has a label};
  \node[fillbox=cgreen, text width=2.6cm, right=5mm of a] (b) {\textbf{Classes 09-10}\\ unsupervised: no labels, find structure};
  \node[fillbox=corange, text width=2.6cm, right=5mm of b] (c) {\textbf{Class 11}\\ autoencoders; then a few labels and many rows without};
  \draw[->] (a) -- (b); \draw[->] (b) -- (c);
\end{tikzpicture}
\end{center}
```

* Labels cost an expert's time
* Rows without labels are almost free
* Today: first networks that rebuild their input from rows alone (autoencoders), then rows without labels that help the few labels (semi-supervised)

## Roadmap of the Class

| Part | Question | Methods |
|:--------------|:------------------------------------|:----------------------------------------|
| **Autoencoders** | can we compress and rebuild? | linear AE and PCA, ReLU AE |
| **Variants** | what else can an AE do? | denoising, sparse, anomaly score, text |
| **Few labels** | what do labels buy? | learning curve, three assumptions |
| **Pseudo-labels** | can the model label for us? | self-training, co-training |
| **Label spreading** | can labels flow along the data? | kNN graph, closed form |
| **Pretraining** | can unlabelled data prepare the model? | pretext tasks, frozen, fine-tuned |

<!-- defines: labelled, unlabelled, pseudo-label, confidence, affinity, graph, autoencoder, encoder, decoder, code, reconstruction error, denoising, sparse code, anomaly score, pretext task, fine-tuning -->

## Terminology

| term | meaning | example |
|:-------------------|:------------------------------------------|:-----------------------------|
| autoencoder | network that rebuilds its input through a small code | $64 \to 8 \to 64$ |
| encoder, decoder | the first and the second half of the autoencoder | $64 \to 8$, $8 \to 64$ |
| code | the narrow middle layer | 8 numbers per digit |
| reconstruction error | squared distance between input and output | 0.021 per pixel |
| labelled, unlabelled | rows with and without a known class | 50 and 1207 digits |
| pseudo-label | class guessed by the model, used as if true | top probability 0.93 |
| confidence | the largest class probability | threshold $\tau = 0.9$ |
| affinity, graph | similarity of two rows; rows as nodes joined by it | 10 nearest neighbours |
| pretext task | a made-up task whose labels come free | rebuild a masked digit |
| fine-tuning | continue training a pretrained network on labels | 100 epochs |

## Notation N11: the Critical Symbols

Other symbols are defined where they are used.

| symbol | meaning | shape | example |
|:---------------------|:-----------------------------------|:-----------------------:|:-----------|
| $X_L$, $y_L$ | labelled rows and their classes | $n_L \times d$ | 50 |
| $X_U$ | unlabelled rows | $n_U \times d$ | 1207 |
| $\tau$ | confidence threshold | scalar | 0.9 |
| $W$, $S$ | affinity matrix, its normalised form | $n \times n$ | $k = 10$ |
| $z$ | code of one row | $k$ | $k = 8$ |
| $\hat{x}$ | reconstruction $g(f(x))$ | $d$ | |

## Compress and Rebuild

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize, node distance=7mm]
  \node[fillbox=cblue, minimum width=1.2cm, minimum height=1.6cm] (x) {$x$\\ 64};
  \node[fillbox=cgreen, minimum width=1.2cm, minimum height=0.8cm, right=20mm of x] (z) {$z$\\ 8};
  \node[fillbox=cblue, minimum width=1.2cm, minimum height=1.6cm, right=20mm of z] (xh) {$\hat{x}$\\ 64};
  \draw[flow] (x) -- node[above] {encoder $f$} (z);
  \draw[flow] (z) -- node[above] {decoder $g$} (xh);
  \node[note, below=3mm of z] {code};
\end{tikzpicture}
\end{center}
```

* Input and target are the same row: no labels
* The narrow code forces the network to keep only what it needs to rebuild

## The Loss

The network is trained to make $\hat{x}$ close to $x$: the reconstruction error.

<!-- derivation: ae-loss -->
$$
\begin{aligned}
L &= \frac{1}{n} \sum_{i=1}^{n} \lVert x_i - g(f(x_i)) \rVert^2 && \text{(definition)}
\end{aligned}
$$

* $f$: encoder, $g$: decoder, $z = f(x)$ has $k < d$ numbers
* $L$ is the squared error of Class 03, and the target is the input
* Hinton and Salakhutdinov [-@hinton2006]: deep autoencoders as non-linear PCA

## One Hidden Unit by Hand

Three points in the plane; one code unit; $z = w^\top x$, $\hat{x} = w z$ with $w = (0.8, 0.6)$.

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize, scale=0.6]
  \draw[->] (-1.8,0) -- (2.8,0) node[right] {$x_1$}; \draw[->] (0,-1.2) -- (0,1.6) node[above] {$x_2$};
  \draw[cgray, dashed] (-1.6,-1.2) -- (2.6,1.95);
  \node[dot=cblue, label=above:$x_1$] at (2,1) {}; \node[dot=cblue, label=below:$x_2$] at (-1,-0.5) {};
  \node[dot=corange, label=below:$\hat{x}_1$] at (1.76,1.32) {};
  \draw[cred] (2,1) -- (1.76,1.32);
\end{tikzpicture}
\end{center}
```

| point | $z$ | $\hat{x}$ | error |
|:-----------:|:-----------:|:------------------:|:-----------:|
| $(2, 1)$ | 2.2 | $(1.76, 1.32)$ | 0.16 |
| $(-1, -0.5)$ | $-1.1$ | $(-0.88, -0.66)$ | 0.04 |
| $(0, 0)$ | 0 | $(0, 0)$ | 0 |

* Mean error 0.0667; the best direction $(0.894, 0.447)$ has error 0 (Demo A1)

## Why a Linear Autoencoder Is PCA

Without activations: $\hat{x} = D E x$, $E$ is $k \times d$, $D$ is $d \times k$.

* The product $D E$ has rank at most $k$
* Best rank-$k$ matrix: projection on the top $k$ singular vectors (Class 09)
* Baldi and Hornik [-@baldi1989]: the minimum of $L$ is that projection
* The network learns the **subspace**, not the individual PCA directions

## Counting the Weights of an Autoencoder

A dense autoencoder $d \to u \to k \to u \to d$ has four layers with biases: $du + u$, $uk + k$, $ku + u$ and $ud + d$ weights.

<!-- derivation: ae-params
defs:
  d: "64"
  u: "32"
  k: "8"
symbols:
  d: {kind: scalar}
  u: {kind: scalar}
  k: {kind: scalar}
-->
$$
\begin{aligned}
m &= 2 d u + 2 u k + 2 u + k + d && \text{(definition)} \\
&= 4096 + 512 + 64 + 8 + 64 && \text{(substitute)} \\
&= 4744 && \text{(evaluate)}
\end{aligned}
$$

(result) 4 744 weights for $64 \to 32 \to 8 \to 32 \to 64$, for 1257 training rows

## A Linear Autoencoder Finds the PCA Subspace

Keras, linear, 300 epochs, against PCA with the same $k$ (test mean squared error per pixel).

:::: {.columns}
::: {.column width="46%"}
| $k$ | PCA | linear AE |
|:--------:|:-----------:|:---------------:|
| 2 | 0.053 | 0.053 |
| 5 | 0.034 | 0.035 |
| 10 | 0.020 | 0.020 |
| 20 | 0.008 | 0.008 |
:::
::: {.column width="52%"}
![](plots/11/a1_linear_pca.pdf)
:::
::::

* Same error: the same subspace (Demo A1)

## The Code Is Not the PCA Score

The AE code mixes the PCA scores: any invertible mixing has the same error.

* $k = 5$: four PCA scores come from the AE codes with $R^2 = 1.00$
* The fifth has $R^2 = 0.39$: its variance nearly equals the sixth
* Principal angle to PCA: 12.9° ($k = 2$), 6.9° (10), 50.6° (5), 87.3° (20)
* Large angle where $\lambda_k / \lambda_{k+1}$ is near 1 (1.04 at $k = 20$); error equal
* Class 09 had this ambiguity for ICA

## Depth and Non-Linearity

A ReLU autoencoder $64 \to 32 \to k \to 32 \to 64$ against PCA, 3 seeds.

:::: {.columns}
::: {.column width="46%"}
| $k$ | PCA | ReLU AE |
|:--------:|:-----------:|:---------------:|
| 2 | 0.053 | 0.043 |
| 4 | 0.038 | 0.029 |
| 8 | 0.025 | 0.021 |
| 16 | 0.011 | 0.011 |
| 32 | 0.003 | 0.006 |
:::
::: {.column width="52%"}
![](plots/11/a2_code_size.pdf)
:::
::::

* Small $k$: the curved manifold helps; $k = 32$: PCA wins, the network is under-trained (Demo A2)

## Reconstructions

Eight test digits rebuilt from codes of 2, 8 and 32 numbers.

![](plots/11/a2_reconstructions.pdf){width=70%}

* Error per pixel: 0.041 ($k = 2$), 0.022 (8), 0.006 (32)
* Two numbers give a blur of the class

## Two-Dimensional Codes

The 540 test digits in a 2-D code; colours are not used in training.

![](plots/11/a2_codes_2d.pdf){width=80%}

* 5-nearest-neighbour accuracy on the two coordinates: PCA 0.606, autoencoder 0.780

## A Code as Large as the Input

If $k = d$ the network can copy.

| $k$ | epochs | test error |
|:--------:|:-------------:|:--------------:|
| 64 | 150 | 0.005 |
| 64 | 3 | 0.046 |
| 8 | 3 | 0.069 |

* A small error does not make the code useful: probe it (Demo A2)

## Demo A

`01_autoencoder.ipynb`

* a unit by hand; a linear autoencoder against the PCA subspace
* a ReLU autoencoder: code size, reconstructions, two-dimensional codes
* failure case: a code as large as the input

# Variants and Anomaly Detection

## Denoising

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize, node distance=7mm]
  \node[fillbox=cgray] (c) {clean $x$};
  \node[fillbox=corange, right=of c] (n) {add noise $\tilde{x}$};
  \node[fillbox=cblue, right=of n] (a) {autoencoder};
  \node[fillbox=cgray, right=of a] (o) {output $\hat{x}$};
  \draw[flow] (c) -- (n); \draw[flow] (n) -- (a); \draw[flow] (a) -- (o);
  \draw[flow, cred, dashed] (o.south) -- ++(0,-0.6) -| node[pos=0.25, below] {compare with the clean $x$} (c.south);
\end{tikzpicture}
\end{center}
```

* New noise every epoch; clean target (Vincent et al. [-@vincent2008])
* It must learn what a digit is, not copy pixels

## Denoising in Numbers

Gaussian noise of 0.3 on every pixel; error against the clean digit.

:::: {.columns}
::: {.column width="46%"}
| model | error |
|:-----------------|:-----------:|
| noisy input itself | 0.052 |
| plain AE | 0.050 |
| denoising AE | 0.029 |
| PCA, $k = 8$ | 0.033 |
:::
::: {.column width="52%"}
![](plots/11/b1_denoising.pdf)
:::
::::

* The plain autoencoder copies the noise; the denoising one removes it (Demo B1)

## Sparse Codes

A large code with a penalty: most units must be zero.

<!-- derivation: sparse-loss -->
$$
\begin{aligned}
L &= \frac{1}{n} \sum_i \lVert x_i - \hat{x}_i \rVert^2 + \lambda \sum_i \lVert z_i \rVert_1 && \text{(definition)}
\end{aligned}
$$

* $\lambda$: penalty weight; $\lVert z \rVert_1$: sum of $|z_j|$
* The L1 penalty of Class 03: many exact zeros
* A few units per digit, like NMF parts (Class 09)

## Sparse Codes in Numbers

Code of 64 units; active = $|z| > 0.01$.

:::: {.columns}
::: {.column width="46%"}
| $\lambda$ | error | active units |
|:--------:|:-----------:|:--------------:|
| 0 | 0.002 | 54.4 |
| 0.0001 | 0.002 | 42.1 |
| 0.001 | 0.006 | 21.0 |
| 0.01 | 0.058 | 1.5 |
:::
::: {.column width="52%"}
![](plots/11/b1_sparse.pdf)
:::
::::

* Sparsity costs reconstruction: pick the knee (Demo B1)

## Other Autoencoders

* **Variational** (Kingma and Welling [-@kingma2014]): the code is a distribution; sample new digits
* **Contractive**: penalise how much the code moves when the input moves
* **Temporal**: 1-D convolutions read text (Demo C); 2-D layers come in Class 12
* Same shape; they differ in penalty and layers

## The Reconstruction Error as a Score

An autoencoder trained on normal rows rebuilds normal rows well and the others badly.

<!-- derivation: anomaly-score -->
$$
\begin{aligned}
s(x) &= \lVert x - g(f(x)) \rVert^2 && \text{(definition)}
\end{aligned}
$$

* Alarm when $s(x) > \theta$; $\theta$ = 95th percentile of the errors of held-out normal rows
* Same recipe as Class 10 for the clustering anomalies
* No anomaly labels are needed to train: they are needed only to measure

## Anomalies Are Badly Reconstructed

Normal digits 0 to 7, anomalies 8 and 9; code of 4; threshold 0.0516.

:::: {.columns}
::: {.column width="52%"}
![](plots/11/b2_error_hist.pdf)
:::
::: {.column width="46%"}
* AUROC 0.912
* Recall 0.500, precision 0.671
* 53 of 106 anomalies found, 26 false alarms in 434 normal digits
:::
::::

## Autoencoder or PCA as a Detector?

Same normal rows; AUROC of the reconstruction error, 3 seeds for the autoencoder.

:::: {.columns}
::: {.column width="46%"}
| $k$ | AE | PCA |
|:--------:|:-----------:|:---------:|
| 2 | 0.696 | 0.636 |
| 4 | 0.885 | 0.726 |
| 8 | 0.918 | 0.849 |
| 16 | 0.915 | 0.893 |
:::
::: {.column width="52%"}
![](plots/11/b2_auroc_k.pdf)
:::
::::

* The non-linear code wins for small $k$ (Demo B2)

## A Room-Temperature Anomaly

Five hourly temperatures; offset of 1.5 standard deviations for 24 h, five times.

![](plots/11/b2_climate_error.pdf){width=85%}

* AUROC: autoencoder 0.651, PCA 0.697: no clear winner
* Test hours are 1.7 to 2.1 deviations warmer: drift hides faults

## The Autoencoder That Reconstructs Everything

| model | error normal | error anomaly | AUROC |
|:-----------------|:-----------:|:-------------:|:-----------:|
| $k = 4$, hidden 32 | 0.026 | 0.054 | 0.912 |
| $k = 64$, none | 0.002 | 0.003 | 0.811 |

* A large code rebuilds the anomalies too: the margin shrinks to 0.001
* Keep the code small enough that only the normal pattern fits

## Demo B

`02_variants.ipynb`

* denoising and sparse autoencoders on noisy digits
* reconstruction error as an anomaly score: digits 8 and 9, room temperature
* failure case: an autoencoder that reconstructs everything

## Autoencoders for Sequences

Text is a sequence: a temporal autoencoder reads 32 characters at a time.

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize, node distance=5mm]
  \node[fillbox=cblue, text width=1.5cm] (x) {32 characters};
  \node[fillbox=cgreen, text width=1.6cm, right=of x] (e) {2 convolutions, stride 2};
  \node[fillbox=corange, text width=1.5cm, right=of e] (z) {code: 32 $\times$ 4 bits};
  \node[fillbox=cgreen, text width=1.6cm, right=of z] (d) {2 transposed convolutions};
  \node[fillbox=cblue, text width=1.5cm, right=of d] (y) {32 characters};
  \draw[flow] (x) -- (e); \draw[flow] (e) -- (z); \draw[flow] (z) -- (d); \draw[flow] (d) -- (y);
\end{tikzpicture}
\end{center}
```

* The code is 16 bytes for 32 characters: ratio 2, lossy
* Noise on the code while training teaches the decoder to survive rounding
* `python -m faa.textae` trains, compresses and decompresses

## A Text Compressor in Numbers

Trained on four BBC files (1.9 MB); tested on an unseen file.

:::: {.columns}
::: {.column width="52%"}
| method | bytes | ratio | exact |
|:-------------|:--------:|:--------:|:--------:|
| gzip -9 | 8548 | 2.34 | yes |
| bzip2 -9 | 7829 | 2.56 | yes |
| lzma | 8272 | 2.42 | yes |
| autoencoder | 10006 | 2.00 | no |
:::
::: {.column width="46%"}
![](plots/11/c1_training.pdf)
:::
::::

* 0.753 of the characters survive the round trip
* Classical tools are smaller and exact; lossy pays only if approximate is enough

## What the Compressor Rebuilds

The same 48 characters before and after the round trip (an unseen file).

```
original: a staunton were both winners at the 2004 evening
rebuilt : anshaunton were boys winvers as the 2000 enining
```

* Short, frequent words survive; names and numbers are blurred
* The decoder predicts each character from the code: it rebuilds what the training files made likely

## How Big Must the Code Be?

Three code sizes, a short run (5 epochs, 20 000 windows) so that the numbers are lower than above.

:::: {.columns}
::: {.column width="52%"}
![](plots/11/c2_code_size.pdf)
:::
::: {.column width="46%"}
* 4 bytes per window (ratio 8): 0.188 of the characters
* 8 bytes (ratio 4): 0.295
* 16 bytes (ratio 2): 0.463
:::
::::

* Every halving of the code costs characters: the code size is the knob

## Text the Training Files Never Showed

Same compressor, four kinds of text, 20 000 characters each.

| text | ratio | characters identical |
|:----------------------------------|:-------:|:----------------------:|
| unseen news | 2.00 | 0.753 |
| Markdown page | 2.00 | 0.622 |
| Python source | 1.99 | 0.500 |
| random characters | 2.00 | 0.081 |

* The ratio is fixed by the architecture
* The quality depends on how like the training data the text is

## Demo C

`03_text_compression.ipynb`

* four training files, one unseen file, windows of 32 characters
* the command line: train, compress, decompress
* gzip, bzip2 and lzma as the lossless reference
* failure case: text unlike the news it was trained on

# Few Labels

## Labels Cost, Rows Do Not

The digits of Class 05 (1797 images of 8 × 8 pixels, ten classes): someone must name each one.

| rows | labelled | cost |
|:--------------------------------|:-----------------|:----------------------|
| photographs of a factory line | a few hundred | a technician's week |
| medical scans | tens | a specialist's hour each |
| text, sensor streams, logs | almost none | free but unlabelled |

* **Semi-supervised learning**: $n_L$ labelled rows, $n_U \gg n_L$ unlabelled rows, one model
* Goal: beat the model that sees only the $n_L$ labels

## How Much Do Labels Buy?

Logistic regression **trained on the $n_L$ labelled digits only**; tested on 540 other digits.

:::: {.columns}
::: {.column width="46%"}
| labels | accuracy |
|:-------------------:|:-------------------:|
| 10 | 0.707 |
| 50 | 0.844 |
| 100 | 0.895 |
| 500 | 0.956 |
| 1257 | 0.970 |
:::
::: {.column width="52%"}
![](plots/11/d1_labels_curve.pdf)
:::
::::

* Mean of 5 draws; steep up to 100 labels (Demo D1)

## Which Digits Are Labelled Matters

Draw 10 labelled digits (one per class) at random, 20 times. Each draw gives one accuracy.

:::: {.columns}
::: {.column width="52%"}
![](plots/11/d1_draws.pdf)
:::
::: {.column width="46%"}
* Mean accuracy 0.668, standard deviation 0.051
* Worst draw 0.541, best 0.737
* With 20 per class the spread is 0.919 to 0.950
:::
::::

* Report the spread over draws, never one draw (Demo D1)

## The Unlabelled Digits Have a Shape

Left: 40 unlabelled digits. Right: 1257 digits on a t-SNE map.

![](plots/11/d2_digits_plane.pdf)

* Groups appear without labels: 0.977 of 5 map neighbours share the hidden class
* PCA (Class 09) keeps 28.4 % and overlaps them

## Three Assumptions

Unlabelled rows help only if the data have a shape the labels follow.

| assumption | statement | method that uses it |
|:--------------|:--------------------------------------|:-------------------------|
| smoothness | close rows share a class | label spreading |
| cluster | a class is a cluster; boundaries cross empty regions | self-training |
| manifold | rows lie on a low-dimensional surface | autoencoders |

* When an assumption is false, unlabelled rows can hurt (Chapelle et al. [-@chapelle2006])

## Two Labels and the Cluster Assumption

Two moons, one label per class. Left: the 2 labels only. Right: also the 298 unlabelled points.

![](plots/11/d2_moons_line.pdf)

* Line and 1-nearest neighbour: same boundary, accuracy 0.703
* Label spreading (Demo F) uses the other points: 0.953

## Demo D

`04_few_labels.ipynb`

* learning curve and the spread over draws of the labels
* the unlabelled digits as images and on a map; two moons with two labels
* failure case: the worst draw

# Pseudo-Labels: Self-Training

## The Model Labels Its Own Data

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize, node distance=8mm]
  \node[fillbox=cblue, text width=2.1cm] (fit) {1. fit on the labelled rows};
  \node[fillbox=corange, text width=2.1cm, right=of fit] (pred) {2. predict the unlabelled rows};
  \node[fillbox=cgreen, text width=2.3cm, right=of pred] (add) {3. keep those with confidence $\ge \tau$};
  \draw[flow] (fit) -- (pred); \draw[flow] (pred) -- (add);
  \draw[flow, cred] (add.south) -- ++(0,-0.7) -| node[pos=0.25, below] {add them as labelled, repeat} (fit.south);
\end{tikzpicture}
\end{center}
```

* The pseudo-label is the most probable class; the confidence is its probability
* Pseudo-labelling for deep networks: Lee [-@lee2013]; the model teaches itself

## Self-Training Step by Step

Let $p_i$ be the class probabilities of unlabelled row $i$ and $\tau$ the threshold.

1. Fit the model on $(X_L, y_L)$
2. For each unlabelled row, compute $\hat{y}_i = \arg\max p_i$ and the confidence $\max p_i$
3. Move every row with $\max p_i \ge \tau$ to the labelled set with label $\hat{y}_i$
4. Stop when no row passes, or after a fixed number of rounds

* Needs a model with calibrated probabilities (Class 04): a logistic regression here, $C = 10$
* One free choice: $\tau$

## Self-Training in Rounds

50 labelled digits, 1207 unlabelled, threshold $\tau = 0.9$: each round adds the sure rows.

:::: {.columns}
::: {.column width="52%"}
| round | added | right |
|:-------------:|:-------------:|:-------------:|
| 1 | 439 | 0.991 |
| 2 | 310 | 0.952 |
| 3 | 141 | 0.801 |
| 4 | 65 | 0.692 |
| 5 | 31 | 0.677 |
| 7 | 9 | 0.222 |
:::
::: {.column width="44%"}
![](plots/11/e1_rounds.pdf)
:::
::::

* "right" = share of the added pseudo-labels that are correct
* Test accuracy 0.831 before, 0.857 after 10 rounds (Demo E1)

## The Threshold $\tau$

Low $\tau$ adds wrong rows; high $\tau$ adds almost none.

:::: {.columns}
::: {.column width="52%"}
![](plots/11/e1_threshold.pdf)
:::
::: {.column width="46%"}
| $\tau$ | accuracy | share added |
|:-----------:|:------------:|:-------------:|
| 0.5 | 0.875 | 0.996 |
| 0.7 | 0.889 | 0.964 |
| 0.9 | 0.882 | 0.843 |
| 0.99 | 0.840 | 0.037 |
:::
::::

* Labelled only: 0.843; best $\tau$ is 0.7 to 0.9

## Confirmation Bias

A wrong pseudo-label makes the model surer of the same mistake, and the next round copies it.

:::: {.columns}
::: {.column width="52%"}
![](plots/11/e1_bias.pdf)
:::
::: {.column width="46%"}
* A share of the 50 labels is wrong
* 0 %: 0.843 to 0.882
* 20 %: 0.652 to 0.688
* The gain survives; the errors do too
:::
::::

* Self-training cannot fix a bad start, only extend it (Demo E1)

## Co-Training: Two Views

Two models, each reading a different part of the row, label rows for each other.

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize, node distance=14mm]
  \node[fillbox=cblue, text width=2.2cm] (va) {view A: top four rows};
  \node[fillbox=cgreen, text width=2.2cm, below=5mm of va] (vb) {view B: bottom four rows};
  \node[fillbox=corange, text width=2.2cm, right=of va] (ca) {classifier A};
  \node[fillbox=corange, text width=2.2cm, right=of vb] (cb) {classifier B};
  \draw[flow] (va) -- (ca); \draw[flow] (vb) -- (cb);
  \draw[flow, cred] ([xshift=-4mm]ca.south) -- ([xshift=-4mm]cb.north);
  \draw[flow, cred] ([xshift=4mm]cb.north) -- ([xshift=4mm]ca.south);
  \node[font=\tiny, text width=2.6cm, right=3mm of ca.east, anchor=west] {A hands pseudo-labels to B, B hands pseudo-labels to A};
\end{tikzpicture}
\end{center}
```

* Each round, a classifier hands its confident pseudo-labels ($\geq \tau$) to the other
* Needs views independent given the class (Blum and Mitchell [-@blum1998])
* Digits: two views of 32 pixels, 50 labels

## Co-Training: Result

5 draws of the 50 labels; accuracy on the 540 test digits.

:::: {.columns}
::: {.column width="52%"}
![](plots/11/e2_cotrain.pdf)
:::
::: {.column width="46%"}
* Top and bottom halves: co-training 0.735, below the two views without unlabelled rows (0.795)
* Random pixel split: 0.816 against 0.831
* A stricter $\tau$ up to 0.999 gives 0.832: no gain
* Self-training on all pixels: 0.882
:::
::::

* Two halves of one digit are not independent given the class; the assumption fails and the pseudo-labels hurt

## Beyond Self-Training

* **Consistency**: a prediction must not change when the input is perturbed
* **FixMatch** (Sohn et al. [-@sohn2020]): pseudo-label a weak augmentation, train on a strong one
* Same loop: guess, filter by confidence, train

## Demo E

`05_self_training.ipynb`

* self-training in rounds; the threshold $\tau$; the scikit-learn twin
* wrong labels: confirmation bias
* co-training with two views of each digit

# Label Spreading

## From Points to a Graph

Each row is a node; each row is joined to its 10 nearest neighbours.

:::: {.columns}
::: {.column width="52%"}
![](plots/11/f1_graph.pdf)
:::
::: {.column width="46%"}
* Two moons, 300 nodes, 1778 edges
* Two labelled nodes (large dots)
* Colours: labels after spreading
* Accuracy 0.953 against 0.703 for a line
:::
::::

## The Affinity of Two Rows

Close rows get a weight near 1, far rows near 0: $w_{ij} = \exp(-\lVert x_i - x_j \rVert^2 / 2\sigma^2)$.

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[dot=cblue, label=above:$x_i$] (a) at (0,0) {};
  \node[dot=cblue, label=above:$x_j$] (b) at (2,0) {};
  \node[dot=cgray, label=above:$x_k$] (c) at (5,0) {};
  \draw (a) -- node[above, pos=0.5, yshift=-1mm, fill=white, inner sep=1pt] {$w$ near 1} (b);
  \draw[dashed] (b) -- node[above, pos=0.5, yshift=-1mm, fill=white, inner sep=1pt] {$w$ near 0} (c);
\end{tikzpicture}
\end{center}
```

* $\sigma$ is the median distance to the 10th neighbour: no free scale
* Keep only the edges to the 10 nearest neighbours, symmetrised
* $W$ is $n \times n$, $w_{ii} = 0$

## Normalising the Graph

Rows with many neighbours must not dominate: $S = D^{-1/2} W D^{-1/2}$, where $D$ is diagonal with $D_{ii} = \sum_j w_{ij}$.

* Path of four points, $w_{12} = w_{23} = w_{34} = 1$: degrees $(1, 2, 2, 1)$
* $s_{12} = 1 / \sqrt{1 \cdot 2} = 0.707$ and $s_{23} = 1 / \sqrt{2 \cdot 2} = 0.5$
* Symmetric, so labels flow both ways

## Spreading the Labels

Each node takes a share $\alpha$ of what its neighbours have, and keeps $1 - \alpha$ of its own start.

<!-- derivation: spreading-loop -->
$$
\begin{aligned}
F^{(t+1)} &= \alpha\, S\, F^{(t)} + (1 - \alpha)\, Y_0 && \text{(definition)}
\end{aligned}
$$

* $Y_0$ is $n \times c$: a one-hot row for a labelled node, zeros otherwise
* $F$ is $n \times c$: the class scores; the class of node $i$ is $\arg\max_c F_{ic}$
* Zhou et al. [-@zhou2004]

## The Closed Form

At the limit of the loop $F = \alpha S F + Y_0$ (up to a factor): solve for $F$.

<!-- derivation: spreading-closed-form
symbols:
  S: {kind: matrix, shape: n x n}
  F: {kind: matrix, shape: n x c}
  Y_0: {kind: matrix, shape: n x c}
  \alpha: {kind: scalar}
-->
$$
\begin{aligned}
Y_0 &= F - \alpha S F && \text{(definition)} \\
&= (I - \alpha S) F && \text{(matrix form)} \\
F &= (I - \alpha S)^{-1} Y_0 && \text{(solve for)}
\end{aligned}
$$

(result) the factor $1 - \alpha$ only rescales $F$; the argmax stays

## Label Spreading by Hand

Four points on a path; node 1 is A, node 4 is B; $\alpha = 0.9$.

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize, node distance=6mm]
  \node[fillbox=cblue, minimum width=1.4cm] (a) {1: A\\ $(2.59, 1.20)$};
  \node[fillbox=csky, minimum width=1.4cm, right=of a] (b) {2\\ $(2.50, 1.89)$};
  \node[fillbox=cpurple, minimum width=1.4cm, right=of b] (c) {3\\ $(1.89, 2.50)$};
  \node[fillbox=corange, minimum width=1.4cm, right=of c] (d) {4: B\\ $(1.20, 2.59)$};
  \draw (a) -- (b); \draw (b) -- (c); \draw (c) -- (d);
\end{tikzpicture}
\end{center}
```

* Scores $(F_{iA}, F_{iB})$: nodes 2, 3 take the nearer label
* Middle edge 0.2: nodes 2, 3 hold their sides harder

## What $\alpha$ and $k$ Do

Digits: all 1797 as nodes, 50 labels, accuracy on the unlabelled nodes, 5 draws.

:::: {.columns}
::: {.column width="52%"}
![](plots/11/f1_alpha_k.pdf)
:::
::: {.column width="46%"}
* Best: $k = 5$ to 10, $\alpha = 0.9$: 0.951 and 0.952
* $\alpha = 0.5$: labels do not travel far: 0.93
* $k = 20$: more edges across classes: 0.941
:::
::::

## Graph or Loop: 50 Labels

Same 50 digits for every method.

:::: {.columns}
::: {.column width="46%"}
| method | accuracy |
|:----------------------|:----------:|
| labelled only | 0.838 |
| self-training | 0.872 |
| label spreading | 0.933 |
:::
::: {.column width="52%"}
![](plots/11/g1_summary.pdf)
:::
::::

* The graph beats the loop here: the digits obey the smoothness assumption (Demo G1)

## When the Graph Is Wrong

Two moons that touch: neighbours cross the gap and the labels follow them.

| noise | spreading | line on 4 labels |
|:------------:|:-------------:|:-------------------:|
| 0.05 | 1.000 | 0.817 |
| 0.15 | 0.890 | 0.815 |
| 0.30 | 0.814 | 0.773 |
| 0.50 | 0.767 | 0.722 |

* The advantage vanishes when the classes overlap (Demo F1)

## Real Data: News Headlines

6000 AG News headlines, four topics, as tf-idf vectors; graph of the 10 nearest neighbours.

:::: {.columns}
::: {.column width="52%"}
![](plots/11/f2_news.pdf)
:::
::: {.column width="46%"}
* 32 labels: spreading 0.723, logistic regression 0.525
* 128 labels: 0.781 against 0.702
* Self-training adds no row ($\tau = 0.9$ never reached)
* The graph needs only a similarity: words here (Demo F)
:::
::::

## Transductive and Inductive

* **Transductive**: labels the given nodes; a new row needs a new solve
* **Inductive**: a model that labels any new row (self-training, autoencoders)
* Spreading solves an $n \times n$ system: thousands of nodes, not millions

## Twins

* `SelfTrainingClassifier`: 0.841 against our 0.857, agreement 0.981 on the test rows
* `LabelSpreading(kernel="knn")`: 0.920 against our 0.950 (other weights)
* Same idea, different details: report the library you used

## Demo F

`06_label_spreading.ipynb`

* four nodes by hand; the graph of the moons; $\alpha$ and $k$ on the digits
* the scikit-learn twin; failure case: touching moons
* real data: 6000 news headlines with 32 labels

# Pretraining and Self-Supervision

## Pretrain, Then Label

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize, node distance=7mm]
  \node[fillbox=cgray, text width=2cm] (u) {1257 unlabelled digits};
  \node[fillbox=cblue, text width=2cm, right=of u] (p) {pretrain the encoder (rebuild)};
  \node[fillbox=cgreen, text width=2cm, right=of p] (l) {50 labels: train a classifier on the code};
  \draw[flow] (u) -- (p); \draw[flow] (p) -- (l);
\end{tikzpicture}
\end{center}
```

* **Frozen**: the encoder stays fixed, only a classifier is trained on the code
* **Fine-tuned**: the labels adjust the encoder as well
* **Scratch**: the same network with a random start

## Pretrain, Then Label: Results

Encoder $64 \to 32 \to 16$ pretrained without labels; 3 draws.

:::: {.columns}
::: {.column width="52%"}
| per class | pixels | frozen | tuned | scratch |
|:----:|:----:|:----:|:----:|:----:|
| 1 | 0.704 | 0.642 | 0.540 | 0.590 |
| 5 | 0.838 | 0.814 | 0.779 | 0.815 |
| 20 | 0.927 | 0.909 | 0.926 | 0.922 |
:::
::: {.column width="46%"}
![](plots/11/g1_pretrain_curve.pdf)
:::
::::

* Pixels are hard to beat: the classes are almost linear (Demo G1)

## What Pretraining Does Buy

Frozen encoder with 50 labels: the pretraining data change the result.

:::: {.columns}
::: {.column width="46%"}
| unlabelled rows | accuracy |
|:-------------------:|:------------:|
| 0 (random encoder) | 0.576 |
| 100 | 0.757 |
| 400 | 0.783 |
| 1257 | 0.814 |
:::
::: {.column width="52%"}
![](plots/11/g1_unlabelled.pdf)
:::
::::

* More unlabelled rows, better code: 0.576 to 0.814
* It does not beat the pixels (0.838): there is little to gain on 8 by 8 digits

## When Pretraining Pays

* The raw features are poor: photographs, text, audio, sensor streams
* The unlabelled set is large: millions of rows
* The pretext resembles the task: the encoder learns what the labels need
* Class 12: BERT (Devlin et al. [-@devlin2019]) pretrains on text, then a few labels fine-tune it

## Self-Supervised Learning

The labels come from the data: hide a part, predict it.

| pretext task | input | target |
|:----------------------|:----------------------|:-----------------------|
| denoising | digit plus noise | the clean digit |
| masking | image with hidden pixels | the hidden pixels |
| next token | the words so far | the next word |
| contrastive | two crops of one image | "same image" |

* No human labels; the encoder is the product

## Mask and Reconstruct

Hide a fraction $p$ of the pixels; the target is the full digit.

:::: {.columns}
::: {.column width="46%"}
| $p$ | accuracy |
|:----------:|:------------:|
| 0 | 0.817 |
| 0.25 | 0.791 |
| 0.5 | 0.799 |
| 0.75 | 0.794 |
| 0.9 | 0.760 |
:::
::: {.column width="52%"}
![](plots/11/g2_mask_rate.pdf)
:::
::::

* Frozen encoder, 50 labels: no gain here (Demo G2)
* At scale, He et al. [-@he2022] mask 75 % of an image

## A Pretext Unrelated to the Task

| pretrained on | accuracy |
|:------------------|:-------------:|
| digits | 0.814 |
| uniform noise | 0.667 |
| no pretraining | 0.576 |

* Rebuilding noise teaches little
* Still above a random encoder: 16 random units keep information
* Pretext and task must share the data

## Which Method for 50 Labels?

| method | accuracy |
|:----------------------|:---------:|
| labelled only (pixels) | 0.838 |
| PCA 16 + LR | 0.829 |
| self-training | 0.872 |
| label spreading | 0.933 |
| AE 16 frozen + LR | 0.814 |
| AE 16 fine-tuned | 0.779 |

* The best method follows the data: here the neighbour graph (Demo G1)

## Demo G

`07_pretraining.ipynb`

* pretrain on unlabelled digits, then 1 to 20 labels; frozen, fine-tuned, scratch
* masking rate of the pretext task
* failure case: a pretext unrelated to the task

# Choosing and Using

## Which Tool for Which Situation?

| situation | try first |
|:-------------------------------|:---------------------------|
| few labels, rows form clusters | label spreading, self-training |
| a graph is natural (links, similarity) | label spreading |
| large unlabelled set, poor features | pretrain an encoder |
| only normal rows | autoencoder anomaly score |
| noisy inputs to clean | denoising autoencoder |

* Always run the labelled-only baseline and report the spread over draws

## Summary

* **Autoencoder**: $z = f(x)$, rebuild $x$; linear = PCA subspace
* **Variants**: denoising, sparse, anomaly score (AUROC 0.91), a text compressor
* **Labels**: 50 labels give 0.84, 1257 give 0.97; the draw matters
* **Self-training**: pseudo-labels above $\tau$; gain 0.84 to 0.88; errors are copied
* **Label spreading**: $F = (I - \alpha S)^{-1} Y_0$ on a kNN graph: 0.93 to 0.95
* **Pretraining**: unlabelled rows prepare the encoder; raw pixels are hard to beat

## Lab 11

| part | task |
|:---------:|:---------------------------------------------------------------------------------|
| A | self-training: the pseudo-label loop against `SelfTrainingClassifier` |
| B | label spreading: the kNN graph and the closed form against `LabelSpreading` |
| C | a linear autoencoder by gradient descent against PCA |
| D | the reconstruction error as an anomaly score, threshold and AUROC |
| E, F | home: pretrain an encoder and compare with scratch; denoise noisy digits |

## Lab 11: How It Connects

* A, B reuse the self-training and graph frames
* C meets PCA of Class 09 on the same digits
* D reuses the threshold of the anomaly frames; E, F reuse the pretraining demos

## Key Takeaways

* Unlabelled rows help only when the data have a shape the labels follow
* A confident model can be confidently wrong; co-training needs views that are independent
* An autoencoder is non-linear PCA; the code size is the knob
* Reconstruction error flags what the training rows did not contain
* A learned compressor is lossy: on text it loses to gzip, so use it where approximate is enough
* On small, clean data a simple baseline can win: measure it first

<!-- refs:begin -->
## References

:::: {.columns}
::: {.column width="50%"}
\fontsize{5.5}{6.4}\selectfont\setlength{\parskip}{2.5pt}

Baldi, Pierre, and Kurt Hornik. 1989. “Neural Networks and Principal Component Analysis: Learning from Examples Without Local Minima.” Neural Networks 2 (1): 53–58.

Blum, Avrim, and Tom Mitchell. 1998. “Combining Labeled and Unlabeled Data with Co-Training.” Conference on Computational Learning Theory, 92–100.

Chapelle, Olivier, Bernhard Schölkopf, and Alexander Zien. 2006. Semi-Supervised Learning. MIT Press.

Devlin, Jacob, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. “BERT: Pre-Training of Deep Bidirectional Transformers for Language Understanding.” North American Chapter of the Association for Computational Linguistics, 4171–86.

He, Kaiming, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. 2022. “Masked Autoencoders Are Scalable Vision Learners.” IEEE Conference on Computer Vision and Pattern Recognition, 16000–16009.

Hinton, Geoffrey E., and Ruslan R. Salakhutdinov. 2006. “Reducing the Dimensionality of Data with Neural Networks.” Science 313 (5786): 504–7.

:::
::: {.column width="50%"}
\fontsize{5.5}{6.4}\selectfont\setlength{\parskip}{2.5pt}

Kingma, Diederik P., and Max Welling. 2014. “Auto-Encoding Variational Bayes.” International Conference on Learning Representations.

Lee, Dong-Hyun. 2013. “Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks.” ICML Workshop on Challenges in Representation Learning.

Sohn, Kihyuk, David Berthelot, Nicholas Carlini, et al. 2020. “FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence.” Advances in Neural Information Processing Systems 33, 596–608.

Vincent, Pascal, Hugo Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol. 2008. “Extracting and Composing Robust Features with Denoising Autoencoders.” International Conference on Machine Learning, 1096–103.

Zhou, Dengyong, Olivier Bousquet, Thomas Navin Lal, Jason Weston, and Bernhard Schölkopf. 2004. “Learning with Local and Global Consistency.” Advances in Neural Information Processing Systems 16, 321–28.

:::
::::
<!-- refs:end -->
