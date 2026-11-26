---
title: Fundamentos de Aprendizagem Automática
subtitle: "Class 12 — Networks with Structure: CNN, RNN, LSTM, GRU and the Transformer"
bibliography: "references.bib"
suppress-bibliography: true
---

# From the MLP to Structure

## Where We Left Off

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cblue, text width=3.4cm] (a) {\textbf{Class 05: the MLP}\\ layers, backpropagation, training};
  \node[fillbox=cgreen, text width=3.4cm, right=8mm of a] (b) {\textbf{Class 12: structure}\\ build what we know about the data into the network};
  \node[fillbox=corange, text width=2.4cm, below=7mm of b] (c) {images $\to$ CNN\\ sequences $\to$ RNN\\ tokens $\to$ transformer};
  \draw[->] (a) -- (b); \draw[->] (b) -- (c);
\end{tikzpicture}
\end{center}
```

* An MLP knows nothing about the shape of its input
* Today: three ways to tell it what the data look like

## Roadmap of the Class

| Part | Question | Methods |
|:--------------|:------------------------------------|:----------------------------------------|
| **MLP limits** | what can a dense layer not do? | shuffled pixels, shifted digits |
| **CNN** | how do we use the neighbourhood? | convolution, pooling, filters |
| **RNN** | how do we use the order? | hidden state, backpropagation through time |
| **LSTM, GRU** | how do we remember? | gates, cell state |
| **Transformer** | how do we look at everything at once? | attention, positions, blocks |
| **Choosing** | which one for which data? | cost, data size, baselines |

<!-- defines: filter, feature map, stride, padding, pooling, receptive field, hidden state, gate, token, attention, query, key, value, embedding -->

## Terminology

| term | meaning | example |
|:-------------------|:------------------------------------------|:-----------------------------|
| filter | small weight array slid over the input | a $3 \times 3$ edge detector |
| feature map | output of one filter at every position | $6 \times 6$ map |
| stride, padding | step of the filter; zeros added at the border | $S = 2$, $P = 1$ |
| pooling | summarise a block by its maximum or mean | $2 \times 2$ max |
| receptive field | input pixels that one unit sees | $5 \times 5$ after two layers |
| hidden state | the memory a recurrent layer carries | $h_t$ |
| gate | a number in $(0, 1)$ that scales a signal | forget gate |
| token | one element of a sequence | a word, an hour |
| attention | weighted average of the values | weights sum to 1 |
| query, key, value | what is asked, what is offered, what is returned | $Q$, $K$, $V$ |
| embedding | token turned into a vector | 32 numbers per word |

## Notation N12: the Critical Symbols

Other symbols are defined where they are used.

| symbol | meaning | shape | example |
|:---------------------|:-----------------------------------|:-----------------------:|:-----------|
| $F$, $P$, $S$ | filter size, padding, stride | scalar | 3, 0, 1 |
| $T$, $t$ | number of steps, step index | scalar | 24 |
| $x_t$, $h_t$ | input ($d$ values), hidden state ($u$ units) | $d$, $u$ | $u = 32$ |
| $W$, $U$ | input weights, recurrent weights | $u \times d$, $u \times u$ | |
| $i_t$, $f_t$, $o_t$, $c_t$ | LSTM gates and cell state | $u$ | 0.82, 1.62 |
| $z_t$, $r_t$ | update, reset gates (GRU) | $u$ | 0.82 |
| $Q$, $K$, $V$ | queries, keys, values (key size $d_k$) | $T \times d_k$ | $d_k = 2$ |
| $A$ | attention weights, rows sum to 1 | $T \times T$ | 0.40, 0.20 |


## Where an MLP Fails: Count the Weights

A dense layer has $(HWC + 1)\,h$ weights on an $H \times W \times C$ image.

:::: {.columns}
::: {.column width="46%"}
| image | inputs | $h = 1000$ |
|:--------------------------------------------------|:----------------|:-----------------------|
| digits $8 \times 8$ | 64 | 65 000 |
| MNIST $28 \times 28$ | 784 | 785 000 |
| photo $224 \times 224 \times 3$ | 150 528 | 150 529 000 |
:::
::: {.column width="52%"}
![](plots/12/m1_params_image.pdf)
:::
::::

* A photo: 150 million weights in one layer (Demo M1)

## Where an MLP Fails: It Ignores the Neighbours

Shuffle the pixels with one fixed permutation: the MLP loses nothing.

:::: {.columns}
::: {.column width="52%"}
![](plots/12/m2_shuffled.pdf)
:::
::: {.column width="46%"}
| digits | test accuracy |
|:-----------------------------------------------|:-------------------------------------------|
| original | 0.969 |
| pixels shuffled | 0.974 |
:::
::::

* A dense layer has no notion of "next to" (Demo M2)

## Where an MLP Fails: A Moved Digit Is a New Digit

Train on centred digits; test on digits moved by up to $s$ pixels.

:::: {.columns}
::: {.column width="46%"}
| max shift $s$ | MLP accuracy |
|:----------------------------------------------:|:--------------------------------------------:|
| 0 | 0.969 |
| 1 | 0.399 |
| 2 | 0.215 |
:::
::: {.column width="52%"}
![](plots/12/m2_shift.pdf)
:::
::::

* One pixel of shift: accuracy 0.97 to 0.40
* Every pixel has its own weights: a moved stroke is new

## Three Kinds of Structure

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cblue, text width=3.0cm] (a) {\textbf{neighbours matter}\\ images, audio\\ $\to$ \textbf{convolution}};
  \node[fillbox=cgreen, text width=3.0cm, right=5mm of a] (b) {\textbf{order matters}\\ series, text\\ $\to$ \textbf{recurrence}};
  \node[fillbox=corange, text width=3.0cm, right=5mm of b] (c) {\textbf{any pair matters}\\ long text, code\\ $\to$ \textbf{attention}};
  \node[fillbox=cgray, text width=9.6cm, below=5mm of b] (d) {the structure is a constraint on the weights: fewer parameters, the right invariance};
\end{tikzpicture}
\end{center}
```

* Each assumption removes weights and data needs
* A wrong assumption cannot be trained away

# Convolutional Networks

## The Idea: One Small Filter, Slid Everywhere

A **filter** is a small array of weights, applied at every position.

```{=latex}
\begin{center}
\begin{tikzpicture}[x=0.62cm, y=0.62cm, font=\scriptsize]
  \foreach \i/\v in {0/1, 1/2, 2/0, 3/-1, 4/3} {
    \node[draw, minimum size=0.6cm, fill=cblue!12] at (\i,0) {$\v$};
  }
  \node[left] at (-0.6,0) {$x$};
  \foreach \i/\v in {0/1, 1/3, 2/-3} {
    \node[draw, minimum size=0.6cm, fill=cgreen!15] at (\i+1,-2) {$\v$};
  }
  \node[left] at (0.4,-2) {$y$};
  \draw[thick, corange] (-0.35,-0.35) rectangle (2.35,0.35);
  \draw[->, corange, thick] (1,-0.45) -- (1,-1.55);
  \node[right, corange] at (4.8,-1) {filter $w = (1, 0, -1)$};
\end{tikzpicture}
\end{center}
```

* $y_i = \sum_{j} w_j\, x_{i+j}$; first output: $(+1)(1) + (0)(2) + (-1)(0) = 1$

## A Convolution by Hand

Signal $x = (1, 2, 0, -1, 3)$, filter $w = (1, 0, -1)$ (difference of the two neighbours).

```{=latex}
\begin{center}
\begin{tikzpicture}[x=0.55cm, y=0.55cm, font=\scriptsize]
  \foreach \i/\v in {0/1, 1/2, 2/0, 3/-1, 4/3} { \node[draw, minimum size=0.5cm, fill=cblue!12] at (\i,0) {$\v$}; }
  \draw[thick, corange] (-0.32,-0.32) rectangle (2.32,0.32);
  \draw[thick, cgreen, dashed] (0.68,-0.4) rectangle (3.32,0.4);
  \draw[thick, cred, dotted] (1.68,-0.48) rectangle (4.32,0.48);
  \foreach \i/\v/\c in {0/1/corange, 1/3/cgreen, 2/-3/cred} { \node[draw=\c, thick, minimum size=0.5cm, fill=\c!10] at (\i+1,-1.5) {$\v$}; }
  \node[left] at (-0.5,0) {$x$}; \node[left] at (0.5,-1.5) {$y$};
\end{tikzpicture}
\end{center}
```

| $i$ | window | $y_i = \sum_j w_j x_{i+j}$ | $y_i$ |
|:-------------:|:----------------------|:------------------------------------------|:-------------:|
| 1 | $(1, 2, 0)$ | $(+1)(1) + (0)(2) + (-1)(0)$ | $+1$ |
| 2 | $(2, 0, -1)$ | $(+1)(2) + (0)(0) + (-1)(-1)$ | $+3$ |
| 3 | $(0, -1, 3)$ | $(+1)(0) + (0)(-1) + (-1)(3)$ | $-3$ |

* Deep learning calls this a convolution; mathematically it is the cross-correlation (the filter is not flipped)
* The same three weights produced all three outputs (Demo C1)

## Two Dimensions: Three Filters on a Digit

The filter is now an $F \times F$ array sliding in two directions.

![](plots/12/c1_kernels.pdf)

* Vertical edge: rows $(1, 0, -1)$, $(2, 0, -2)$, $(1, 0, -1)$
* Blur: nine weights of $1/9$; an $8 \times 8$ digit gives a $6 \times 6$ map

## How Big Is the Output?

How many positions fit? Filter $F$, input $L$, $P$ zeros per side, step $S$, output length $L_{out}$.

```{=latex}
\begin{center}
\begin{tikzpicture}[x=0.42cm, y=0.42cm, font=\scriptsize]
  \foreach \i in {0,...,7} { \node[draw, minimum size=0.4cm, fill=cblue!12] at (\i,0) {}; }
  \foreach \i in {-1,8} { \node[draw, minimum size=0.4cm, fill=cgray!25] at (\i,0) {\scriptsize 0}; }
  \draw[thick, corange] (-0.35,-0.35) rectangle (1.35,0.35);
  \draw[<->] (-0.4,0.9) -- node[above] {$L + 2P$} (7.4,0.9);
  \draw[->, corange, thick] (0.5,-0.7) -- node[below] {step $S$} (2.5,-0.7);
  \node[corange, right] at (8.6,0) {filter $F$};
\end{tikzpicture}
\end{center}
```

<!-- derivation: conv-size -->
$$
\begin{aligned}
\text{last start} &= L + 2P - F && \text{(definition)} \\
\text{starts} &= 0, S, 2S, \dots, mS \le L + 2P - F && \text{(definition)} \\
L_{out} &= m + 1 && \text{(definition)}
\end{aligned}
$$

(result) $m = \lfloor (L + 2P - F)/S \rfloor$; $L = 8$, $F = 3$, $P = 0$, $S = 1$ gives $L_{out} = 6$

## Output Size: the Formula and Keras

:::: {.columns}
::: {.column width="46%"}
| $L$ | $F$ | $P$ | $S$ | $L_{out}$ | Keras |
|:--------------:|:--------------:|:--------------:|:--------------:|:------------------:|:--------------:|
| 8 | 3 | 0 | 1 | 6 | 6 |
| 8 | 3 | 1 | 1 | 8 | 8 |
| 8 | 3 | 0 | 2 | 3 | 3 |
| 28 | 5 | 0 | 1 | 24 | 24 |
| 28 | 3 | 1 | 2 | 14 | 14 |
| 224 | 7 | 3 | 2 | 112 | 112 |
:::
::: {.column width="52%"}
![](plots/12/c1_out_size.pdf)
:::
::::

* $P = (F - 1)/2$ with $S = 1$ keeps the size ("same" padding)

## Channels and Weights

$C$ channels in; $C'$ filters of size $F \times F \times C$, one bias each.

<!-- derivation: conv-params
symbols:
  C: {kind: scalar}
  C_prime: {kind: scalar}
  F: {kind: scalar}
defs:
  C_prime: "8"
  F: "3"
  C: "1"
-->
$$
\begin{aligned}
m_{conv} &= C'\,(F\,F\,C + 1) && \text{(definition)} \\
&= 8\,(3 \cdot 3 \cdot 1 + 1) && \text{(substitute)} \\
&= 80 && \text{(evaluate)}
\end{aligned}
$$

(result) 80 weights for any image size; dense: 4 160

```{=latex}
\begin{center}
\begin{tikzpicture}[x=0.5cm, y=0.5cm, font=\scriptsize]
  \foreach \k in {0,1,2} { \draw[fill=cblue!15] (\k*0.25,\k*0.25) rectangle (\k*0.25+1.6,\k*0.25+1.6); }
  \node[below] at (1,-0.2) {input $C$};
  \foreach \k in {0,1,2,3} { \draw[fill=corange!20] (4+\k*0.3,\k*0.3) rectangle (4+\k*0.3+0.9,\k*0.3+0.9); }
  \node[below] at (5,-0.2) {$C'$ filters};
  \foreach \k in {0,1,2,3} { \draw[fill=cgreen!20] (8+\k*0.3,\k*0.3) rectangle (8+\k*0.3+1.6,\k*0.3+1.6); }
  \node[below] at (9,-0.2) {output $C'$};
  \draw[->, thick] (2.4,0.9) -- (3.8,0.9); \draw[->, thick] (5.6,0.9) -- (7.8,0.9);
\end{tikzpicture}
\end{center}
```

## Same Weights Everywhere: Equivariance

If the digit moves, the feature map moves with it.

```{=latex}
\begin{center}
\begin{tikzpicture}[x=0.32cm, y=0.5cm, font=\scriptsize]
  \foreach \i/\h in {0/0.1,1/0.1,2/0.6,3/1.0,4/0.6,5/0.1,6/0.1,7/0.1,8/0.1,9/0.1} { \fill[cblue!50] (\i,0) rectangle (\i+0.8,\h); }
  \foreach \i/\h in {0/0.1,1/0.1,2/0.1,3/0.1,4/0.1,5/0.6,6/1.0,7/0.6,8/0.1,9/0.1} { \fill[corange!60] (\i,-1.6) rectangle (\i+0.8,-1.6+\h); }
  \node[left] at (-0.3,0.4) {$x$}; \node[left] at (-0.3,-1.2) {$x'$};
  \draw[->, thick, cred] (4.5,-0.2) -- node[right] {shift $s$} (6.5,-1.0);
\end{tikzpicture}
\end{center}
```

<!-- derivation: conv-equivariance -->
$$
\begin{aligned}
y_i &= \sum_j w_j\, x_{i+j} && \text{(definition)} \\
x'_i &= x_{i-s} && \text{(definition)} \\
y'_i &= \sum_j w_j\, x'_{i+j} && \text{(definition)} \\
&= \sum_j w_j\, x_{i-s+j} && \text{(definition of } x' \text{)} \\
&= y_{i-s} && \text{(identity)}
\end{aligned}
$$

(result) a shifted input gives a shifted output (Demo C2)

## Pooling: a Little Invariance

**Max pooling** keeps the largest value of each $2 \times 2$ block.

```{=latex}
\begin{center}
\begin{tikzpicture}[x=0.6cm, y=0.6cm, font=\scriptsize]
  \foreach \r/\c/\v in {0/0/1, 0/1/3, 0/2/0, 0/3/2, 1/0/4, 1/1/2, 1/2/1, 1/3/0} {
    \node[draw, minimum size=0.55cm, fill=cblue!10] at (\c,-\r) {$\v$};
  }
  \draw[thick, corange] (-0.35,0.35) rectangle (1.35,-1.35);
  \draw[thick, cgreen] (1.65,0.35) rectangle (3.35,-1.35);
  \node[draw, minimum size=0.55cm, fill=corange!25] at (6,-0.5) {$4$};
  \node[draw, minimum size=0.55cm, fill=cgreen!25] at (7,-0.5) {$2$};
  \draw[->, thick] (4,-0.5) -- node[above] {max} (5.2,-0.5);
\end{tikzpicture}
\end{center}
```

* A one-pixel shift changes the map by 0.77 before, 0.34 after (Demo C2)

## Stride and the Limit of Equivariance

Stride $S > 1$ computes only every $S$-th position.

![](plots/12/c2_stride.pdf){width=80%}

* Shift 2: mismatch 0; shift 1: mismatch 3 (Demo C2)
* A strided CNN is only approximately equivariant

## The Receptive Field Grows with Depth

A unit of layer $L$ sees $r_L = 1 + \sum_{l \le L} (F_l - 1) \prod_{m < l} S_m$ pixels.

:::: {.columns}
::: {.column width="46%"}
| layers | $r$ |
|:-----------------------------------------------------------------------|:-------------------:|
| $3 \times 3$ | 3 |
| two $3 \times 3$ | 5 |
| three $3 \times 3$ | 7 |
| $3 \times 3$, pool 2, $3 \times 3$ | 8 |
| 3, pool, 3, pool, 3 | 18 |
:::
::: {.column width="52%"}
![](plots/12/c2_receptive.pdf)
:::
::::

* Two $3 \times 3$ layers see a $5 \times 5$ patch with 18 weights, not 25
* Pooling grows the field faster (Demo C2)

## Learning the Filter: the Gradient

Write $g_i = \partial \mathcal{L} / \partial y_i$; in $y_i = \sum_{j'} w_{j'} x_{i+j'}$ only the term $j' = j$ depends on $w_j$: $\partial y_i / \partial w_j = x_{i+j}$.

```{=latex}
\begin{center}
\begin{tikzpicture}[x=0.75cm, y=0.8cm, font=\scriptsize]
  \foreach \i in {0,...,5} { \node[draw, circle, inner sep=1pt, fill=cblue!12] (x\i) at (\i,0) {\scriptsize $x_{\i}$}; }
  \foreach \i in {0,...,3} { \node[draw, circle, inner sep=1pt, fill=cgreen!15] (y\i) at (\i+1,1.3) {\scriptsize $y_{\i}$}; }
  \foreach \i in {0,...,3} {
    \pgfmathtruncatemacro{\a}{\i+1} \pgfmathtruncatemacro{\b}{\i+2}
    \draw[corange, thick] (x\i) -- (y\i);
    \draw[cgreen, thick] (x\a) -- (y\i);
    \draw[cred, thick] (x\b) -- (y\i);
  }
  \node[corange, right] at (5.3,0.9) {$w_0$}; \node[cgreen, right] at (5.3,0.55) {$w_1$}; \node[cred, right] at (5.3,0.2) {$w_2$};
\end{tikzpicture}
\end{center}
```

<!-- derivation: conv-gradient -->
$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial w_j} &= \sum_i \frac{\partial \mathcal{L}}{\partial y_i}\,\frac{\partial y_i}{\partial w_j} && \text{(chain rule)} \\
&= \sum_i g_i\, x_{i+j} && \text{(definition of the layer)}
\end{aligned}
$$

(result) a correlation; `jax.grad` agrees (Demo C2)

## A Small CNN

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize, node distance=3mm]
  \node[fillbox=cgray, text width=0.9cm] (i) {image $8{\times}8{\times}1$};
  \node[fillbox=cblue, text width=1.15cm, right=of i] (c1) {conv 3x3\\ 8 filters\\ ReLU};
  \node[fillbox=cgreen, text width=0.9cm, right=of c1] (p1) {pool 2\\ $4{\times}4{\times}8$};
  \node[fillbox=cblue, text width=1.15cm, right=of p1] (c2) {conv 3x3\\ 16 filters\\ ReLU};
  \node[fillbox=cgreen, text width=0.9cm, right=of c2] (p2) {pool 2\\ $2{\times}2{\times}16$};
  \node[fillbox=corange, text width=1.2cm, right=of p2] (d) {flatten 64\\ dense 10\\ softmax};
  \draw[flow] (i) -- (c1); \draw[flow] (c1) -- (p1); \draw[flow] (p1) -- (c2); \draw[flow] (c2) -- (p2); \draw[flow] (p2) -- (d);
\end{tikzpicture}
\end{center}
```

* 1 898 weights against 6 570 for the MLP
* Trained as in Class 05: loss, backpropagation, Adam

## What the First Layer Learns

![](plots/12/c3_filters.pdf)

* Eight $3 \times 3$ filters after training: edges and blobs
* Deeper layers combine them into strokes and loops

## MLP against CNN on the Digits

:::: {.columns}
::: {.column width="46%"}
| model | weights | test accuracy |
|:-----------------------------|:------------------------:|:--------------------------------------:|
| MLP | 6 570 | 0.969 |
| CNN | 1 898 | 0.974 |
| CNN, pool | 5 130 | 0.977 |
:::
::: {.column width="52%"}
![](plots/12/c3_accuracy.pdf)
:::
::::

* Mean of 3 seeds: the CNN matches the MLP with a third of the weights
* The difference appears when the digit moves

## Shifted Digits: the CNN Helps, Not Enough

:::: {.columns}
::: {.column width="50%"}
| shift | MLP | CNN | + pool |
|:----------------------:|:----------------------:|:----------------------:|:----------------------:|
| 0 | 0.969 | 0.974 | 0.977 |
| 1 | 0.399 | 0.567 | 0.599 |
| 2 | 0.215 | 0.294 | 0.370 |
:::
::: {.column width="48%"}
![](plots/12/c4_shift.pdf)
:::
::::

* `Flatten` keeps positions; the global pool (+ pool) drops them

## Data Augmentation: Show the Shifts

Four copies of each digit, three shifted by up to 2 pixels.

| test shift | MLP | CNN 8--16 | CNN + global pool |
|:----------------------:|:---------------:|:--------------------:|:---------------------------------:|
| 0 | 0.963 | 0.962 | 0.969 |
| 1 | 0.872 | 0.924 | 0.935 |
| 2 | 0.877 | 0.887 | 0.921 |

* Every model gains; the pooled CNN most (0.37 to 0.92 at $s = 2$)
* Augmentation is free structure

## Variants: What Matters in the Architecture

:::: {.columns}
::: {.column width="46%"}
| CNN variant | weights | $s = 0$ | $s = 1$ |
|:-----------------------------|:--------------------:|:--------------------:|:--------------------:|
| 1 filter | 180 | 0.833 | 0.388 |
| 4 filters | 690 | 0.953 | 0.465 |
| no pool | 11 498 | 0.971 | 0.422 |
| 8--16 | 1 898 | 0.970 | 0.569 |
| 16--32 | 6 090 | 0.978 | 0.534 |
:::
::: {.column width="52%"}
![](plots/12/c4_variants.pdf)
:::
::::

* One filter reads 83%; pooling beats width on shifts (Demo C4)

## From LeNet to ResNet

* **LeNet** [@lecun1998]: the design of this class
* **AlexNet** [@krizhevsky2012]: ReLU, GPUs, ImageNet 2012
* **ResNet** [@he2016]: skip connection $y = x + F(x)$
* **BatchNorm**: rescale each channel per minibatch
* **1-D convolution**: audio and series

## DEMO C1--C4: `02_convolution.ipynb`, `03_cnn_digits.ipynb`

Demos 12b, 12c: convolution by hand, CNN against MLP.

# Recurrent Networks

## Order Matters: Sequences

A **sequence**: observations whose order carries information.

```{=latex}
\begin{center}
\begin{tikzpicture}[x=0.5cm, y=0.5cm, font=\scriptsize]
  \draw[thick, cblue] (0,0.2) .. controls (1,1.2) and (2,-0.4) .. (3,0.6) .. controls (4,1.3) and (5,0) .. (6,0.5);
  \node[below] at (3,-0.5) {temperatures};
  \foreach \i/\w in {0/the, 1/cat, 2/sat} { \node[draw, fill=cgreen!15, minimum size=0.55cm, inner sep=2pt] at (8.5+2.0*\i,0.3) {\w}; }
  \node[below] at (10.5,-0.5) {words};
  \draw[thick, corange] (14,0.3) -- (14.4,1.0) -- (14.8,-0.2) -- (15.2,0.9) -- (15.6,-0.3) -- (16,0.8) -- (16.4,0.1) -- (16.8,0.5);
  \node[below] at (15.4,-0.5) {sound};
\end{tikzpicture}
\end{center}
```

| data | one element $x_t$ | question |
|:-----------------------|:--------------------------------|:-----------------------------------|
| room temperature | five readings of one hour | what in six hours? |
| text | a word | which topic? what next word? |

* An MLP on a window of $T$ values has $T\,d$ inputs
* A pattern learned at step 5 is lost at step 9

## The Hidden State

A recurrent layer reads one element at a time and keeps a **hidden state** $h_t$.

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize, node distance=7mm]
  \node[fillbox=cblue, minimum width=1.0cm] (h1) {$h_1$};
  \node[fillbox=cblue, minimum width=1.0cm, right=of h1] (h2) {$h_2$};
  \node[fillbox=cblue, minimum width=1.0cm, right=of h2] (h3) {$h_3$};
  \node[right=7mm of h3] (dots) {$\cdots$};
  \node[fillbox=cblue, minimum width=1.0cm, right=7mm of dots] (hT) {$h_T$};
  \node[below=6mm of h1] (x1) {$x_1$}; \node[below=6mm of h2] (x2) {$x_2$}; \node[below=6mm of h3] (x3) {$x_3$}; \node[below=6mm of hT] (xT) {$x_T$};
  \node[left=6mm of h1] (h0) {$h_0 = 0$};
  \node[above=6mm of hT, fillbox=corange] (y) {output};
  \draw[flow] (h0) -- (h1); \draw[flow] (h1) -- (h2); \draw[flow] (h2) -- (h3); \draw[flow] (h3) -- (dots); \draw[flow] (dots) -- (hT);
  \draw[flow] (x1) -- (h1); \draw[flow] (x2) -- (h2); \draw[flow] (x3) -- (h3); \draw[flow] (xT) -- (hT); \draw[flow] (hT) -- (y);
\end{tikzpicture}
\end{center}
```

* The **same** weights at every step

## The Recurrent Layer

<!-- derivation: rnn-layer -->
$$
\begin{aligned}
h_t &= \tanh\big(W x_t + U h_{t-1} + b\big) && \text{(definition)} \\
h_0 &= 0 && \text{(definition)}
\end{aligned}
$$

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize, node distance=9mm]
  \node[fillbox=cblue, minimum width=2.2cm] (c) {$\tanh(W x_t + U h_{t-1} + b)$};
  \node[left=of c] (x) {$x_t$}; \node[right=of c] (h) {$h_t$};
  \draw[flow] (x) -- (c); \draw[flow] (c) -- (h);
  \draw[flow, cred] (c.south) -- ++(0,-0.5) -| node[pos=0.25, below] {delay: $h_{t-1}$} (c.south west) ;
\end{tikzpicture}
\end{center}
```

* Sizes: $x_t$ is $d$, $h_t$ is $u$, $W$ is $u \times d$, $U$ is $u \times u$
* The simple RNN of Elman [@elman1990]
* Keras stores the transposes (`x @ W + h @ U`)

## An RNN by Hand

$d = 1$, $u = 2$, $W = \begin{pmatrix} 0.5 \\ -0.4 \end{pmatrix}$, $U = \begin{pmatrix} 0.3 & -0.1 \\ 0.2 & 0.4 \end{pmatrix}$, $b = 0$, input $x = (1, -1, 0.5)$.

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize, node distance=5mm]
  \node[fillbox=cblue, text width=1.9cm] (h1) {$h_1 = (0.46, -0.38)$};
  \node[fillbox=cblue, text width=1.9cm, right=of h1] (h2) {$h_2 = (-0.31, 0.33)$};
  \node[fillbox=cblue, text width=1.9cm, right=of h2] (h3) {$h_3 = (0.12, -0.13)$};
  \node[below=4mm of h1] {$x_1 = 1$}; \node[below=4mm of h2] {$x_2 = -1$}; \node[below=4mm of h3] {$x_3 = 0.5$};
  \draw[flow] (h1) -- (h2); \draw[flow] (h2) -- (h3);
\end{tikzpicture}
\end{center}
```

| $t$ | $x_t$ | $W x_t$ | $U h_{t-1}$ | $h_t = \tanh(\cdot)$ |
|:-----------:|:-----------:|:---------------------:|:-----------------------:|:-------------------------:|
| 1 | $+1.0$ | $(0.50, -0.40)$ | $(0, 0)$ | $(0.462, -0.380)$ |
| 2 | $-1.0$ | $(-0.50, 0.40)$ | $(0.177, -0.060)$ | $(-0.313, 0.328)$ |
| 3 | $+0.5$ | $(0.25, -0.20)$ | $(-0.127, 0.069)$ | $(0.123, -0.131)$ |

* Keras `SimpleRNN`, same weights: difference $5 \cdot 10^{-8}$ (Demo R1)

## How Many Weights?

```{=latex}
\begin{center}
\begin{tikzpicture}[x=0.5cm, y=0.5cm, font=\scriptsize]
  \foreach \r in {0,1} { \node[draw, minimum size=0.5cm, fill=cblue!20] at (0,-\r) {}; }
  \foreach \r in {0,1} { \foreach \c in {0,1} { \node[draw, minimum size=0.5cm, fill=cgreen!20] at (\c+2.5,-\r) {}; } }
  \foreach \r in {0,1} { \node[draw, minimum size=0.5cm, fill=corange!25] at (5.5,-\r) {}; }
  \node at (0,0.9) {$W$}; \node at (3,0.9) {$U$}; \node at (5.5,0.9) {$b$};
  \node[below] at (0,-1.4) {$u \times d$}; \node[below] at (3,-1.4) {$u \times u$}; \node[below] at (5.5,-1.4) {$u$};
\end{tikzpicture}
\end{center}
```

<!-- derivation: rnn-params
defs:
  u: "2"
  d: "1"
-->
$$
\begin{aligned}
m_{rnn} &= u \cdot d + u \cdot u + u && \text{(definition)} \\
&= u \cdot (d + u + 1) && \text{(factor)} \\
&= 2 \cdot (1 + 2 + 1) && \text{(substitute)} \\
&= 8 && \text{(evaluate)}
\end{aligned}
$$

(result) 8 weights here; $u = 32$, $d = 5$: 1 216 for any $T$ (Demo R1)

## What a Recurrent Net Can Output

```{=latex}
\begin{center}
\begin{tikzpicture}[x=0.45cm, y=0.55cm, font=\scriptsize]
  % many to one
  \foreach \i in {0,...,3} { \node[dot=cblue] (a\i) at (\i,0) {}; }
  \node[dot=corange] (ao) at (3,2) {};
  \foreach \i in {0,...,3} { \draw[->, cgray] (a\i) -- (ao); }
  \node[below] at (1.5,-0.3) {many to one};
  % many to many
  \foreach \i in {0,...,3} { \node[dot=cblue] (b\i) at (\i+6,0) {}; \node[dot=corange] (bo\i) at (\i+6,2) {}; \draw[->, cgray] (b\i) -- (bo\i); }
  \node[below] at (7.5,-0.3) {many to many};
  % one to many
  \node[dot=cblue] (c0) at (12,0) {};
  \foreach \i in {0,...,3} { \node[dot=corange] (co\i) at (\i+12,2) {}; \draw[->, cgray] (c0) -- (co\i); }
  \node[below] at (13.5,-0.3) {one to many};
\end{tikzpicture}
\end{center}
```

| task | example |
|:---------------------|:---------------------------------------------------------------------|
| many to one | topic of a text; the temperature in 6 hours |
| many to many | label every word; next-word prediction ($x_{1..t} \to x_{t+1}$) |
| one to many | caption of an image |

* Our demos are many to one

## Training: Backpropagation Through Time

Unroll into $T$ layers sharing their weights; $a_t = W x_t + U h_{t-1} + b$.

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize, node distance=7mm]
  \node[fillbox=cgray] (hp) {$h_{t-1}$};
  \node[fillbox=corange, right=of hp] (a) {$a_t$};
  \node[fillbox=cblue, right=of a] (h) {$h_t$};
  \node[above=2mm of a] (x) {$x_t$};
  \draw[flow] (hp) -- node[above] {$U$} (a); \draw[flow] (x) -- node[left] {$W$} (a); \draw[flow] (a) -- node[above] {$\tanh$} (h);
  \draw[flow, cred] (h.south) to[out=-90, in=-90, looseness=0.6] node[below, pos=0.5] {$\partial h_t / \partial h_{t-1}$} (hp.south);
\end{tikzpicture}
\end{center}
```

<!-- derivation: bptt-step -->
$$
\begin{aligned}
\frac{\partial h_t}{\partial h_{t-1}} &= \frac{\partial \tanh(a_t)}{\partial a_t}\,\frac{\partial a_t}{\partial h_{t-1}} && \text{(chain rule)} \\
&= \mathrm{diag}\big(1 - h_t^2\big)\, U && \text{(identity)}
\end{aligned}
$$

(result) $\tanh'(a) = 1 - \tanh^2(a) \in (0, 1]$

## Gradient Through Many Steps

$\partial h_T / \partial h_t$: the product of the one-step factors.

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize, node distance=9mm]
  \node[fillbox=cgray] (h1) {$h_t$}; \node[fillbox=cgray, right=of h1] (h2) {$h_{t+1}$}; \node[right=6mm of h2] (d) {$\cdots$}; \node[fillbox=cblue, right=6mm of d] (hT) {$h_T$};
  \draw[flow] (h1) -- (h2); \draw[flow] (h2) -- (d); \draw[flow] (d) -- (hT);
  \draw[<-, thick, cred] ([yshift=3mm]h1.north) -- node[above] {product of $T - t$ factors} ([yshift=3mm]hT.north);
\end{tikzpicture}
\end{center}
```

<!-- derivation: bptt-product -->
$$
\begin{aligned}
\frac{\partial h_T}{\partial h_t} &= \frac{\partial h_T}{\partial h_{T-1}} \cdots \frac{\partial h_{t+1}}{\partial h_t} && \text{(chain rule)} \\
&= \prod_{k=t+1}^{T} \mathrm{diag}\big(1 - h_k^2\big)\, U && \text{(identity)}
\end{aligned}
$$

(result) the lag is the number of factors; `jax.jacobian` gives 0.115465 both ways (Demo R2)

## Why the Gradient Vanishes or Explodes

Norms: $\lVert \mathrm{diag}(1 - h^2) \rVert \le 1$; $\sigma_{\max}$ is the largest singular value of $U$.

<!-- derivation: bptt-bound -->
$$
\begin{aligned}
\Big\lVert \frac{\partial h_T}{\partial h_t} \Big\rVert &\le \prod_{k=t+1}^{T} \big\lVert \mathrm{diag}(1 - h_k^2) \big\rVert\, \lVert U \rVert && \text{(assumption)} \\
&\le \big(1 \cdot \sigma_{\max}\big)^{T - t} && \text{(assumption)}
\end{aligned}
$$

(result) $\sigma_{\max} < 1$: the bound shrinks, the gradient vanishes; $\sigma_{\max} > 1$: explosion is possible

```{=latex}
\begin{center}
\begin{tikzpicture}[x=0.5cm, y=0.55cm, font=\scriptsize]
  \foreach \k in {0,...,7} { \fill[cgreen!60] (\k*0.7,0) rectangle (\k*0.7+0.5,{0.8^\k}); \fill[cred!60] (\k*0.7+7,0) rectangle (\k*0.7+7.5,{0.35*1.25^\k}); }
  \node[below] at (2.5,0) {$\sigma_{\max} < 1$: shrinks}; \node[below] at (9.5,0) {$\sigma_{\max} > 1$: grows};
\end{tikzpicture}
\end{center}
```

## Vanishing and Exploding: Numbers

:::: {.columns}
::: {.column width="46%"}
| lag | $\rho = 0.5$ | 0.9 | 1.1 | 1.5 |
|:--------------:|:------------------------:|:-----------------------:|:--------------:|:--------------:|
| 1 | 0.5 | 0.90 | 1.10 | 1.50 |
| 5 | 0.026 | 0.39 | 0.94 | 3.64 |
| 10 | 0.001 | 0.12 | 0.62 | 5.18 |
| 20 | $<10^{-3}$ | 0.009 | 0.19 | 6.34 |
| 40 | $\approx 0$ | $\approx 0$ | 0.002 | 1.29 |
:::
::: {.column width="52%"}
![](plots/12/r2_gradient_lag.pdf)
:::
::::

* $U = \rho Q$, $Q$ orthogonal, 32 units: measured $\lVert \partial h_T/\partial h_t \rVert$ for four values of $\rho$
* $\rho = 1.5$ explodes first, then $\tanh$ saturates and it falls: the bound is not tight

## Remedy 1: Clip the Gradient

If $\lVert g \rVert > c$, rescale to norm $c$: $g \leftarrow g \min(1, c / \lVert g \rVert)$.

:::: {.columns}
::: {.column width="46%"}
| SGD lr 0.5, lag 20 | final loss | accuracy |
|:-----------------------------------------|:--------------------------:|:----------------------:|
| no clipping | 2.94 | 0.25 |
| clip norm 1 | 1.03 | 0.52 |
:::
::: {.column width="52%"}
![](plots/12/r2_clipping.pdf)
:::
::::

* Cures explosion [@pascanu2013], not vanishing

## DEMO R1--R2: `04_rnn.ipynb`

Demo 12d: the RNN by hand, gradient against the lag, clipping.

# LSTM and GRU

## A Task That Needs Memory

Step 1 holds one of four symbols, then $T$ steps of noise; answer the symbol (chance 0.25).

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize, node distance=4mm]
  \node[fillbox=corange, minimum width=1.0cm] (a) {symbol};
  \node[fillbox=cgray, minimum width=0.9cm, right=of a] (n1) {noise};
  \node[fillbox=cgray, minimum width=0.9cm, right=of n1] (n2) {noise};
  \node[right=3mm of n2] (d) {$\cdots$};
  \node[fillbox=cgray, minimum width=0.9cm, right=3mm of d] (n3) {noise};
  \node[fillbox=cgreen, minimum width=1.0cm, right=8mm of n3] (y) {answer};
  \draw[->, thick, cred] (a.north) to[out=50, in=130] node[above] {lag $T$} (y.north);
\end{tikzpicture}
\end{center}
```

* Only the distance changes with the lag
* A vanilla RNN carries the symbol through $T$ Jacobian products

## The Idea: Write, Keep, Read

The LSTM adds a **cell state** $c_t$. **Gates** scale signals by $(0, 1)$.

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize, node distance=6mm]
  \node[fillbox=cblue, text width=1.6cm] (f) {forget $f_t$\\ keep how much?};
  \node[fillbox=cgreen, text width=1.6cm, right=of f] (i) {input $i_t$\\ write how much?};
  \node[fillbox=corange, text width=1.6cm, right=of i] (o) {output $o_t$\\ read how much?};
  \node[fillbox=cgray, text width=7.0cm, above=5mm of i] (c) {cell state $c_t$: a conveyor belt};
  \draw[flow] (f) -- (c); \draw[flow] (i) -- (c); \draw[flow] (c) to[out=0, in=90] (o);
\end{tikzpicture}
\end{center}
```

* Each gate is a sigmoid of $x_t$ and $h_{t-1}$

## LSTM: the Equations

Each gate reads $x_t$ and $h_{t-1}$ ($[\cdot, \cdot]$ joins them):

<!-- derivation: lstm-def -->
$$
\begin{aligned}
i_t, f_t, o_t &= \sigma\big(W_\bullet [x_t, h_{t-1}] + b_\bullet\big) && \text{(let)} \\
g_t &= \tanh\big(W_g [x_t, h_{t-1}] + b_g\big) && \text{(let)} \\
c_t &= f_t \odot c_{t-1} + i_t \odot g_t && \text{(let)} \\
h_t &= o_t \odot \tanh(c_t) && \text{(let)}
\end{aligned}
$$

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize, node distance=5mm]
  \node[fillbox=cgray] (x) {$[x_t, h_{t-1}]$};
  \node[fillbox=cblue, right=of x] (g) {gates $i, f, o, g$};
  \node[fillbox=cgreen, right=of g] (c) {$c_t$};
  \node[fillbox=corange, right=of c] (h) {$h_t$};
  \draw[flow] (x) -- (g); \draw[flow] (g) -- (c); \draw[flow] (c) -- (h);
\end{tikzpicture}
\end{center}
```

* $g_t$ is the candidate; $\odot$ is element-wise [@hochreiter1997]

## One LSTM Step by Hand

$d = u = 1$, $h_{t-1} = 0.5$, $c_{t-1} = 1$, $x_t = 1$.

:::: {.columns}
::: {.column width="46%"}
| quantity | value |
|:---------------------------------------------------------------|:---------------------------:|
| $i_t = \sigma(1.5)$ | 0.818 |
| $f_t = \sigma(2.0)$ | 0.881 |
| $g_t = \tanh(1.5)$ | 0.905 |
| $o_t = \sigma(1.0)$ | 0.731 |
:::
::: {.column width="52%"}
![](plots/12/g1_lstm_step.pdf)
:::
::::

* $c_t = 0.881 \cdot 1.0 + 0.818 \cdot 0.905 = 1.621$; $h_t = 0.731 \tanh(1.621) = 0.676$
* Keras `LSTM`, same weights: difference $2 \cdot 10^{-8}$ (Demo G1)

## Why the Cell State Keeps the Gradient

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize, node distance=10mm]
  \node (a) {$c_{t-1}$};
  \node[draw, circle, fill=cblue!15, inner sep=1pt, right=of a] (m) {$\times f_t$};
  \node[draw, circle, fill=cgreen!15, inner sep=1pt, right=of m] (p) {$+$};
  \node[right=of p] (b) {$c_t$};
  \node[above=5mm of p] (w) {$i_t \odot g_t$};
  \draw[flow, thick, cblue] (a) -- (m); \draw[flow, thick, cblue] (m) -- (p); \draw[flow, thick, cblue] (p) -- (b); \draw[flow] (w) -- (p);
\end{tikzpicture}
\end{center}
```

<!-- derivation: lstm-cell-gradient -->
$$
\begin{aligned}
c_t &= f_t \odot c_{t-1} + i_t \odot g_t && \text{(definition)} \\
\frac{\partial c_t}{\partial c_{t-1}} &= \mathrm{diag}(f_t) && \text{(identity)} \\
\frac{\partial c_T}{\partial c_t} &= \prod_{k=t+1}^{T} \mathrm{diag}(f_k) && \text{(chain rule)}
\end{aligned}
$$

(result) along $c$ only gates in $(0, 1)$ multiply: $f \approx 1$ keeps the gradient

## Forget Gates and the Lag

:::: {.columns}
::: {.column width="46%"}
| constant $f$ | lag 20 | lag 50 |
|:----------------------------------:|:---------------------------:|:-----------------------------:|
| 0.5 | $10^{-6}$ | $10^{-15}$ |
| 0.9 | 0.122 | 0.005 |
| 0.95 | 0.358 | 0.077 |
| 0.99 | 0.818 | 0.605 |
| 1.0 | 1 | 1 |
:::
::: {.column width="52%"}
![](plots/12/g1_forget.pdf)
:::
::::

* The layer can learn $f \approx 1$ where it must remember
* Keras starts the forget bias at 1 [@gers2000]

## GRU: Two Gates

The GRU [@cho2014]: $h_t$ blends the old state and a candidate.

<!-- derivation: gru-def -->
$$
\begin{aligned}
z_t &= \sigma(W_z [x_t, h_{t-1}] + b_z) && \text{(let)} \\
r_t &= \sigma(W_r [x_t, h_{t-1}] + b_r) && \text{(let)} \\
\tilde{h}_t &= \tanh\big(W_h x_t + r_t \odot (U_h h_{t-1}) + b_h\big) && \text{(let)} \\
h_t &= z_t \odot h_{t-1} + (1 - z_t) \odot \tilde{h}_t && \text{(let)}
\end{aligned}
$$

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize, node distance=5mm]
  \node[fillbox=cgray] (hp) {$h_{t-1}$};
  \node[fillbox=cblue, right=of hp] (z) {gates $z, r$};
  \node[fillbox=corange, right=of z] (c) {$\tilde{h}_t$};
  \node[fillbox=cgreen, right=of c] (h) {$h_t$};
  \draw[flow] (hp) -- (z); \draw[flow] (z) -- (c); \draw[flow] (c) -- (h);
  \draw[flow, cgray] (hp.north) to[out=50, in=130] node[above] {copy if $z = 1$} (h.north);
\end{tikzpicture}
\end{center}
```

* $z_t$ keeps old state; $r_t$ gates what the candidate reads

## One GRU Step by Hand

$d = u = 1$, $h_{t-1} = 0.5$, $x_t = 1$, all weights 1, biases 0.

:::: {.columns}
::: {.column width="46%"}
| quantity | value |
|:-----------------------------------------------------------------|:-------------------------:|
| update $z_t$ | 0.818 |
| reset $r_t$ | 0.818 |
| candidate $\tilde{h}_t$ | 0.887 |
| state $h_t$ | 0.571 |
:::
::: {.column width="52%"}
![](plots/12/g1_gru_step.pdf)
:::
::::

* $h_t = 0.818 \cdot 0.5 + 0.182 \cdot 0.887$
* Keras `GRU`: difference $3 \cdot 10^{-8}$ (Demo G1)

## Counting Weights: RNN, LSTM, GRU

Four weight sets for the LSTM, three for the GRU.

<!-- derivation: gate-params -->
$$
\begin{aligned}
m_{lstm} &= 4\,u\,(d + u + 1) && \text{(definition)} \\
m_{gru} &= 3\,(u\,d + u\,u + 2u) && \text{(definition)}
\end{aligned}
$$

(result) $u = 16$, $d = 5$: 1 408 and 1 104 (plus 68 for the head)

![](plots/12/r1_params.pdf){width=40%}


## Memory Against the Lag

:::: {.columns}
::: {.column width="46%"}
| lag | RNN | LSTM | GRU |
|:----------------------:|:----------------------:|:----------------------:|:----------------------:|
| 10 | 1.00 | 1.00 | 1.00 |
| 20 | 1.00 | 1.00 | 1.00 |
| 40 | 0.26 | 1.00 | 0.51 |
| 60 | 0.24 | 0.75 | 0.25 |
:::
::: {.column width="52%"}
![](plots/12/g2_memory_lag.pdf)
:::
::::

* Mean of 3 seeds, 32 units; chance 0.25
* The LSTM lasts longest; all fail at some lag (Demo G2)

## What the Gates Actually Do

:::: {.columns}
::: {.column width="52%"}
![](plots/12/g2_gates.pdf)
:::
::: {.column width="46%"}
* LSTM, lag 20, accuracy 1.00
* Mean forget gate 0.76, input gate 0.52
* Best unit: forget gates multiply to 0.028
:::
::::

* Gates are not clean switches: look before you explain

## Using Them in Practice

* **Stack** layers; **bidirectional** if the whole input is known
* **Dropout**; clip the gradient at norm 1
* **LSTM or GRU**: try both; the GRU is smaller
* **Seq2seq** [@bahdanau2015]: one vector is the bottleneck
* Still sequential: no parallel steps

## Forecasting a Room Temperature: the Protocol

Room 1, six hours ahead (Demo G3).

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize, x=0.1cm, y=0.5cm]
  \fill[cblue!20] (0,0) rectangle (70,1); \fill[corange!30] (70,0) rectangle (85,1); \fill[cgreen!25] (85,0) rectangle (100,1);
  \node at (35,0.5) {train 70\% (2 282 windows)}; \node at (77.5,0.5) {valid.}; \node at (92.5,0.5) {test};
  \draw[->, thick] (0,-0.3) -- (100,-0.3) node[right] {time};
  \draw[thick, cred] (30,1.2) rectangle (36,1.9) node[right] {window: 24 h};
\end{tikzpicture}
\end{center}
```

* Input: 24 hours of five temperatures; target: the change of room 1
* Chronological split; metric: MAE in °C on the test part
* Baselines: persistence, yesterday, ridge on the window

## Forecasting: Does a Recurrent Net Win?

:::: {.columns}
::: {.column width="46%"}
| model | test MAE (°C) |
|:-----------------------------------------------|:-------------------------------------------:|
| persistence | 0.450 |
| same hour yesterday | 0.430 |
| ridge on the window | **0.358** |
| RNN, 16 units | $0.422 \pm 0.046$ |
| LSTM, 16 units | $0.854 \pm 0.056$ |
| GRU, 16 units | $0.909 \pm 0.061$ |
:::
::: {.column width="52%"}
![](plots/12/g3_forecast.pdf)
:::
::::

* Three seeds; no recurrent model beats ridge
* LSTM: 0.41 °C on training, 0.85 °C on test
* The test hours are 2.5 standard deviations warmer than the training mean
* A linear model extrapolates; a saturating network does not

## DEMO G1--G3: `05_lstm_gru.ipynb`

Demo 12e: LSTM and GRU steps by hand, memory against the lag, gates, a room forecast.

## A Nonlinear Series: Lorenz

The room temperature is almost linear, so ridge wins. A nonlinear system is where recurrence pays off (Demo F1).

![](plots/12/f1_series.pdf){width=70%}

* Lorenz system, three coupled equations; we see only $x$, plus noise
* Target: $x$ ten steps ahead; 20 000 points, chronological split
* Metric: RMSE (the standard deviation of $x$ is 1)

## Recurrent Layers on the Lorenz Series

:::: {.columns}
::: {.column width="46%"}
| model | test RMSE |
|:----------------------------------------|:--------------------------------------------------:|
| persistence | 1.342 |
| ridge on the window | 0.992 |
| MLP on the window | $0.499 \pm 0.019$ |
| RNN, 32 units | $0.408 \pm 0.016$ |
| LSTM, 32 units | $0.406 \pm 0.032$ |
| **GRU, 32 units** | $\mathbf{0.388} \pm 0.012$ |
:::
::: {.column width="52%"}
![](plots/12/f1_forecast.pdf)
:::
::::

* Ridge explains almost nothing; every network halves the error
* The three recurrent layers are close: the seed spread is as large as the gap

## Freeze the Encoders, Train an MLP Head

Three trained recurrent layers see the same window in three ways (Demo F2).

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize, node distance=6mm]
  \node[fillbox=cgray] (x) {window};
  \node[fillbox=cblue, right=12mm of x, yshift=8mm] (r) {RNN, 32};
  \node[fillbox=cgreen, right=12mm of x] (l) {LSTM, 32};
  \node[fillbox=corange, right=12mm of x, yshift=-8mm] (g) {GRU, 32};
  \node[fillbox=cgray, right=12mm of l] (c) {concat, 96};
  \node[fillbox=cred, right=of c] (h) {Dense 32};
  \node[fillbox=cgray, right=of h] (o) {$\hat y$};
  \draw[flow] (x) -- (r); \draw[flow] (x) -- (l); \draw[flow] (x) -- (g);
  \draw[flow] (r) -- (c); \draw[flow] (l) -- (c); \draw[flow] (g) -- (c);
  \draw[flow] (c) -- (h); \draw[flow] (h) -- (o);
  \node[below=3mm of g] {frozen: not trained again};
  \node[below=3mm of h, yshift=-8mm] {trained: 3 137 weights};
\end{tikzpicture}
\end{center}
```

* Each encoder is a trained regressor without its last `Dense` layer
* `encoder.trainable = False`: Keras keeps its 8 800 weights fixed
* The head sees 96 numbers per window and learns how to combine them

## Stacked Model: Result

The three encoders are frozen; the last two rows train a head on their 96 features.

:::: {.columns}
::: {.column width="46%"}
| model | test RMSE |
|:-----------------------------------------|:-------------------------------------------------:|
| GRU (best) | $0.388 \pm 0.012$ |
| average | $0.367 \pm 0.014$ |
| linear head | $0.360 \pm 0.006$ |
| **MLP head** | $\mathbf{0.345} \pm 0.004$ |
:::
::: {.column width="52%"}
![](plots/12/f2_stack.pdf)
:::
::::

* The head beats the best single layer and the average
* The hidden layer matters: 0.360 with a linear head
* Encoder weights unchanged (Demo F2)

## DEMO F1--F2: `08_forecast_stack.ipynb`

Demo 12h: the Lorenz series, three recurrent layers, frozen encoders with an MLP head.

# Attention

## The Bottleneck

A recurrent layer squeezes everything it has read into $h_t$.

* Step 1 reaches step 100 through 99 products
* Old and new content compete for $u$ numbers
* **Attention**: every step looks back at every other step
* Step 100 reads step 1 in one hop

## Attention Is a Soft Lookup

A dictionary returns the value of the matching key. Attention returns a **weighted average**.

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize, node distance=5mm]
  \node[fillbox=cblue, text width=1.3cm] (q) {query $q$\\ what I need};
  \node[fillbox=cgreen, text width=1.3cm, right=14mm of q] (k) {keys $K$\\ what each offers};
  \node[fillbox=corange, text width=1.3cm, right=10mm of k] (v) {values $V$\\ what each holds};
  \node[fillbox=cgray, text width=2.3cm, below=6mm of k] (w) {weights $A$\\ softmax of $q \cdot k_j$};
  \draw[flow] (q) -- (k); \draw[flow] (k) -- (w); \draw[flow] (w) -| (v);
  \node[note, below=2mm of w] {output $= \sum_j A_j v_j$};
\end{tikzpicture}
\end{center}
```

* Soft lookup: every key contributes, in proportion to its match

## Query, Key, Value

The tokens are the rows of $X$ ($T \times d$). Each is projected three times: what it asks, offers, returns.

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize, node distance=7mm]
  \node[fillbox=cgray] (x) {$X$};
  \node[fillbox=cblue, right=14mm of x, yshift=8mm] (q) {$Q = XW_Q$};
  \node[fillbox=cgreen, right=14mm of x] (k) {$K = XW_K$};
  \node[fillbox=corange, right=14mm of x, yshift=-8mm] (v) {$V = XW_V$};
  \draw[flow] (x) -- (q); \draw[flow] (x) -- (k); \draw[flow] (x) -- (v);
\end{tikzpicture}
\end{center}
```

* $W_Q$, $W_K$ are $d \times d_k$; $W_V$ is $d \times d_v$; all learned
* Match of token $i$ with token $j$: the dot product $q_i \cdot k_j$
* Self-attention: $Q$, $K$, $V$ come from the same sequence

## Scaled Dot-Product Attention

<!-- derivation: attention-def -->
$$
\begin{aligned}
A &= \mathrm{softmax}\Big(\frac{Q K^\top}{\sqrt{d_k}}\Big) && \text{(definition)} \\
\text{output} &= A\,V && \text{(definition)}
\end{aligned}
$$

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize, node distance=5mm]
  \node[fillbox=cblue] (s) {$Q K^\top$};
  \node[fillbox=cgray, right=of s] (d) {$/ \sqrt{d_k}$};
  \node[fillbox=cgreen, right=of d] (m) {softmax};
  \node[fillbox=corange, right=of m] (v) {$\times V$};
  \draw[flow] (s) -- (d); \draw[flow] (d) -- (m); \draw[flow] (m) -- node[above] {$A$} (v);
\end{tikzpicture}
\end{center}
```

* Row $i$ of $QK^\top$: how well token $i$ matches every token
* Softmax per row: weights sum to 1
* Shapes: $Q$, $K$ are $T \times d_k$; $A$ is $T \times T$

## Attention by Hand

$X = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}$, $W_Q = W_K = W_V = I$, $d_k = 2$ (three tokens, two numbers each).

| step | result for token 1 |
|:-------------------------------------------|:-----------------------------------------------|
| scores $QK^\top$ | row $(1, 0, 1)$ |
| divide by $\sqrt{2}$ | $(0.707, 0, 0.707)$ |
| softmax | $(0.401, 0.198, 0.401)$ |
| output | $(0.802, 0.599)$ |

![](plots/12/t1_attention_hand.pdf){width=50%}

* Token 3 attends most to itself (0.50); Keras agrees to $10^{-7}$ (Demo T1)

## Why Divide by $\sqrt{d_k}$

<!-- derivation: attention-scale -->
$$
\begin{aligned}
\mathrm{Var}(q \cdot k) &= \sum_{j=1}^{d_k} \mathrm{Var}(q_j k_j) && \text{(assumption)} \\
&= d_k && \text{(assumption)} \\
\mathrm{Var}\Big(\frac{q \cdot k}{\sqrt{d_k}}\Big) &= \frac{\mathrm{Var}(q \cdot k)}{d_k} && \text{(identity)} \\
&= 1 && \text{(from row 2)}
\end{aligned}
$$

(result) independent entries, mean 0, variance 1: scores keep std 1

```{=latex}
\begin{center}
\begin{tikzpicture}[x=0.45cm, y=0.75cm, font=\scriptsize]
  \draw[thick, cred, domain=-8:8, samples=60] plot (\x, {0.25*exp(-\x*\x/32)});
  \draw[thick, cblue, domain=-8:8, samples=60] plot (\x, {exp(-\x*\x/2)});
  \node[cred, right] at (4,0.15) {$q \cdot k$}; \node[cblue, right] at (1,0.8) {$q \cdot k / \sqrt{d_k}$};
\end{tikzpicture}
\end{center}
```

## The Scale, Measured

:::: {.columns}
::: {.column width="46%"}
| $d_k$ | std of $q \cdot k$ | largest weight |
|:-----------------|:----------------------------------------:|:---------------------------------:|
| 4 | 2.0 | 0.50 |
| 64 | 7.9 | 0.87 |
| 1024 | 32.1 | 0.97 |
| scaled | 1.0 | 0.31 |
:::
::: {.column width="52%"}
![](plots/12/t2_scaling.pdf)
:::
::::

* Unscaled, the largest weight tends to 1 and the gradient vanishes

## Masks: Do Not Look at the Future

Predicting the next token must not read it: scores with $j > i$ become $-\infty$.

The mask adds $0$ where $j \le i$ and $-\infty$ where $j > i$.

![](plots/12/t1_mask.pdf){width=50%}

* A **padding mask** does the same for the empty positions

## Self-Attention Does Not Know the Order

Shuffle the tokens: the output is the same rows, shuffled (a **set**).

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize, node distance=4mm]
  \node[fillbox=cgray] (x) {$X$}; \node[fillbox=cblue, right=of x] (a) {attention}; \node[fillbox=cgreen, right=of a] (y) {$Y$};
  \node[fillbox=cgray, right=12mm of y] (px) {$P X$}; \node[fillbox=cblue, right=of px] (a2) {attention}; \node[fillbox=cgreen, right=of a2] (py) {$P Y$};
  \draw[flow] (x) -- (a); \draw[flow] (a) -- (y); \draw[flow] (px) -- (a2); \draw[flow] (a2) -- (py);
\end{tikzpicture}
\end{center}
```

<!-- derivation: attention-permutation -->
$$
\begin{aligned}
Q' &= P\,Q, \quad K' = P\,K, \quad V' = P\,V && \text{(definition)} \\
A' &= \mathrm{softmax}\big(P\,Q K^\top P^\top / \sqrt{d_k}\big) = P\,A\,P^\top && \text{(identity)} \\
A'\,V' &= P\,A\,P^\top P\,V && \text{(from rows 1 and 2)} \\
&= P\,(A V) && \text{(identity)}
\end{aligned}
$$

(result) measured $6.7 \cdot 10^{-16}$ without positions, $1.72$ with (Demo T2)

## Positional Encoding

Token $p$ gets a vector $PE_p$ [@vaswani2017]:

<!-- derivation: positional-encoding -->
$$
\begin{aligned}
PE_{p,2i} &= \sin\big(p / 10000^{2i/d}\big) && \text{(definition)} \\
PE_{p,2i+1} &= \cos\big(p / 10000^{2i/d}\big) && \text{(definition)}
\end{aligned}
$$

![](plots/12/t2_positional.pdf)

* $PE_p \cdot PE_{p+k}$ depends only on $k$ (Demo T2)

# The Transformer

## Many Heads, Same Cost

**Multi-head attention**: $H$ heads in parallel, each of size $d_k = d / H$, then joined.

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize, node distance=5mm]
  \node[fillbox=cgray] (x) {$X$};
  \node[fillbox=cblue, right=8mm of x, yshift=6mm] (h1) {head 1};
  \node[fillbox=cgreen, right=8mm of x, yshift=-6mm] (h2) {head $H$};
  \node[fillbox=corange, right=12mm of x] (c) {concat};
  \node[fillbox=cgray, right=of c] (o) {$\times W_O$};
  \draw[flow] (x) -- (h1); \draw[flow] (x) -- (h2); \draw[flow] (h1) -- (c); \draw[flow] (h2) -- (c); \draw[flow] (c) -- (o);
\end{tikzpicture}
\end{center}
```

<!-- derivation: multihead-params -->
$$
\begin{aligned}
m_{qkv} &= H \cdot 3\,d\,\frac{d}{H} && \text{(definition)} \\
&= 3\,d^2 && \text{(simplify)} \\
m_{attn} &= 3\,d^2 + d^2 && \text{(definition)}
\end{aligned}
$$

(result) $d_k = d / H$: $4 d^2$ for any $H$, 4 096 weights at $d = 32$

## What Different Heads Can Do

* One head follows the previous token, one the first token...
* Each head has its own $W_Q$, $W_K$
* Outputs are joined and mixed by $W_O$
* We use $H = 2$; the largest models 32 to 128

## The Feed-Forward Layer and the Skip

Attention mixes the tokens; a small MLP ($d \to 4d \to d$) then works on each one.

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize, node distance=7mm]
  \node (x) {$x$}; \node[fillbox=cblue, right=of x] (f) {sub-layer $F$}; \node[draw, circle, inner sep=1pt, right=of f] (p) {$+$}; \node[fillbox=cgreen, right=of p] (n) {LayerNorm};
  \draw[flow] (x) -- (f); \draw[flow] (f) -- (p); \draw[flow] (p) -- (n);
  \draw[flow, cred] (x) to[out=60, in=120] (p);
\end{tikzpicture}
\end{center}
```

<!-- derivation: block-def -->
$$
\begin{aligned}
x' &= \mathrm{LayerNorm}\big(x + \mathrm{Attention}(x)\big) && \text{(let)} \\
y &= \mathrm{LayerNorm}\big(x' + W_2\,\mathrm{ReLU}(W_1 x' + b_1) + b_2\big) && \text{(let)}
\end{aligned}
$$

* Skip $x + F(x)$ [@he2016]; LayerNorm [@ba2016] standardises each token

## The Transformer Block

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize, node distance=4mm]
  \node[fillbox=cgray, text width=3.4cm] (x) {tokens $+$ positions ($T \times d$)};
  \node[fillbox=cblue, text width=3.4cm, above=of x] (a) {multi-head self-attention};
  \node[fillbox=cgreen, text width=3.4cm, above=of a] (n1) {add \& LayerNorm};
  \node[fillbox=corange, text width=3.4cm, above=of n1] (f) {feed-forward $d \to 4d \to d$};
  \node[fillbox=cgreen, text width=3.4cm, above=of f] (n2) {add \& LayerNorm};
  \draw[flow] (x) -- (a); \draw[flow] (a) -- (n1); \draw[flow] (n1) -- (f); \draw[flow] (f) -- (n2);
  \draw[flow, cred] (x.east) -- ++(2.2,0) |- (n1.east);
  \draw[flow, cred] (n1.west) -- ++(-2.2,0) |- (n2.west);
  \node[right=1.0cm of n2] {$\times N$ blocks};
\end{tikzpicture}
\end{center}
```

## Weights of One Block

Attention $4d^2 + 4d$; feed-forward $8d^2 + 5d$; two norms $4d$.

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize, node distance=3mm]
  \node[fillbox=cblue, text width=1.9cm] (a) {attention\\ $4d^2 + 4d$};
  \node[fillbox=cgreen, text width=1.0cm, right=of a] (n1) {norm\\ $2d$};
  \node[fillbox=corange, text width=1.9cm, right=of n1] (f) {feed-forward\\ $8d^2 + 5d$};
  \node[fillbox=cgreen, text width=1.0cm, right=of f] (n2) {norm\\ $2d$};
  \draw[flow] (a) -- (n1); \draw[flow] (n1) -- (f); \draw[flow] (f) -- (n2);
\end{tikzpicture}
\end{center}
```

<!-- derivation: block-params -->
$$
\begin{aligned}
m_{block} &= (4\,d^2 + 4\,d) + (8\,d^2 + 5\,d) + 4\,d && \text{(definition)} \\
&= 12\,d^2 + 13\,d && \text{(simplify)}
\end{aligned}
$$

(result) $d = 32$: $12\,704$ weights, as Keras counts (also for $d = 64, 128$); $d = 512$: 3.15 million [@vaswani2017]

## A Block Against a Recurrent Layer

| | LSTM layer | transformer block |
|:------------------------------|:-----------------------------|:-------------------------------|
| weights ($u = d = 32$) | $8\,d^2 + 4\,d = 8\,320$ | $12\,d^2 + 13\,d = 12\,704$ |
| operations per pass | $O(T\,d^2)$ | $O(T^2\,d + T\,d^2)$ |
| steps in sequence | $T$ (sequential) | 1 (all positions at once) |
| path between two tokens | up to $T$ steps | 1 step |
| memory for the weights $A$ | none | $T^2$ numbers |

```{=latex}
\begin{center}
\begin{tikzpicture}[x=0.5cm, y=0.5cm, font=\scriptsize]
  \foreach \i in {0,...,5} { \node[dot=cblue] (r\i) at (\i,0) {}; }
  \foreach \i in {0,...,4} { \pgfmathtruncatemacro{\j}{\i+1} \draw[->, cgray] (r\i) -- (r\j); }
  \node[below] at (2.5,-0.3) {LSTM: $T$ steps in a row};
  \foreach \i in {0,...,5} { \node[dot=corange] (t\i) at (\i+8,0) {}; }
  \foreach \i in {1,...,5} { \draw[cgray] (t0) to[out=30, in=150] (t\i); }
  \node[below] at (10.5,-0.3) {attention: every pair, one hop};
\end{tikzpicture}
\end{center}
```

* Parallel steps scale; the $T^2$ matrix limits length

## The Cost of Attention

:::: {.columns}
::: {.column width="46%"}
| $T$ | entries $T^2$ | seconds |
|:-----------------------:|:-----------------------------------------:|:--------------------------:|
| 512 | 262 144 | 0.004 |
| 1 024 | 1 048 576 | 0.022 |
| 2 048 | 4 194 304 | 0.070 |
| 4 096 | 16 777 216 | 0.285 |
:::
::: {.column width="52%"}
![](plots/12/t2_cost.pdf)
:::
::::

* Doubling $T$ multiplies the time by about 4 (Demo T2)
* Remedies: windowed, linear, flash attention

## Encoder, Decoder, Both

| design | attention | trained to | examples |
|:------------------|:-----------------------------|:---------------------|:-----------------------|
| encoder only | every token sees all | fill a masked word | BERT, text classifiers |
| decoder only | causal mask | next token | GPT, chat models |
| encoder--decoder | decoder also reads the encoder | translate, summarise | the original model |

```{=latex}
\begin{center}
\begin{tikzpicture}[x=0.28cm, y=0.28cm, font=\scriptsize]
  \foreach \i in {0,...,3} { \foreach \j in {0,...,3} { \fill[cblue!50] (\j,-\i) rectangle (\j+0.9,-\i-0.9); } }
  \node at (1.7,0.9) {encoder};
  \foreach \i in {0,...,3} { \foreach \j in {0,...,3} { \ifnum\j>\i \fill[cgray!15] (\j+8,-\i) rectangle (\j+8.9,-\i-0.9); \else \fill[cgreen!60] (\j+8,-\i) rectangle (\j+8.9,-\i-0.9); \fi } }
  \node at (9.7,0.9) {decoder};
  \foreach \i in {0,...,3} { \foreach \j in {0,...,4} { \fill[corange!50] (\j+16,-\i) rectangle (\j+16.9,-\i-0.9); } }
  \node at (18.2,0.9) {cross};
\end{tikzpicture}
\end{center}
```

* **Cross-attention**: queries from the decoder, keys and values from the encoder
* Our demos use the **encoder**

## DEMO T1--T2: `06_attention.ipynb`

Demo 12f: attention by hand, masks, scale, positions, cost.

# The Transformer at Work

## Memory at Lag 100: One Hop

The memory task with 100 steps of noise.

:::: {.columns}
::: {.column width="46%"}
| model (32 units) | accuracy |
|:----------------------------------------------------------------|:--------------------------:|
| vanilla RNN | 0.247 |
| LSTM | 0.274 |
| GRU | 0.258 |
| transformer, 12 100 weights | 1.000 |
:::
::: {.column width="52%"}
![](plots/12/t3_lag100.pdf)
:::
::::

* Chance 0.25; recurrent: one seed; transformer: 3 seeds (Demo T3)

## Where the Last Position Looks

:::: {.columns}
::: {.column width="52%"}
![](plots/12/t3_last_query.pdf)
:::
::: {.column width="46%"}
* Lag 20: 0.822 of the attention on step 0
* Each noise step: 0.009 (uniform 0.048)
* Attention shows *where* the model looked
:::
::::

## Order: Positions Matter

Two markers hide in noise; which comes first? (chance 0.5)

:::: {.columns}
::: {.column width="46%"}
| length | with positions | no positions | LSTM |
|:----------------:|:-------------------------------:|:---------------------------:|:----------------:|
| 8 | 0.998 | 0.506 | 0.999 |
| 16 | 0.991 | 0.498 | 0.998 |
:::
::: {.column width="52%"}
![](plots/12/t4_order.pdf)
:::
::::

* Without positions attention sees the same *set*: chance
* The LSTM gets the order from its recurrence (Demo T4)

## Forecasting with a Transformer Block

:::: {.columns}
::: {.column width="46%"}
| model | test MAE (°C) |
|:-----------------------------------------------|:-------------------------------------------:|
| persistence | 0.450 |
| ridge on the window | **0.358** |
| transformer block | $0.408 \pm 0.101$ |
:::
::: {.column width="52%"}
![](plots/12/t4_forecast.pdf)
:::
::::

* Better than LSTM and GRU, worse than ridge
* The spread over seeds (0.10) is as large as the gap (Demo T4)

## Text: Transformer from Scratch or tf-idf?

1 000 BBC articles, five categories, 600 to train.

:::: {.columns}
::: {.column width="46%"}
| model | test accuracy |
|:-----------------------------------------------------|:-------------------------------------:|
| tf-idf + logistic regression | **0.970** |
| transformer, 111 173 weights | $0.945 \pm 0.007$ |
:::
::: {.column width="52%"}
![](plots/12/t4_text.pdf)
:::
::::

* The old baseline wins: 600 articles are too few
* How many does the transformer need? Next frame

## The Transformer Needs More Data

AG News: $n$ training articles of four categories (tf-idf with logistic regression), 4 000 test articles (Demo T4).

:::: {.columns}
::: {.column width="46%"}
| $n$ | tf-idf | transformer |
|:----------------:|:-------------------------:|:------------------------------------------------:|
| 1k | 0.840 | $0.736 \pm 0.010$ |
| 4k | 0.878 | $0.848 \pm 0.006$ |
| 16k | 0.892 | $0.878 \pm 0.006$ |
| 32k | 0.894 | $\mathbf{0.892} \pm 0.004$ |
:::
::: {.column width="52%"}
![](plots/12/t4_more_data.pdf)
:::
::::

* The gap shrinks from 0.104 ($n = 1\,000$) to 0.002
* The bag of words saturates; the transformer still rises
* Pre-training replaces the missing data

## Where the Transformer Goes From Here

* **Language models**: decoder-only, next-token prediction
* **BERT-style encoders**: pre-trained by masking words
* **Vision transformers** [@dosovitskiy2021]: $16 \times 16$ patches as tokens
* Cost: $O(T^2)$ attention

## DEMO T3--T4: `07_transformer.ipynb`

Demo 12g: block weights, memory at lag 100, order, forecasting, text, learning curve.

# Choosing and Using

## Which Architecture for Which Data

| data | first choice | why | watch out |
|:-----------------|:---------------------------|:------------------------|:----------------------|
| table | trees, linear (Classes 03, 06, 07) | strong baselines | an MLP rarely wins |
| image | CNN | local patterns, shared weights | few images: augment |
| series, short | ridge on a window, then RNN | seasonality is linear | a baseline can win |
| series, long memory | LSTM, GRU or transformer | gates or direct attention | cost of long inputs |
| text, many examples | transformer | any pair of words | needs data or pre-training |
| text, few examples | tf-idf, logistic regression | the baseline of Class 09 | transformers overfit |

* Start simple; add structure when the baseline fails (@goodfellow2016, chapters 9, 10, 12)

## Summary: One Map

```{=latex}
\begin{center}
\begin{tikzpicture}[font=\scriptsize]
  \node[fillbox=cblue, text width=2.6cm] (a) {\textbf{CNN}\\ same filter everywhere\\ equivariance};
  \node[fillbox=cgreen, text width=2.6cm, right=5mm of a] (b) {\textbf{RNN}\\ same cell at every step\\ hidden state};
  \node[fillbox=corange, text width=2.6cm, right=5mm of b] (c) {\textbf{LSTM, GRU}\\ gates and a cell\\ additive path};
  \node[fillbox=cred, text width=2.6cm, below=5mm of b] (d) {\textbf{Transformer}\\ attention between all pairs\\ positions are added};
  \draw[->] (b) -- (c); \draw[->] (b) -- (d);
\end{tikzpicture}
\end{center}
```

* Each fixes something the MLP left free

## Summary: One Page of Equations

| block | equation |
|:---------------|:---------------------------------------------------------------------------|
| convolution | $y_i = \sum_j w_j x_{i+j}$; $L_{out} = \lfloor (L + 2P - F)/S \rfloor + 1$; $m = C'(F^2 C + 1)$ |
| RNN | $h_t = \tanh(W x_t + U h_{t-1} + b)$; $\partial h_T/\partial h_t = \prod \mathrm{diag}(1 - h_k^2) U$ |
| LSTM | $c_t = f_t \odot c_{t-1} + i_t \odot g_t$; $h_t = o_t \odot \tanh(c_t)$; $m = 4u(d + u + 1)$ |
| GRU | $h_t = z_t \odot h_{t-1} + (1 - z_t) \odot \tilde{h}_t$ |
| attention | $A = \mathrm{softmax}(QK^\top / \sqrt{d_k})$; output $AV$ |
| block | $m = 12\,d^2 + 13\,d$; cost $O(T^2 d + T d^2)$ |

* Shapes: $X$ is $T \times d$ for one sequence; $Q$, $K$, $V$ are $T \times d_k$

## Recap: Library Calls

| task | call | note |
|:----------------|:-------------------------------------------|:------------------------------|
| convolution | `keras.layers.Conv2D(C', F, strides=S, padding="same")` | channels last |
| pooling | `MaxPooling2D(2)`, `GlobalAveragePooling2D()` | |
| RNN, LSTM, GRU | `SimpleRNN(u)`, `LSTM(u)`, `GRU(u)` | `return_sequences=True` for stacking |
| gradient clip | `Adam(lr, clipnorm=1.0)` | |
| attention | `MultiHeadAttention(H, d_k)(x, x)` | can return the weights |
| normalise | `LayerNormalization()` | per token, not per batch |
| positions | add a `PositionalEmbedding` (`faa.neural`) | sinusoids |

## Key Takeaways (1/2)

* An MLP has one weight per input; structure removes weights
* **Convolution**: one filter everywhere; equivariant to shifts
* **RNN**: one cell at every step; gradients vanish or explode
* **LSTM, GRU**: gated cell path; longer memory, not infinite

## Key Takeaways (2/2)

* **Attention**: a weighted average; every pair is one hop
* Blind to order: add positions; cost $T^2$
* **Block**: attention, feed-forward, skips, norm
* On small data a baseline can win: report the seed spread

## Lab 12

| part | task |
|:---------:|:---------------------------------------------------------------------------------|
| A | `conv2d` and `max_pool` by hand against Keras `Conv2D` |
| B | one LSTM step and a run over a sequence against Keras `LSTM`; the GRU |
| C | scaled dot-product attention, masks and positions against `MultiHeadAttention` |
| D | the memory task: RNN, LSTM, GRU and transformer at two lags |
| E, F | home: CNN against MLP on shifted digits; forecast a room temperature, then a nonlinear series with frozen encoders |

## Lab 12: How It Connects

* A, B, C: the convolution, LSTM and attention frames
* Each function meets its Keras twin with the **same weights** (difference near $10^{-7}$)
* D to F reuse the demos: memory, shifted digits, forecasts, the frozen stack

<!-- refs:begin -->
## References

:::: {.columns}
::: {.column width="50%"}
\fontsize{5.5}{6.4}\selectfont\setlength{\parskip}{2.5pt}

Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton. 2016. “Layer Normalization.” arXiv Preprint arXiv:1607.06450.

Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. 2015. “Neural Machine Translation by Jointly Learning to Align and Translate.” International Conference on Learning Representations.

Cho, Kyunghyun, Bart van Merriënboer, Çağlar Gülçehre, et al. 2014. “Learning Phrase Representations Using RNN Encoder-Decoder for Statistical Machine Translation.” Conference on Empirical Methods in Natural Language Processing, 1724–34.

Dosovitskiy, Alexey, Lucas Beyer, Alexander Kolesnikov, et al. 2021. “An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale.” International Conference on Learning Representations.

Elman, Jeffrey L. 1990. “Finding Structure in Time.” Cognitive Science 14 (2): 179–211.

Gers, Felix A., Jürgen Schmidhuber, and Fred Cummins. 2000. “Learning to Forget: Continual Prediction with LSTM.” Neural Computation 12 (10): 2451–71.

Goodfellow, Ian, Yoshua Bengio, and Aaron Courville. 2016. Deep Learning. MIT Press.

:::
::: {.column width="50%"}
\fontsize{5.5}{6.4}\selectfont\setlength{\parskip}{2.5pt}

He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. “Deep Residual Learning for Image Recognition.” IEEE Conference on Computer Vision and Pattern Recognition, 770–78.

Hochreiter, Sepp, and Jürgen Schmidhuber. 1997. “Long Short-Term Memory.” Neural Computation 9 (8): 1735–80.

Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. 2012. “ImageNet Classification with Deep Convolutional Neural Networks.” Advances in Neural Information Processing Systems 25: 1097–105.

LeCun, Yann, Léon Bottou, Yoshua Bengio, and Patrick Haffner. 1998. “Gradient-Based Learning Applied to Document Recognition.” Proceedings of the IEEE 86 (11): 2278–324.

Pascanu, Razvan, Tomas Mikolov, and Yoshua Bengio. 2013. “On the Difficulty of Training Recurrent Neural Networks.” International Conference on Machine Learning, 1310–18.

Vaswani, Ashish, Noam Shazeer, Niki Parmar, et al. 2017. “Attention Is All You Need.” Advances in Neural Information Processing Systems 30: 5998–6008.

:::
::::
<!-- refs:end -->
