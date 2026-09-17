---
title: Fundamentos de Aprendizagem Automática
subtitle: "Class 01 — Learning as Compression, Pitfalls & Taxonomies"
---

# Course Introduction

## Professor

:::: {.columns}
::: {.column width="62%"}
* **Mário Antunes**
* Associate Professor, UA
* Researcher, Instituto de Telecomunicações
* `mario.antunes@ua.pt`
* Office: 19.2.15 (IT1)
* Research:
  - Machine learning, time series
  - Optimization and knee detection
  - Maintains the Python libraries used in this course
:::
::: {.column width="38%"}
![](mantunes.jpg){width=100%}
:::
::::

## A Fundamentals Course

```{=latex}
\begin{center}
\begin{tikzpicture}
  \fill[cblue!20]   (-4,0) -- (4,0) -- (3,1.1) -- (-3,1.1) -- cycle;
  \fill[corange!25] (-3,1.1) -- (3,1.1) -- (2,2.2) -- (-2,2.2) -- cycle;
  \fill[cgreen!25]  (-2,2.2) -- (2,2.2) -- (1,3.3) -- (-1,3.3) -- cycle;
  \node at (0,0.55) {\textbf{Foundations:} compression, optimization, evaluation};
  \node at (0,1.65) {\textbf{Classic models:} linear, SVM, NN, trees, ensembles};
  \node at (0,2.75) {\textbf{Modern}};
  \draw[->, thick, cgray] (4.4,0.2) -- node[right, note, text width=2.2cm, align=left] {build\\bottom-up} (4.4,3.2);
\end{tikzpicture}
\end{center}
```

* Understand **why** learning works before using the latest architectures
* Every model in the course rests on the same foundations

## Course Topics

1. Learning as Compression and Optimization
2. Learning Taxonomies
3. Supervised Learning: Linear and Probabilistic Models
4. Supervised Learning: SVM and Neural Networks
5. Supervised Learning: Decision Trees and KNN
6. Supervised Learning: Ensemble Models
7. Unsupervised Learning: Blind Signal Separation
8. Unsupervised Learning: Soft/Hard Clustering
9. Semi-Supervised and Self-Supervised Learning
10. Reinforcement Learning

## How the Topics Connect

```{=latex}
\begin{center}
\begin{tikzpicture}[node distance=4mm and 3mm]
  \node[fillbox=cblue, text width=8.6cm] (f) {\textbf{Foundations}\\ 1 Compression \& Optimization \quad 2 Taxonomies};
  \node[fillbox=corange, text width=4.1cm, below left=6mm and -4.3cm of f] (s) {\textbf{Supervised}\\ 3 Linear \& Probabilistic\\ 4 SVM \& NN\\ 5 Trees \& KNN\\ 6 Ensembles};
  \node[fillbox=cgreen, text width=4.1cm, below right=6mm and -4.3cm of f] (u) {\textbf{Unsupervised}\\ 7 Blind Signal Separation\\ 8 Clustering};
  \node[fillbox=cpurple, text width=4.1cm, below=4mm of u] (ss) {\textbf{Semi/Self-Supervised}\\ 9 AE, Transformers, JEPA};
  \node[fillbox=cred, text width=8.6cm, below=18mm of f.south, anchor=north, yshift=-17mm] (rl) {\textbf{Reinforcement Learning}\\ 10 Learning from rewards};
  \draw[flow] (f) -- (s);
  \draw[flow] (f) -- (u);
  \draw[flow] (u) -- (ss);
  \draw[flow] (s.south) -- (s.south |- rl.north);
  \draw[flow] (ss) -- (ss |- rl.north);
\end{tikzpicture}
\end{center}
```

## Schedule I: Classes 01--07

```{=latex}
\begin{center}\small
\begin{tabular}{@{}c l l L{4.7cm} l@{}}
\toprule
\# & TP1 & TP2 & Topic & Milestone \\
\midrule
01 & 17 Sep & 18 Sep & T1 + T2: Compression, Pitfalls, Taxonomies & \\
02 & 24 Sep & 25 Sep & T1: Blind vs. Gradient Optimization & \\
03 & 01 Oct & 02 Oct & T3: Linear Models & \textbf{P1 released} \\
04 & 08 Oct & 09 Oct & T3: Probabilistic Models & \\
05 & 15 Oct & 16 Oct & T4: SVM and Neural Networks & \\
06 & 22 Oct & 23 Oct & T5: Decision Trees and KNN & P1 checkpoint \\
07 & 29 Oct & 30 Oct & T6: Ensemble Models & \\
\bottomrule
\end{tabular}
\end{center}
```

* T = Topic, P = Project; TP1 on Thursdays, TP2 on Fridays

## Schedule II: Classes 08--14

```{=latex}
\begin{center}\small
\begin{tabular}{@{}c l l L{4.7cm} l@{}}
\toprule
\# & TP1 & TP2 & Topic & Milestone \\
\midrule
08 & 05 Nov & 06 Nov & \textbf{Test 1} (T1--T6) + P1 synthesis & \textbf{P1 due} \\
09 & 12 Nov & 13 Nov & T7: Blind Signal Separation & \textbf{P2 released} \\
10 & 19 Nov & 20 Nov & T8: Soft/Hard Clustering & \\
11 & 26 Nov & 27 Nov & T9: AutoEncoders, Semi-Supervised & \\
12 & 03 Dec & 04 Dec & T9: Transformers and JEPA & \\
13 & 10 Dec & 11 Dec & T10: Reinforcement Learning & P2 clinic \\
14 & 17 Dec & 18 Dec & \textbf{Test 2} (T7--T10) + P2 presentations & \textbf{P2 due} \\
\bottomrule
\end{tabular}
\end{center}
```

* Classes end with the tests; projects are defended in Class 14

## Evaluation Scheme

```{=latex}
\begin{center}
\begin{tikzpicture}
  \node[note] at (2.5,2.3) {\textbf{Discrete evaluation} (recommended)};
  \fill[cblue!70]   (0,1) rectangle (1.25,1.8); \node[white] at (0.625,1.4) {T1};
  \fill[cblue!45]   (1.25,1) rectangle (2.5,1.8); \node at (1.875,1.4) {T2};
  \fill[corange!80] (2.5,1) rectangle (3.75,1.8); \node at (3.125,1.4) {P1};
  \fill[corange!50] (3.75,1) rectangle (5,1.8); \node at (4.375,1.4) {P2};
  \draw[decorate, decoration={brace, mirror, amplitude=4pt}] (0,0.9) -- node[below=4pt, note] {Theory 50\%} (2.5,0.9);
  \draw[decorate, decoration={brace, mirror, amplitude=4pt}] (2.5,0.9) -- node[below=4pt, note] {Practice 50\%} (5,0.9);
  \node[note] at (8.5,2.3) {\textbf{Final evaluation}};
  \fill[cblue!60]   (6,1) rectangle (8.5,1.8); \node at (7.25,1.4) {Exam 50\%};
  \fill[corange!65] (8.5,1) rectangle (11,1.8); \node at (9.75,1.4) {Project 50\%};
\end{tikzpicture}
\end{center}
```

* **Test 1** (25%): Class 08, Topics 1--6 $\cdot$ **Test 2** (25%): Class 14, Topics 7--10
* **Project 1** (25%): released Class 03, due Class 08
* **Project 2** (25%): released Class 09, due Class 14

## Class Format

```{=latex}
\begin{center}
\begin{tikzpicture}
  \node[note, anchor=east] at (-0.2,1.4) {Regular class};
  \fill[cblue!60] (0,1.1) rectangle (3,1.7); \node[white] at (1.5,1.4) {1h lecture};
  \fill[cgreen!60] (3,1.1) rectangle (9,1.7); \node at (6,1.4) {2h lab (practice guide)};
  \node[note, anchor=east] at (-0.2,0.4) {Today};
  \fill[cblue!60] (0,0.1) rectangle (9,0.7); \node[white] at (4.5,0.4) {$\sim$2h lecture with live Jupyter demos};
  \foreach \h in {0,1,2,3} { \draw[cgray] (3*\h,-0.05) -- (3*\h,-0.2) node[below, note] {\h h}; }
\end{tikzpicture}
\end{center}
```

* Lectures introduce the ideas; labs turn them into code
* Today there is no lab guide: the demos run **during** the lecture

## Setting Up the Environment

```bash
git clone https://github.com/detiuaveiro/faa.git && cd faa
export KERAS_BACKEND=jax   # add to ~/.bashrc
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/01-foundations/
```

* Today's demos in `notebooks/01-foundations/`:
  1. `01_learning_as_compression.ipynb`
  2. `02_evaluation_validation.ipynb`
  3. `03_pitfalls.ipynb`
  4. `04_taxonomies.ipynb`

## The Software Stack

```{=latex}
\begin{center}
\begin{tikzpicture}[every node/.style={box, text width=9cm, minimum height=8mm}]
  \node[fill=cblue!20, draw=cblue] at (0,0) {\textbf{Acceleration:} NumPy (OpenBLAS) \quad JAX};
  \node[fill=corange!20, draw=corange] at (0,1) {\textbf{Models:} scikit-learn \quad Keras 3 (JAX backend) \quad joblib};
  \node[fill=cgreen!20, draw=cgreen] at (0,2) {\textbf{Data \& plots:} polars \quad matplotlib \quad seaborn};
  \node[fill=cpurple!20, draw=cpurple] at (0,3) {\textbf{Presentation:} Jupyter Notebook};
  \node[fill=cgray!15, draw=cgray] at (0,-1.3) {\textbf{Group libraries:} pyBlindOpt, EmptySpaceSearch,\\ kneeliverse, pyUTSAlgorithms, pyNNMF};
\end{tikzpicture}
\end{center}
```

# What is Learning?

## The Core Question

```{=latex}
\begin{center}
\begin{tikzpicture}[node distance=9mm]
  \node[fillbox=cgray] (d) {\textbf{Data}\\ past examples};
  \node[fillbox=cblue, right=of d] (l) {\textbf{Learning}\\ algorithm};
  \node[fillbox=corange, right=of l] (m) {\textbf{Model}\\ a function};
  \node[fillbox=cgreen, right=of m] (p) {\textbf{Predictions}\\ new cases};
  \draw[flow] (d) -- (l); \draw[flow] (l) -- (m); \draw[flow] (m) -- (p);
  \node[note, below=3mm of m] {what did it \emph{learn}?};
\end{tikzpicture}
\end{center}
```

* What does it mean for a machine to **learn**?
* It receives examples from the past
* It must act well on situations it has **never seen**
* Today: why this is possible at all, and how it fails

## A Classic Definition

> "A computer program is said to **learn** from experience $E$ with respect to some class of tasks $T$ and performance measure $P$, if its performance at tasks in $T$, as measured by $P$, improves with experience $E$."
>
> --- Tom Mitchell, *Machine Learning*, 1997

* Learning is defined by **improvement**
* Improvement needs a **measure**
* Improvement comes from **experience** (data)

## Task, Experience, Performance

```{=latex}
\begin{center}
\begin{tikzpicture}
  \node[fillbox=cblue, minimum width=2.6cm] (t) at (90:1.9) {\textbf{Task} $T$\\ what to do};
  \node[fillbox=cgreen, minimum width=2.6cm] (e) at (210:2.3) {\textbf{Experience} $E$\\ the data};
  \node[fillbox=corange, minimum width=2.6cm] (p) at (330:2.3) {\textbf{Performance} $P$\\ how to score};
  \draw[flow, <->] (t) -- (e); \draw[flow, <->] (e) -- (p); \draw[flow, <->] (p) -- (t);
  \node[fillbox=cgray, text width=3.1cm, align=left] at (5.3,0.3) {\textbf{Spam filter:}\\ $T$ = label e-mails\\ $E$ = labelled e-mails\\ $P$ = \% correct};
\end{tikzpicture}
\end{center}
```

* Every model in this course is described by these three ingredients

# The Signal and the Noise

## Why So Many Predictions Fail

:::: {.columns}
::: {.column width="55%"}
* *The Signal and the Noise: Why So Many Predictions Fail --- but Some Don't*
* Nate Silver, 2012
* Case studies: weather, elections, baseball, poker, earthquakes, the 2008 financial crisis
* The practical art of building models with **probability and statistics**
:::
::: {.column width="45%"}
```{=latex}
\begin{center}
\begin{tikzpicture}
  \draw[thick, fill=cblue!10] (0,0) rectangle (3,4);
  \draw[thick, fill=cblue!40] (0,0) rectangle (0.3,4);
  \node[align=center, font=\small\bfseries] at (1.65,3.2) {The Signal\\ and the\\ Noise};
  \draw[cred, thick, domain=0.5:2.8, samples=40, smooth] plot (\x, {1.4+0.35*sin(360*\x/1.2)+0.12*sin(360*\x*3.7)});
  \node[font=\scriptsize] at (1.65,0.5) {Nate Silver};
\end{tikzpicture}
\end{center}
```
:::
::::

## Signal and Noise

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=4.6cm, xlabel={time}, ylabel={value}, ytick=\empty, xtick=\empty, legend pos=north west]
  \addplot[only marks, mark=*, mark size=1.2pt, cgray] coordinates {(0,0.3) (0.5,0.9) (1,0.5) (1.5,1.6) (2,1.2) (2.5,1.1) (3,2.1) (3.5,1.4) (4,2.3) (4.5,2.6) (5,1.9) (5.5,2.9) (6,2.5) (6.5,3.4) (7,2.8) (7.5,3.3) (8,3.9)};
  \addlegendentry{what we observe}
  \addplot[cblue, very thick, domain=0:8] {0.45*x+0.4};
  \addlegendentry{signal (the pattern)}
\end{axis}
\end{tikzpicture}
\end{center}
```

* **Signal:** the regularity that repeats and lets us predict
* **Noise:** random variation that does not repeat
* Every dataset mixes both, and they are not labelled

## Mistaking Noise for Signal

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=4.6cm, xtick=\empty, ytick=\empty, xlabel={time}, legend pos=north west]
  \addplot[only marks, mark=*, mark size=1.2pt, cgray] coordinates {(0,0.3) (1,0.5) (2,1.2) (3,2.1) (4,2.3) (5,1.9) (6,2.5) (7,2.8) (8,3.9)};
  \addlegendentry{data}
  \addplot[cred, thick, smooth, tension=0.7] coordinates {(0,0.3) (1,0.5) (2,1.2) (3,2.1) (4,2.3) (5,1.9) (6,2.5) (7,2.8) (8,3.9)};
  \addlegendentry{follows every wiggle}
  \addplot[cblue, very thick, domain=0:8] {0.42*x+0.4};
  \addlegendentry{captures the trend}
\end{axis}
\end{tikzpicture}
\end{center}
```

* The world has far more **noise** than **signal**
* A story that explains *every* past wiggle usually predicts the future badly
* More data does not help if we cannot tell the two apart

## Uncertainty: Ranges, Not Single Numbers

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=4.2cm, width=0.8\textwidth, xlabel={vote share (\%)}, ytick=\empty, xmin=44, xmax=56, ymin=0, ymax=1.35, axis y line=none, clip=false]
  \addplot[cblue, very thick, fill=cblue!20, domain=44:56, samples=80] {exp(-((x-50.1)^2)/(2*1^2))} \closedcycle;
  \addplot[cred, very thick, mark=none] coordinates {(50.1,0) (50.1,1.05)};
  \node[cred, font=\scriptsize] at (axis cs:50.1,1.15) {point estimate 50.1\%};
  \draw[<->, thick] (axis cs:48.1,0.3) -- node[fill=white, inner sep=1pt, font=\scriptsize, above=1pt] {$\pm 2\%$} (axis cs:52.1,0.3);
\end{axis}
\end{tikzpicture}
\end{center}
```

* "Candidate A has an edge" is vague
* "A wins with **probability 83%**, vote share **50.1% $\pm$ 2%**" is useful
* A good model states **how sure** it is, not only its best guess

## Calibration

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, width=5.6cm, height=5cm, xmin=0, xmax=1, ymin=0, ymax=1, xlabel={forecast probability of rain}, ylabel={observed frequency}, legend pos=north west, legend style={font=\tiny}]
  \addplot[cgray, dashed, domain=0:1] {x}; \addlegendentry{perfect calibration}
  \addplot[cblue, mark=*, mark size=1.5pt] coordinates {(0.1,0.12) (0.3,0.28) (0.5,0.52) (0.7,0.69) (0.9,0.88)}; \addlegendentry{well calibrated}
  \addplot[cred, mark=square*, mark size=1.5pt] coordinates {(0.1,0.02) (0.3,0.12) (0.5,0.3) (0.7,0.45) (0.9,0.62)}; \addlegendentry{over-confident}
\end{axis}
\end{tikzpicture}
\end{center}
```

* Weather forecasters: when they say **70% rain**, it should rain on about **70%** of those days
* Calibration checks whether stated probabilities **match reality**

## Updating Beliefs with Data

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=4.5cm, xlabel={probability that the coin lands heads}, ylabel={belief}, ytick=\empty, xmin=0, xmax=1, ymin=0, ymax=9.5, legend pos=north west, samples=150]
  \addplot[cgray, very thick, domain=0:1] {6*x*(1-x)}; \addlegendentry{before any toss: vague}
  \addplot[corange, very thick, domain=0:1] {1320*x^7*(1-x)^3}; \addlegendentry{after 10 tosses (7 heads)}
  \addplot[cblue, very thick, domain=0.001:0.999] {exp(64.82 + 71*ln(x) + 31*ln(1-x))}; \addlegendentry{after 100 tosses (70 heads)}
\end{axis}
\end{tikzpicture}
\end{center}
```

* Start with a belief, observe data, **update** the belief (Bayes' rule, Topic 3)
* With more data the belief becomes **sharper**
* Learning is gradual, not a single perfect model

## Lessons for Machine Learning

```{=latex}
\begin{center}
\begin{tikzpicture}[node distance=5mm]
  \node[fillbox=cblue, text width=3cm] (a) {Separate\\ \textbf{signal} from \textbf{noise}};
  \node[fillbox=corange, text width=3cm, right=of a] (b) {Report\\ \textbf{uncertainty}};
  \node[fillbox=cgreen, text width=3cm, right=of b] (c) {Check\\ \textbf{calibration}};
  \node[fillbox=cpurple, text width=3cm, below=6mm of b] (d) {\textbf{Update} with\\ new data};
  \draw[flow] (a) -- (b); \draw[flow] (b) -- (c); \draw[flow] (c) |- (d); \draw[flow] (d) -| (a);
\end{tikzpicture}
\end{center}
```

* These ideas return throughout the course:
  - Compression separates signal from noise (today)
  - Statistical validation measures uncertainty (today)
  - Probabilistic models and Bayes' rule (Topic 3)

# Pattern, Noise, and Memorization

## Data = Pattern + Noise

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=4.4cm, xmin=0, xmax=1, ymin=-1.7, ymax=1.5, xlabel={$x$}, ylabel={$y$}, xtick={0,0.5,1}, ytick={-1,0,1}]
  \addplot[cblue, very thick, domain=0:1, samples=80] {sin(deg(2*pi*x))};
  \foreach \px/\py in {0.04/0.48, 0.13/0.29, 0.19/0.86, 0.31/0.42, 0.37/0.94, 0.44/0.61, 0.48/-0.12, 0.67/-1.0, 0.7/-0.81, 0.77/-1.35, 0.82/-0.75, 0.86/-0.9, 0.97/-0.58} {
    \edef\temp{\noexpand\draw[cred, thin] (axis cs:\px,\py) -- (axis cs:\px,{sin(deg(2*pi*\px))});}\temp
    \edef\temp{\noexpand\fill[black] (axis cs:\px,\py) circle (1.6pt);}\temp
  }
  \node[cblue, font=\scriptsize] at (axis cs:0.25,1.3) {$f(x)$: the pattern};
  \node[cred, font=\scriptsize] at (axis cs:0.62,0.6) {red: noise $\varepsilon$};
\end{axis}
\end{tikzpicture}
\end{center}
```

$$y = f(x) + \varepsilon$$

* $x$: the input we observe; $y$: the output we want to predict
* $f(x)$: the unknown pattern; $\varepsilon$ (epsilon): the noise, a random error

## Reading the Formula

$$\underbrace{y}_{\text{what we measure}} \;=\; \underbrace{f(x)}_{\text{the rule we want}} \;+\; \underbrace{\varepsilon}_{\text{random error}}$$

* **Example:** house price
  - $x$ = area of the house
  - $f(x)$ = the "fair" price for that area
  - $\varepsilon$ = everything else: the seller's mood, a nice view, luck
* We **never** see $f$ directly, only pairs $(x, y)$
* Learning means building an estimate $\hat{f}$ ("f-hat") of $f$

## We Only See a Finite Sample

```{=latex}
\begin{center}
\begin{tikzpicture}
  \draw[thick, fill=cgray!10] (0,0) ellipse (2.4cm and 1.5cm);
  \node[note] at (0,1.8) {all possible examples};
  \foreach \i in {1,...,45} { \fill[cgray] ({2.1*rand},{1.2*rand}) circle (1.1pt); }
  \foreach \px/\py in {-1/0.3, 0.4/-0.6, 1.2/0.5, -0.3/0.8, 0.8/0.1, -1.5/-0.4} { \fill[cblue] (\px,\py) circle (2.2pt); }
  \draw[flow] (2.6,0) -- node[above, note] {we get} (4.2,0);
  \node[fillbox=cblue, text width=3.3cm] at (6.1,0) {dataset $D$\\ $n$ examples\\ $(x_1,y_1),\dots$\\ $\dots,(x_n,y_n)$};
\end{tikzpicture}
\end{center}
```

* $D = \{(x_1, y_1), \dots, (x_n, y_n)\}$: $n$ observed pairs; the index $i$ numbers them
* The goal is **not** to reproduce $D$
* The goal is to predict well on **new** $x$ from the same process

## Memorization: the Lookup Table

```{=latex}
\begin{center}
\begin{tikzpicture}
  \matrix (t) [matrix of nodes, nodes={draw, minimum width=1.4cm, minimum height=5mm, font=\scriptsize}, column sep=-\pgflinewidth, row sep=-\pgflinewidth, ampersand replacement=\&] {
    |[fill=cgray!25]| $x$ \& |[fill=cgray!25]| $y$ \\
    0.04 \& 0.48 \\ 0.13 \& 0.29 \\ 0.19 \& 0.86 \\ \dots \& \dots \\ 0.97 \& $-0.58$ \\
  };
  \node[fillbox=corange, right=1.4cm of t, text width=3.4cm] (q) {new input $x = 0.15$\\ copy the answer of the\\ \textbf{closest} stored $x$};
  \draw[flow] (q.west) -- (t.east);
\end{tikzpicture}
\end{center}
```

* Stores **every** training example
* Perfect on data it has seen: training error = 0
* Stores the noise $\varepsilon$ as faithfully as the pattern $f$

## Generalization

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=4.4cm, xmin=0, xmax=1, ymin=-1.7, ymax=1.5, xlabel={$x$}, ylabel={$y$}, xtick={0,0.5,1}, ytick={-1,0,1}, legend pos=south west]
  \addplot[only marks, mark=*, mark size=1.4pt, black] coordinates {(0.04,0.48) (0.13,0.29) (0.19,0.86) (0.31,0.42) (0.37,0.94) (0.44,0.61) (0.48,-0.12) (0.67,-1.0) (0.7,-0.81) (0.77,-1.35) (0.82,-0.75) (0.86,-0.9) (0.97,-0.58)};
  \addlegendentry{training data}
  \addplot[cred, thick, const plot mark mid] coordinates {(0,0.48) (0.085,0.48) (0.085,0.29) (0.16,0.29) (0.16,0.86) (0.25,0.86) (0.25,0.42) (0.34,0.42) (0.34,0.94) (0.405,0.94) (0.405,0.61) (0.46,0.61) (0.46,-0.12) (0.575,-0.12) (0.575,-1.0) (0.685,-1.0) (0.685,-0.81) (0.735,-0.81) (0.735,-1.35) (0.795,-1.35) (0.795,-0.75) (0.84,-0.75) (0.84,-0.9) (0.915,-0.9) (0.915,-0.58) (1,-0.58)};
  \addlegendentry{lookup table}
  \addplot[cblue, very thick, domain=0:1, samples=60] {1.1*sin(deg(2*pi*x))-0.05};
  \addlegendentry{smooth model}
\end{axis}
\end{tikzpicture}
\end{center}
```

* **Generalization:** performing well on data **not** seen during training
* The lookup table jumps with every noisy point; the smooth model captures the pattern

## The Dilemma: Too Flexible or Too Rigid

```{=latex}
\begin{center}
\begin{tikzpicture}
  \draw[thick, <->] (0,0) -- (10,0);
  \node[fillbox=cred, text width=2.8cm] at (1.5,1) {\textbf{too rigid}\\ misses the pattern\\ \emph{underfitting}};
  \node[fillbox=cgreen, text width=2.8cm] at (5,1) {\textbf{just right}\\ pattern, not noise};
  \node[fillbox=cred, text width=2.8cm] at (8.5,1) {\textbf{too flexible}\\ memorizes noise\\ \emph{overfitting}};
  \node[note, anchor=west] at (0,-0.35) {simple model};
  \node[note, anchor=east] at (10,-0.35) {complex model};
\end{tikzpicture}
\end{center}
```

* We need a **principled** way to prefer simple explanations
* Answer today: **compression** (and later, validation)
* Under- and overfitting are revisited in detail as a pitfall

## Live Demo D1: Memorization vs. Generalization

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, ybar, height=4.2cm, width=0.75\textwidth, bar width=16pt, symbolic x coords={lookup table, cubic model}, xtick=data, ylabel={mean squared error}, ymin=0, ymax=0.2, ytick={0,0.05,0.1,0.15}, yticklabel style={/pgf/number format/fixed}, enlarge x limits=0.5, legend style={at={(0.5,1.02)}, anchor=south, legend columns=2}, nodes near coords, nodes near coords style={font=\tiny, /pgf/number format/fixed, /pgf/number format/precision=3}]
  \addplot[fill=cblue!60, draw=cblue] coordinates {(lookup table,0) (cubic model,0.05)}; \addlegendentry{training}
  \addplot[fill=corange!70, draw=corange] coordinates {(lookup table,0.149) (cubic model,0.110)}; \addlegendentry{test}
  \draw[cgray, dashed] (rel axis cs:0,0.45) -- (rel axis cs:1,0.45);
  \node[font=\tiny, cgray, anchor=south west] at (rel axis cs:0,0.45) {noise floor $\sigma^2 = 0.09$};
\end{axis}
\end{tikzpicture}
\end{center}
```

* Notebook `01_learning_as_compression.ipynb`, section **D1**: 50 noisy points
* Lookup table: **100 numbers** stored; cubic polynomial: **4 numbers**
* Which one has lower training error? Lower test error? Why?

# A Small Math Toolbox

## Summation: Adding Many Things

```{=latex}
\begin{center}
\begin{tikzpicture}
  \foreach \v [count=\i] in {2,5,3} {
    \node[fillbox=cblue, minimum width=1cm] (b\i) at (2.1*\i,0) {$x_{\i} = \v$};
  }
  \node at (3.15,0) {$+$}; \node at (5.25,0) {$+$};
  \node[fillbox=corange, minimum width=1.4cm] (r) at (9.4,0) {$10$};
  \draw[flow] (b3) -- node[above, note] {add all} (r);
  \node[note] at (4.2,-0.8) {$i = 1, 2, 3$ \quad (the index counts the items)};
\end{tikzpicture}
\end{center}
```

$$\sum_{i=1}^{n} x_i = x_1 + x_2 + \dots + x_n$$

* $\sum$ (capital sigma) means **"add up"**
* $i = 1$ below: start at the first item; $n$ on top: stop at the last item
* Example with $n = 3$: $\sum_{i=1}^{3} x_i = 2 + 5 + 3 = 10$

## The Mean: a Typical Value

```{=latex}
\begin{center}
\begin{tikzpicture}[scale=0.9]
  \draw[thick, ->] (0,0) -- (10.5,0);
  \foreach \t in {0,2,...,10} { \draw (\t,0.1) -- (\t,-0.1) node[below, note] {\t}; }
  \foreach \v in {2,5,3,6} { \fill[cblue] (\v,0.3) circle (3pt); }
  \draw[cred, very thick] (4,-0.5) -- (4,0.9) node[above, font=\scriptsize] {mean = 4};
  \fill[cgray] (3.7,-0.55) -- (4.3,-0.55) -- (4,-0.15) -- cycle;
\end{tikzpicture}
\end{center}
```

$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$$

* $\bar{x}$ ("x-bar"): the average; $\frac{1}{n}$: divide the total by how many items there are
* Example: values $2, 5, 3, 6$ $\Rightarrow$ $\bar{x} = \frac{2+5+3+6}{4} = \frac{16}{4} = 4$
* The mean is the **balance point** of the values

## Errors: How Wrong Is a Prediction?

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=4.2cm, width=0.7\textwidth, xmin=0, xmax=4, ymin=0, ymax=5, xlabel={$x$}, ylabel={$y$}, xtick={1,2,3}, ytick={1,2,3,4}]
  \addplot[cblue, very thick, domain=0:4] {x+0.5};
  \foreach \px/\py in {1/2.5, 2/2, 3/4} {
    \edef\temp{\noexpand\draw[cred, very thick] (axis cs:\px,\py) -- (axis cs:\px,{\px+0.5});}\temp
    \edef\temp{\noexpand\fill (axis cs:\px,\py) circle (2pt);}\temp
  }
  \node[cred, font=\scriptsize, anchor=east] at (axis cs:0.95,2) {$e_1 = 1$};
  \node[cred, font=\scriptsize, anchor=west] at (axis cs:2.08,1.8) {$e_2 = -0.5$};
  \node[cred, font=\scriptsize, anchor=east] at (axis cs:2.92,3.8) {$e_3 = 0.5$};
  \node[cblue, font=\scriptsize, anchor=west] at (axis cs:0.1,4.5) {model $\hat{y} = x + 0.5$};
\end{axis}
\end{tikzpicture}
\end{center}
```

* **Error (residual):** $e_i = y_i - \hat{y}_i$ = true value $-$ prediction
* $\hat{y}_i$ ("y-hat"): what the model predicts for example $i$
* Errors can be positive or negative, so we cannot just add them: they cancel out

## Mean Squared Error (MSE)

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n} \left(y_i - \hat{y}_i\right)^2$$

```{=latex}
\begin{center}
\begin{tikzpicture}
  \node[fillbox=cred, text width=2.2cm] (a) {errors\\ $1,\ -0.5,\ 0.5$};
  \node[fillbox=corange, text width=2.3cm, right=4mm of a] (b) {square\\ $1,\ 0.25,\ 0.25$};
  \node[fillbox=cblue, text width=1.1cm, right=4mm of b] (c) {add\\ $1.5$};
  \node[fillbox=cgreen, text width=1.9cm, right=4mm of c] (d) {divide by 3\\ $\text{MSE} = 0.5$};
  \draw[flow] (a) -- (b); \draw[flow] (b) -- (c); \draw[flow] (c) -- (d);
\end{tikzpicture}
\end{center}
```

* **Squaring** makes every error positive and punishes large errors more
* MSE = 0 means perfect predictions; larger means worse
* We use MSE throughout the course to measure regression models

## Probability: How Likely Is Each Outcome?

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, ybar, height=4.2cm, width=0.7\textwidth, symbolic x coords={sun, cloud, rain, snow}, xtick=data, ymin=0, ymax=0.7, ylabel={$p(x)$}, bar width=18pt, nodes near coords, nodes near coords style={font=\scriptsize}, enlarge x limits=0.2]
  \addplot[fill=cblue!60, draw=cblue] coordinates {(sun,0.5) (cloud,0.25) (rain,0.2) (snow,0.05)};
\end{axis}
\end{tikzpicture}
\end{center}
```

* $p(x)$: the probability of outcome $x$, a number between **0** (never) and **1** (always)
* All probabilities add up to one: $\sum_{x} p(x) = 0.5 + 0.25 + 0.2 + 0.05 = 1$
* Probability 0.25 means "happens about 1 time in 4"

## Logarithm Base 2: Counting Yes/No Questions

```{=latex}
\begin{center}
\begin{tikzpicture}[level distance=9mm, level 1/.style={sibling distance=40mm}, level 2/.style={sibling distance=20mm}, level 3/.style={sibling distance=10mm}, every node/.style={font=\scriptsize}]
  \node[fillbox=cblue] {Q1: $\geq 5$?}
    child { node[fillbox=cblue] {Q2: $\geq 3$?}
      child { node[fillbox=cblue] {Q3} child {node {1}} child {node {2}} }
      child { node[fillbox=cblue] {Q3} child {node {3}} child {node {4}} } }
    child { node[fillbox=cblue] {Q2: $\geq 7$?}
      child { node[fillbox=cblue] {Q3} child {node {5}} child {node {6}} }
      child { node[fillbox=cblue] {Q3} child {node {7}} child {node {8}} } };
\end{tikzpicture}
\end{center}
```

* Guess a number from 1 to 8 with yes/no questions: **3 questions** always suffice
* $\log_2 8 = 3$ because $2^3 = 8$; in general $\log_2 N$ = questions needed for $N$ equally likely options
* One yes/no answer = **1 bit** of information

## Logarithms of Probabilities

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=4.4cm, width=0.62\textwidth, xmin=0, xmax=1.05, ymin=0, ymax=5, xlabel={probability $p$}, ylabel={$-\log_2 p$ (bits)}, samples=100]
  \addplot[cblue, very thick, domain=0.03:1] {-ln(x)/ln(2)};
  \foreach \p/\b in {1/0, 0.5/1, 0.25/2, 0.125/3} {
    \edef\temp{\noexpand\fill[cred] (axis cs:\p,\b) circle (2pt);}\temp
  }
\end{axis}
\end{tikzpicture}
\quad
\begin{tabular}[b]{cc}
\toprule
$p$ & $-\log_2 p$ \\ \midrule
1 & 0 bits \\ 1/2 & 1 bit \\ 1/4 & 2 bits \\ 1/8 & 3 bits \\ \bottomrule
\end{tabular}
\end{center}
```

* For $p < 1$, $\log_2 p$ is negative, so we use $-\log_2 p$ to get a positive number
* **Certain** events ($p = 1$) carry 0 bits; **rare** events carry many bits

## argmin: Where Is the Minimum?

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=4.3cm, width=0.65\textwidth, xmin=-1, xmax=5, ymin=0, ymax=6, xlabel={$\theta$}, ylabel={$L(\theta)$}, xtick={2}, ytick={1}]
  \addplot[cblue, very thick, domain=-1:5, samples=60] {0.5*(x-2)^2+1};
  \draw[cred, dashed] (axis cs:2,0) -- (axis cs:2,1);
  \draw[cgreen!60!black, dashed] (axis cs:-1,1) -- (axis cs:2,1);
  \fill[cred] (axis cs:2,1) circle (2.5pt);
  \node[cred, font=\scriptsize, anchor=west] at (axis cs:2.1,0.4) {$\arg\min_\theta L = 2$};
  \node[cgreen!60!black, font=\scriptsize, anchor=south west] at (axis cs:-0.9,1.05) {$\min_\theta L = 1$};
\end{axis}
\end{tikzpicture}
\end{center}
```

* $L(\theta)$: a function of a parameter $\theta$ (theta), e.g. the error of a model
* $\min_\theta L(\theta)$: the **smallest value** of the function (here 1)
* $\arg\min_\theta L(\theta)$: the **parameter that achieves it** (here $\theta = 2$)

## Variance: How Spread Out Are the Values?

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=4cm, width=0.75\textwidth, xmin=-5, xmax=5, ymin=0, ymax=0.85, axis y line=none, xlabel={value}, samples=100, legend style={at={(0.02,0.98)}, anchor=north west}]
  \addplot[cblue, very thick, domain=-5:5] {exp(-x^2/(2*0.5^2))/(0.5*sqrt(2*pi))}; \addlegendentry{small variance}
  \addplot[corange, very thick, domain=-5:5] {exp(-x^2/(2*1.8^2))/(1.8*sqrt(2*pi))}; \addlegendentry{large variance}
\end{axis}
\end{tikzpicture}
\end{center}
```

$$s^2 = \frac{1}{n-1}\sum_{i=1}^{n}\left(x_i - \bar{x}\right)^2 \qquad s = \sqrt{s^2}$$

* Variance $s^2$: average squared distance to the mean; standard deviation $s$: its square root
* Example: $2, 5, 3, 6$ with $\bar{x} = 4$: $\frac{(4 + 1 + 1 + 4)}{3} = 3.33$, so $s \approx 1.83$

# Learning as Compression

## To Learn Is to Compress

```{=latex}
\begin{center}
\begin{tikzpicture}
  \fill[cgray!40] (0,0) rectangle (2.2,3); \node[align=center] at (1.1,1.5) {raw data\\ \textbf{1000 bits}};
  \draw[flow, very thick] (2.6,1.5) -- node[above, note] {learn} (4,1.5);
  \fill[cblue!50] (4.4,1.6) rectangle (6.6,2.4); \node[align=center] at (5.5,2) {model\\ 60 bits};
  \fill[corange!50] (4.4,0.6) rectangle (6.6,1.4); \node[align=center] at (5.5,1) {errors\\ 140 bits};
  \draw[decorate, decoration={brace, amplitude=5pt}] (6.8,2.4) -- node[right=6pt, note, align=left] {200 bits\\ same data} (6.8,0.6);
\end{tikzpicture}
\end{center}
```

* If a short model plus its errors **reproduces** the data, the model found regularities
* The model is a **compressed** description of the data
* A model that is as big as the data (a lookup table) has **learned nothing**

## Compression as Communication

```{=latex}
\begin{center}
\begin{tikzpicture}[node distance=12mm]
  \node[fillbox=cblue, text width=2.2cm] (s) {sender\\ has the data};
  \node[fillbox=cgray, text width=2.6cm, right=of s] (c) {channel\\ send as few bits\\ as possible};
  \node[fillbox=cgreen, text width=2.2cm, right=of c] (r) {receiver\\ rebuilds the data};
  \draw[flow] (s) -- (c); \draw[flow] (c) -- (r);
  \node[note, below=4mm of c, text width=7cm] {"temperature rises 2\,°C per hour, plus these small corrections"\\ is much shorter than sending every reading};
\end{tikzpicture}
\end{center}
```

* To send less, the sender must **understand** the data
* Understanding = finding the rule that generates most of it
* Only the part the rule cannot explain (the noise) must be sent in full

## Pure Noise Cannot Be Compressed

```{=latex}
\begin{center}
\begin{tikzpicture}[every node/.style={font=\ttfamily\small}]
  \node[anchor=west] at (0,1.5) {ABABABABABABABABABABABAB};
  \node[anchor=west, cgreen!60!black] at (7.2,1.5) {\normalfont $\rightarrow$ "AB $\times$ 12"};
  \node[anchor=west] at (0,0.8) {THE CAT SAT ON THE MAT.};
  \node[anchor=west, corange] at (7.2,0.8) {\normalfont $\rightarrow$ some savings};
  \node[anchor=west] at (0,0.1) {Q7\#pZ2!mK9vX@4rT\&1wL8};
  \node[anchor=west, cred] at (7.2,0.1) {\normalfont $\rightarrow$ no shorter description};
\end{tikzpicture}
\end{center}
```

* **Structure** can be described briefly; **randomness** cannot
* If data were pure noise, there would be **nothing to learn**
* Learning is possible **because** real data contain regularities

## Surprise: Information of One Event

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=4.2cm, width=0.7\textwidth, xmin=0, xmax=1.05, ymin=0, ymax=5.2, xlabel={probability of the event}, ylabel={surprise (bits)}, samples=100]
  \addplot[cblue, very thick, domain=0.03:1] {-ln(x)/ln(2)};
  \node[fillbox=cgreen, font=\tiny, anchor=east] at (axis cs:1,2.2) {"sun rises": $p \approx 1$, no surprise};
  \node[fillbox=cred, font=\tiny, anchor=west] at (axis cs:0.1,4.3) {"snow in Aveiro": rare, big surprise};
\end{axis}
\end{tikzpicture}
\end{center}
```

$$h(x) = -\log_2 p(x)$$

* $h(x)$: the information (surprise) of observing outcome $x$, measured in **bits**
* Likely events tell us little; unlikely events tell us a lot

## Shannon Entropy

$$H(X) = -\sum_{x} p(x)\,\log_2 p(x)$$

```{=latex}
\begin{center}
\begin{tikzpicture}
  \node[fillbox=cblue, text width=1.7cm] (a) {for each\\ outcome $x$};
  \node[fillbox=corange, text width=2.1cm, right=5mm of a] (b) {its surprise\\ $-\log_2 p(x)$};
  \node[fillbox=cgreen, text width=2.3cm, right=5mm of b] (c) {weight by how\\ often: $p(x)$};
  \node[fillbox=cpurple, text width=1.2cm, right=5mm of c] (d) {add up\\ $\sum_x$};
  \draw[flow] (a) -- (b); \draw[flow] (b) -- (c); \draw[flow] (c) -- (d);
\end{tikzpicture}
\end{center}
```

* $X$: a random variable, e.g. "the result of a coin toss"
* $H(X)$: the **average surprise**, in bits per outcome (Claude Shannon, 1948)
* High entropy = unpredictable; low entropy = predictable

## Worked Example: a Fair Coin

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, ybar, height=3.6cm, width=0.45\textwidth, symbolic x coords={heads, tails}, xtick=data, ymin=0, ymax=1, ylabel={$p(x)$}, bar width=20pt, nodes near coords, nodes near coords style={font=\scriptsize}, enlarge x limits=0.5]
  \addplot[fill=cblue!60, draw=cblue] coordinates {(heads,0.5) (tails,0.5)};
\end{axis}
\end{tikzpicture}
\end{center}
```

1. Probabilities: $p(\text{heads}) = 0.5$, $p(\text{tails}) = 0.5$
2. Surprise of each: $-\log_2 0.5 = 1$ bit
3. Weight and add: $H = 0.5 \times 1 + 0.5 \times 1 = 1$ **bit**

* A fair coin needs one full bit per toss: nothing can be predicted

## Worked Example: a Biased Coin

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, ybar, height=3.6cm, width=0.45\textwidth, symbolic x coords={heads, tails}, xtick=data, ymin=0, ymax=1, ylabel={$p(x)$}, bar width=20pt, nodes near coords, nodes near coords style={font=\scriptsize}, enlarge x limits=0.5]
  \addplot[fill=corange!70, draw=corange] coordinates {(heads,0.9) (tails,0.1)};
\end{axis}
\end{tikzpicture}
\end{center}
```

1. Probabilities: $p(\text{heads}) = 0.9$, $p(\text{tails}) = 0.1$
2. Surprise: $-\log_2 0.9 \approx 0.152$ bits; $-\log_2 0.1 \approx 3.32$ bits
3. Weight and add: $H = 0.9 \times 0.152 + 0.1 \times 3.32 \approx 0.137 + 0.332 = 0.47$ **bits**

* Mostly predictable: on average less than half a bit per toss

## Entropy Across All Coins

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=4.6cm, width=0.72\textwidth, xmin=0, xmax=1, ymin=0, ymax=1.25, xlabel={$p(\text{heads})$}, ylabel={$H$ (bits)}, samples=120, clip=false]
  \addplot[cblue, very thick, domain=0.001:0.999] {-(x*ln(x)+(1-x)*ln(1-x))/ln(2)};
  \fill[cblue] (axis cs:0.5,1) circle (2.5pt) node[above, font=\scriptsize] {fair: 1 bit};
  \fill[corange] (axis cs:0.9,0.469) circle (2.5pt) node[right, font=\scriptsize] {0.47 bits};
  \fill[cred] (axis cs:0.999,0.01) circle (2.5pt) node[above left, font=\scriptsize] {always heads: 0 bits};
\end{axis}
\end{tikzpicture}
\end{center}
```

* Entropy is **maximal** when all outcomes are equally likely
* Entropy is **zero** when the outcome is certain

## Entropy Is the Limit of Compression

```{=latex}
\begin{center}
\begin{tikzpicture}[level distance=10mm, level 1/.style={sibling distance=26mm}, level 2/.style={sibling distance=16mm}, every node/.style={font=\scriptsize}]
  \node[fillbox=cgray] {start}
    child { node[fillbox=cblue] {A \quad code \texttt{0}} edge from parent node[left] {0} }
    child { node[fillbox=cgray] {}
      child { node[fillbox=corange] {B \quad \texttt{10}} edge from parent node[left] {0} }
      child { node[fillbox=cgreen] {C \quad \texttt{11}} edge from parent node[right] {1} }
      edge from parent node[right] {1} };
  \node[note, text width=4.6cm, align=flush left] at (5.6,-1.2) {$p(A) = 0.5,\ p(B) = 0.25,\ p(C) = 0.25$\\[2pt] average length:\\ $0.5 \times 1 + 0.25 \times 2 + 0.25 \times 2 = 1.5$ bits\\[2pt] entropy: $H = 1.5$ bits};
\end{tikzpicture}
\end{center}
```

* Give **short codes to frequent symbols** and long codes to rare ones
* No lossless code can use fewer bits on average than $H$ (Shannon's source coding theorem)

## Kolmogorov Complexity

```{=latex}
\begin{center}
\begin{tikzpicture}[node distance=12mm]
  \node[fillbox=cblue, text width=3.9cm, align=flush left, font=\ttfamily\scriptsize] (p) {for i in range(2000):\\ \ \ print("AB", end="")};
  \node[fillbox=cgray, right=8mm of p] (u) {computer};
  \node[fillbox=corange, right=8mm of u, text width=3.1cm, font=\ttfamily\scriptsize] (x) {ABABAB...AB\\ (4000 characters)};
  \draw[flow] (p) -- node[above, note] {run} (u); \draw[flow] (u) -- (x);
  \draw[decorate, decoration={brace, mirror, amplitude=4pt}] (p.south west) -- node[below=5pt, note] {$|p|$ = length of the program} (p.south east);
\end{tikzpicture}
\end{center}
```

$$K(x) = \min_{p} \{\, |p| : U(p) = x \,\}$$

* $K(x)$: length of the **shortest program** $p$ that prints the string $x$
* $U(p) = x$: "running program $p$ on a computer $U$ outputs $x$"; $|p|$: length of $p$
* The ultimate measure of how much structure a string contains (Kolmogorov, 1965)

## Two Strings, Two Shortest Programs

```{=latex}
\begin{center}
\begin{tikzpicture}
  \node[anchor=west, font=\small] at (0,2) {\texttt{ABAB...AB} (4000 chars)};
  \fill[cblue!60] (0,1.3) rectangle (0.5,1.7); \node[anchor=west, font=\scriptsize] at (0.6,1.5) {program: $\approx$ 40 characters};
  \node[anchor=west, font=\small] at (0,0.8) {4000 random characters};
  \fill[cred!60] (0,0.1) rectangle (9,0.5); \node[anchor=west, font=\scriptsize, white] at (0.1,0.3) {program: print("...all 4000 characters...") $\approx$ 4000 characters};
\end{tikzpicture}
\end{center}
```

* Regular string: $K$ is **tiny** compared with its length
* Random string: $K \approx$ its length, it is **incompressible**
* Learning looks for the short program behind the data

## Kolmogorov Complexity in Practice

```{=latex}
\begin{center}
\begin{tikzpicture}
  \fill[cgreen!50] (0,1.6) rectangle (3,2.1); \node[anchor=west, font=\scriptsize] at (3.1,1.85) {$K(x)$: true shortest program (unknown)};
  \fill[corange!60] (0,0.9) rectangle (5,1.4); \node[anchor=west, font=\scriptsize] at (5.1,1.15) {size after \texttt{zlib} / \texttt{gzip}};
  \fill[cgray!60] (0,0.2) rectangle (8,0.7); \node[anchor=west, font=\scriptsize] at (8.1,0.45) {raw size};
\end{tikzpicture}
\end{center}
```

* $K(x)$ **cannot be computed**: no algorithm finds the shortest program for every string
* Real compressors give an **upper bound**: $K(x) \leq$ compressed size
* Machine learning models play the same role: practical, imperfect compressors

## Live Demo D2: Entropy and Compression

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, ybar, height=4.4cm, width=0.95\textwidth, bar width=9pt, symbolic x coords={ABAB..., text $\times$10, unique text, random bytes}, xtick=data, ylabel={bits per byte}, ymin=0, ymax=9.5, legend style={at={(0.5,1.03)}, anchor=south, legend columns=2}, enlarge x limits=0.15]
  \addplot[fill=cblue!60, draw=cblue] coordinates {(ABAB...,1) (text $\times$10,4.2) (unique text,4.21) (random bytes,7.95)}; \addlegendentry{entropy $H$}
  \addplot[fill=corange!70, draw=corange] coordinates {(ABAB...,0.06) (text $\times$10,0.96) (unique text,4.15) (random bytes,8.02)}; \addlegendentry{\texttt{zlib} size}
\end{axis}
\end{tikzpicture}
\end{center}
```

* Notebook `01_learning_as_compression.ipynb`, section **D2**
* Why does `zlib` beat the entropy on `ABAB...`? Why does random data **grow**?

## Occam's Razor

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=4.3cm, xmin=0, xmax=1, ymin=-1.9, ymax=1.6, xtick=\empty, ytick=\empty, legend style={at={(0.5,1.03)}, anchor=south, legend columns=3}]
  \addplot[only marks, mark=*, mark size=1.4pt] coordinates {(0.04,0.48) (0.13,0.29) (0.19,0.86) (0.31,0.42) (0.37,0.94) (0.44,0.61) (0.48,-0.12) (0.67,-1.0) (0.7,-0.81) (0.77,-1.35) (0.82,-0.75) (0.86,-0.9) (0.97,-0.58)};
  \addlegendentry{data}
  \addplot[cblue, very thick, domain=0:1, samples=60] {1.05*sin(deg(2*pi*x))};
  \addlegendentry{simple explanation}
  \addplot[cred, thick, smooth, tension=0.6] coordinates {(0,0.9) (0.04,0.48) (0.08,0.1) (0.13,0.29) (0.19,0.86) (0.25,0.1) (0.31,0.42) (0.37,0.94) (0.44,0.61) (0.48,-0.12) (0.58,-0.2) (0.67,-1.0) (0.7,-0.81) (0.77,-1.35) (0.82,-0.75) (0.86,-0.9) (0.97,-0.58) (1,0.2)};
  \addlegendentry{complicated explanation}
\end{axis}
\end{tikzpicture}
\end{center}
```

* *"Entities should not be multiplied beyond necessity"* (William of Ockham, 14th century)
* Among explanations that fit the data similarly well, **prefer the simplest**
* But how do we measure "simple" and "fits well" on the same scale?

## Minimum Description Length (MDL)

```{=latex}
\begin{center}
\begin{tikzpicture}
  \fill[cblue!55] (0,0) rectangle (3,0.8); \node at (1.5,0.4) {$L(H)$};
  \fill[corange!65] (3,0) rectangle (7,0.8); \node at (5,0.4) {$L(D \mid H)$};
  \draw[decorate, decoration={brace, amplitude=5pt}] (0,0.9) -- node[above=5pt, note] {total message length $L(D, H)$} (7,0.9);
  \node[note, text width=3cm] at (1.5,-0.5) {bits to describe\\ the \textbf{model}};
  \node[note, text width=3.5cm] at (5,-0.5) {bits to describe the data\\ \textbf{given} the model (its errors)};
\end{tikzpicture}
\end{center}
```

$$L(D, H) = L(H) + L(D \mid H)$$

* $H$: a hypothesis (a model); $D$: the data; $L(\cdot)$: length in bits
* $D \mid H$ reads "$D$ given $H$": what is left to describe once the model is known
* **MDL principle** (Rissanen, 1978): choose the model with the **shortest total** description

## The MDL Trade-Off

```{=latex}
\begin{center}
\begin{tikzpicture}
  \foreach \name/\lh/\ld/\y/\c in {too simple/0.6/5.6/2/cred, good model/1.6/2.2/1/cgreen, too complex/4.6/1.2/0/cred} {
    \node[anchor=east, font=\scriptsize] at (-0.1,\y+0.3) {\name};
    \fill[cblue!55] (0,\y) rectangle (\lh,\y+0.6);
    \fill[corange!65] (\lh,\y) rectangle ({\lh+\ld},\y+0.6);
    \node[anchor=west, font=\scriptsize, \c] at ({\lh+\ld+0.1},\y+0.3) {total $= \pgfmathparse{\lh+\ld}\pgfmathprintnumber{\pgfmathresult}$};
  }
  \fill[cblue!55] (0,-0.9) rectangle (0.4,-0.6); \node[anchor=west, font=\scriptsize] at (0.45,-0.75) {model $L(H)$};
  \fill[corange!65] (2.6,-0.9) rectangle (3,-0.6); \node[anchor=west, font=\scriptsize] at (3.05,-0.75) {errors $L(D \mid H)$};
\end{tikzpicture}
\end{center}
```

* **Too simple:** tiny model, but the errors take many bits
* **Too complex:** tiny errors, but the model itself is huge (it encodes the noise)
* The minimum total balances **fit** and **complexity** automatically

## MDL in Practice: BIC

$$L(D, H) \approx \underbrace{\frac{k}{2}\log_2 n}_{\text{model cost } L(H)} \;+\; \underbrace{\frac{n}{2}\log_2 \text{MSE}}_{\text{error cost } L(D \mid H)}$$

```{=latex}
\begin{center}
\begin{tikzpicture}
  \node[fillbox=cblue, text width=4.4cm, align=left] (a) {$k$: number of parameters\\ each costs $\frac{1}{2}\log_2 n$ bits\\ grows with model size};
  \node[fillbox=corange, text width=4.4cm, align=left, right=6mm of a] (b) {$n$: number of examples\\ smaller MSE $\Rightarrow$ fewer bits\\ shrinks as the fit improves};
\end{tikzpicture}
\end{center}
```

* This approximation is the **Bayesian Information Criterion (BIC)**, written in bits
* It uses **only the training data** to choose model complexity
* Constants are dropped, so values can be negative: only **differences** between models matter

## Worked Example: Choosing a Polynomial Degree

```{=latex}
\begin{center}\small
\begin{tabular}{@{}lccccc@{}}
\toprule
Model & $k$ & model cost & MSE & error cost & \textbf{total} \\
 & & $\frac{k}{2} \times 5.64$ & & $25 \times \log_2 \text{MSE}$ & \\
\midrule
degree 1 (line) & 2 & 5.6 & 0.167 & $-64.6$ & $-59.0$ \\
degree 5 & 6 & 16.9 & 0.042 & $-114.8$ & $\mathbf{-97.9}$ \\
degree 15 & 16 & 45.2 & 0.029 & $-127.6$ & $-82.5$ \\
\bottomrule
\end{tabular}
\end{center}
```

* Data from demo D1: $n = 50$ points, so $\log_2 50 \approx 5.64$
* Going from degree 5 to 15: errors save **12.8 bits**, the model costs **28.3 bits** more
* **Degree 5 wins**: the lowest total description length

## The MDL Curve

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=4.8cm, xlabel={polynomial degree $d$}, ylabel={description length (bits)}, xmin=0, xmax=15, legend pos=north east]
  \addplot[cpurple, mark=*, mark size=1.5pt] coordinates {(0,-14.5) (1,-59.0) (2,-56.2) (3,-96.7) (4,-94.4) (5,-97.9) (6,-96.1) (7,-93.3) (8,-90.5) (9,-90.9) (10,-90.7) (11,-88.0) (12,-86.5) (13,-84.0) (14,-84.4) (15,-82.5)};
  \addlegendentry{MDL / BIC}
  \draw[cred, dashed, thick] (axis cs:5,-100) -- (axis cs:5,-10) node[pos=0.9, right, font=\scriptsize] {minimum: $d = 5$};
\end{axis}
\end{tikzpicture}
\end{center}
```

* The curve drops while extra parameters capture **pattern**
* It rises again once extra parameters only encode **noise**

## Live Demo D3: Model Selection as Compression

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=4.4cm, ymode=log, xlabel={polynomial degree $d$}, ylabel={MSE (log scale)}, xmin=0, xmax=15, legend style={at={(0.5,1.03)}, anchor=south, legend columns=3}]
  \addplot[cblue, mark=*, mark size=1.2pt] coordinates {(0,0.618) (1,0.167) (2,0.167) (3,0.05) (4,0.049) (5,0.042) (6,0.040) (7,0.040) (8,0.040) (9,0.037) (10,0.034) (11,0.034) (12,0.033) (13,0.033) (14,0.030) (15,0.029)};
  \addlegendentry{training}
  \addplot[corange, mark=square*, mark size=1.2pt] coordinates {(0,0.626) (1,0.313) (2,0.313) (3,0.110) (4,0.111) (5,0.110) (6,0.115) (7,0.119) (8,0.118) (9,0.133) (10,0.169) (11,0.157) (12,0.241) (13,0.123) (14,2.94) (15,8.42)};
  \addlegendentry{test}
  \addplot[cgray, dashed, domain=0:15] {0.09};
  \addlegendentry{noise $\sigma^2$}
\end{axis}
\end{tikzpicture}
\end{center}
```

* Notebook `01_learning_as_compression.ipynb`, section **D3**
* Training error **always** decreases; test error explodes for $d \geq 14$
* MDL picked $d = 5$ **without** looking at the test data

# Compression in Modern Machine Learning

## Likelihood Is a Code Length

```{=latex}
\begin{center}
\begin{tikzpicture}
  \node[fillbox=cgreen, text width=3.4cm] (a) {model gives the data\\ \textbf{high} probability};
  \node[fillbox=cblue, text width=2.5cm, right=10mm of a] (b) {\textbf{short} code\\ few bits};
  \node[fillbox=cred, text width=3.4cm, below=5mm of a] (c) {model gives the data\\ \textbf{low} probability};
  \node[fillbox=corange, text width=2.5cm, right=10mm of c] (d) {\textbf{long} code\\ many bits};
  \draw[flow] (a) -- (b); \draw[flow] (c) -- (d);
\end{tikzpicture}
\end{center}
```

$$L(D \mid H) = -\log_2 P(D \mid \theta)$$

* $P(D \mid \theta)$: the probability the model with parameters $\theta$ assigns to the data (the **likelihood**)
* Maximizing likelihood = **minimizing code length**: the same objective

## Regularization Is a Description Length

```{=latex}
\begin{center}
\begin{tikzpicture}
  \fill[corange!65] (0,0) rectangle (5,0.8); \node at (2.5,0.4) {$\mathcal{L}_{data}(D;\theta)$: fit errors};
  \fill[cblue!55] (5,0) rectangle (7.5,0.8); \node at (6.25,0.4) {$\lambda\,\Omega(\theta)$};
  \node[note] at (2.5,-0.35) {$\approx L(D \mid H)$};
  \node[note] at (6.25,-0.35) {$\approx L(H)$};
  \draw[decorate, decoration={brace, amplitude=5pt}] (0,0.9) -- node[above=5pt, note] {total training loss $\mathcal{L}_{total}(\theta)$} (7.5,0.9);
\end{tikzpicture}
\end{center}
```

$$\mathcal{L}_{total}(\theta) = \mathcal{L}_{data}(D; \theta) + \lambda\, \Omega(\theta)$$

* $\Omega(\theta)$: a **penalty** for complex models, e.g. the sum of squared weights
* $\lambda$ (lambda): how much we care about simplicity versus fit
* The same two-part trade-off as MDL, in the language of optimization (Topic 3)

## Compression Across the Course

```{=latex}
\begin{center}
\begin{tikzpicture}
  % autoencoder
  \fill[cblue!40] (0,0) -- (1.2,0.5) -- (1.2,1.5) -- (0,2) -- cycle;
  \fill[cblue!70] (1.2,0.5) rectangle (1.6,1.5);
  \fill[cblue!40] (1.6,0.5) -- (2.8,0) -- (2.8,2) -- (1.6,1.5) -- cycle;
  \node[note, text width=3cm] at (1.4,-0.5) {AutoEncoder:\\ bottleneck (Topic 9)};
  % PCA
  \foreach \i in {1,...,25} { \fill[cgray] ({4.4+1.2*rand*0.8+0.8*rand*0.25},{1+0.6*rand*0.8+0.2*rand}) circle (1pt); }
  \draw[cred, very thick, ->] (3.6,0.5) -- (5.4,1.5);
  \node[note, text width=3cm] at (4.5,-0.5) {PCA: keep the main\\ direction (Topic 7)};
  % clustering
  \foreach \cx/\cy in {7.2/1.5, 8.5/0.6} { \foreach \i in {1,...,10} { \fill[cgray] ({\cx+0.35*rand},{\cy+0.35*rand}) circle (1pt); } \fill[cgreen] (\cx,\cy) circle (3pt); }
  \node[note, text width=3cm] at (7.8,-0.5) {Clustering: a few\\ prototypes (Topic 8)};
\end{tikzpicture}
\end{center}
```

* The same idea appears again and again: **summarize the data with less**

# From Compression to Optimization

## Finding the Model Is an Optimization Problem

```{=latex}
\begin{center}
\begin{tikzpicture}[scale=0.72]
\draw[->, thick] (-3.2,-2.2) -- (3.2,-2.2) node[below left, font=\scriptsize] {$w$};
  \draw[->, thick] (-3.2,-2.2) -- (-3.2,2.2) node[below left, font=\scriptsize] {$b$};
  \begin{scope}[rotate=25]
    \foreach \r/\c in {0.3/90, 0.7/70, 1.15/50, 1.6/35, 2.05/20} { \draw[cblue!\c!cgray, thick] (0.4,-0.2) ellipse ({1.5*\r} and {0.85*\r}); }
  \end{scope}
  \draw[cred, very thick, ->] (-2.6,1.7) -- (-1.6,1.2) -- (-0.7,0.6) -- (0,0.15) -- (0.35,-0.05);
  \foreach \px/\py in {-2.6/1.7, -1.6/1.2, -0.7/0.6, 0/0.15} { \fill[cred] (\px,\py) circle (1.8pt); }
  \node[cred, font=\scriptsize, anchor=north west] at (0.45,-0.1) {$\theta^\star$};
\end{tikzpicture}
\end{center}
```

$$\theta^\star = \arg\min_{\theta}\, \mathcal{L}(\theta)$$

* MDL tells us **what** a good model is; optimization tells us **how** to find its parameters
* Contours: lines of equal loss; the red path walks downhill to the minimum $\theta^\star$

## Why Not Try Every Value?

```{=latex}
\begin{center}
\begin{tikzpicture}
  \foreach \i in {0,...,9} { \fill[cblue] (\i*0.3,0) circle (2pt); }
  \node[note] at (1.35,-0.5) {1 parameter: 10 tries};
  \foreach \i in {0,...,9} { \foreach \j in {0,...,9} { \fill[corange] (4+\i*0.2,\j*0.2-0.6) circle (1.2pt); } }
  \node[note] at (4.9,-1.1) {2 parameters: 100 tries};
  \node[fillbox=cred, text width=3.5cm] at (9.3,0.3) {1000 parameters:\\ $10^{1000}$ tries\\ \textbf{impossible}};
\end{tikzpicture}
\end{center}
```

* A grid with 10 values per parameter needs $10^{k}$ evaluations for $k$ parameters
* Real models have thousands to billions of parameters
* We need **smarter search**

## Next Class: Two Ways to Search

```{=latex}
\begin{center}
\begin{tikzpicture}
  \begin{scope}
    \foreach \r in {0.4,0.9,1.4} { \draw[cgray] (0,0) ellipse ({\r*1.3} and \r); }
    \foreach \px/\py in {-1.5/0.8, 1.2/1.1, -0.6/-1.2, 1.6/-0.5, 0.3/0.9, -1.1/-0.3, 0.7/-0.8} { \fill[corange] (\px,\py) circle (2.5pt); }
    \fill[cred] (0.1,0.1) circle (3pt);
    \fill[cred] (0,0) circle (3pt); \node[note, text width=4cm] at (0,-2.2) {\textbf{Blind optimization}\\ a population of guesses,\\ no derivatives needed};
  \end{scope}
  \begin{scope}[xshift=6.2cm]
    \foreach \r in {0.4,0.9,1.4} { \draw[cgray] (0,0) ellipse ({\r*1.3} and \r); }
    \draw[cblue, very thick, ->] (-1.7,1.1) -- (-0.9,0.5) -- (-0.4,0.2) -- (-0.1,0.05);
    \node[note, text width=4cm] at (0,-2.2) {\textbf{Gradient-based}\\ follow the slope downhill,\\ derivatives by hand and JAX};
  \end{scope}
\end{tikzpicture}
\end{center}
```

# Evaluation and Statistical Validation

## Training Finished. Are We Done?

```{=latex}
\begin{center}
\begin{tikzpicture}
  \draw[thick, rounded corners=2pt] (0,0) rectangle (5,0.6);
  \fill[cgreen!60, rounded corners=2pt] (0,0) rectangle (5,0.6);
  \node at (2.5,0.3) {training: 100\% \quad loss = 0.001};
  \draw[flow, very thick] (5.4,0.3) -- (6.6,0.3);
  \node[fillbox=corange, text width=3.2cm] at (8.5,0.3) {deploy it on\\ real patients?};
  \node[font=\Huge, cred] at (8.5,-1) {?};
\end{tikzpicture}
\end{center}
```

* The optimizer stopped: it found parameters with **low training loss**
* That only says the model fits the **examples it was given**
* It says **nothing yet** about the cases the model will actually face

## Training Loss Is Not the Goal

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=4.5cm, xlabel={training time / model complexity}, ylabel={error}, xtick=\empty, ytick=\empty, xmin=0, xmax=10, ymin=0, ymax=1.1, legend style={at={(0.5,1.03)}, anchor=south, legend columns=2}]
  \addplot[cblue, very thick, domain=0:10, samples=60] {0.9*exp(-0.45*x)+0.03}; \addlegendentry{training error}
  \addplot[corange, very thick, domain=0:10, samples=60] {0.9*exp(-0.45*x)+0.012*x^1.8+0.12}; \addlegendentry{error on new data}
  \draw[cred, dashed] (axis cs:4.2,0) -- (axis cs:4.2,1.05) node[pos=0.95, right, font=\scriptsize] {best model};
\end{axis}
\end{tikzpicture}
\end{center}
```

* Training error keeps going **down**
* Error on new data goes down, then **up**: the model starts memorizing noise
* We need an **independent** measurement to see the second curve

## The Exam Analogy

```{=latex}
\begin{center}
\begin{tikzpicture}
  \node[fillbox=cblue, text width=4.2cm, minimum height=2.6cm] (a) {\textbf{Practice exam}\\[3pt] student memorized\\ all the answers\\[3pt] score: \textbf{100\%}};
  \node[fillbox=corange, text width=4.2cm, minimum height=2.6cm, right=12mm of a] (b) {\textbf{Real exam}\\[3pt] new questions,\\ same topics\\[3pt] score: \textbf{?}};
  \draw[flow, very thick] (a) -- (b);
\end{tikzpicture}
\end{center}
```

* A perfect score on questions you have already seen proves **memory**, not understanding
* To grade understanding, the teacher uses **questions the student never saw**
* Evaluation of a model follows exactly the same logic

## Why Measure Performance at All?

```{=latex}
\begin{center}
\begin{tikzpicture}[node distance=5mm]
  \node[fillbox=cblue, text width=4.2cm] (a) {\textbf{Is it good enough?}\\ deploy or not};
  \node[fillbox=corange, text width=4.2cm, right=of a] (b) {\textbf{Is it better than}\\ \textbf{what we have?}\\ baseline, current system};
  \node[fillbox=cgreen, text width=4.2cm, below=of a] (c) {\textbf{Which model?}\\ which settings?};
  \node[fillbox=cpurple, text width=4.2cm, below=of b] (d) {\textbf{How sure are we?}\\ could it be luck?};
\end{tikzpicture}
\end{center}
```

* Every decision about a model needs an **honest estimate** of its future performance
* Mistakes are costly: a missed diagnosis, a fraud not detected, a wrong forecast

## Hold Out Data

```{=latex}
\begin{center}
\begin{tikzpicture}
  \fill[cblue!55] (0,0) rectangle (7.5,0.8); \node at (3.75,0.4) {training set (75\%)};
  \fill[corange!70] (7.5,0) rectangle (10,0.8); \node at (8.75,0.4) {test (25\%)};
  \node[note] at (3.75,-0.4) {the model learns from these};
  \node[note] at (8.75,-0.4) {hidden until the end};
  \node[fillbox=cgray, text width=6cm] at (5,-1.5) {Decision tree (demo D4a):\\ training accuracy \textbf{1.000} \quad test accuracy \textbf{0.923}};
\end{tikzpicture}
\end{center}
```

* Split the data **before** training
* Accuracy on training data measures **memorization**; on the test set, **generalization**

## Training, Validation, and Test Sets

```{=latex}
\begin{center}
\begin{tikzpicture}
  \fill[cblue!55] (0,0) rectangle (6,0.8); \node at (3,0.4) {training};
  \fill[cgreen!55] (6,0) rectangle (8,0.8); \node at (7,0.4) {validation};
  \fill[corange!70] (8,0) rectangle (10,0.8); \node at (9,0.4) {test};
  \node[note, text width=3cm] at (3,-0.55) {fit the parameters};
  \node[note, text width=2.2cm] at (7,-0.55) {compare models,\\ tune settings};
  \node[note, text width=2.2cm] at (9,-0.55) {final estimate,\\ used \textbf{once}};
\end{tikzpicture}
\end{center}
```

* **Training set:** the model learns its parameters here
* **Validation set:** we make choices here (which model, which complexity)
* **Test set:** a final, unbiased estimate of performance on new data

## The Test Set Is Used Once

```{=latex}
\begin{center}
\begin{tikzpicture}
  \node[fillbox=cblue] (t) {train};
  \node[fillbox=cgreen, right=15mm of t] (v) {validate};
  \draw[flow] (t) to[bend left=35] node[above, note] {try another model} (v);
  \draw[flow] (v) to[bend left=35] node[below, note] {adjust and retrain} (t);
  \node[fillbox=corange, right=25mm of v] (x) {test};
  \draw[flow, very thick] (v) -- node[above, note] {once, at the end} (x);
  \node[fillbox=cred, text width=5cm, below=12mm of v] (w) {looking at the test score and then changing the model turns the test set into a validation set};
\end{tikzpicture}
\end{center}
```

* Every decision based on the test set makes its estimate **optimistic**

## One Split Is Not Enough

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, ybar interval, height=4.3cm, width=0.8\textwidth, xlabel={test accuracy}, ylabel={number of splits}, ymin=0, xticklabel style={/pgf/number format/precision=2}, xtick={0.874,0.897,0.920,0.942,0.965}]
  \addplot[fill=cblue!55, draw=cblue] coordinates {(0.874,2) (0.885,3) (0.897,1) (0.908,8) (0.920,14) (0.931,7) (0.942,4) (0.954,11) (0.965,0)};
\end{axis}
\end{tikzpicture}
\end{center}
```

* Same model, same data, **50 different random splits** (demo D4b)
* Test accuracy ranged from **0.874** to **0.965**
* A single number like "92.3%" hides this variation

## k-Fold Cross-Validation

```{=latex}
\begin{center}
\begin{tikzpicture}[yscale=0.55]
  \foreach \r in {1,...,5} {
    \node[anchor=east, font=\scriptsize] at (-0.1,-\r) {round \r};
    \foreach \c in {1,...,5} {
      \ifnum\c=\r \fill[corange!75] ({(\c-1)*1.6},-\r-0.4) rectangle ({\c*1.6-0.05},-\r+0.4);
      \else \fill[cblue!45] ({(\c-1)*1.6},-\r-0.4) rectangle ({\c*1.6-0.05},-\r+0.4); \fi
    }
    \node[anchor=west, font=\scriptsize] at (8.1,-\r) {score$_\r$};
  }
  \fill[cblue!45] (0,-6.4) rectangle (0.4,-5.9); \node[anchor=west, font=\scriptsize] at (0.45,-6.15) {train};
  \fill[corange!75] (2,-6.4) rectangle (2.4,-5.9); \node[anchor=west, font=\scriptsize] at (2.45,-6.15) {test fold};
\end{tikzpicture}
\end{center}
```

* Split the data into $k$ parts (**folds**); here $k = 5$
* Each round trains on $k - 1$ folds and tests on the remaining one
* Every example is tested **exactly once**; we get $k$ scores instead of one

## Cross-Validation Variants

```{=latex}
\begin{center}
\begin{tikzpicture}
  \node[note] at (2,1.2) {\textbf{Stratified}: same class mix per fold};
  \foreach \c in {0,1,2} { \fill[cblue!55] (\c*1.4,0) rectangle (\c*1.4+0.9,0.8); \fill[cred!60] (\c*1.4+0.9,0) rectangle (\c*1.4+1.3,0.8); }
  \node[note] at (8,1.2) {\textbf{Temporal}: never train on the future};
  \foreach \r in {0,1,2} {
    \fill[cblue!55] (5.5,-\r*0.45) rectangle ({6.5+\r*0.9},-\r*0.45+0.35);
    \fill[corange!75] ({6.5+\r*0.9},-\r*0.45) rectangle ({7.4+\r*0.9},-\r*0.45+0.35);
  }
  \draw[->, cgray] (5.5,-1.2) -- (10.5,-1.2) node[below left, note] {time};
\end{tikzpicture}
\end{center}
```

* **Stratified:** keep class proportions equal in every fold
* **Grouped:** all samples of the same patient/user stay in the same fold
* **Temporal:** for time series, always test on data **after** the training period

## Summarizing the Folds

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=3.8cm, width=0.85\textwidth, xmin=0.5, xmax=10.5, ymin=0.88, ymax=0.98, xlabel={fold}, ylabel={accuracy}, xtick={1,...,10}, ytick={0.9,0.93,0.96}]
  \addplot[only marks, mark=*, cblue] coordinates {(1,0.912) (2,0.930) (3,0.965) (4,0.895) (5,0.895) (6,0.965) (7,0.930) (8,0.912) (9,0.930) (10,0.929)};
  \addplot[cred, dashed, domain=0.5:10.5] {0.926};
  \node[cred, font=\scriptsize, anchor=south east] at (axis cs:10.4,0.927) {mean $\bar{s} = 0.926$};
\end{axis}
\end{tikzpicture}
\end{center}
```

* Decision tree, 10-fold cross-validation (demo D4c): 10 accuracy scores $s_1, \dots, s_{10}$
* **Mean:** $\bar{s} = \frac{1}{10}\sum_{i=1}^{10} s_i = 0.926$
* **Standard deviation:** $s = 0.025$ (the folds disagree by a few points)

## Confidence Interval

$$\bar{s} \;\pm\; t_{0.975,\,k-1} \times \frac{s}{\sqrt{k}}$$

```{=latex}
\begin{center}
\begin{tikzpicture}
  \node[fillbox=cblue, text width=2.6cm] (a) {$\bar{s}$\\ best guess\\ (mean score)};
  \node[fillbox=corange, text width=3.4cm, right=4mm of a] (b) {$s / \sqrt{k}$\\ standard error:\\ uncertainty of the mean};
  \node[fillbox=cgreen, text width=3.2cm, right=4mm of b] (c) {$t_{0.975,\,k-1}$\\ multiplier for 95\%\\ confidence ($\approx 2$)};
\end{tikzpicture}
\end{center}
```

* $k$: number of folds; $s$: standard deviation of the fold scores
* **95% confidence interval:** a range that, in repeated experiments, contains the true performance 95% of the time

## Worked Example: Confidence Interval

1. Mean and standard deviation: $\bar{s} = 0.926$, $s = 0.025$, $k = 10$
2. Standard error: $\frac{0.025}{\sqrt{10}} = \frac{0.025}{3.16} = 0.0079$
3. Multiplier (table value for $k-1 = 9$): $t_{0.975,\,9} = 2.26$
4. Half-width: $2.26 \times 0.0079 = 0.018$

```{=latex}
\begin{center}
\begin{tikzpicture}
  \draw[thick, ->] (0,0) -- (9,0);
  \foreach \v/\l in {0.5/0.90, 3.5/0.92, 6.5/0.94} { \draw (\v,0.1) -- (\v,-0.1) node[below, note] {\l}; }
  \draw[cblue, very thick, |-|] (1.7,0.6) -- (7.1,0.6);
  \fill[cred] (4.4,0.6) circle (3pt);
  \node[above, font=\scriptsize] at (4.4,0.75) {$0.926 \pm 0.018$ \quad i.e. $[0.908,\ 0.944]$};
\end{tikzpicture}
\end{center}
```

* Report: "accuracy **0.926 ± 0.018** (95% CI, 10-fold CV)", not just "92.6%"

## Always Compare with a Baseline

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, xbar, height=3.8cm, width=0.75\textwidth, symbolic y coords={logistic regression, decision tree, majority class}, ytick=data, xmin=0, xmax=1.1, xlabel={10-fold CV accuracy}, bar width=11pt, nodes near coords, nodes near coords style={font=\scriptsize}, enlarge y limits=0.25]
  \addplot[fill=cblue!55, draw=cblue] coordinates {(0.977,logistic regression) (0.926,decision tree) (0.627,majority class)};
\end{axis}
\end{tikzpicture}
\end{center}
```

* **Baseline:** the simplest possible predictor, e.g. "always predict the most common class"
* On this dataset the baseline already scores **62.7%**: "70% accuracy" would be poor
* A model is only useful if it clearly **beats the baseline**

## The Accuracy Paradox

```{=latex}
\begin{center}
\begin{tikzpicture}
  \foreach \i in {0,...,49} {
    \pgfmathtruncatemacro{\row}{\i/10}
    \pgfmathtruncatemacro{\col}{mod(\i,10)}
    \ifnum\i=37 \fill[cred] (\col*0.42,-\row*0.42) rectangle ++(0.36,0.36);
    \else \fill[cgreen!55] (\col*0.42,-\row*0.42) rectangle ++(0.36,0.36); \fi
  }
  \node[fillbox=cgray, text width=4.6cm, align=flush left] at (7.2,-0.7) {50 patients, \textcolor{cred}{1 sick}\\ model: "everyone is healthy"\\[3pt] accuracy = 49/50 = \textbf{98\%}\\ sick patients found = \textbf{0}};
\end{tikzpicture}
\end{center}
```

* When one class is rare, accuracy can look excellent for a **useless** model
* Demo D4d: 98% accuracy, balanced accuracy 50%, F1 = 0
* **The metric must match the goal**

## The Confusion Matrix

```{=latex}
\begin{center}
\begin{tikzpicture}
  \node[font=\scriptsize] at (2.5,2.35) {\textbf{predicted}};
  \node[font=\scriptsize, rotate=90] at (-1.25,0.8) {\textbf{actual}};
  \node[font=\scriptsize] at (1.5,1.95) {sick}; \node[font=\scriptsize] at (3.5,1.95) {healthy};
  \node[font=\scriptsize, anchor=east] at (0.45,1.3) {sick}; \node[font=\scriptsize, anchor=east] at (0.45,0.3) {healthy};
  \draw[fill=cgreen!45] (0.5,0.8) rectangle (2.5,1.8); \node[align=center, font=\scriptsize] at (1.5,1.3) {\textbf{TP} = 6\\ true positive};
  \draw[fill=cred!35] (2.5,0.8) rectangle (4.5,1.8); \node[align=center, font=\scriptsize] at (3.5,1.3) {\textbf{FN} = 4\\ missed};
  \draw[fill=corange!40] (0.5,-0.2) rectangle (2.5,0.8); \node[align=center, font=\scriptsize] at (1.5,0.3) {\textbf{FP} = 2\\ false alarm};
  \draw[fill=cgreen!25] (2.5,-0.2) rectangle (4.5,0.8); \node[align=center, font=\scriptsize] at (3.5,0.3) {\textbf{TN} = 88\\ true negative};
\end{tikzpicture}
\end{center}
```

* 100 patients, 10 of them sick; the model flags 8 patients as sick
* Each cell counts one kind of **right** or **wrong** decision

## Precision, Recall, and F1

$$\text{precision} = \frac{TP}{TP + FP} \qquad \text{recall} = \frac{TP}{TP + FN}$$

$$F_1 = \frac{2 \times \text{precision} \times \text{recall}}{\text{precision} + \text{recall}}$$

```{=latex}
\begin{center}
\begin{tikzpicture}
  \node[fillbox=corange, text width=3.1cm] (p) {\textbf{precision}\\ of those flagged,\\ how many are sick?\\ $\frac{6}{6+2} = 0.75$};
  \node[fillbox=cblue, text width=3.1cm, right=5mm of p] (r) {\textbf{recall}\\ of the sick,\\ how many were found?\\ $\frac{6}{6+4} = 0.60$};
  \node[fillbox=cgreen, text width=3.1cm, right=5mm of r] (f) {\textbf{F1}\\ balance of both\\ $\frac{2 \times 0.75 \times 0.6}{0.75 + 0.6}$\\ $= 0.67$};
\end{tikzpicture}
\end{center}
```

* Screening for a disease: high **recall** matters (do not miss sick patients)
* Spam filter: high **precision** matters (do not hide real e-mails)

## Comparing Two Models: Pair the Scores

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=4cm, width=0.8\textwidth, xmin=0.5, xmax=10.5, ymin=0.86, ymax=1.01, xlabel={fold}, ylabel={accuracy}, xtick={1,...,10}, legend style={at={(0.5,1.03)}, anchor=south, legend columns=2}]
  \addplot[only marks, mark=*, corange] coordinates {(1,0.912) (2,0.930) (3,0.965) (4,0.895) (5,0.895) (6,0.965) (7,0.930) (8,0.912) (9,0.930) (10,0.929)}; \addlegendentry{decision tree}
  \addplot[only marks, mark=square*, cblue] coordinates {(1,0.982) (2,0.982) (3,0.947) (4,0.947) (5,0.982) (6,0.947) (7,1.000) (8,0.982) (9,0.982) (10,1.000)}; \addlegendentry{logistic regression}
  \foreach \f/\a/\b in {1/0.912/0.982, 2/0.930/0.982, 3/0.965/0.947, 4/0.895/0.947, 5/0.895/0.982, 6/0.965/0.947, 7/0.930/1.000, 8/0.912/0.982, 9/0.930/0.982, 10/0.929/1.000} {
    \edef\temp{\noexpand\draw[cgray] (axis cs:\f,\a) -- (axis cs:\f,\b);}\temp
  }
\end{axis}
\end{tikzpicture}
\end{center}
```

* Evaluate both models on **the same folds** and look at the difference per fold: $d_i = a_i - b_i$
* Pairing removes the variation caused by "easy" and "hard" folds
* Logistic regression wins in 8 of 10 folds, but the tree wins in folds 3 and 6

## Why a Naive t-Test Is Over-Confident

```{=latex}
\begin{center}
\begin{tikzpicture}[yscale=0.5]
  \foreach \r in {1,...,4} {
    \foreach \c in {1,...,4} {
      \ifnum\c=\r \fill[corange!75] ({(\c-1)*1.5},-\r-0.4) rectangle ({\c*1.5-0.05},-\r+0.4);
      \else \fill[cblue!45] ({(\c-1)*1.5},-\r-0.4) rectangle ({\c*1.5-0.05},-\r+0.4); \fi
    }
  }
  \draw[cred, very thick] (1.5,-0.4) rectangle (5.95,-2.6);
  \node[fillbox=cred, text width=4.2cm, anchor=west] at (6.6,-2.5) {rounds 1 and 2 share\\ \textbf{2 of their 3} training folds:\\ their scores are \textbf{not independent}};
\end{tikzpicture}
\end{center}
```

* A standard t-test assumes the $k$ scores are **independent** measurements
* In cross-validation the training sets **overlap**, so the scores are correlated
* The naive test underestimates the variance and finds "significant" differences too easily

## The Corrected Resampled t-Test

$$t = \frac{\bar{d}}{\sqrt{\left(\dfrac{1}{k} + \dfrac{n_{test}}{n_{train}}\right) s_d^2}}$$

* $\bar{d}$: mean of the per-fold differences; $s_d^2$: their variance; $k$: number of scores
* $\frac{n_{test}}{n_{train}}$: extra term that **inflates the variance** to account for overlap (Nadeau & Bengio, 2003)
* Demo D4e, logistic regression vs. decision tree, 5 $\times$ 10-fold CV, mean difference $+0.052$:

```{=latex}
\begin{center}
\begin{tikzpicture}
  \node[fillbox=cred, text width=4cm] (a) {naive paired t-test\\ $p = 3.6 \times 10^{-15}$\\ (far too confident)};
  \node[fillbox=cgreen, text width=4cm, right=8mm of a] (b) {corrected test\\ $p = 6.0 \times 10^{-5}$\\ (still significant, honestly)};
\end{tikzpicture}
\end{center}
```

## McNemar's Test: One Test Set, Two Models

```{=latex}
\begin{center}
\begin{tikzpicture}
  \node[font=\scriptsize, anchor=east] at (-0.1,0.6) {model A};
  \node[font=\scriptsize, anchor=east] at (-0.1,0) {model B};
  \foreach \i/\a/\b in {0/1/1, 1/1/1, 2/1/0, 3/1/1, 4/0/1, 5/1/1, 6/1/0, 7/1/1, 8/0/0, 9/1/1, 10/1/0, 11/1/1} {
    \ifnum\a=1 \fill[cgreen!60] (\i*0.62,0.4) rectangle ++(0.5,0.4); \else \fill[cred!60] (\i*0.62,0.4) rectangle ++(0.5,0.4); \fi
    \ifnum\b=1 \fill[cgreen!60] (\i*0.62,-0.2) rectangle ++(0.5,0.4); \else \fill[cred!60] (\i*0.62,-0.2) rectangle ++(0.5,0.4); \fi
  }
  \foreach \i in {2,4,6,10} { \draw[cblue, very thick] (\i*0.62-0.06,-0.26) rectangle ++(0.62,1.12); }
  \node[note] at (3.7,-0.6) {test examples: \textcolor{cgreen!70!black}{correct}, \textcolor{cred}{wrong}; \textcolor{cblue}{boxes}: the models disagree};
\end{tikzpicture}
\end{center}
```

* Sometimes cross-validation is too expensive: we train once and have **one test set**
* Both models predict the **same** test examples
* Only the examples where they **disagree** tell us which model is better

## McNemar's Contingency Table

```{=latex}
\begin{center}
\begin{tikzpicture}
  \node[font=\scriptsize] at (2.8,2.35) {\textbf{decision tree}};
  \node[font=\scriptsize, rotate=90] at (-1.55,0.8) {\textbf{logistic regression}};
  \node[font=\scriptsize] at (1.65,1.95) {correct}; \node[font=\scriptsize] at (3.95,1.95) {wrong};
  \node[font=\scriptsize, anchor=east] at (0.45,1.3) {correct}; \node[font=\scriptsize, anchor=east] at (0.45,0.3) {wrong};
  \draw[fill=cgray!20] (0.5,0.8) rectangle (2.8,1.8); \node[align=center, font=\scriptsize] at (1.65,1.3) {$a = 131$\\ both right};
  \draw[fill=cblue!40, very thick] (2.8,0.8) rectangle (5.1,1.8); \node[align=center, font=\scriptsize] at (3.95,1.3) {$b = 10$\\ only LR right};
  \draw[fill=corange!45, very thick] (0.5,-0.2) rectangle (2.8,0.8); \node[align=center, font=\scriptsize] at (1.65,0.3) {$c = 1$\\ only tree right};
  \draw[fill=cgray!20] (2.8,-0.2) rectangle (5.1,0.8); \node[align=center, font=\scriptsize] at (3.95,0.3) {$d = 1$\\ both wrong};
\end{tikzpicture}
\end{center}
```

* Demo D4f: the 143 test examples of the breast cancer dataset
* $a$ and $d$ say **nothing** about which model is better: both behave the same
* If the models were equally good, the disagreements would split evenly: $b \approx c$

## McNemar's Statistic

$$\chi^2 = \frac{\left(\,|b - c| - 1\,\right)^2}{b + c}$$

```{=latex}
\begin{center}
\begin{tikzpicture}
  \node[fillbox=cblue, text width=3.3cm] (a) {$|b - c|$\\ how unbalanced\\ they are};
  \node[fillbox=corange, text width=2.6cm, right=5mm of a] (b) {$-1$\\ small-sample\\ correction};
  \node[fillbox=cgreen, text width=3cm, right=5mm of b] (c) {$b + c$\\ total number of\\ disagreements};
\end{tikzpicture}
\end{center}
```

* $\chi^2$ ("chi-squared"): large when one model wins most disagreements
* If both models are equally good, $\chi^2$ follows a **chi-squared distribution with 1 degree of freedom**
* **p-value:** probability of a $\chi^2$ this large (or larger) if the models were equally good
* If $b + c$ is small (below about 25), use the exact **binomial** version

## Worked Example: McNemar

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=3.8cm, width=0.7\textwidth, xmin=0, xmax=10, ymin=0, ymax=0.45, xlabel={$\chi^2$}, ylabel={density}, ytick=\empty, samples=120, clip=false]
  \addplot[cblue, very thick, domain=0.35:10] {exp(-x/2)/sqrt(2*pi*x)};
  \addplot[fill=cred!50, draw=none, domain=5.82:10] {exp(-x/2)/sqrt(2*pi*x)} \closedcycle;
  \draw[cred, thick] (axis cs:5.82,0) -- (axis cs:5.82,0.2) node[above, font=\scriptsize] {$\chi^2 = 5.82$};
  \node[cred, font=\scriptsize, anchor=west] at (axis cs:6.3,0.08) {shaded area $= p = 0.016$};
\end{axis}
\end{tikzpicture}
\end{center}
```

1. Disagreements: $b = 10$, $c = 1$, so $|b - c| = 9$ and $b + c = 11$
2. Statistic: $\chi^2 = \frac{(9 - 1)^2}{11} = \frac{64}{11} = 5.82$
3. p-value $= 0.016 < 0.05$ (exact binomial: $0.012$): logistic regression is **significantly** better

## Which Test When?

```{=latex}
\begin{center}
\begin{tikzpicture}[node distance=7mm]
  \node[fillbox=cgray, text width=4cm] (q) {How was the model evaluated?};
  \node[fillbox=cblue, text width=3.6cm, below left=9mm and -1.2cm of q] (cv) {repeated $k$-fold\\ cross-validation};
  \node[fillbox=corange, text width=3.6cm, below right=9mm and -1.2cm of q] (ts) {a single held-out\\ test set};
  \node[fillbox=cblue, text width=3.6cm, below=of cv] (t) {\textbf{corrected resampled}\\ \textbf{t-test}};
  \node[fillbox=corange, text width=3.6cm, below=of ts] (m) {\textbf{McNemar's test}};
  \draw[flow] (q) -- (cv); \draw[flow] (q) -- (ts); \draw[flow] (cv) -- (t); \draw[flow] (ts) -- (m);
\end{tikzpicture}
\end{center}
```

* Both answer: "is the difference **real**, or could it be **luck**?"
* A significant difference must also be **large enough to matter** in practice

## Live Demo D4: Evaluation and Validation

```{=latex}
\begin{center}
\begin{tikzpicture}[node distance=3mm]
  \node[fillbox=cblue, text width=2.6cm] (a) {\textbf{D4a} training\\ vs. test};
  \node[fillbox=cblue, text width=2.6cm, right=of a] (b) {\textbf{D4b} 50 random\\ splits};
  \node[fillbox=cblue, text width=2.6cm, right=of b] (c) {\textbf{D4c} 10-fold CV\\ and CI};
  \node[fillbox=corange, text width=2.6cm, below=of a] (d) {\textbf{D4d} baselines\\ and metrics};
  \node[fillbox=corange, text width=2.6cm, right=of d] (e) {\textbf{D4e} corrected\\ t-test};
  \node[fillbox=corange, text width=2.6cm, right=of e] (f) {\textbf{D4f} McNemar's\\ test};
\end{tikzpicture}
\end{center}
```

* Notebook `02_evaluation_validation.ipynb`, breast cancer dataset (569 patients, 30 features)
* Evaluation will be **revisited in every topic**: regression metrics, clustering without labels, time series

# Limitations and Pitfalls

## A Map of the Pitfalls

```{=latex}
\begin{center}
\begin{tikzpicture}[scale=0.85, every node/.append style={transform shape}]
  \node[fillbox=cred, text width=2.9cm] (c) at (0,0) {\textbf{Why models fail}};
  \foreach \ang/\txt/\col in {90/Under- and overfitting/cblue, 30/Curse of dimensionality/corange, -30/No Free Lunch/cgreen, -90/Data leakage/cpurple, -150/Correlation $\neq$ causation/csky, 150/Shortcuts and shift/cgray} {
    \node[fillbox=\col, text width=2.5cm] (n\ang) at (\ang:2.6cm) {\txt};
    \draw[thick, cgray] (c) -- (n\ang);
  }
\end{tikzpicture}
\end{center}
```

* A model can score well in the lab and **still fail** in the real world
* Knowing these pitfalls is part of doing machine learning **correctly**

# Under- and Overfitting

## Underfitting: Too Simple

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=4.4cm, width=0.75\textwidth, xmin=0, xmax=1, ymin=-1.8, ymax=1.5, xtick=\empty, ytick=\empty, xlabel={$x$}, ylabel={$y$}]
  \addplot[only marks, mark=*, mark size=1.4pt] coordinates {(0.04,0.48) (0.13,0.29) (0.19,0.86) (0.31,0.42) (0.37,0.94) (0.44,0.61) (0.48,-0.12) (0.67,-1.0) (0.7,-0.81) (0.77,-1.35) (0.82,-0.75) (0.86,-0.9) (0.97,-0.58)};
  \addplot[cgray, dashed, domain=0:1, samples=60] {sin(deg(2*pi*x))};
  \addplot[cred, very thick, domain=0:1] {0.9-1.8*x};
\end{axis}
\end{tikzpicture}
\end{center}
```

* A straight line cannot follow a wave: the model is **too rigid**
* High error on training data **and** on new data
* The model misses real **pattern** (it compresses too much)

## A Good Fit

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=4.4cm, width=0.75\textwidth, xmin=0, xmax=1, ymin=-1.8, ymax=1.5, xtick=\empty, ytick=\empty, xlabel={$x$}, ylabel={$y$}]
  \addplot[only marks, mark=*, mark size=1.4pt] coordinates {(0.04,0.48) (0.13,0.29) (0.19,0.86) (0.31,0.42) (0.37,0.94) (0.44,0.61) (0.48,-0.12) (0.67,-1.0) (0.7,-0.81) (0.77,-1.35) (0.82,-0.75) (0.86,-0.9) (0.97,-0.58)};
  \addplot[cgray, dashed, domain=0:1, samples=60] {sin(deg(2*pi*x))};
  \addplot[cgreen!70!black, very thick, domain=0:1, samples=60] {1.05*sin(deg(2*pi*x))-0.05};
\end{axis}
\end{tikzpicture}
\end{center}
```

* The model follows the **pattern** and ignores the **noise**
* Small training error, and similar error on new data
* This is the goal: the shortest description that still explains the data

## Overfitting: Too Flexible

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=4.4cm, width=0.75\textwidth, xmin=0, xmax=1, ymin=-1.8, ymax=1.5, xtick=\empty, ytick=\empty, xlabel={$x$}, ylabel={$y$}]
  \addplot[only marks, mark=*, mark size=1.4pt] coordinates {(0.04,0.48) (0.13,0.29) (0.19,0.86) (0.31,0.42) (0.37,0.94) (0.44,0.61) (0.48,-0.12) (0.67,-1.0) (0.7,-0.81) (0.77,-1.35) (0.82,-0.75) (0.86,-0.9) (0.97,-0.58)};
  \addplot[cgray, dashed, domain=0:1, samples=60] {sin(deg(2*pi*x))};
  \addplot[cred, very thick, smooth, tension=0.8] coordinates {(0,1.3) (0.04,0.48) (0.085,-0.3) (0.13,0.29) (0.19,0.86) (0.25,-0.2) (0.31,0.42) (0.37,0.94) (0.44,0.61) (0.48,-0.12) (0.58,0.4) (0.67,-1.0) (0.7,-0.81) (0.77,-1.35) (0.82,-0.75) (0.86,-0.9) (0.92,0.1) (0.97,-0.58) (1,-1.6)};
\end{axis}
\end{tikzpicture}
\end{center}
```

* The curve passes through **every** point, including the noise
* Training error near zero, but **large** error on new data
* The model memorized the sample instead of learning the pattern

## Error Versus Model Complexity

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=4.6cm, xmin=0, xmax=10, ymin=0, ymax=1.1, xtick=\empty, ytick=\empty, xlabel={model complexity}, ylabel={error}, legend style={at={(0.5,1.03)}, anchor=south, legend columns=2}]
  \fill[cred!10] (axis cs:0,0) rectangle (axis cs:2.8,1.1);
  \fill[cgreen!12] (axis cs:2.8,0) rectangle (axis cs:5.5,1.1);
  \fill[cred!10] (axis cs:5.5,0) rectangle (axis cs:10,1.1);
  \addplot[cblue, very thick, domain=0:10, samples=60] {0.9*exp(-0.4*x)+0.03}; \addlegendentry{training error}
  \addplot[corange, very thick, domain=0:10, samples=60] {0.9*exp(-0.4*x)+0.01*x^2+0.1}; \addlegendentry{test error}
  \node[font=\scriptsize] at (axis cs:1.4,0.95) {underfitting};
  \node[font=\scriptsize] at (axis cs:4.15,0.95) {good};
  \node[font=\scriptsize] at (axis cs:7.7,0.95) {overfitting};
\end{axis}
\end{tikzpicture}
\end{center}
```

* **Training error** always decreases as the model becomes more complex
* **Test error** is U-shaped: the best model is in the middle

## Bias and Variance: the Dartboard

```{=latex}
\begin{center}
\begin{tikzpicture}[scale=0.62]
  \foreach \bx/\by/\title/\cx/\cy/\spread in {0/0/{low bias, low variance}/0/0/0.25, 4.6/0/{low bias, high variance}/0/0/1.0, 9.2/0/{high bias, low variance}/0.9/0.8/0.25, 13.8/0/{high bias, high variance}/0.9/0.8/1.0} {
    \begin{scope}[shift={(\bx,\by)}]
      \foreach \r/\c in {1.8/cgray!15, 1.2/cgray!30, 0.6/cred!40} { \fill[\c] (0,0) circle (\r); \draw[cgray] (0,0) circle (\r); }
      \foreach \i in {1,...,9} { \fill[cblue] ({\cx+\spread*rand},{\cy+\spread*rand}) circle (2.5pt); }
      \node[font=\scriptsize, align=center, text width=2.8cm] at (0,-2.3) {\title};
    \end{scope}
  }
\end{tikzpicture}
\end{center}
```

* Centre of the target = the true pattern; each dart = a model trained on a **different** sample
* **Bias:** the darts land off-centre **systematically**
* **Variance:** the darts are **scattered**

## Bias: Wrong in the Same Way

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=4.2cm, width=0.75\textwidth, xmin=0, xmax=1, ymin=-1.6, ymax=1.6, xtick=\empty, ytick=\empty, xlabel={$x$}]
  \foreach \s in {-0.15,-0.08,0,0.07,0.14} {
    \edef\temp{\noexpand\addplot[cblue!60, thick, domain=0:1] {0.95+\s-1.9*x};}\temp
  }
  \addplot[black, dashed, very thick, domain=0:1, samples=60] {sin(deg(2*pi*x))};
\end{axis}
\end{tikzpicture}
\end{center}
```

* Five straight lines, each fitted to a **different** noisy sample (blue); dashed: the truth
* The lines agree with each other, but all miss the wave in the same way
* **Bias** = the error that remains even with the **average** of many models

## Variance: Different Every Time

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=4.2cm, width=0.75\textwidth, xmin=0, xmax=1, ymin=-2, ymax=2, xtick=\empty, ytick=\empty, xlabel={$x$}]
  \addplot[cblue!60, thick, smooth] coordinates {(0,1.4) (0.1,0.2) (0.2,1.3) (0.3,0.6) (0.4,1.0) (0.5,-0.4) (0.6,-0.4) (0.7,-1.3) (0.8,-0.6) (0.9,-1.2) (1,0.8)};
  \addplot[corange!80, thick, smooth] coordinates {(0,-0.6) (0.1,0.9) (0.2,0.6) (0.3,1.4) (0.4,0.2) (0.5,0.3) (0.6,-1.2) (0.7,-0.6) (0.8,-1.4) (0.9,-0.2) (1,-1.5)};
  \addplot[cgreen!70!black, thick, smooth] coordinates {(0,0.5) (0.1,0.4) (0.2,1.1) (0.3,0.8) (0.4,0.7) (0.5,-0.1) (0.6,-0.9) (0.7,-0.8) (0.8,-1.1) (0.9,-0.9) (1,1.2)};
  \addplot[black, dashed, very thick, domain=0:1, samples=60] {sin(deg(2*pi*x))};
\end{axis}
\end{tikzpicture}
\end{center}
```

* Three very flexible models, each trained on a **different** noisy sample
* On average they follow the wave, but each one wiggles **differently**
* **Variance** = how much the model changes when the training sample changes

## The Bias--Variance Decomposition

$$\underbrace{\mathbb{E}\big[(y - \hat{f}(x))^2\big]}_{\text{expected test error}} = \underbrace{\text{bias}^2}_{\text{too rigid}} + \underbrace{\text{variance}}_{\text{too sensitive}} + \underbrace{\sigma^2}_{\text{noise}}$$

```{=latex}
\begin{center}
\begin{tikzpicture}
  \fill[cred!55] (0,0) rectangle (2.5,0.7); \node at (1.25,0.35) {bias$^2$};
  \fill[cblue!55] (2.5,0) rectangle (5,0.7); \node at (3.75,0.35) {variance};
  \fill[cgray!50] (5,0) rectangle (7,0.7); \node at (6,0.35) {noise $\sigma^2$};
  \node[note] at (6,-0.35) {cannot be reduced};
\end{tikzpicture}
\end{center}
```

* $\mathbb{E}[\cdot]$: the **average** over many possible training sets
* **bias** = (average prediction $-$ truth); **variance** = spread of predictions around their average
* The noise $\sigma^2$ sets a floor that **no model** can beat

## The Bias--Variance Trade-Off

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=4.6cm, xmin=0, xmax=10, ymin=0, ymax=1.2, xtick=\empty, ytick=\empty, xlabel={model complexity}, ylabel={error}, legend style={at={(0.5,1.03)}, anchor=south, legend columns=4}]
  \addplot[cred, very thick, domain=0.3:10, samples=60] {0.9*exp(-0.5*x)}; \addlegendentry{bias$^2$}
  \addplot[cblue, very thick, domain=0.3:10, samples=60] {0.008*x^2}; \addlegendentry{variance}
  \addplot[cgray, dashed, domain=0.3:10] {0.1}; \addlegendentry{noise}
  \addplot[black, very thick, domain=0.3:10, samples=60] {0.9*exp(-0.5*x)+0.008*x^2+0.1}; \addlegendentry{total}
  \draw[cgreen!60!black, dashed, thick] (axis cs:4.1,0) -- (axis cs:4.1,1.1) node[pos=0.92, right, font=\scriptsize] {sweet spot};
\end{axis}
\end{tikzpicture}
\end{center}
```

* Making the model more complex **lowers bias** but **raises variance**
* The best complexity **minimizes the sum**

## What to Do About It

```{=latex}
\begin{center}
\begin{tikzpicture}
  \node[fillbox=cred, text width=4.4cm, align=flush left, minimum height=2.6cm] (u) {\textbf{Underfitting (high bias)}\\[3pt] $\bullet$ a more flexible model\\ $\bullet$ better features\\ $\bullet$ less regularization\\ $\bullet$ train longer};
  \node[fillbox=cblue, text width=4.4cm, align=flush left, minimum height=2.6cm, right=8mm of u] (o) {\textbf{Overfitting (high variance)}\\[3pt] $\bullet$ more training data\\ $\bullet$ a simpler model\\ $\bullet$ more regularization\\ $\bullet$ ensembles (Topic 6)};
\end{tikzpicture}
\end{center}
```

* Diagnose first: compare training error and validation error
* Both high $\Rightarrow$ underfitting
* Training low, validation high $\Rightarrow$ overfitting

## Live Demo D5: Bias and Variance

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, ybar, height=4.2cm, width=0.8\textwidth, bar width=14pt, symbolic x coords={degree 1, degree 5, degree 15}, xtick=data, ymode=log, ymin=0.0005, ymax=3, ylabel={value (log scale)}, legend style={at={(0.5,1.03)}, anchor=south, legend columns=2}, enlarge x limits=0.25, log origin=infty]
  \addplot[fill=cred!60, draw=cred] coordinates {(degree 1,0.206) (degree 5,0.001) (degree 15,0.019)}; \addlegendentry{bias$^2$}
  \addplot[fill=cblue!60, draw=cblue] coordinates {(degree 1,0.007) (degree 5,0.025) (degree 15,1.084)}; \addlegendentry{variance}
\end{axis}
\end{tikzpicture}
\end{center}
```

* Notebook `03_pitfalls.ipynb`, section **D5**: 30 training sets, each with fresh noise
* Degree 1: **bias** dominates (0.206)
* Degree 15: **variance** explodes (1.084)
* Degree 5: both small, the best compromise

# The Curse of Dimensionality

## Intuition Breaks in High Dimensions

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=4.3cm, width=0.72\textwidth, xmin=1, xmax=100, ymin=0, ymax=1.05, xmode=log, xlabel={number of dimensions $d$}, ylabel={edge length needed}, log basis x=10]
  \addplot[cblue, very thick, domain=1:100, samples=60] {0.1^(1/x)};
  \node[font=\scriptsize, anchor=west] at (axis cs:1.6,0.08) {$d=1$: 10\% of the range};
  \node[font=\scriptsize, anchor=east] at (axis cs:95,0.72) {$d=100$: 98\% of every axis};
\end{axis}
\end{tikzpicture}
\end{center}
```

* To collect the **10% nearest** examples in a unit cube, the neighbourhood edge must be $0.1^{1/d}$
* In high dimensions a "local" neighbourhood covers almost the **whole range** of every feature
* "Near" stops meaning "similar"

## Covering the Space Needs Exponential Data

```{=latex}
\begin{center}
\begin{tikzpicture}
  \foreach \i in {0,...,9} { \draw[fill=cblue!30] (\i*0.25,0) rectangle ++(0.25,0.25); }
  \node[note] at (1.25,-0.4) {$d = 1$: 10 cells};
  \foreach \i in {0,...,9} { \foreach \j in {0,...,9} { \draw[fill=corange!30] (3.5+\i*0.2,\j*0.2-0.9) rectangle ++(0.2,0.2); } }
  \node[note] at (4.5,-1.3) {$d = 2$: 100 cells};
  \begin{scope}[shift={(7.2,-0.9)}]
    \foreach \i in {0,...,10} { \draw[cgreen!70!black, thin] (\i*0.2,0) -- (\i*0.2,2); \draw[cgreen!70!black, thin] (0,\i*0.2) -- (2,\i*0.2);
      \draw[cgreen!70!black, thin] (\i*0.2,2) -- ++(0.7,0.7); \draw[cgreen!70!black, thin] (2,\i*0.2) -- ++(0.7,0.7); }
    \draw[cgreen!70!black, thin] (0.7,2.7) -- (2.7,2.7) -- (2.7,0.7);
  \end{scope}
  \node[note] at (8.3,-1.3) {$d = 3$: 1000 cells};
\end{tikzpicture}
\end{center}
```

* Split each feature into 10 intervals: the number of cells is $10^d$
* To have **one** example per cell we need $10^d$ examples
* With 20 features: $10^{20}$ examples, far more than any dataset

## Data Needed Grows Exponentially

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=4.4cm, width=0.75\textwidth, xmin=1, xmax=20, ymode=log, xlabel={number of features $d$}, ylabel={examples needed ($10^d$)}, ytick={10,1e5,1e10,1e15,1e20}]
  \addplot[cred, very thick, mark=*, mark size=1.2pt, domain=1:20, samples=20] {10^x};
  \draw[cblue, dashed, thick] (axis cs:1,1e6) -- (axis cs:20,1e6) node[pos=0.02, above, anchor=south west, font=\scriptsize] {a large dataset: one million examples};
\end{axis}
\end{tikzpicture}
\end{center}
```

* A straight line on a log scale means **exponential** growth
* Beyond 6 features, one million examples can no longer fill the grid
* Real data survive only because they live near **lower-dimensional structure**

## Distances Concentrate

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=4.3cm, width=0.75\textwidth, xmode=log, ymode=log, xlabel={number of dimensions $d$}, ylabel={relative contrast}]
  \addplot[cblue, very thick, mark=*, mark size=1.5pt] coordinates {(1,1609) (2,193) (5,6.26) (10,2.78) (20,1.12) (50,0.573) (100,0.395) (500,0.142) (1000,0.110)};
\end{axis}
\end{tikzpicture}
\end{center}
```

$$\text{relative contrast} = \frac{d_{\max} - d_{\min}}{d_{\min}}$$

* $d_{\min}$, $d_{\max}$: distance from a query to its **nearest** and **farthest** of 500 random points
* In 1000 dimensions the farthest point is only **11%** farther than the nearest

## Almost All Volume Is Near the Border

```{=latex}
\begin{center}
\begin{tikzpicture}
  \fill[cred!45] (0,0) rectangle (2.4,2.4); \fill[cblue!40] (0.12,0.12) rectangle (2.28,2.28);
  \node[align=center, font=\scriptsize] at (1.2,1.2) {inner\\ square};
  \node[note, text width=2.8cm] at (1.2,-0.45) {red: within 5\% of the border};
\end{tikzpicture}
\hspace{4mm}
\begin{tikzpicture}
\begin{axis}[faa, height=4cm, width=0.52\textwidth, xmode=log, xmin=1, xmax=1000, ymin=0, ymax=1.05, xlabel={dimension $d$}, ylabel={fraction near border}]
  \addplot[cred, very thick, domain=1:1000, samples=80] {1-0.9^x};
\end{axis}
\end{tikzpicture}
\end{center}
```

$$\text{fraction within 5\% of the border} = 1 - 0.9^d$$

* The inner cube has side $0.9$, so its volume is $0.9^d$; the rest is the border region
* $d = 2$: 19% of the volume; $d = 50$: **99.5%**
* Almost every point is an "edge case"

## Irrelevant Features Hurt

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=4.3cm, width=0.75\textwidth, xmode=log, xmin=1, xmax=1000, ymin=0.45, ymax=1, xlabel={number of pure-noise features added (log scale)}, ylabel={$k$-NN accuracy}, log origin=infty]
  \addplot[corange, very thick, mark=*, mark size=1.5pt] coordinates {(1,0.98) (5,0.965) (10,0.922) (50,0.755) (100,0.693) (500,0.58) (1000,0.555)};
  \addplot[cgray, dashed, domain=1:1000] {0.5};
  \node[font=\scriptsize, cgray, anchor=south west] at (axis cs:1.1,0.5) {chance};
\end{axis}
\end{tikzpicture}
\end{center}
```

* 2 informative features + an increasing number of **random** features
* The signal is still there, but distances are dominated by noise
* More features is **not** automatically better

## Live Demo D6: Curse of Dimensionality

```{=latex}
\begin{center}
\begin{tikzpicture}[node distance=4mm]
  \node[fillbox=cblue, text width=3cm] (a) {\textbf{distance contrast}\\ 500 random points\\ $d = 1 \dots 1000$};
  \node[fillbox=corange, text width=3cm, right=of a] (b) {\textbf{volume near border}\\ $1 - 0.9^d$};
  \node[fillbox=cgreen, text width=3cm, right=of b] (c) {\textbf{$k$-NN accuracy}\\ 0.98 $\rightarrow$ 0.555\\ with noise features};
\end{tikzpicture}
\end{center}
```

* Notebook `03_pitfalls.ipynb`, section **D6**
* Practical consequences: feature selection, dimensionality reduction (Topic 7), more data

# No Free Lunch

## The No Free Lunch Theorem

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, ybar, height=4.2cm, width=0.85\textwidth, bar width=8pt, symbolic x coords={problem 1, problem 2, problem 3, problem 4, average}, xtick=data, ymin=0, ymax=1.05, ylabel={performance}, legend style={at={(0.5,1.03)}, anchor=south, legend columns=2}, enlarge x limits=0.12]
  \addplot[fill=cblue!60, draw=cblue] coordinates {(problem 1,0.9) (problem 2,0.4) (problem 3,0.8) (problem 4,0.5) (average,0.65)}; \addlegendentry{algorithm A}
  \addplot[fill=corange!70, draw=corange] coordinates {(problem 1,0.5) (problem 2,0.85) (problem 3,0.45) (problem 4,0.8) (average,0.65)}; \addlegendentry{algorithm B}
\end{axis}
\end{tikzpicture}
\end{center}
```

* **Theorem** (Wolpert, 1996): averaged over **all** possible problems, every learning algorithm has the same performance on unseen data
* Winning on some problems is always paid for by losing on others

## Inductive Bias: the Assumptions a Model Makes

```{=latex}
\begin{center}
\begin{tikzpicture}[scale=0.85]
  \begin{scope}
    \foreach \a in {0,20,...,340} { \fill[cblue] ({0.6*cos(\a)+0.08*rand},{0.6*sin(\a)+0.08*rand}) circle (2pt); \fill[corange] ({1.5*cos(\a+10)+0.08*rand},{1.5*sin(\a+10)+0.08*rand}) circle (2pt); }
    \draw[cred, very thick] (-1.9,-1.2) -- (1.9,1.2);
    \node[note, text width=3.6cm] at (0,-2.3) {linear model: assumes a straight boundary \textbf{(fails)}};
  \end{scope}
  \begin{scope}[xshift=5.5cm]
    \foreach \a in {0,20,...,340} { \fill[cblue] ({0.6*cos(\a)+0.08*rand},{0.6*sin(\a)+0.08*rand}) circle (2pt); \fill[corange] ({1.5*cos(\a+10)+0.08*rand},{1.5*sin(\a+10)+0.08*rand}) circle (2pt); }
    \draw[cgreen!60!black, very thick] (0,0) circle (1.05);
    \node[note, text width=3.6cm] at (0,-2.3) {$k$-NN: assumes nearby points share labels \textbf{(works)}};
  \end{scope}
\end{tikzpicture}
\end{center}
```

* Every model carries **assumptions** about what patterns look like
* A model works when its assumptions **match** the problem

## Live Demo D7: The Ranking Flips

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, ybar, height=4.2cm, width=0.8\textwidth, bar width=14pt, symbolic x coords={concentric circles, linear (10 of 50 useful)}, xtick=data, ymin=0, ymax=1.1, ylabel={CV accuracy}, legend style={at={(0.5,1.03)}, anchor=south, legend columns=2}, enlarge x limits=0.35, nodes near coords, nodes near coords style={font=\tiny}]
  \addplot[fill=cblue!60, draw=cblue] coordinates {(concentric circles,0.453) (linear (10 of 50 useful),0.935)}; \addlegendentry{logistic regression}
  \addplot[fill=corange!70, draw=corange] coordinates {(concentric circles,0.997) (linear (10 of 50 useful),0.790)}; \addlegendentry{$k$-NN}
\end{axis}
\end{tikzpicture}
\end{center}
```

* Notebook `03_pitfalls.ipynb`, section **D7**
* No model is best everywhere: this is why the course covers **many model families**

# Data Leakage

## What Is Data Leakage?

```{=latex}
\begin{center}
\begin{tikzpicture}
  \fill[cblue!50] (0,0) rectangle (6,1); \node at (3,0.5) {training data};
  \fill[corange!70] (7,0) rectangle (10,1); \node at (8.5,0.5) {test data};
  \draw[cred, very thick, ->] (8.5,1.1) to[bend right=40] node[above, font=\scriptsize] {information leaks into training} (3,1.1);
  \node[fillbox=cred, text width=8.4cm] at (5,-1) {the test score becomes \textbf{too optimistic}:\\ excellent in the lab, disappointing in production};
\end{tikzpicture}
\end{center}
```

* **Leakage:** information that would not be available at prediction time reaches the model during training
* The test set stops being "unseen", so it no longer measures generalization

## Wrong and Right Pipelines

```{=latex}
\begin{center}
\begin{tikzpicture}[node distance=4mm]
  \node[font=\scriptsize\bfseries, cred] at (-1.2,0) {wrong};
  \node[fillbox=cgray, text width=1.7cm] (a1) at (0.6,0) {all data};
  \node[fillbox=cred, text width=2cm, right=of a1] (a2) {select best features};
  \node[fillbox=cgray, text width=1.4cm, right=of a2] (a3) {split};
  \node[fillbox=cgray, text width=1.8cm, right=of a3] (a4) {train and test};
  \draw[flow] (a1) -- (a2); \draw[flow] (a2) -- (a3); \draw[flow] (a3) -- (a4);
  \node[font=\scriptsize\bfseries, cgreen!60!black] at (-1.2,-1.5) {right};
  \node[fillbox=cgray, text width=1.7cm] (b1) at (0.6,-1.5) {all data};
  \node[fillbox=cgray, text width=1.4cm, right=of b1] (b2) {split};
  \node[fillbox=cgreen, text width=2.7cm, right=of b2] (b3) {select features\\ on the\\ \textbf{training part}};
  \node[fillbox=cgray, text width=1.8cm, right=of b3] (b4) {train and test};
  \draw[flow] (b1) -- (b2); \draw[flow] (b2) -- (b3); \draw[flow] (b3) -- (b4);
\end{tikzpicture}
\end{center}
```

* In the wrong pipeline, the feature selection has **already seen** the test labels
* **Rule:** every step that learns from data (scaling, selection, imputation, tuning) goes **inside** the training part

## Other Forms of Leakage

```{=latex}
\begin{center}
\begin{tikzpicture}
  \draw[->, thick] (0,0) -- (10,0) node[below left, note] {time};
  \fill[cblue!50] (0,0.2) rectangle (6,0.7); \node[font=\scriptsize] at (3,0.45) {training period};
  \fill[corange!70] (6,0.2) rectangle (9,0.7); \node[font=\scriptsize] at (7.5,0.45) {test period};
  \fill[cred!70] (7,0.9) rectangle (8,1.3); \node[font=\scriptsize, cred, anchor=east] at (6.9,1.1) {future data used in training};
\end{tikzpicture}
\end{center}
```

* **Temporal leakage:** random splits of time series let the model use the **future** to predict the past
* **Duplicates:** the same patient or image in both training and test
* **Target leakage:** a feature only known **after** the outcome (e.g. "treatment given" to predict the disease)

## Live Demo D8: 89% Accuracy on Pure Noise

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, xbar, height=3.6cm, width=0.75\textwidth, symbolic y coords={selection inside CV, selection before CV}, ytick=data, xmin=0, xmax=1.05, xlabel={cross-validation accuracy}, bar width=12pt, nodes near coords, nodes near coords style={font=\scriptsize}, enlarge y limits=0.4]
  \addplot[fill=cblue!55, draw=cblue] coordinates {(0.53,selection inside CV) (0.89,selection before CV)};
  \draw[cgray, dashed, thick] (axis cs:0.5,{[normalized]-1}) -- (axis cs:0.5,{[normalized]2});
\end{axis}
\end{tikzpicture}
\end{center}
```

* 100 samples, 10 000 **random** features, **random** labels: the honest answer is 50%
* Selecting the 20 "best" features on all data first gives **89%** accuracy, from nothing
* Notebook `03_pitfalls.ipynb`, section **D8**

# Correlation Is Not Causation

## Correlation: Moving Together

```{=latex}
\begin{center}
\begin{tikzpicture}
  \foreach \sx/\r/\lab in {0/0/{$r \approx 0$\\ no relation}, 3.8/0.6/{$r \approx 0.6$\\ moderate}, 7.6/0.95/{$r \approx 0.95$\\ strong}} {
    \begin{scope}[xshift=\sx cm]
      \draw[->, cgray] (0,0) -- (2.6,0); \draw[->, cgray] (0,0) -- (0,2.2);
      \foreach \i in {1,...,30} {
        \pgfmathsetmacro{\xx}{1.3+0.9*rand}
        \pgfmathsetmacro{\yy}{1.1+\r*0.8*(\xx-1.3)+sqrt(1-\r*\r)*0.7*rand}
        \fill[cblue] (\xx,\yy) circle (1.5pt);
      }
      \node[note, align=center] at (1.3,-0.5) {\lab};
    \end{scope}
  }
\end{tikzpicture}
\end{center}
```

* **Correlation coefficient** $r$: a number from $-1$ to $1$ describing how strongly two variables move **together**
* $r$ near 0: no linear relation; $r$ near $\pm 1$: points lie close to a line
* Correlation says **nothing** about which variable causes which

## Ice Cream and Drownings

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=4.3cm, width=0.72\textwidth, xlabel={ice-cream sales per day}, ylabel={drownings per day}, xtick=\empty, ytick=\empty]
  \addplot[only marks, mark=*, mark size=1.1pt, cblue, samples=70, domain=10:35] ({20*x+60*rand}, {max(0,0.3*x-3+rand)});
\end{axis}
\end{tikzpicture}
\end{center}
```

* A beach town, one year of data (demo D9): correlation $r = 0.85$
* A model predicting drownings from ice-cream sales scores $R^2 = 0.73$: a **good predictor**
* Does ice cream **cause** drownings?

## The Hidden Common Cause

```{=latex}
\begin{center}
\begin{tikzpicture}[node distance=14mm]
  \node[fillbox=cred, text width=2.4cm] (t) {\textbf{temperature}\\ (confounder)};
  \node[fillbox=cblue, text width=2.4cm, below left=of t] (i) {ice-cream sales};
  \node[fillbox=corange, text width=2.4cm, below right=of t] (d) {drownings};
  \draw[flow, very thick] (t) -- node[left, note] {hot days} (i);
  \draw[flow, very thick] (t) -- node[right, note] {more swimming} (d);
  \draw[cgray, dashed, thick, <->] (i) -- node[below, note] {correlated, but no causal link} (d);
\end{tikzpicture}
\end{center}
```

* A **confounder** causes both variables and makes them move together
* Arrows show **causes**; the dashed line is only an **association**
* Adding temperature to the model: the weight of ice cream drops to about **0**

## Prediction Versus Intervention

```{=latex}
\begin{center}
\begin{tikzpicture}
  \node[fillbox=cblue, text width=4.2cm, minimum height=2.3cm] (p) {\textbf{Prediction}\\[3pt] "I see high ice-cream sales;\\ how many drownings\\ should I expect today?"\\[3pt] \textcolor{cgreen!60!black}{correlation is enough}};
  \node[fillbox=corange, text width=4.2cm, minimum height=2.3cm, right=8mm of p] (i) {\textbf{Intervention}\\[3pt] "If I \textbf{force} sales to halve,\\ will drownings go down?"\\[3pt] \textcolor{cred}{needs the causal structure}};
\end{tikzpicture}
\end{center}
```

* Models trained on observational data answer **prediction** questions
* Decisions that **change** the world are intervention questions
* Using a correlational model to make interventions leads to wrong decisions

## Live Demo D9: Halving Ice-Cream Sales

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, ybar, height=4cm, width=0.8\textwidth, bar width=22pt, symbolic x coords={before, predicted by model, actual result}, xtick=data, ymin=0, ymax=1600, ylabel={drownings per year}, nodes near coords, nodes near coords style={font=\scriptsize}, enlarge x limits=0.25, ytick={0,500,1000,1500}]
  \addplot[fill=cblue!55, draw=cblue] coordinates {(before,1361) (predicted by model,337) (actual result,1345)};
\end{axis}
\end{tikzpicture}
\end{center}
```

* The model predicts that halving ice-cream sales prevents **75%** of drownings
* In the simulation nothing changes: temperature, the real cause, is the same
* Notebook `03_pitfalls.ipynb`, section **D9**

# Shortcuts and Distribution Shift

## Shortcut Learning

```{=latex}
\begin{center}
\begin{tikzpicture}
  \begin{scope}
    \fill[csky!25] (0,0) rectangle (3,2.2);
    \foreach \i in {1,...,25} { \fill[white] ({3*rnd},{2.2*rnd}) circle (1.3pt); }
    \fill[cgray!70] (0.9,0.5) -- (1.2,1.3) -- (1.4,1.0) -- (1.8,1.0) -- (2.0,1.3) -- (2.2,0.5) -- cycle;
    \node[font=\scriptsize] at (1.55,0.3) {wolf}; \node[note] at (1.5,-0.3) {training: wolf in snow};
  \end{scope}
  \begin{scope}[xshift=3.6cm]
    \fill[cgreen!25] (0,0) rectangle (3,2.2);
    \fill[cgray!70] (0.9,0.5) -- (1.2,1.3) -- (1.4,1.0) -- (1.8,1.0) -- (2.0,1.3) -- (2.2,0.5) -- cycle;
    \node[font=\scriptsize] at (1.55,0.3) {wolf}; \node[note] at (1.5,-0.3) {test: wolf on grass};
  \end{scope}
  \node[fillbox=cred, text width=3.2cm] at (9.2,1.1) {model says: \textbf{"husky"}\\[2pt] it learned\\ \textbf{"snow = wolf"}};
\end{tikzpicture}
\end{center}
```

* Models take the **easiest** pattern that reduces training error: a **shortcut**
* The shortcut works on data like the training set and fails elsewhere

## Documented Shortcuts

```{=latex}
\begin{center}
\begin{tikzpicture}[node distance=3mm]
  \node[font=\scriptsize, fillbox=cblue, text width=3.05cm, minimum height=2.2cm] (a) {\textbf{Pneumonia from X-rays}\\[3pt] learned which\\ \textbf{hospital}\\ took the image};
  \node[font=\scriptsize, fillbox=corange, text width=3.05cm, minimum height=2.2cm, right=of a] (b) {\textbf{Skin cancer from photos}\\[3pt] learned that a \textbf{ruler}\\ appears next to\\ suspicious lesions};
  \node[font=\scriptsize, fillbox=cgreen, text width=3.05cm, minimum height=2.2cm, right=of b] (c) {\textbf{Wolves vs. huskies}\\[3pt] learned to\\ detect \textbf{snow}\\ in the background};
\end{tikzpicture}
\end{center}
```

* All of them looked excellent on a validation set from the **same** source
* See Geirhos et al., "Shortcut Learning in Deep Neural Networks", 2020

## Distribution Shift

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=4.2cm, width=0.8\textwidth, xmin=-4, xmax=7, ymin=0, ymax=0.5, axis y line=none, xlabel={feature value (e.g. patient age)}, samples=100, legend style={at={(0.5,1.03)}, anchor=south, legend columns=2}]
  \addplot[cblue, very thick, fill=cblue!20, domain=-4:7] {exp(-(x-0)^2/2)/sqrt(2*pi)} \closedcycle; \addlegendentry{training data}
  \addplot[corange, very thick, fill=corange!20, fill opacity=0.6, domain=-4:7] {exp(-(x-3)^2/2)/sqrt(2*pi)} \closedcycle; \addlegendentry{data in deployment}
\end{axis}
\end{tikzpicture}
\end{center}
```

* Machine learning assumes training and deployment data come from the **same distribution**
* In practice the world **changes**: new users, new sensors, a new season
* Where the two distributions do not overlap, the model is **guessing**

## Kinds of Shift

```{=latex}
\begin{center}
\begin{tikzpicture}[node distance=3mm]
  \node[font=\scriptsize, fillbox=cblue, text width=3.05cm, minimum height=2.3cm] (a) {\textbf{Covariate shift}\\ $P(x)$ changes\\[3pt] a new camera\\ produces darker images};
  \node[font=\scriptsize, fillbox=corange, text width=3.05cm, minimum height=2.3cm, right=of a] (b) {\textbf{Label shift}\\ $P(y)$ changes\\[3pt] a disease becomes more common during an outbreak};
  \node[font=\scriptsize, fillbox=cgreen, text width=3.05cm, minimum height=2.3cm, right=of b] (c) {\textbf{Concept drift}\\ $P(y \mid x)$ changes\\[3pt] fraudsters change their tactics};
\end{tikzpicture}
\end{center}
```

* $P(x)$: how often each input occurs; $P(y)$: how often each label occurs
* $P(y \mid x)$: the relationship between input and label, "the rule" itself

## Sampling Bias

```{=latex}
\begin{center}
\begin{tikzpicture}
  \draw[thick, fill=cgray!8] (0,0) ellipse (2.6cm and 1.5cm);
  \node[note] at (0,1.75) {population};
  \foreach \i in {1,...,35} { \fill[cblue] ({-2.2+2.2*rnd},{1.1*rand}) circle (1.8pt); }
  \foreach \i in {1,...,35} { \fill[corange] ({2.2*rnd},{1.1*rand}) circle (1.8pt); }
  \draw[cred, very thick, dashed] (-1.2,0) circle (0.9);
  \draw[flow] (2.9,0) -- node[above, note] {sample} (4.3,0);
  \node[fillbox=cred, text width=3.4cm] at (6.3,0) {training data contain\\ almost only \textcolor{cblue}{\textbf{blue}}\\ the model fails on \textcolor{corange}{\textbf{orange}}};
\end{tikzpicture}
\end{center}
```

* **Sampling bias:** the training data do not represent the population where the model is used
* Under-represented groups receive **worse** predictions
* Historical data can also encode historical **discrimination**

## Live Demo D10: Shortcuts Under Shift

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, ybar, height=4.2cm, width=0.8\textwidth, bar width=14pt, symbolic x coords={same distribution, after shift}, xtick=data, ymin=0.4, ymax=1.05, ylabel={accuracy}, legend style={at={(0.5,1.03)}, anchor=south, legend columns=2}, enlarge x limits=0.4, nodes near coords, nodes near coords style={font=\tiny}]
  \addplot[fill=cred!55, draw=cred] coordinates {(same distribution,0.938) (after shift,0.559)}; \addlegendentry{uses the shortcut}
  \addplot[fill=cgreen!55, draw=cgreen] coordinates {(same distribution,0.746) (after shift,0.747)}; \addlegendentry{uses only the real cause}
\end{axis}
\end{tikzpicture}
\end{center}
```

* The shortcut model looks **better** in validation, and collapses to near chance after the shift
* Only data from **deployment conditions** reveal the problem
* Notebook `03_pitfalls.ipynb`, section **D10**

# Learning Taxonomies

## Where Does the Learning Signal Come From?

```{=latex}
\begin{center}
\begin{tikzpicture}[scale=0.86, every node/.append style={font=\scriptsize, transform shape}]
  \node[fillbox=cgray] (root) at (4.6,1.8) {\textbf{Machine learning}};
  \foreach \k/\col/\txt [count=\n] in {su/cblue/{\textbf{Supervised}\\ labels}, un/cgreen/{\textbf{Unsupervised}\\ structure}, se/cpurple/{\textbf{Semi-supervised}\\ few labels}, sf/corange/{\textbf{Self-supervised}\\ data as labels}, rl/cred/{\textbf{Reinforcement}\\ rewards}} {
    \node[fillbox=\col, text width=1.95cm, minimum height=1cm] (\k) at ({(\n-1)*2.3},0) {\txt};
    \draw[thick, cgray] (root.south) -- (\k.north);
  }
\end{tikzpicture}
\end{center}
```

* Learning types differ in **what tells the model it is doing well**
* The same data can be used in several ways

## Supervised Learning

```{=latex}
\begin{center}
\begin{tikzpicture}
  \foreach \px/\py in {0.3/0.4, 0.8/1.1, 1.2/0.3, 0.5/1.6, 1.5/0.9, 1.0/2.0, 0.2/2.3} { \fill[cblue] (\px,\py) circle (3pt); }
  \foreach \px/\py in {3.0/1.2, 3.4/2.2, 2.6/2.6, 3.8/0.8, 2.9/0.2, 3.6/1.7, 4.0/2.5} { \fill[corange] (\px,\py) rectangle ++(0.18,0.18); }
  \draw[cred, very thick] (1.4,2.9) -- (2.4,-0.1);
  \node[note] at (0.8,-0.5) {class A (label)};
  \node[note] at (3.4,-0.5) {class B (label)};
  \draw[flow] (4.7,1.3) -- node[above, note] {learn} (5.9,1.3);
  \node[fillbox=cred, text width=3cm] at (7.6,1.3) {a function\\ $f: x \mapsto y$\\ (the boundary)};
\end{tikzpicture}
\end{center}
```

* Every training example comes with the **correct answer** $y$ (the label)
* The model learns to map inputs $x$ to outputs $y$
* The loss compares the prediction with the label

## Classification and Regression

```{=latex}
\begin{center}
\begin{tikzpicture}
  \begin{scope}
    \foreach \px/\py in {0.3/0.4, 0.8/1.1, 1.2/0.3, 0.5/1.6} { \fill[cblue] (\px,\py) circle (2.5pt); }
    \foreach \px/\py in {2.6/1.2, 3.0/2.2, 2.3/2.4, 3.2/0.8} { \fill[corange] (\px,\py) rectangle ++(0.15,0.15); }
    \draw[cred, very thick] (1.1,2.7) -- (2.1,0);
    \node[note, text width=4cm] at (1.7,-0.8) {\textbf{classification}: $y$ is a category\\ spam / not spam, digit 0--9};
  \end{scope}
  \begin{scope}[xshift=5.6cm]
    \draw[->, cgray] (0,0) -- (3.6,0) node[right, note] {area}; \draw[->, cgray] (0,0) -- (0,2.7) node[above, note] {price};
    \foreach \px/\py in {0.4/0.5, 0.9/0.8, 1.3/1.0, 1.8/1.5, 2.2/1.4, 2.7/2.0, 3.1/2.2} { \fill[cblue] (\px,\py) circle (2.5pt); }
    \draw[cred, very thick] (0.2,0.35) -- (3.4,2.35);
    \node[note, text width=4cm] at (1.7,-0.8) {\textbf{regression}: $y$ is a number\\ price, temperature};
  \end{scope}
\end{tikzpicture}
\end{center}
```

* Same idea, different kind of output
* Topics 3--6: linear and probabilistic models, SVM, neural networks, trees, $k$-NN, ensembles

## The Price of Labels

```{=latex}
\begin{center}
\begin{tikzpicture}
  \foreach \i in {0,...,4} { \foreach \j in {0,...,2} { \draw[fill=cgray!25] (\i*0.55,\j*0.55) rectangle ++(0.45,0.45); } }
  \node[note, text width=3cm] at (1.3,-0.5) {millions of unlabelled\\ images: cheap};
  \draw[flow] (3.0,0.7) -- (3.8,0.7);
  \node[fillbox=corange, text width=2.4cm] at (5.2,0.7) {human expert\\ labels each one};
  \draw[flow] (6.6,0.7) -- (7.4,0.7);
  \node[fillbox=cred, text width=2.1cm] at (8.6,0.7) {time, money,\\ privacy,\\ disagreement};
\end{tikzpicture}
\end{center}
```

* Labels often require **experts** (radiologists, lawyers, engineers)
* Experts **disagree**, and labels contain errors
* Most of the world's data has **no labels**: this motivates the other learning types

## Unsupervised Learning

```{=latex}
\begin{center}
\begin{tikzpicture}
  \begin{scope}
    \foreach \cx/\cy in {0.8/0.8, 2.6/1.9, 2.8/0.4} { \foreach \i in {1,...,9} { \fill[cgray] ({\cx+0.35*rand},{\cy+0.3*rand}) circle (2.2pt); } }
    \node[note] at (1.7,-0.4) {data without labels};
  \end{scope}
  \draw[flow] (3.8,1.1) -- node[above, note] {discover} (5.0,1.1);
  \begin{scope}[xshift=5.4cm]
    \foreach \cx/\cy/\c in {0.8/0.8/cblue, 2.6/1.9/cgreen, 2.8/0.4/corange} { \foreach \i in {1,...,9} { \fill[\c] ({\cx+0.35*rand},{\cy+0.3*rand}) circle (2.2pt); } \draw[\c, thick, dashed] (\cx,\cy) ellipse (0.62 and 0.52); }
    \node[note] at (1.7,-0.4) {groups found by the algorithm};
  \end{scope}
\end{tikzpicture}
\end{center}
```

* Only inputs $x$, **no labels** $y$
* The model finds **structure**: groups, directions, hidden sources
* Harder to evaluate: there is no correct answer to compare with

## Unsupervised Tasks

```{=latex}
\begin{center}
\begin{tikzpicture}
  \begin{scope}
    \foreach \cx/\cy/\c in {0.5/0.5/cblue, 1.7/1.4/cgreen} { \foreach \i in {1,...,7} { \fill[\c] ({\cx+0.3*rand},{\cy+0.3*rand}) circle (1.8pt); } }
    \node[note] at (1.1,-0.4) {\textbf{clustering}};
  \end{scope}
  \begin{scope}[xshift=3.6cm]
    \foreach \i in {1,...,18} { \pgfmathsetmacro{\t}{2.2*rnd} \fill[cgray] ({\t+0.12*rand},{0.2+0.6*\t+0.15*rand}) circle (1.6pt); }
    \draw[cred, very thick, ->] (0,0.2) -- (2.3,1.6);
    \node[note] at (1.1,-0.4) {\textbf{dimensionality reduction}};
  \end{scope}
  \begin{scope}[xshift=7.4cm]
    \draw[cblue, thick, domain=0:2.2, samples=40] plot (\x, {1.6+0.25*sin(360*\x)});
    \draw[corange, thick] (0,0.9) -- (0.4,1.2) -- (0.8,0.9) -- (1.2,1.2) -- (1.6,0.9) -- (2.0,1.2) -- (2.2,1.05);
    \draw[cgray, thick, domain=0:2.2, samples=60] plot (\x, {0.25+0.18*sin(360*\x)+0.1*sin(720*\x+40)});
    \node[note, text width=3cm] at (1.1,-0.4) {\textbf{blind signal separation}};
    \node[note] at (2.65,0.25) {mix};
  \end{scope}
\end{tikzpicture}
\end{center}
```

* **Clustering:** group similar examples (Topic 8)
* **Dimensionality reduction:** keep the few directions that matter (Topic 7)
* **Blind signal separation:** recover the sources hidden in a mixture (Topic 7)

## Semi-Supervised Learning

```{=latex}
\begin{center}
\begin{tikzpicture}
  \begin{scope}
    \foreach \cx/\cy in {0.9/1.0, 3.1/1.0} { \foreach \i in {1,...,14} { \fill[cgray!70] ({\cx+0.55*rand},{\cy+0.6*rand}) circle (2pt); } }
    \fill[cblue] (0.8,1.2) circle (3.5pt); \fill[corange] (3.2,0.8) circle (3.5pt);
    \node[note] at (2,-0.2) {2 labels + many unlabelled points};
  \end{scope}
  \draw[flow] (4.4,1) -- node[above, note] {spread} (5.5,1);
  \begin{scope}[xshift=5.9cm]
    \foreach \cx/\cy/\c in {0.9/1.0/cblue, 3.1/1.0/corange} { \foreach \i in {1,...,14} { \fill[\c!70] ({\cx+0.55*rand},{\cy+0.6*rand}) circle (2pt); } }
    \draw[cred, very thick] (2,-0) -- (2,2.1);
    \node[note] at (2,-0.2) {labels follow the structure};
  \end{scope}
\end{tikzpicture}
\end{center}
```

* A **few** labelled examples and **many** unlabelled ones
* Unlabelled data reveal the **shape** of the data; the few labels name the groups
* Methods: **self-training** (pseudo-labelling), label propagation on a similarity graph

## The Cluster Assumption

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=4cm, width=0.8\textwidth, xmin=-4, xmax=4, ymin=0, ymax=0.45, axis y line=none, xlabel={feature}, samples=100, xtick=\empty, clip=false]
  \addplot[cgray, very thick, fill=cgray!20, domain=-4:4] {0.5*exp(-(x+1.8)^2/(2*0.6^2))/(0.6*sqrt(2*pi)) + 0.5*exp(-(x-1.8)^2/(2*0.6^2))/(0.6*sqrt(2*pi))} \closedcycle;
  \draw[cred, very thick] (axis cs:0,0) -- (axis cs:0,0.42) node[pos=0.9, right, font=\scriptsize] {good boundary: low density};
  \draw[cred, dashed, thick] (axis cs:-1.8,0) -- (axis cs:-1.8,0.42) node[pos=0.9, left, font=\scriptsize] {bad: cuts a group};
\end{axis}
\end{tikzpicture}
\end{center}
```

* **Assumption:** points in the same dense group usually share a label
* So decision boundaries should pass through **empty** regions
* When this assumption fails, unlabelled data can **hurt** instead of help

## Self-Supervised Learning

```{=latex}
\begin{center}
\begin{tikzpicture}[scale=0.3]
  \begin{scope}
    \foreach \x/\y in {2/7,3/7,4/7,5/7,1/6,6/6,6/5,5/4,4/4,3/4,6/3,6/2,1/1,6/1,2/0,3/0,4/0,5/0} { \fill[black] (\x,\y) rectangle ++(1,1); }
    \draw[cgray] (0,0) grid (8,8);
    \node[note] at (4,-1.2) {original};
  \end{scope}
  \begin{scope}[xshift=11cm]
    \foreach \x/\y in {2/7,3/7,4/7,5/7,1/6,6/6,6/5,5/4,4/4,3/4} { \fill[black] (\x,\y) rectangle ++(1,1); }
    \fill[corange!60] (0,0) rectangle (8,4);
    \node[font=\scriptsize] at (4,2) {hidden};
    \draw[cgray] (0,0) grid (8,8);
    \node[note] at (4,-1.2) {input: top half};
  \end{scope}
  \draw[flow, very thick] (20,4) -- node[above, note] {predict} (23,4);
  \begin{scope}[xshift=24cm]
    \foreach \x/\y in {2/7,3/7,4/7,5/7,1/6,6/6,6/5,5/4,4/4,3/4} { \fill[black] (\x,\y) rectangle ++(1,1); }
    \foreach \x/\y in {6/3,6/2,1/1,6/1,2/0,3/0,4/0,5/0} { \fill[cblue] (\x,\y) rectangle ++(1,1); }
    \draw[cgray] (0,0) grid (8,8);
    \node[note] at (4,-1.2) {target: bottom half};
  \end{scope}
\end{tikzpicture}
\end{center}
```

* No human labels: the **targets are created from the data itself**
* Hide part of the input and train the model to predict it (a **pretext task**)
* To succeed, the model must learn how the data are **structured**

## Pretext Tasks in Practice

```{=latex}
\begin{center}
\begin{tikzpicture}[node distance=4mm]
  \node[fillbox=corange, text width=3.2cm, minimum height=2.2cm] (a) {\textbf{Masked words}\\[3pt] "the cat \colorbox{cgray!40}{???}\\ on the mat"\\[2pt] predict "sat"\\ (Transformers)};
  \node[fillbox=cblue, text width=3.2cm, minimum height=2.2cm, right=of a] (b) {\textbf{Reconstruction}\\[3pt] squeeze the input\\ through a bottleneck\\ and rebuild it\\ (AutoEncoders)};
  \node[fillbox=cpurple, text width=3.2cm, minimum height=2.2cm, right=of b] (c) {\textbf{Predict in}\\ \textbf{latent space}\\[3pt] predict the\\ \emph{representation}\\ of a hidden region\\ (JEPA)};
\end{tikzpicture}
\end{center}
```

* The engine behind modern **foundation models**: learning as compression, at scale
* Topic 9 covers these three families

## Reinforcement Learning

```{=latex}
\begin{center}
\begin{tikzpicture}
  \node[fillbox=cblue, text width=2.4cm, minimum height=1.2cm] (a) at (0,0) {\textbf{Agent}\\ chooses actions};
  \node[fillbox=cgreen, text width=2.8cm, minimum height=1.2cm] (e) at (6,0) {\textbf{Environment}\\ the world};
  \draw[flow, very thick] (a.north) to[bend left=35] node[above, font=\scriptsize] {action $a_t$} (e.north);
  \draw[flow, very thick] (e.south) to[bend left=35] node[below, font=\scriptsize, align=center] {new state $s_{t+1}$\\ reward $r_{t+1}$} (a.south);
\end{tikzpicture}
\end{center}
```

* No dataset of correct answers: the agent **acts** and receives a **reward**
* $s_t$: the situation at time $t$; $a_t$: the action taken; $r_{t+1}$: the reward received
* Goal: learn a **policy** (what to do in each state) that maximizes the total reward

## Rewards, Exploration, and Exploitation

```{=latex}
\begin{center}
\begin{tikzpicture}[scale=0.62]
  \foreach \r in {0,...,4} { \foreach \c in {0,...,4} { \draw[cgray] (\c,-\r) rectangle ++(1,1); } }
  \foreach \c/\r in {1/1, 3/2, 1/3} { \fill[cred!60] (\c,-\r) rectangle ++(1,1); \node[font=\tiny] at (\c+0.5,-\r+0.5) {pit}; }
  \fill[cgreen!60] (4,-4) rectangle ++(1,1); \node[font=\tiny] at (4.5,-3.5) {goal};
  \foreach \c/\r/\dx/\dy in {0/0/0/-0.3, 1/0/0.3/0, 2/0/0/-0.3, 3/0/0/-0.3, 4/0/0/-0.3, 0/1/0/-0.3, 2/1/0/-0.3, 3/1/-0.3/0, 4/1/0/-0.3, 0/2/0.3/0, 1/2/0.3/0, 2/2/0/-0.3, 4/2/0/-0.3, 0/3/0/-0.3, 2/3/0/-0.3, 3/3/0/-0.3, 4/3/0/-0.3, 0/4/0.3/0, 1/4/0.3/0, 2/4/0.3/0, 3/4/0.3/0} {
    \draw[->, thick, cblue] ({\c+0.5-\dx},{-\r+0.5-\dy}) -- ({\c+0.5+\dx},{-\r+0.5+\dy});
  }
  \node[note, text width=4cm] at (2.5,-4.9) {policy learned by Q-learning (demo D11)};
  \node[fillbox=corange, text width=4.2cm, align=flush left] at (10.5,-0.3) {\textbf{Exploit:} take the best action known so far};
  \node[fillbox=cpurple, text width=4.2cm, align=flush left] at (10.5,-2.4) {\textbf{Explore:} try a random action, it might be better};
\end{tikzpicture}
\end{center}
```

* Reward: $+1$ at the goal, $-1$ in a pit, $-0.01$ per step; nobody says which move is correct
* The agent must balance **exploration** and **exploitation** (Topic 10)

## Summary of Learning Types

```{=latex}
\begin{center}\small\renewcommand{\arraystretch}{1.35}
\begin{tabular}{@{}l L{2.9cm} L{3.2cm} c@{}}
\toprule
Type & Data & Learning signal & Topics \\
\midrule
\textcolor{cblue}{\textbf{Supervised}} & $(x, y)$ pairs & human labels & 3--6 \\
\textcolor{cgreen!70!black}{\textbf{Unsupervised}} & only $x$ & structure of the data & 7--8 \\
\textcolor{cpurple}{\textbf{Semi-supervised}} & few $(x,y)$, many $x$ & labels + structure & 9 \\
\textcolor{corange}{\textbf{Self-supervised}} & only $x$ & targets built from $x$ & 9 \\
\textcolor{cred}{\textbf{Reinforcement}} & interaction & rewards & 10 \\
\bottomrule
\end{tabular}
\end{center}
```

* The **model families** (linear, trees, neural networks, ...) can be used in several of these settings
* The learning type describes **how** the model is trained, not **which** model it is

## Live Demo D11: One Dataset, Many Signals

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, xbar, height=4cm, width=0.72\textwidth, symbolic y coords={all 1257 labels, label spreading with 126, self-training with 126, only 126 labels}, ytick=data, xmin=0.8, xmax=1.02, xlabel={test accuracy (digits)}, bar width=9pt, nodes near coords, nodes near coords style={font=\tiny}, enlarge y limits=0.18, y tick label style={font=\tiny}]
  \addplot[fill=cpurple!50, draw=cpurple] coordinates {(0.981,all 1257 labels) (0.950,label spreading with 126) (0.924,self-training with 126) (0.894,only 126 labels)};
\end{axis}
\end{tikzpicture}
\end{center}
```

* Notebook `04_taxonomies.ipynb`: supervised, $k$-means clustering, **10% of the labels**, masked-half prediction
* **Reinforcement learning:** average return rises from $-0.26$ to $+0.64$ in 500 episodes

# Wrap-Up

## Key Takeaways

```{=latex}
\begin{center}
\begin{tikzpicture}[node distance=3mm]
  \node[fillbox=cblue, text width=3.3cm, minimum height=1.3cm] (a) {\textbf{1. Compression}\\ keep the pattern,\\ drop the noise};
  \node[fillbox=cblue, text width=3.3cm, minimum height=1.3cm, right=of a] (b) {\textbf{2. MDL}\\ balance fit and complexity};
  \node[fillbox=cblue, text width=3.3cm, minimum height=1.3cm, right=of b] (c) {\textbf{3. Optimization}\\ finds the parameters};
  \node[fillbox=corange, text width=3.3cm, minimum height=1.3cm, below=of a] (d) {\textbf{4. Evaluation}\\ CV, intervals,\\ statistical tests};
  \node[fillbox=cred, text width=3.3cm, minimum height=1.3cm, right=of d] (e) {\textbf{5. Pitfalls}\\ bias--variance, curse,\\ leakage, causation};
  \node[fillbox=cgreen, text width=3.3cm, minimum height=1.3cm, right=of e] (f) {\textbf{6. Learning types}\\ where the signal\\ comes from};
\end{tikzpicture}
\end{center}
```

* These foundations apply to **every** model in the rest of the course

## Next Class: Learning as Optimization

```{=latex}
\begin{center}
\begin{tikzpicture}[node distance=6mm]
  \node[fillbox=corange, text width=4.4cm, minimum height=1.9cm] (b) {\textbf{Blind optimization}\\[2pt] random search, hill climbing,\\ genetic algorithms,\\ differential evolution, PSO};
  \node[fillbox=cblue, text width=4.4cm, minimum height=1.9cm, right=of b] (g) {\textbf{Gradient-based}\\[2pt] derivatives and chain rule\\ by hand, gradient descent,\\ SGD, \textbf{JAX}};
  \node[fillbox=cgreen, text width=9.4cm, below=of b.south west, anchor=north west] {\textbf{Lab:} fit the same model with \texttt{pyBlindOpt} and with gradient descent, then compare};
\end{tikzpicture}
\end{center}
```

* Learning as **search** over parameters: loss functions and search spaces

## References

```{=latex}
\footnotesize
```

* C. E. Shannon, "A Mathematical Theory of Communication", *Bell System Technical Journal*, 1948
* A. N. Kolmogorov, "Three Approaches to the Quantitative Definition of Information", *Problems of Information Transmission*, 1965
* J. Rissanen, "Modeling by Shortest Data Description", *Automatica*, 1978
* D. H. Wolpert, "The Lack of A Priori Distinctions Between Learning Algorithms", *Neural Computation*, 1996
* T. M. Mitchell, *Machine Learning*, McGraw-Hill, 1997
* C. Nadeau and Y. Bengio, "Inference for the Generalization Error", *Machine Learning*, 2003
* D. J. C. MacKay, *Information Theory, Inference, and Learning Algorithms*, Cambridge University Press, 2003
* N. Silver, *The Signal and the Noise*, Penguin, 2012
* R. Geirhos et al., "Shortcut Learning in Deep Neural Networks", *Nature Machine Intelligence*, 2020
