---
title: Fundamentos de Aprendizagem Automática
subtitle: "Class 02 — Blind vs. Gradient-Based Optimization"
---

# From Compression to Optimization

## Where We Left Off

```{=latex}
\begin{center}
\begin{tikzpicture}
  \node[fillbox=cgray, text width=2.3cm] (d) {\textbf{data}\\ signal + noise};
  \node[fillbox=cblue, text width=2.6cm, right=16mm of d] (m) {\textbf{model} $f_\theta$\\ a short description};
  \node[fillbox=corange, text width=2.6cm, right=16mm of m] (o) {\textbf{optimizer}\\ finds $\theta^\star$};
  \draw[flow] (d) -- node[above, note] {compress} (m);
  \draw[flow] (o) -- node[above, note] {tunes} (m);
  \node[note, below=4mm of m, text width=7cm] {Class 01: \emph{what} a good model is (MDL, evaluation)\\ Class 02: \emph{how} to find its parameters};
\end{tikzpicture}
\end{center}
```

* **Signal and noise:** the data hide a pattern behind noise
* **Learning = compression:** a good model keeps the pattern in few numbers
* **Compression needs optimization:** the numbers $\theta$ must be *found*

## The Loss Is the Code Length

$$\theta^\star = \arg\min_{\theta}\; \underbrace{\mathcal{L}_{data}(D;\theta)}_{\approx\, L(D \mid H)} \;+\; \lambda\,\underbrace{\Omega(\theta)}_{\approx\, L(H)}$$

* The two-part MDL code from Class 01, written as a **loss** to minimize
* Every model in this course is trained by minimizing a loss like this one
* The model family fixes the *shape* of $\mathcal{L}$; the optimizer walks on it

# What Is Optimization?

## The Optimization Problem

$$\min_{x \in \mathcal{X}}\; f(x) \qquad \text{subject to} \quad g_i(x) \le 0, \quad h_j(x) = 0$$

* $x$: the **decision variables** (in ML: the parameters $\theta$)
* $f$: the **objective** (in ML: the loss $\mathcal{L}$)
* $\mathcal{X}$ with $g_i, h_j$: the **feasible set** (e.g. box bounds $\ell \le x \le u$)
* Maximizing is the same problem: $\max f = -\min(-f)$
* $x^\star = \arg\min f$ is the **minimizer**; $f(x^\star)$ is the **minimum**

## A Numerical Example

Fit a line $\hat y = w\,x + b$ to three points: $(0, 1)$, $(1, 2)$, $(2, 4)$

```{=latex}
\renewcommand{\arraystretch}{1.5}
```

| Concept | In this example |
|:----------|:------------------------------------|
| decision variables | $\theta = (w, b)$ |
| objective | $\mathcal{L}(w, b) = \frac13 \sum_{i=1}^{3} (w x_i + b - y_i)^2$ |
| feasible set | the box $-5 \le w \le 5$, $\;-5 \le b \le 5$ |
| one guess | $\theta = (1, 1)$: errors $0, 0, -1$, so $\mathcal{L} = 1/3 = 0.333$ |
| minimizer | $\theta^\star = (1.5,\; 0.833)$ |
| minimum | $\mathcal{L}(\theta^\star) = 0.056$ |

* Constraint as $g(x) \le 0$: e.g. $g_1(\theta) = w - 5 \le 0$
* **Maximizing** accuracy is the same as **minimizing** the error rate $1 - \text{accuracy}$

## Global and Local Minima

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=4.3cm, width=0.85\textwidth, xmin=-4, xmax=4, ymin=-1.4, ymax=4.3, xlabel={$x$}, ylabel={$f(x)$}]
  \addplot[cblue, domain=-4:4, samples=400] {x^2/5 + sin(deg(3*x)) + 0.3*sin(deg(11*x))};
  \addplot[only marks, mark=*, cred, mark size=2.5pt] coordinates {(-0.67,-1.08)};
  \addplot[only marks, mark=o, corange, mark size=2.5pt, thick] coordinates {(-2.44,0.03) (-0.21,-0.80) (1.08,-0.06) (1.56,-0.81) (3.32,1.42)};
  \node[cred, font=\scriptsize, anchor=north] at (axis cs:-0.67,-1.12) {global};
  \node[corange, font=\scriptsize, anchor=south] at (axis cs:1.56,-0.7) {local};
\end{axis}
\end{tikzpicture}
\end{center}
```

* **Global minimum:** $f(x^\star) \le f(x)$ for **every** feasible $x$
* **Local minimum:** $f(x^\star) \le f(x)$ for every $x$ **near** $x^\star$
* Most optimizers only *guarantee* a local minimum

## Optimality Conditions

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=3.3cm, width=0.32\textwidth, xmin=-2, xmax=2, ymin=-2.2, ymax=2.2, xtick=\empty, ytick=\empty, title={\scriptsize minimum}]
  \addplot[cblue, domain=-2:2] {x^2/2-1}; \addplot[only marks, mark=*, cred] coordinates {(0,-1)};
\end{axis}
\end{tikzpicture}
\begin{tikzpicture}
\begin{axis}[faa, height=3.3cm, width=0.32\textwidth, xmin=-2, xmax=2, ymin=-2.2, ymax=2.2, xtick=\empty, ytick=\empty, title={\scriptsize maximum}]
  \addplot[cblue, domain=-2:2] {1-x^2/2}; \addplot[only marks, mark=*, cred] coordinates {(0,1)};
\end{axis}
\end{tikzpicture}
\begin{tikzpicture}
\begin{axis}[faa, height=3.3cm, width=0.32\textwidth, xmin=-2, xmax=2, ymin=-2.2, ymax=2.2, xtick=\empty, ytick=\empty, title={\scriptsize inflection / saddle}]
  \addplot[cblue, domain=-2:2] {x^3/3}; \addplot[only marks, mark=*, cred] coordinates {(0,0)};
\end{axis}
\end{tikzpicture}
\end{center}
```

* **First order:** at an interior minimum the slope vanishes, $f'(x^\star) = 0$ (in $n$-D: $\nabla f(x^\star) = 0$)
* **Second order:** $f''(x^\star) > 0$ (in $n$-D: the Hessian $H$ is positive definite)
* $f' = 0$ alone also holds at maxima and saddles: a **stationary point** is not always a minimum

## What the Second Derivative Tells Us

:::: {.columns}
::: {.column width="48%"}
```{=latex}
\begin{tikzpicture}
\begin{axis}[faa, height=3.6cm, width=\textwidth, xmin=0, xmax=4, ymin=0, ymax=5.5, xlabel={$x$}, title={\scriptsize $f(x) = (x-2)^2 + 1$}]
  \addplot[cblue, domain=0:4] {(x-2)^2+1};
  \addplot[corange, thick, domain=0.4:1.6] {2 - 2*(x-1)};
  \addplot[cgreen, thick, domain=2.4:3.6] {2 + 2*(x-3)};
  \addplot[cred, thick, domain=1.5:2.5] {1};
  \node[font=\tiny, corange] at (axis cs:1.2,4.9) {$f' = -2$};
  \node[font=\tiny, cred] at (axis cs:2,0.45) {$f' = 0$};
  \node[font=\tiny, cgreen!70!black] at (axis cs:2.8,4.9) {$f' = +2$};
\end{axis}
\end{tikzpicture}
```
:::
::: {.column width="52%"}
* Through the minimum, $f'$ goes from **negative** to **positive**
* $f'' = 2 > 0$: the slope **increases**, the curve bends **up** (a cup)
* $f'' < 0$: a cap (maximum); $f'' = 0$: no verdict ($x^3$ at 0)
:::
::::

* $f''$ is the **curvature**: it says how far the bottom is, $\delta = -f'/f''$, not only which way
* **Newton's method** uses it: few steps, no learning rate, very stable near a minimum
* The price: $f$ must be **smooth** (twice differentiable); on rugged or noisy losses curvature misleads

## Vectors and Matrices: What We Need

:::: {.columns}
::: {.column width="58%"}
* **Vector** $v \in \mathbb{R}^n$: a point, a direction, a gradient
* **Matrix** $H \in \mathbb{R}^{n\times n}$; **symmetric** if $H = H^\top$
* **Eigenpair:** $Hv = \lambda v$; $H$ only stretches $v$ by $\lambda$
* Hessian: $\lambda_i$ = **curvature** along direction $v_i$
* **Positive definite:** all $\lambda_i > 0$
* **Condition number** $\kappa = \lambda_{max}/\lambda_{min}$
* Solve $H\delta = -g$; never compute $H^{-1}$
:::
::: {.column width="42%"}
```{=latex}
\begin{tikzpicture}
  \foreach \r in {0.35,0.7,1.05} { \draw[cgray] (0,0) ellipse ({\r*2.2} and {\r*0.45}); }
  \draw[->, thick, cblue] (0,0) -- (2.4,0);
  \node[font=\scriptsize, cblue, anchor=north] at (1.6,-0.55) {$v_1$: $\lambda_1 = 2$ (flat)};
  \draw[->, thick, cred] (0,0) -- (0,0.8) node[above, font=\scriptsize] {$v_2$: $\lambda_2 = 50$ (steep)};
\end{tikzpicture}
```
$$H = \begin{bmatrix} 2 & 0 \\ 0 & 50 \end{bmatrix},\; \kappa = 25$$
:::
::::

* $f(x, y) = x^2 + 25y^2$ has this Hessian: a bowl 25 times more curved along $y$ than along $x$

## Gradient and Hessian

$$\nabla f(x) = \begin{bmatrix} \dfrac{\partial f}{\partial x_1} \\ \vdots \\ \dfrac{\partial f}{\partial x_n} \end{bmatrix}
\qquad\qquad
H(x) = \begin{bmatrix} \dfrac{\partial^2 f}{\partial x_1^2} & \cdots & \dfrac{\partial^2 f}{\partial x_1 \partial x_n} \\ \vdots & \ddots & \vdots \\ \dfrac{\partial^2 f}{\partial x_n \partial x_1} & \cdots & \dfrac{\partial^2 f}{\partial x_n^2} \end{bmatrix}$$

* $\nabla f$: $n$ numbers, the direction of **steepest ascent**; $-\nabla f$ points downhill
* $H$: $n^2$ numbers, the **curvature** in every pair of directions
* Eigenvalues of $H$: all $> 0$ minimum, all $< 0$ maximum, mixed signs **saddle**

## Convexity

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=3.8cm, width=0.45\textwidth, xmin=-2.2, xmax=2.2, ymin=-0.3, ymax=4.5, xtick=\empty, ytick=\empty, title={\scriptsize convex}]
  \addplot[cblue, domain=-2.2:2.2] {x^2 - 0.3*x + 0.2};
  \addplot[cred, thick, mark=*] coordinates {(-1.6,3.24) (1.8,2.9)};
\end{axis}
\end{tikzpicture}
\begin{tikzpicture}
\begin{axis}[faa, height=3.8cm, width=0.45\textwidth, xmin=-2.2, xmax=2.2, ymin=-0.3, ymax=4.5, xtick=\empty, ytick=\empty, title={\scriptsize not convex}]
  \addplot[cblue, domain=-2.2:2.2, samples=100] {0.25*x^4 - x^2 + 0.3*x + 1.5};
  \addplot[cred, thick, mark=*] coordinates {(-1.4,0.0804) (1.4,0.9204)};
\end{axis}
\end{tikzpicture}
\end{center}
```

$$f\big(t x + (1-t) y\big) \le t f(x) + (1-t) f(y) \qquad \forall x, y,\; t \in [0,1]$$

* The chord between any two points lies **above** the curve
* Convex $\Rightarrow$ every local minimum is **global**: nothing can trap the search
* Linear and logistic regression are convex; neural networks are not

## Common Landscapes

```{=latex}
\begin{center}
\begin{tikzpicture}
\pgfplotsset{thumb/.style={width=2.75cm, height=2.7cm, axis lines=box, xtick=\empty, ytick=\empty, title style={yshift=-1.5mm, font=\scriptsize}}}
\begin{axis}[thumb, at={(0cm,0cm)}, xmin=-3, xmax=3, ymin=-0.5, ymax=9.5, title={convex}]
  \addplot[cblue, thick, domain=-3:3] {x^2};
\end{axis}
\begin{axis}[thumb, at={(2.2cm,0cm)}, xmin=-3, xmax=3, ymin=-3, ymax=3, title={ill-conditioned}]
  \foreach \r in {0.6,1.2,1.8,2.4} { \edef\tmp{\noexpand\addplot[cblue, thick, domain=0:360, samples=60] ({\r*1.2*cos(x)}, {\r*0.25*sin(x)});}\tmp }
\end{axis}
\begin{axis}[thumb, at={(4.4cm,0cm)}, xmin=-3, xmax=3, ymin=-1.5, ymax=3, title={multimodal}]
  \addplot[cblue, thick, domain=-3:3, samples=200] {x^2/5 + sin(deg(3*x))};
\end{axis}
\begin{axis}[thumb, at={(6.6cm,0cm)}, xmin=-3, xmax=3, ymin=-1, ymax=3, title={noisy}]
  \addplot[cblue, thick, domain=-3:3, samples=120] {x^2/5 + 0.35*rand};
\end{axis}
\begin{axis}[thumb, at={(8.8cm,0cm)}, xmin=-3, xmax=3, ymin=-0.3, ymax=1.3, title={flat \& steps}]
  \addplot[cblue, thick, domain=-3:3, samples=200] {floor(abs(x))/2};
\end{axis}
\end{tikzpicture}
\end{center}
```

* **Convex:** one basin; any downhill method reaches the minimum
* **Ill-conditioned** (2D contours): steep in one direction, flat in the other
* **Multimodal:** many basins; the start decides the result
* **Noisy:** the same point returns a different value each time
* **Flat or discontinuous:** the gradient is zero or undefined
* Also common, often mixed: curved valleys (Rosenbrock), ridges, deceptive basins

# Optimization in Machine Learning

## Training Is Optimization

$$\theta^\star = \arg\min_\theta\; \frac{1}{N}\sum_{i=1}^{N} \ell\big(f_\theta(x_i),\, y_i\big) \;+\; \lambda\,\Omega(\theta)$$

* $\ell$: the loss on **one** example; the average is the **empirical risk**
* $\lambda\,\Omega(\theta)$: an optional **regularization** term that penalizes complex models (the $L(H)$ part of MDL); $\lambda = 0$ removes it (Topic 3)
* Minimizing the training loss is only a *means*: the goal is low **test** error (Class 01)
* So we rarely need the exact $\theta^\star$; a good $\theta$ found cheaply is often enough

## Common Losses

```{=latex}
\renewcommand{\arraystretch}{1.6}
```

| Loss | $\ell(\hat y, y)$ | Used for | Smooth? |
|:------|:--------------------|:--------|:--:|
| Squared error | $(\hat y - y)^2$ | regression | yes |
| Huber | $r^2/2$ near 0, linear beyond $\delta$ | robust regression | yes |
| Cross-entropy | $-\log \hat p_y$ (prob. of the true class) | classification | yes |
| 0/1 loss | $[\hat y \ne y]$ | what we report | **no** |

* The loss we *care* about (errors) is often not the loss we can *optimize* (smooth surrogates)
* A smooth loss has a useful gradient; a step-shaped loss does not

## Parameters and Hyperparameters

| | Parameters | Hyperparameters |
|:--|:--|:--|
| Examples | weights, biases, centroids | learning rate, depth, $k$, $\lambda$ |
| How many | thousands to billions | a handful |
| Gradient available? | usually yes | usually **no** |
| One evaluation costs | microseconds | a full training run |
| Typical optimizer | gradient-based | blind (search) |

* Both are optimization problems; they sit at different levels
* Blind methods also fit models directly when the loss has no gradient

## Two Families of Optimizers

```{=latex}
\begin{center}
\begin{tikzpicture}
  \begin{scope}
    \foreach \r in {0.4,0.9,1.4} { \draw[cgray] (0,0) ellipse ({\r*1.3} and \r); }
    \draw[cblue, very thick, ->] (-1.7,1.1) -- (-0.9,0.5) -- (-0.4,0.2) -- (-0.1,0.05);
    \node[note, text width=4.6cm, anchor=north] at (0,-1.6) {\textbf{Gradient-based}\\ one point, follows $-\nabla f$\\ needs derivatives, very efficient};
  \end{scope}
  \begin{scope}[xshift=6.2cm]
    \foreach \r in {0.4,0.9,1.4} { \draw[cgray] (0,0) ellipse ({\r*1.3} and \r); }
    \foreach \px/\py in {-1.5/0.8, 1.2/1.1, -0.6/-1.2, 1.6/-0.5, 0.3/0.9, -1.1/-0.3, 0.7/-0.8} { \fill[corange] (\px,\py) circle (2.5pt); }
    \fill[cred] (0.1,0.1) circle (3pt);
    \node[note, text width=4.6cm, anchor=north] at (0,-1.6) {\textbf{Blind (derivative-free)}\\ only values $f(x)$, often a population\\ works on any black box, needs more evaluations};
  \end{scope}
\end{tikzpicture}
\end{center}
```

## Two Test Functions

:::: {.columns}
::: {.column width="50%"}
```{=latex}
\begin{tikzpicture}
\begin{axis}[faa, height=4cm, width=\textwidth, xmin=-4, xmax=4, ymin=0, ymax=20, xlabel={$x$}, title={\scriptsize $F_1$ (convex)}]
  \addplot[cblue, domain=-4:4] {(x-2)^2+1};
  \addplot[only marks, mark=*, cred] coordinates {(2,1)};
\end{axis}
\end{tikzpicture}
```
:::
::: {.column width="50%"}
```{=latex}
\begin{tikzpicture}
\begin{axis}[faa, height=4cm, width=\textwidth, xmin=-4, xmax=4, ymin=-1.4, ymax=4.3, xlabel={$x$}, title={\scriptsize $F_2$ (rugged)}]
  \addplot[cblue, domain=-4:4, samples=400] {x^2/5 + sin(deg(3*x)) + 0.3*sin(deg(11*x))};
  \addplot[only marks, mark=*, cred] coordinates {(-0.67,-1.08)};
\end{axis}
\end{tikzpicture}
```
:::
::::

* $F_1(x) = (x-2)^2 + 1$: convex, minimum at $x^\star = 2$
* $F_2(x) = x^2/5 + \sin 3x + 0.3 \sin 11x$: a smooth bowl (the signal) plus fast ripples that look like **noise**; global minimum $x^\star \approx -0.67$, surrounded by local minima
* The **same two functions** are used by both demo notebooks (`f_convex`, `f_rugged`), so the two families can be compared

# Gradient-Based Optimization

## The Derivative Is a Local Slope

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=4.2cm, width=0.75\textwidth, xmin=-3, xmax=4, ymin=0, ymax=27, xlabel={$x$}, ylabel={$f(x)$}]
  \addplot[cblue, domain=-3:4] {(x-2)^2+1};
  \addplot[corange, thick, domain=-2.5:0.6] {10 - 6*(x+1)};
  \addplot[only marks, mark=*, cred] coordinates {(-1,10)};
  \node[font=\scriptsize, cred, anchor=south west] at (axis cs:-1,10.5) {$x_k = -1$};
  \node[font=\scriptsize, corange, anchor=west] at (axis cs:0.6,20) {tangent, slope $f'(x_k) = -6$};
  \draw[->, very thick, cgreen] (axis cs:-1,4) -- (axis cs:0.4,4) node[right, font=\scriptsize] {move against the slope};
\end{axis}
\end{tikzpicture}
\end{center}
```

* $f'(x) = \lim_{h\to 0} \frac{f(x+h) - f(x)}{h}$: how much $f$ changes per unit step in $x$
* Negative slope: $f$ decreases to the **right**, so step right; positive slope: step left
* In $n$-D, $-\nabla f$ is the direction of **steepest descent**

## Gradient Descent

$$x_{k+1} = x_k - \eta\, \nabla f(x_k)$$

* $\eta > 0$: the **learning rate** (step size)
* It follows from the first-order Taylor model $f(x + \delta) \approx f(x) + \nabla f(x)^\top \delta$: for a small step of fixed length, $\delta \propto -\nabla f$ decreases $f$ the most
* The step is **proportional to the slope**: large far away, small near the minimum
* Stop when $\lVert \nabla f \rVert$ is tiny, the loss stops improving, or the budget runs out

## Worked Example: GD by Hand

:::: {.columns}
::: {.column width="50%"}
$f(x) = (x-2)^2 + 1$, $\;f'(x) = 2(x-2)$

$x_0 = -2$, $\;\eta = 0.1$

| $k$ | $x_k$ | $f(x_k)$ | $f'(x_k)$ |
|:-:|--:|--:|--:|
| 0 | $-2.000$ | 17.000 | $-8.000$ |
| 1 | $-1.200$ | 11.240 | $-6.400$ |
| 2 | $-0.560$ | 7.554 | $-5.120$ |
| 3 | $-0.048$ | 5.194 | $-4.096$ |
| 4 | $0.362$ | 3.684 | $-3.277$ |
| 5 | $0.689$ | 2.718 | $-2.621$ |
:::
::: {.column width="50%"}
```{=latex}
\begin{tikzpicture}
\begin{axis}[faa, height=5cm, width=\textwidth, xmin=-3, xmax=4, ymin=0, ymax=20, xlabel={$x$}]
  \addplot[cblue, domain=-3:4] {(x-2)^2+1};
  \addplot[cred, mark=*, mark size=1.8pt] coordinates {(-2,17) (-1.2,11.24) (-0.56,7.554) (-0.048,5.194) (0.362,3.684) (0.689,2.718) (0.951,2.100) (1.161,1.704) (1.329,1.451) (1.463,1.289)};
\end{axis}
\end{tikzpicture}
```
:::
::::

* $x_1 = -2 - 0.1 \cdot (-8) = -1.2$; the steps shrink as the slope flattens

## The Learning Rate

$$x_{k+1} - 2 = (1 - 2\eta)(x_k - 2) \;\Rightarrow\; \lvert x_k - x^\star \rvert = \lvert 1 - 2\eta\rvert^k \lvert x_0 - x^\star \rvert$$

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=3.5cm, width=0.62\textwidth, ymode=log, xmin=0, xmax=20, ymin=1e-8, ymax=100, xlabel={iteration $k$}, ylabel={$|x_k - x^\star|$}, legend pos=outer north east, yticklabel style={/pgf/number format/sci}]
  \addplot[cblue, domain=0:20, samples=21, mark=*, mark size=1pt] {4*0.9^x}; \addlegendentry{$\eta=0.05$: slow}
  \addplot[cgreen, domain=0:20, samples=21, mark=*, mark size=1pt] {4*0.1^x}; \addlegendentry{$\eta=0.45$: fast}
  \addplot[corange, domain=0:20, samples=21, mark=*, mark size=1pt] {4*0.9^x}; \addlegendentry{$\eta=0.95$: oscillates}
  \addplot[cred, domain=0:20, samples=21, mark=*, mark size=1pt] {4*1.1^x}; \addlegendentry{$\eta=1.05$: diverges}
\end{axis}
\end{tikzpicture}
\end{center}
```

* Too small: slow. Too large: overshoots and oscillates (same error size as $\eta = 0.05$, but the sign flips), and finally **diverges**
* The safe range depends on the **curvature**: here $f'' = 2$ and GD is stable for $\eta < 2/f'' = 1$

## A Rugged Function: Local Minima

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=4.3cm, width=0.85\textwidth, xmin=-4, xmax=4, ymin=-1.4, ymax=4.3, xlabel={$x$}, ylabel={$f(x)$}]
  \addplot[cblue, domain=-4:4, samples=400] {x^2/5 + sin(deg(3*x)) + 0.3*sin(deg(11*x))};
  \foreach \s/\e/\fe in {-3.5/-3.45/3.10, -2.0/-1.91/1.01, -0.4/-0.21/-0.80, 1.0/1.08/-0.06, 3.0/3.32/1.42} {
    \edef\tmp{\noexpand\addplot[only marks, mark=*, cred, mark size=2.2pt] coordinates {(\e,\fe)};}\tmp
    \edef\tmp{\noexpand\draw[->, cgray, thick] (axis cs:\s,{\fe+0.9}) -- (axis cs:\e,{\fe+0.25});}\tmp
  }
  \addplot[only marks, mark=star, cgreen, mark size=4pt, thick] coordinates {(-0.67,-1.08)};
\end{axis}
\end{tikzpicture}
\end{center}
```

$$f'_{rugged}(x) = \tfrac{2x}{5} + 3\cos 3x + 3.3\cos 11x$$

* From 5 starting points, GD stops in the **nearest valley** (red); none reaches the global minimum (green)
* Over 400 starts on $[-4, 4]$, only **15%** end in the global basin: GD is a **local** method

## Where Does the Gradient Come From?

| Method | How | Accuracy | Cost for $n$ parameters |
|:--|:--|:--|:--|
| Symbolic | derive by hand (or CAS) | exact | human time; error-prone |
| Numerical | $\frac{f(x+h) - f(x-h)}{2h}$ | $\approx$ 6--8 digits | $2n$ evaluations of $f$ |
| **Automatic** | chain rule on the program | exact (machine precision) | $\approx$ 2--4 $\times$ one evaluation |

* **Reverse-mode automatic differentiation** (backpropagation) gets *all* $n$ partial derivatives for the price of a few evaluations of $f$
* This is why models with millions of parameters are trainable at all
* In this course: **JAX** (`jax.grad`), also the backend of Keras 3

## Live Demo G1--G3: Gradient Descent

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=4cm, width=0.7\textwidth, xmin=-4, xmax=4, ymin=-4.3, ymax=4.3, xlabel={starting point $x_0$}, ylabel={final $x$}]
  \addplot[cgray, dashed] coordinates {(-4,-0.67) (4,-0.67)};
  \addplot[only marks, mark=*, mark size=1pt, cblue] coordinates {(-4.00,-4.18) (-3.80,-3.45) (-3.60,-3.45) (-3.40,-2.87) (-3.20,-2.87) (-3.00,-2.87) (-2.80,-2.87) (-2.60,-2.44) (-2.40,-2.44) (-2.20,-2.44) (-2.00,-1.91) (-1.80,-1.91) (-1.60,-1.91) (-1.40,-0.67) (-1.20,-0.67) (-1.00,-0.67) (-0.80,-0.67) (-0.60,-0.67) (-0.40,-0.21) (-0.20,-0.21) (0.00,-0.21) (0.20,-0.21) (0.40,0.39) (0.60,0.39) (0.80,1.08) (1.00,1.08) (1.20,1.08) (1.40,1.56) (1.60,1.56) (1.80,1.56) (2.00,1.56) (2.20,1.56) (2.40,1.56) (2.60,2.70) (2.80,2.70) (3.00,3.32) (3.20,3.32) (3.40,3.32) (3.60,3.78) (3.80,3.78) (4.00,3.78)};
  \node[font=\scriptsize, cgray, anchor=south west] at (axis cs:-4,-0.6) {global minimum};
\end{axis}
\end{tikzpicture}
\end{center}
```

* Notebook `01_gradient_based.ipynb`: **G1** derivative by hand, by finite differences and with `jax.grad`; **G2** four learning rates; **G3** basins of attraction
* GD from 41 starts ($\eta = 0.02$, 300 steps). Each flat step is one **basin of attraction**: the start decides the answer

# Newton's Method

## Use the Curvature Too

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=3.6cm, width=0.75\textwidth, xmin=-0.5, xmax=2.3, ymin=0, ymax=6, xlabel={$x$}]
  \addplot[cblue, domain=-0.5:2.3, samples=100] {exp(x) - 2*x}; \addlegendentry{$f(x) = e^x - 2x$}
  \addplot[corange, dashed, domain=-0.2:2.3, samples=100] {1.4817 + 2.4817*(x-1.5) + 0.5*4.4817*(x-1.5)^2}; \addlegendentry{parabola at $x_k = 1.5$}
  \addplot[only marks, mark=*, cred] coordinates {(1.5,1.4817)};
  \addplot[only marks, mark=*, cgreen] coordinates {(0.9463,0.7942)};
  \draw[->, thick, cgray] (axis cs:1.45,1.2) -- (axis cs:1.0,0.95);
  \node[font=\scriptsize, cgreen, anchor=north] at (axis cs:0.95,0.6) {$x_{k+1}$};
\end{axis}
\end{tikzpicture}
\end{center}
```

$$f(x_k + \delta) \approx f(x_k) + f'(x_k)\,\delta + \tfrac12 f''(x_k)\,\delta^2$$
$$\Longrightarrow\quad x_{k+1} = x_k - \frac{f'(x_k)}{f''(x_k)}$$

* Fit a **parabola** (second-order Taylor) at $x_k$ and jump to its minimum
* No learning rate: the step length comes from the curvature

## Newton in $n$ Dimensions

$$\theta_{k+1} = \theta_k - H(\theta_k)^{-1}\, \nabla \mathcal{L}(\theta_k)$$

* $H$: the $n \times n$ Hessian. In practice solve $H\,\delta = -\nabla\mathcal{L}$ (never invert $H$)
* On a **quadratic** loss (e.g. linear regression with MSE) the Taylor model is exact: **one step** reaches the minimum
* Each step: $n^2$ second derivatives and an $O(n^3)$ solve
* **Damped** Newton: $(H + \lambda I)^{-1}$ blends Newton ($\lambda \to 0$) with a small gradient step ($\lambda$ large)
* **Quasi-Newton** (BFGS, L-BFGS): build an approximation of $H^{-1}$ from successive gradients

## Worked Example: Quadratic Convergence

:::: {.columns}
::: {.column width="48%"}
$f(x) = e^x - 2x$, $\;x^\star = \ln 2$, $\;x_0 = 2$

| $k$ | Newton $\lvert x_k - x^\star\rvert$ | GD ($\eta = 0.1$) |
|:-:|--:|--:|
| 0 | $1.3$ | $1.3$ |
| 1 | $5.8 \cdot 10^{-1}$ | $7.7 \cdot 10^{-1}$ |
| 2 | $1.4 \cdot 10^{-1}$ | $5.4 \cdot 10^{-1}$ |
| 3 | $9.2 \cdot 10^{-3}$ | $4.0 \cdot 10^{-1}$ |
| 4 | $4.2 \cdot 10^{-5}$ | $3.0 \cdot 10^{-1}$ |
| 5 | $8.9 \cdot 10^{-10}$ | $2.3 \cdot 10^{-1}$ |
| 6 | $0$ (machine) | $1.8 \cdot 10^{-1}$ |
:::
::: {.column width="52%"}
```{=latex}
\begin{tikzpicture}
\begin{axis}[faa, height=5.2cm, width=\textwidth, ymode=log, xmin=0, xmax=6, ymin=1e-16, ymax=10, xlabel={iteration}, legend pos=south west]
  \addplot[cblue, mark=*, mark size=1.5pt] coordinates {(0,1.31) (1,0.578) (2,0.139) (3,9.2e-3) (4,4.22e-5) (5,8.91e-10) (6,1e-16)}; \addlegendentry{Newton}
  \addplot[corange, mark=*, mark size=1.5pt] coordinates {(0,1.31) (1,0.768) (2,0.537) (3,0.395) (4,0.298) (5,0.229) (6,0.177)}; \addlegendentry{GD}
\end{axis}
\end{tikzpicture}
```
:::
::::

* Near the minimum Newton **doubles the correct digits** at every step: $9\!\cdot\!10^{-3} \to 4\!\cdot\!10^{-5} \to 9\!\cdot\!10^{-10}$

## Newton on Rosenbrock

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=4.6cm, width=0.55\textwidth, xmin=-1.6, xmax=1.6, ymin=-3.4, ymax=2.2, xlabel={$x$}, ylabel={$y$}, legend pos=outer north east]
  \addplot[cgray!60, line width=6pt, domain=-1.5:1.48] {x^2}; \addlegendentry{valley floor $y = x^2$}
  \addplot[corange, mark=*, mark size=1.2pt] coordinates {(-1.20,1.00) (-0.51,0.27) (0.33,0.10) (0.60,0.36) (0.72,0.52) (0.79,0.63) (0.84,0.71) (0.88,0.77) (0.90,0.82) (0.92,0.85) (0.94,0.88)}; \addlegendentry{GD: 5000 steps (every 500th)}
  \addplot[cred, thick, mark=*, mark size=1.8pt] coordinates {(-1.20,1.00) (-1.18,1.38) (0.76,-3.18) (0.76,0.58) (1.00,0.94) (1.00,1.00)}; \addlegendentry{Newton: 8 steps}
  \addplot[only marks, mark=star, black, mark size=4pt, thick] coordinates {(1,1)};
\end{axis}
\end{tikzpicture}
\end{center}
```

* $f(x,y) = (1-x)^2 + 100\,(y-x^2)^2$, minimum at $(1,1)$, start at $(-1.2, 1)$
* Newton: exact after **5 steps** (with a wild jump on the way). GD: still $f \approx 4\cdot10^{-3}$ after 5000

## When Newton Fails

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=3.6cm, width=0.6\textwidth, xmin=-2, xmax=1, ymin=-1.5, ymax=1.3, xlabel={$x$}, legend pos=outer north east]
  \addplot[cblue, domain=-2:1, samples=300, forget plot] {x^2/5 + sin(deg(3*x)) + 0.3*sin(deg(11*x))};
  \addplot[cgreen, mark=*, mark size=1.8pt] coordinates {(-0.8,-0.723) (-0.67,-1.081)}; \addlegendentry{$x_0 = -0.8$: a minimum}
  \addplot[cred, mark=*, mark size=1.8pt] coordinates {(-0.5,-0.736) (-0.403,-0.614)}; \addlegendentry{$x_0 = -0.5$: a \textbf{maximum}}
\end{axis}
\end{tikzpicture}
\end{center}
```

* Newton solves $f'(x) = 0$: it cannot tell minima, maxima and saddles apart
* Where $f'' < 0$ the parabola opens downwards and the step goes **uphill**
* Where $f'' \approx 0$ the step explodes: from $x_0 = -0.9$ it ends at $x = 7.28$
* Cost: $O(n^2)$ memory and $O(n^3)$ time per step, impossible for millions of weights

## Live Demo G4: Newton's Method

* Notebook `01_gradient_based.ipynb`, section **G4**
* One step on a quadratic; quadratic convergence on $e^x - 2x$
* Minimum, maximum and explosion on the rugged function
* `jax.hessian` and `jnp.linalg.solve` on Rosenbrock
* Question: which of these would you use to train a network with $10^6$ weights?

# Momentum and Adaptive Methods

## Ill-Conditioning: the Zig-Zag

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=4.4cm, width=0.6\textwidth, xmin=-5, xmax=1, ymin=-1.6, ymax=1.7, xlabel={$x$}, ylabel={$y$}, legend pos=outer north east]
  \foreach \c in {0.5,2,5,10} { \edef\tmp{\noexpand\addplot[cgray!50, domain=0:360, samples=80, forget plot] ({sqrt(2*\c)*cos(x)}, {sqrt(2*\c/25)*sin(x)});}\tmp }
  \addplot[cblue, mark=*, mark size=0.9pt] coordinates {(-4.50,1.50) (-4.16,-1.31) (-3.85,1.15) (-3.56,-1.00) (-3.29,0.88) (-3.05,-0.77) (-2.82,0.67) (-2.61,-0.59) (-2.41,0.52) (-2.23,-0.45) (-2.06,0.39) (-1.91,-0.35) (-1.77,0.30) (-1.63,-0.26) (-1.51,0.23) (-1.40,-0.20) (-1.29,0.18) (-1.20,-0.15) (-1.11,0.14) (-1.02,-0.12) (-0.95,0.10) (-0.88,-0.09) (-0.81,0.08) (-0.75,-0.07) (-0.69,0.06)}; \addlegendentry{GD, $\eta = 0.075$}
  \addplot[corange, mark=*, mark size=0.9pt] coordinates {(-4.50,1.50) (-4.37,0.38) (-4.14,-0.69) (-3.86,-0.92) (-3.54,-0.39) (-3.22,0.27) (-2.89,0.53) (-2.58,0.31) (-2.28,-0.07) (-2.01,-0.29) (-1.75,-0.22) (-1.52,-0.01) (-1.32,0.15) (-1.13,0.15) (-0.97,0.04) (-0.83,-0.07) (-0.70,-0.09) (-0.59,-0.04) (-0.50,0.03) (-0.42,0.05) (-0.35,0.03) (-0.29,-0.01) (-0.24,-0.03) (-0.20,-0.02) (-0.16,-0.00)}; \addlegendentry{Momentum, $\eta=0.03,\ \beta=0.7$}
\end{axis}
\end{tikzpicture}
\end{center}
```

* $q(x,y) = \tfrac12(x^2 + 25 y^2)$: 25 times steeper in $y$ than in $x$ (first 25 steps)
* GD's step is capped by the **steep** direction ($\eta < 2/25$), so it crawls along the **flat** one
* Momentum averages out the zig-zag and accumulates speed along the valley

## Momentum: a Heavy Ball

```{=latex}
\begin{center}
\begin{tikzpicture}
  \draw[cblue, thick, domain=-3:3, samples=100] plot (\x, {0.18*\x*\x + 0.15*sin(deg(5*\x))});
  \fill[cred] (-2.2,{0.18*4.84 + 0.15*sin(deg(-11)) + 0.2}) circle (0.2);
  \draw[->, very thick, corange] (-1.9,1.25) -- (-0.8,0.65) node[pos=0.1, above=3mm, note, anchor=south west] {velocity $v$ carries it over small bumps};
\end{tikzpicture}
\end{center}
```

$$v_{k+1} = \beta\, v_k + \nabla\mathcal{L}(\theta_k), \qquad \theta_{k+1} = \theta_k - \eta\, v_{k+1}$$

* $v$: an exponentially weighted **sum of past gradients**; $\beta \approx 0.9$ is the "friction"
* Consistent directions add up; oscillating directions cancel
* **Nesterov:** evaluate the gradient at the look-ahead point $\theta_k - \eta\beta v_k$ (corrects earlier)

## Adaptive Step Sizes: RMSProp

$$s_{k+1} = \rho\, s_k + (1-\rho)\, g_k^2, \qquad \theta_{k+1} = \theta_k - \eta\, \frac{g_k}{\sqrt{s_{k+1}} + \epsilon}$$

* $g_k = \nabla\mathcal{L}(\theta_k)$; squares and divisions are **per coordinate**
* $s$ tracks the recent **size** of each gradient coordinate
* Steep coordinates get smaller steps, flat ones larger: every direction moves at a similar pace
* A cheap, **diagonal** stand-in for Newton's curvature scaling ($n$ numbers instead of $n^2$)
* History: AdaGrad (2011) sums all past $g^2$; RMSProp (Hinton, 2012) forgets old ones

## Adam = Momentum + RMSProp

\begin{align*}
m_{k+1} &= \beta_1 m_k + (1-\beta_1)\, g_k, & s_{k+1} &= \beta_2 s_k + (1-\beta_2)\, g_k^2\\
\hat m &= m_{k+1}/(1-\beta_1^{k+1}), & \hat s &= s_{k+1}/(1-\beta_2^{k+1})\\
\theta_{k+1} &= \theta_k - \eta\, \hat m / (\sqrt{\hat s} + \epsilon)
\end{align*}

* $m$: the **direction** (momentum); $s$: the **scale** (RMSProp)
* Defaults: $\eta = 10^{-3}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$ (Kingma and Ba, 2015)
* Bias correction: $m$ and $s$ start at 0, so early averages are too small without it
* The default optimizer of deep learning: `keras.optimizers.Adam`

## Comparing the Optimizers

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, ybar, height=4.5cm, width=0.9\textwidth, ymode=log, log origin=infty, ymin=1e-20, ymax=10, bar width=16pt, x tick label style={font=\scriptsize}, symbolic x coords={GD, Momentum, Nesterov, RMSProp, Adam}, xtick=data, ylabel={loss after 3000 steps}, nodes near coords, nodes near coords style={font=\tiny, /pgf/number format/sci, /pgf/number format/precision=1}, point meta=rawy, enlarge x limits=0.15]
  \addplot[fill=cblue!60, draw=cblue] coordinates {(GD,2.5e-2) (Momentum,1.5e-12) (Nesterov,2.2e-12) (RMSProp,2.3e-3) (Adam,5.1e-19)};
\end{axis}
\end{tikzpicture}
\end{center}
```

* Rosenbrock from $(-1.2, 1)$, learning rates tuned roughly by hand: $10^{-3}$ (GD, momentum), $3\!\cdot\!10^{-3}$ (RMSProp), $0.05$ (Adam)
* With a constant $\eta$, RMSProp and Adam **hover** around the minimum with steps of order $\eta$: in practice $\eta$ is decayed

## Stochastic Gradient Descent

$$\nabla\mathcal{L}(\theta) = \frac1N\sum_{i=1}^{N} \nabla\ell_i(\theta) \;\approx\; \frac1B\sum_{i \in \mathcal{B}} \nabla\ell_i(\theta)$$

```{=latex}
\begin{center}
\begin{tikzpicture}
  \foreach \r in {0.4,0.9,1.4} { \draw[cgray] (0,0) ellipse ({\r*1.6} and \r); }
  \draw[cblue, thick, ->] (-2.2,1.0) -- (-1.4,0.6) -- (-0.8,0.3) -- (-0.3,0.1) -- (-0.05,0.02);
  \draw[cred, thick, ->] (-2.2,1.0) -- (-1.7,0.35) -- (-1.2,0.8) -- (-0.9,0.1) -- (-0.4,0.45) -- (-0.3,-0.2) -- (0.1,0.2) -- (0.05,-0.05);
  \node[note, cblue, anchor=west] at (2.6,0.6) {full batch: smooth, expensive};
  \node[note, cred, anchor=west] at (2.6,0.1) {mini-batch: noisy, cheap};
\end{tikzpicture}
\end{center}
```

* A **mini-batch** $\mathcal{B}$ of $B \ll N$ examples gives a cheap, unbiased, **noisy** estimate of the gradient
* One pass over all data = one **epoch**; $B = 32$--$256$ is typical
* The noise is useful: it helps escape sharp minima and saddle points

## Live Demo G5: Momentum, RMSProp, Adam, SGD

* Notebook `01_gradient_based.ipynb`, section **G5**
* All four optimizers written in a few lines each, on the ill-conditioned bowl and on Rosenbrock
* Momentum on the rugged 1D function: from $x_0 = -3.7$ it rolls into the global basin where GD stops early; over 400 starts, the mean final loss drops from $0.38$ to $0.05$
* SGD with batch sizes 1000, 16 and 1 on a line fit: the smaller the batch, the noisier the path near the minimum

# Automatic Differentiation with JAX

## JAX: NumPy That Can Differentiate

| Transform | Returns |
|:--|:--|
| `jax.grad(f)` | a function computing $\nabla f$ (reverse mode) |
| `jax.value_and_grad(f)` | $f$ and $\nabla f$ in one pass |
| `jax.hessian(f)` | the matrix of second derivatives |
| `jax.jit(f)` | $f$ compiled with XLA (CPU or GPU) |
| `jax.vmap(f)` | $f$ vectorized over a batch axis |

* Write the model and the loss with `jax.numpy` (same API as NumPy); JAX derives the rest
* Transforms **compose**: `jax.jit(jax.vmap(jax.grad(loss)))`
* Keras 3 on the JAX backend uses exactly this machinery to train networks

## The Chain Rule on a Graph

```{=latex}
\begin{center}
\begin{tikzpicture}[node distance=9mm, every node/.style={font=\footnotesize}]
  \node[fillbox=cgray] (w) {$w$};
  \node[fillbox=cgray, below=5mm of w] (b) {$b$};
  \node[fillbox=cblue, right=of w, yshift=-6mm] (z) {$z = w x + b$};
  \node[fillbox=cblue, right=of z] (r) {$r = z - y$};
  \node[fillbox=cblue, right=of r] (l) {$\ell = r^2$};
  \draw[flow] (w) -- (z); \draw[flow] (b) -- (z); \draw[flow] (z) -- (r); \draw[flow] (r) -- (l);
  \draw[->, thick, cred, bend left=35] (l.south) to node[below, note, text=cred] {$\partial\ell/\partial r = 2r$} (r.south);
  \draw[->, thick, cred, bend left=35] (r.south) to node[below, note, text=cred] {$\cdot\, 1$} (z.south);
  \draw[->, thick, cred, bend left=20] (z.west) to node[left, note, text=cred, pos=0.7] {$\cdot\, x$} (w.east);
  \draw[->, thick, cred, bend right=20] (z.west) to node[below, note, text=cred, pos=0.8] {$\cdot\, 1$} (b.east);
  \node[note, above=3mm of r] {\textcolor{cblue}{forward: compute values}\quad \textcolor{cred}{backward: multiply local derivatives}};
\end{tikzpicture}
\end{center}
```

* **Forward pass:** evaluate the program and remember intermediate values
* **Backward pass:** apply the chain rule from the output back to every input: $\frac{\partial\ell}{\partial w} = 2r \cdot 1 \cdot x$
* One backward pass gives **all** partial derivatives: this is backpropagation (Topic 4)

## JAX in Five Lines

```python
import jax, jax.numpy as jnp

def loss(theta, x, y):                     # any model, any loss
    return jnp.mean((theta[0] * x + theta[1] - y) ** 2)

grad = jax.jit(jax.grad(loss))              # derived and compiled
for _ in range(100):
    theta = theta - 0.1 * grad(theta, x, y) # gradient descent
```

* No derivative written by hand; changing the model or the loss changes nothing else
* `jax.jit`: the gradient of Rosenbrock drops from about **8 ms** to about **7 $\mu$s** per call, $\approx 1000\times$ faster (the first call compiles)

## A Custom Model with a Custom Loss

:::: {.columns}
::: {.column width="52%"}
```{=latex}
\begin{tikzpicture}
\begin{axis}[faa, height=4.4cm, width=\textwidth, xmin=0, xmax=10, ymin=-2.4, ymax=2.6, xlabel={$t$}, legend style={font=\tiny, at={(0.5,-0.35)}, anchor=north, legend columns=2}]
  \addplot[cgreen, dashed, domain=0:10, samples=200] {2*exp(-0.3*x)*cos(deg(2*x+0.5))}; \addlegendentry{true signal}
  \addplot[cblue, domain=0:10, samples=200] {2.017*exp(-0.307*x)*cos(deg(2.002*x+0.483))}; \addlegendentry{Huber fit}
  \addplot[only marks, mark=*, black, mark size=0.8pt] coordinates {(0.00,1.79) (0.20,1.26) (0.40,0.50) (0.60,-0.30) (0.80,-0.80) (1.06,-1.28) (1.26,-1.55) (1.46,-1.22) (1.66,-0.75) (1.91,-0.42) (2.11,0.00) (2.31,0.40) (2.61,0.85) (2.91,0.71) (3.12,0.61) (3.32,0.51) (3.57,0.12) (3.77,-0.03) (3.97,-0.38) (4.17,-0.60) (4.37,-0.53) (4.57,-0.47) (4.77,-0.61) (4.97,-0.20) (5.18,0.01) (5.38,-0.05) (5.58,0.24) (5.78,0.31) (6.03,0.48) (6.23,0.54) (6.43,0.12) (6.63,0.06) (6.83,-0.04) (7.09,0.02) (7.29,-0.22) (7.49,-0.32) (7.69,-0.29) (7.99,-0.13) (8.24,0.02) (8.44,-0.04) (8.64,0.07) (8.84,0.05) (9.05,0.18) (9.25,0.21) (9.45,-0.05) (9.65,0.16) (9.85,0.13)}; \addlegendentry{data (every 4th)}
  \addplot[only marks, mark=*, cred, mark size=1.5pt] coordinates {(0.85,0.55) (1.76,-2.19) (2.41,-0.88) (2.51,-0.78) (2.76,-0.76) (2.81,2.44) (3.37,2.00) (5.83,-1.25) (7.04,1.29) (7.74,1.27) (7.89,1.36) (8.19,-1.54)}; \addlegendentry{12 outliers}
\end{axis}
\end{tikzpicture}
```
:::
::: {.column width="48%"}
$$\hat y(t) = A\, e^{-\lambda t} \cos(\omega t + \varphi)$$

| | $A$ | $\lambda$ | $\omega$ | $\varphi$ |
|:--|--:|--:|--:|--:|
| true | 2.000 | 0.300 | 2.000 | 0.500 |
| MSE | 2.023 | 0.326 | 1.963 | 0.458 |
| Huber | 2.017 | 0.307 | 2.002 | 0.483 |
:::
::::

* 200 noisy points and 12 outliers; **Huber loss**: quadratic for small residuals, linear for large ones
* Same model, two losses, zero derivatives by hand: Huber is pulled less by the outliers

## Differentiating Through a Simulation

```{=latex}
\begin{center}
\begin{tikzpicture}[node distance=6mm]
  \node[fillbox=cgray, text width=1.8cm] (p) {$r, K$};
  \node[fillbox=cblue, text width=5.2cm, right=of p] (s) {100 Euler steps\\ $y_{t+1} = y_t + \Delta t\, r\, y_t (1 - y_t/K)$};
  \node[fillbox=corange, text width=2cm, right=of s] (l) {MSE vs.\\ observations};
  \draw[flow] (p) -- (s); \draw[flow] (s) -- (l);
  \draw[->, thick, cred] (l.south) -- ++(0,-0.5) -| node[pos=0.25, below, note, text=cred] {\texttt{jax.grad} through the whole loop (\texttt{lax.scan})} (p.south);
\end{tikzpicture}
\end{center}
```

* Some models have no formula, only a **numerical integration** (an ODE solver, a physics simulation)
* JAX differentiates the solver itself: from $(r, K) = (0.5, 3)$, Adam recovers $r = 0.805$ (true 0.8) and $K = 4.958$ (true 5.0)
* Use this whenever a model or a loss is not in a library

## Live Demo G6: JAX for Custom Models

* Notebook `01_gradient_based.ipynb`, section **G6**
* `jit` timing; the damped oscillator with MSE and Huber; logistic growth fitted through its ODE solver
* Exercise: change the Huber $\delta$, or the model, and rerun: nothing else changes

# Blind Optimization

## When There Is No Useful Gradient

```{=latex}
\begin{center}
\begin{tikzpicture}[every node/.style={fillbox=corange, text width=2.5cm, minimum height=1.2cm, font=\scriptsize}]
  \node at (0,0) {\textbf{black box}\\ simulator, lab experiment, game};
  \node at (3,0) {\textbf{no derivative}\\ 0/1 loss, ranks, discrete choices};
  \node at (6,0) {\textbf{noisy}\\ each evaluation differs};
  \node at (9,0) {\textbf{hyper-parameters}\\ one value = one training run};
  \node[fillbox=cred] at (4.5,-1.7) {\textbf{multimodal}\\ gradients find only the nearest valley};
\end{tikzpicture}
\end{center}
```

* A **blind** (derivative-free, zeroth-order) optimizer only asks: *what is $f(x)$ here?*
* It needs no formula, tolerates discontinuities and noise, and can search globally
* The price: many more evaluations, and weaker guarantees

## Activity: Be the Optimizer

```{=latex}
\begin{center}
\begin{tikzpicture}
  \fill[black!85] (0,0) rectangle (5,3.2);
  \foreach \x/\y/\c/\n in {1.0/2.4/cred/1, 3.8/0.8/corange/2, 2.6/1.9/cgreen/3, 2.9/1.6/cgreen!70!black/4, 3.3/1.3/corange/5} {
    \fill[\c] (\x,\y) circle (5pt); \node[white, font=\tiny] at (\x,\y) {\n};
  }
  \node[note, anchor=west, text width=5.5cm] at (5.4,1.6) {Hidden 2D function\\ 5 evaluations per problem\\ no landscape, no $f^\star$, no gradient\\ only the numbers you asked for};
\end{tikzpicture}
\end{center}
```

* **blindgame**: minimize hidden functions by hand, with a class leaderboard (link given in class)
* Inspired by the **GECCO 2025 Fun Competition**: *manual* optimization by human experts
* Afterwards, the algorithms of this lecture get the same budget: who does better, you or them?
* Think about it while you play: where do you place the first point? When do you stop exploring?

## Exploration versus Exploitation

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=3.8cm, width=0.8\textwidth, xmin=-4, xmax=4, ymin=-1.4, ymax=4.3, xtick=\empty, ytick=\empty]
  \addplot[cblue, domain=-4:4, samples=400] {x^2/5 + sin(deg(3*x)) + 0.3*sin(deg(11*x))};
  \addplot[only marks, mark=*, corange, mark size=2pt] coordinates {(-3.3,3.07) (-2.1,1.10) (0.3,0.77) (2.2,1.50) (3.6,1.87)};
  \addplot[only marks, mark=*, cgreen, mark size=2pt] coordinates {(1.45,-0.66) (1.52,-0.78) (1.58,-0.80) (1.63,-0.73)};
  \node[font=\scriptsize, corange, anchor=west] at (axis cs:-1.5,3.6) {explore: wide jumps};
  \node[font=\scriptsize, cgreen!70!black, anchor=east] at (axis cs:1.2,-1.0) {exploit: small steps};
\end{axis}
\end{tikzpicture}
\end{center}
```

* **Exploration:** sample new regions; finds the right basin, but is imprecise
* **Exploitation:** refine around the best point; precise, but can be trapped
* Every blind algorithm is a **schedule** between the two: first explore, then exploit

## Random Search and Hill Climbing

:::: {.columns}
::: {.column width="50%"}
**Random search**

1. sample $x \sim \mathcal{U}(\text{box})$
2. keep the best seen so far

No memory of *where* good points are.
:::
::: {.column width="50%"}
**Hill climbing**

1. $x' = x + \mathcal{N}(0, \sigma^2)$
2. if $f(x') < f(x)$: $x \leftarrow x'$

A local search: the step size $\sigma$ sets how far it looks.
:::
::::

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, xbar, height=3.5cm, width=0.62\textwidth, xmin=0, xmax=100, bar width=6pt, symbolic y coords={HC $\sigma{=}1.0$, SA $\sigma{=}0.5$, HC $\sigma{=}0.5$, HC $\sigma{=}0.1$, Random search}, ytick=data, xlabel={\% of 100 runs reaching the global basin}, nodes near coords, nodes near coords style={font=\tiny}, enlarge y limits=0.15]
  \addplot[fill=cblue!60, draw=cblue] coordinates {(94,Random search) (21,HC $\sigma{=}0.1$) (36,HC $\sigma{=}0.5$) (78,SA $\sigma{=}0.5$) (95,HC $\sigma{=}1.0$)};
\end{axis}
\end{tikzpicture}
\end{center}
```

* $f_{rugged}$, 200 evaluations, random start. In **1D** random search is excellent; in 10D it will be the worst

## Simulated Annealing

:::: {.columns}
::: {.column width="45%"}
Accept a **worse** point ($\Delta f > 0$) with probability
$$P(\text{accept}) = e^{-\Delta f / T}$$

Cooling schedule (pyBlindOpt):
$$T_t = \frac{T_0}{t + 1}$$
:::
::: {.column width="55%"}
```{=latex}
\begin{tikzpicture}
\begin{axis}[faa, height=4.2cm, width=\textwidth, xmin=0, xmax=5, ymin=0, ymax=1.05, xlabel={$\Delta f$ (how much worse)}, ylabel={$P$(accept)}, legend style={at={(0.5,-0.45)}, anchor=north, legend columns=3}]
  \addplot[cred, domain=0:5, samples=60] {exp(-x/30)}; \addlegendentry{$T = 30$ (start)}
  \addplot[corange, domain=0:5, samples=60] {exp(-x/3)}; \addlegendentry{$T = 3$}
  \addplot[cblue, domain=0:5, samples=60] {exp(-x/0.3)}; \addlegendentry{$T = 0.3$ (late)}
\end{axis}
\end{tikzpicture}
```
:::
::::

* Hot: accepts almost anything (**explores**); cold: only improvements (**exploits**, like hill climbing)
* Inspired by annealing in metallurgy: slow cooling lets the atoms find a low-energy state
* On $f_{rugged}$: 78% of runs reach the global basin against 36% for hill climbing with the same $\sigma$

## Live Demo B1: Local Search

* Notebook `02_blind_optimization.ipynb`, section **B1**
* Random search, hill climbing and simulated annealing from $x_0 = 3$, 200 evaluations each
* Success rates over 100 random starts, for several step sizes
* On $f_{convex}$ all three work, but each needs **200 evaluations** where GD needed about 25 gradient steps

# Initialization

## Where Do We Start?

```{=latex}
\begin{center}
\begin{tikzpicture}
\foreach \name/\pts [count=\i] in {
  Random/{(0.81,0.81) (0.52,0.29) (0.05,0.38) (0.41,0.05) (0.05,1.00) (0.65,0.23) (0.43,0.97) (0.90,0.84) (0.39,0.49) (0.68,0.06) (0.56,0.27) (0.88,0.06) (0.68,0.87) (0.23,0.90) (0.87,0.02) (0.71,0.00)},
  Latin hypercube/{(0.11,0.79) (0.62,0.64) (0.16,0.70) (0.40,0.84) (0.99,0.61) (0.50,0.45) (0.78,0.40) (0.34,0.06) (0.44,0.25) (0.87,0.23) (0.67,0.12) (0.25,0.34) (0.70,0.13) (0.21,0.92) (0.05,0.52) (0.93,0.95)},
  Sobol/{(0.67,0.81) (0.17,0.31) (0.42,0.56) (0.92,0.06) (0.80,0.68) (0.30,0.18) (0.05,0.93) (0.55,0.43) (0.61,0.62) (0.11,0.12) (0.36,0.87) (0.86,0.37) (0.98,0.99) (0.48,0.49) (0.23,0.74) (0.73,0.24)}} {
  \begin{scope}[xshift={(\i-1)*3.6cm}]
  \begin{axis}[width=3.6cm, height=3.6cm, axis lines=box, xtick=\empty, ytick=\empty, xmin=0, xmax=1, ymin=0, ymax=1, title={\scriptsize \name}, grid=major, grid style={cgray!25}, xtick={0.25,0.5,0.75}, ytick={0.25,0.5,0.75}, xticklabels={}, yticklabels={}]
    \addplot[only marks, mark=*, cblue, mark size=1.6pt] coordinates {\pts};
  \end{axis}
  \end{scope}
}
\end{tikzpicture}
\end{center}
```

* 16 points from pyBlindOpt's samplers (`RandomSampler`, `HLCSampler`, `SobolSampler`)
* **Random:** clumps and holes. **Latin hypercube:** one point per row strip and per column strip
* **Sobol:** a low-discrepancy sequence, spread evenly at every scale. **Chaotic:** a logistic map

## Opposition-Based Strategies

```{=latex}
\begin{center}
\begin{tikzpicture}[x=1.1cm]
  \draw[thick] (0,0) -- (10,0);
  \foreach \p/\l in {0/$\ell$, 5/centre $c$, 10/$u$} { \draw (\p,0.12) -- (\p,-0.12) node[below, font=\scriptsize] {\l}; }
  \fill[cblue] (2,0) circle (3pt) node[above=2pt, font=\scriptsize, cblue] {sample $x$};
  \fill[cred] (8,0) circle (3pt) node[above=2pt, font=\scriptsize, cred] {opposite $\breve x = \ell + u - x$};
  \draw[corange, line width=4pt, opacity=0.5] (5,0) -- (8,0);
  \node[font=\scriptsize, corange] at (6.5,-0.85) {quasi-opposite: uniform in $[c, \breve x]$};
\end{tikzpicture}
\end{center}
```

* **OBL:** evaluate every sample and its opposite, keep the best $N$ of the $2N$
* **QOBL:** the opposite is drawn between the centre and $\breve x$
* **OBLESA:** OBL plus probes placed in the **empty space** no sample covered (EmptySpaceSearch), then the best $N$ with a diversity bonus
* Samplers are free; strategies **spend evaluations** ($2N$, $3N$) before the search starts

## A Benchmark Pitfall: the Centre

```{=latex}
\begin{center}
\begin{tikzpicture}
  \draw[thick] (0,0) rectangle (3,3); \fill[cred] (1.5,1.5) circle (3pt); \node[note, below] at (1.5,-0.1) {textbook: $x^\star$ at the centre};
  \draw[thick] (5,0) rectangle (8,3); \fill[cred] (6.9,0.6) circle (3pt); \node[note, below] at (6.5,-0.1) {realistic: $x^\star$ anywhere};
  \draw[->, cgray, thick] (3.4,1.5) -- node[above, note] {shift $s$} (4.6,1.5);
\end{tikzpicture}
\end{center}
```

* Sphere, Rastrigin, Ackley have $x^\star = 0$ at the centre of a symmetric box $[-5, 5]^d$
* There **OBL is useless** ($f(-x) = f(x)$: the opposite has the same value) and **QOBL looks brilliant** (it samples towards the centre)
* Always test on **shifted** functions $f(x - s)$, as the BBOB benchmarks do

## Does Initialization Matter?

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, ybar, height=4.3cm, width=0.85\textwidth, ymin=0, ymax=150, bar width=9pt, symbolic x coords={Random, LHS, Sobol, OBL, QOBL, OBLESA}, xtick=data, ylabel={median best $f$}, legend style={at={(0.5,1.02)}, anchor=south, legend columns=2}, nodes near coords, nodes near coords style={font=\tiny, /pgf/number format/fixed, /pgf/number format/precision=0}, enlarge x limits=0.1]
  \addplot[fill=cgray!40, draw=cgray] coordinates {(Random,126.8) (LHS,128.6) (Sobol,129.4) (OBL,121.8) (QOBL,88.5) (OBLESA,87.7)}; \addlegendentry{initial population}
  \addplot[fill=cblue!60, draw=cblue] coordinates {(Random,28.0) (LHS,26.0) (Sobol,29.8) (OBL,28.4) (QOBL,19.8) (OBLESA,22.9)}; \addlegendentry{after 40 generations of DE}
\end{axis}
\end{tikzpicture}
\end{center}
```

* Shifted Rastrigin, $d = 10$, 20 individuals, **30 seeds** per method (lower is better)
* Better spreading (LHS, Sobol) barely matters with 20 points in 10D; **selecting** the start (QOBL, OBLESA) gives DE a head start that survives
* The effect depends on the algorithm: EGWO hardly responds to a better start

## Live Demo B2: Initialization

* Notebook `02_blind_optimization.ipynb`, section **B2**
* The four samplers in 2D, with the closest pair of points as a crude coverage measure
* OBL, QOBL and OBLESA on a shifted Rastrigin landscape
* 180 DE runs: initial and final best for each initializer, as box plots

# Population-Based Algorithms

## One Loop, Many Algorithms

```{=latex}
\begin{center}
\begin{tikzpicture}[node distance=7mm]
  \node[fillbox=cgray, text width=2cm] (i) {initialize\\ $N$ points};
  \node[fillbox=cblue, text width=2cm, right=of i] (e) {evaluate\\ $f$ of each};
  \node[fillbox=corange, text width=2.2cm, right=of e] (g) {generate\\ new candidates};
  \node[fillbox=cgreen, text width=2cm, right=of g] (s) {select\\ who survives};
  \draw[flow] (i) -- (e); \draw[flow] (e) -- (g); \draw[flow] (g) -- (s);
  \draw[flow] (s.south) -- ++(0,-0.6) -| node[pos=0.25, below, note] {repeat until the budget is spent; return the best ever seen} (e.south);
\end{tikzpicture}
\end{center}
```

* The algorithms differ only in **generate** and **select**
* pyBlindOpt implements this loop once (`Optimizer.optimize`); each algorithm supplies its operators
* Budget = $N \times (\text{epochs} + 1)$ evaluations: compare algorithms at **equal budget**

## Genetic Algorithm

```{=latex}
\begin{center}
\begin{tikzpicture}[node distance=5mm]
  \node[fillbox=cgray, text width=2.1cm] (p) {population\\ (with losses)};
  \node[fillbox=cblue, text width=2.1cm, right=of p] (t) {\textbf{selection}\\ tournament of $k$};
  \node[fillbox=corange, text width=2.1cm, right=of t] (c) {\textbf{crossover}\\ blend two parents};
  \node[fillbox=cred, text width=2.1cm, right=of c] (m) {\textbf{mutation}\\ small random change};
  \node[fillbox=cgreen, text width=2.1cm, below=6mm of c] (n) {new generation\\ + \textbf{elites}};
  \draw[flow] (p) -- (t); \draw[flow] (t) -- (c); \draw[flow] (c) -- (m); \draw[flow] (m) |- (n); \draw[flow] (n) -| (p);
\end{tikzpicture}
\end{center}
```

* **Selection** (who reproduces) sets the pressure; **crossover** mixes good solutions
* **Mutation** reaches values no parent has; **elitism** copies the best unchanged
* Real-coded GA: an individual is simply the vector $\theta \in \mathbb{R}^d$

## GA Operators

:::: {.columns}
::: {.column width="50%"}
**Tournament ($k$):** pick $k$ at random, keep the best

* $k=1$: random, no pressure
* $k=2$--$3$: mild (default)
* large $k$: fast, loses diversity

**Gaussian mutation:** each gene, with probability $p_m$, gets $+\,\mathcal{N}(0, \sigma_j^2)$ with $\sigma_j = 0.1\,(u_j - \ell_j)$
:::
::: {.column width="50%"}
**Blend crossover (BLX-$\alpha$)**

```{=latex}
\begin{tikzpicture}[x=0.9cm]
  \draw[thick] (0,0) -- (6,0);
  \fill[corange!40] (1.25,-0.15) rectangle (4.75,0.15);
  \fill[cblue] (2,0) circle (3pt) node[above=3pt, font=\scriptsize] {$p_1$};
  \fill[cblue] (4,0) circle (3pt) node[above=3pt, font=\scriptsize] {$p_2$};
  \draw[<->] (2,-0.4) -- node[below, font=\scriptsize] {$d$} (4,-0.4);
  \draw[<->] (1.25,-0.9) -- node[below, font=\scriptsize] {$\alpha d$} (2,-0.9);
  \draw[<->] (4,-0.9) -- node[below, font=\scriptsize] {$\alpha d$} (4.75,-0.9);
\end{tikzpicture}
```

$$c_j \sim \mathcal{U}\big(\ell_j - \alpha d_j,\; u_j + \alpha d_j\big)$$

$\alpha = 0.5$: children may land **outside** the parents, which keeps the population from shrinking too fast
:::
::::

## Differential Evolution

```{=latex}
\begin{center}
\begin{tikzpicture}[scale=0.9]
  \foreach \r in {0.5,1.1,1.7} { \draw[cgray!60] (0,0) ellipse ({\r*1.6} and \r); }
  \fill[cgray] (2.2,1.2) circle (2.5pt) node[above, font=\scriptsize] {$x_{r_1}$};
  \fill[cgray] (-2.3,-0.2) circle (2.5pt) node[left, font=\scriptsize] {$x_{r_2}$};
  \fill[cred] (0.5,0.2) circle (3pt) node[below, font=\scriptsize, cred] {$x_{best}$};
  \draw[->, thick, corange] (-2.3,-0.2) -- (2.2,1.2) node[midway, above, font=\scriptsize] {$x_{r_1} - x_{r_2}$};
  \draw[->, very thick, cblue] (0.5,0.2) -- (2.3,0.76) node[right, font=\scriptsize, cblue] {$v$};
  \fill[cblue] (2.3,0.76) circle (2.5pt);
\end{tikzpicture}
\end{center}
```

1. **Mutant:** $v = x_{best} + F\,(x_{r_1} - x_{r_2})$ (variant `best/1`; `rand/1` uses $x_{r_3}$ as the base)
2. **Crossover:** each gene of the trial comes from $v$ with probability $CR$, else from the parent
3. **Selection:** the trial replaces its parent only if it is **better** (greedy)

* The step size comes from the **population's own spread**: large while scattered, small once converged

## DE Variants and Adaptive DE

| Variant | Mutant | Behaviour |
|:--|:--|:--|
| `rand/1` | $x_{r_3} + F(x_{r_1} - x_{r_2})$ | diverse |
| `best/1` | $x_{best} + F(x_{r_1} - x_{r_2})$ | greedy, can stall |
| `c-to-pbest/1` | $x + F(x_{pb} - x) + F(x_{r_1} - x_{r_2})$ | towards top $p\%$ |

* **JADE**, **SHADE**: learn $F$ and $CR$ from the settings that produced survivors (a memory of $h$ settings in SHADE)
* pyBlindOpt: `differential_evolution(f, bounds, variant="current-to-pbest/1/bin", policy="shade")`
* pyBlindOpt's own benchmark ($d = 10$, 25 seeds): on Rastrigin, `best/1` has median 7.96 and SHADE 0.00015

## Grey Wolf Optimization

:::: {.columns}
::: {.column width="40%"}
```{=latex}
\begin{tikzpicture}
  \fill[cred] (0,2.2) circle (4pt) node[right=3pt, font=\scriptsize] {$\alpha$ (best)};
  \fill[corange] (-0.8,1.4) circle (3.5pt) node[left=3pt, font=\scriptsize] {$\beta$};
  \fill[corange] (0.8,1.4) circle (3.5pt) node[right=3pt, font=\scriptsize] {$\delta$};
  \foreach \x in {-1.5,-0.75,0,0.75,1.5} { \fill[cgray] (\x,0.4) circle (2.5pt); }
  \node[font=\scriptsize] at (0,0) {$\omega$ wolves (the rest)};
\end{tikzpicture}
```
:::
::: {.column width="60%"}
* The three best wolves **estimate** where the prey (the optimum) is
* Every wolf moves to a random point **around** that estimate
* $a$ falls linearly from 2 to 0: $|A| > 1$ **explores** (overshoots the prey), $|A| < 1$ **exploits**
:::
::::

$$X_{t+1} = X_{prey} - A \odot \lvert C \odot X_{prey} - X_t\rvert$$

with $A \sim \mathcal{U}(-a, a)$ and $C \sim \mathcal{U}(0, 2)$ drawn per coordinate

## Enhanced Grey Wolf (EGWO)

$$X_{prey} = w_1 X_\alpha + w_2 X_\beta + w_3 X_\delta + \mathcal{N}(0, \sigma^2), \quad w_1 \ge w_2 \ge w_3$$

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, height=3cm, width=0.5\textwidth, xmin=0, xmax=100, ymin=0, ymax=2.2, xlabel={epoch (\% of the run)}, ylabel={$a$}]
  \addplot[cblue, domain=0:100] {2*(1-x/100)};
  \node[font=\scriptsize, anchor=west] at (axis cs:5,0.5) {explore $\rightarrow$ exploit};
\end{axis}
\end{tikzpicture}
\end{center}
```

* The prey is a **weighted** mix of the leaders (the alpha counts most), plus an estimation error
* $\sigma$ is proportional to how much the three leaders **disagree**: wide while they are scattered, zero once they agree
* Reference: Luo (2019), *Applied Soft Computing* 77

## pyBlindOpt in Practice

```python
import numpy as np, pyBlindOpt as pbo
from pyBlindOpt import init, utils

bounds = np.array([[-5.0, 5.0]] * 10)              # 10 dimensions
rng = np.random.default_rng(42)
pop = init.oblesa(f, bounds, n_pop=20, seed=rng)   # initialization
best_x, best_f = pbo.differential_evolution(
    f, bounds, population=pop, n_iter=100, seed=rng,
    variant="current-to-pbest/1/bin", policy="shade")
```

* `f` takes a point or a population: write it with `x[..., i]`
* Same signature for `genetic_algorithm`, `enhanced_grey_wolf_optimization`, `hill_climbing`, ...
* Class interface (`pbo.DifferentialEvolution(...)`) + callbacks to record every population

## Live Demo B3: Populations at Work

* Notebook `02_blind_optimization.ipynb`, section **B3**
* GA, DE and EGWO on a shifted **Sphere** and a shifted **Rastrigin**: snapshots at epochs 0, 3, 10 and 30
* Watch how each population **contracts**: GA by recombination, DE by shrinking difference vectors, EGWO by shrinking $a$
* **pyOptViewer:** the same algorithms animated live, with the surface each one has explored

```bash
git clone https://github.com/mariolpantunes/pyOptViewer
cd pyOptViewer && python -m venv venv && venv/bin/pip install .
venv/bin/python -m optviewer    # http://127.0.0.1:8000
```

# Comparing Stochastic Optimizers

## Rules for a Fair Comparison

```{=latex}
\begin{center}
\begin{tikzpicture}[every node/.style={fillbox=cblue, text width=2.3cm, minimum height=1.25cm}]
  \node at (0,0) {\textbf{same budget}\\ in evaluations, not epochs};
  \node at (2.8,0) {\textbf{many seeds}\\ one run is an anecdote};
  \node at (5.6,0) {\textbf{spread}\\ medians, quartiles, box plots};
  \node at (8.4,0) {\textbf{shifted}\\ problems, not centred};
\end{tikzpicture}
\end{center}
```

* The same statistics as for models in Class 01: a stochastic optimizer is an experiment
* Under **noise**, report the *true* $f$ of the returned point: the best *observed* value is optimistically biased (a lucky evaluation wins)
* Tune every contender with the same effort, or none of them

## Seven Optimizers, Four Problems

| Median true $f$ | Sphere | Rastrigin | Ackley | Noisy Sphere |
|:--|--:|--:|--:|--:|
| Random | 18.05 | 85.65 | 6.20 | 18.05 |
| Hill climb. | 10.31 | 52.87 | 6.36 | 15.43 |
| Sim. anneal. | 10.01 | 51.56 | 6.56 | 15.00 |
| GA | 0.001 | 9.37 | 0.037 | **0.32** |
| DE (`best/1`) | 0.015 | 18.07 | 1.65 | 0.98 |
| DE (SHADE) | **3e-5** | 9.69 | **0.009** | 0.36 |
| EGWO | 0.010 | 13.78 | 0.103 | 0.63 |

* 15 seeds, shifted functions, $d = 10$, 20 individuals $\times$ 100 epochs ($\approx$ 2000 evaluations); lower is better
* In 10D, single-point methods are far behind every population method
* GA wins on Rastrigin by a small margin: **no algorithm wins everywhere**

## No Free Lunch, Again

```{=latex}
\begin{center}
\begin{tikzpicture}
\begin{axis}[faa, ybar, height=4.2cm, width=0.85\textwidth, ymode=log, log origin=infty, ymin=1e-5, ymax=100, bar width=7pt, symbolic x coords={Sphere, Rastrigin, Ackley, noisy Sphere}, xtick=data, ylabel={median true $f$}, legend style={at={(0.5,1.02)}, anchor=south, legend columns=4}, enlarge x limits=0.15]
  \addplot[fill=cgray!50, draw=cgray] coordinates {(Sphere,10.01) (Rastrigin,51.56) (Ackley,6.56) (noisy Sphere,15.00)}; \addlegendentry{SA}
  \addplot[fill=cgreen!60, draw=cgreen] coordinates {(Sphere,0.001) (Rastrigin,9.37) (Ackley,0.037) (noisy Sphere,0.32)}; \addlegendentry{GA}
  \addplot[fill=cblue!60, draw=cblue] coordinates {(Sphere,3.3e-5) (Rastrigin,9.69) (Ackley,0.009) (noisy Sphere,0.36)}; \addlegendentry{SHADE}
  \addplot[fill=corange!70, draw=corange] coordinates {(Sphere,0.010) (Rastrigin,13.78) (Ackley,0.103) (noisy Sphere,0.63)}; \addlegendentry{EGWO}
\end{axis}
\end{tikzpicture}
\end{center}
```

* The ranking changes with the landscape (No Free Lunch theorem, Class 01)
* Greedy methods (DE `best/1`) are excellent on some seeds and stuck on others: look at the **spread**
* On the noisy Sphere, GA and SHADE are close: noise blurs the differences

## Best of Both Worlds: Hybrids

| seed | DE (48 evaluations) $x$ | $f$ | + 50 GD steps $x$ | $f$ |
|:-:|--:|--:|--:|--:|
| 0 | $+1.530$ | $-0.795$ | $+1.557$ | $-0.811$ |
| 1 | $-0.647$ | $-1.070$ | $-0.670$ | **$-1.081$** |
| 2 | $-0.629$ | $-1.049$ | $-0.670$ | **$-1.081$** |
| 3 | $-0.669$ | $-1.081$ | $-0.670$ | **$-1.081$** |
| 4 | $+1.561$ | $-0.810$ | $+1.557$ | $-0.811$ |

* $f_{rugged}$: the **blind** stage chooses the basin, the **gradient** stage reaches its bottom cheaply
* Seeds 0 and 4 chose the second-best basin: 48 evaluations are not enough to be sure
* Blind global search + local gradient refinement = **memetic** algorithm

## Live Demo B4--B5: Comparing and Combining

* Notebook `02_blind_optimization.ipynb`, sections **B4** and **B5**
* 420 runs: 7 optimizers $\times$ 4 problems $\times$ 15 seeds, as a table (polars) and box plots
* The noisy Sphere: why we evaluate the returned point on the *true* function
* DE followed by gradient descent with `jax.grad`

# Summary

## Choosing an Optimizer

| Situation | First choice |
|:--|:--|
| Differentiable loss, many parameters (ML training) | Adam or SGD + momentum (via JAX / Keras) |
| Differentiable, few parameters, smooth | Newton or L-BFGS |
| Convex | any gradient method: local = global |
| No gradient, few dimensions, cheap $f$ | DE (SHADE), GA, EGWO |
| Multimodal, gradient available | population search + gradient polish (hybrid) |
| Expensive $f$ (e.g. hyperparameters) | few, well-placed evaluations: good initialization |

## Gradient-Based versus Blind

| | Gradient-based | Blind (population) |
|:--|:--|:--|
| Needs | $\nabla f$ (JAX gives it for free) | only $f(x)$ |
| Scales to | millions of parameters | tens to hundreds |
| Finds | a local minimum, fast | a good basin, slowly |
| Cost per step | one gradient ($\approx$ a few $f$) | $N$ evaluations |
| Hyperparameters | learning rate ($\beta$'s) | $N$, operators, rates |
| Weak on | plateaus, steps, noise, many basins | precision, high dimension |

## Key Takeaways

* **Training is optimization:** $\theta^\star = \arg\min \mathcal{L}(\theta)$, and the loss is the code length of Class 01
* **Gradients** are the most efficient signal when they exist; the learning rate and the curvature decide the speed
* **Newton** uses curvature: few steps, expensive, fooled by maxima and saddles; **Adam** is a cheap diagonal imitation
* **JAX** derives gradients of any model and loss, even through a simulation
* **Blind** methods need only values: they handle black boxes, steps, noise and many basins, at the cost of evaluations
* **Initialization** matters most with small budgets; always test on shifted problems
* **Compare** stochastic optimizers like models: equal budgets, many seeds, spreads
