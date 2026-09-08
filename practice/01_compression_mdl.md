---
title: Lab 01 - Learning as Compression & MDL
---

# Laboratory Guide 01: Information, Compression, and Learning

## 1. Objectives
* Understand the mathematical connection between learning, information, and data compression.
* Compute empirical Shannon Entropy on discrete distributions.
* Implement polynomial curve fitting and analyze the trade-off between model complexity and residual fit.
* Apply the Minimum Description Length (MDL) principle / Bayesian Information Criterion (BIC) to select optimal models without a validation set.

## 2. Theoretical Background

### 2.1 Shannon Entropy
Given a discrete random variable $X$ with alphabet $\mathcal{X}$ and probability mass function $p(x)$:
$$H(X) = -\sum_{x \in \mathcal{X}} p(x) \log_2 p(x)$$
Entropy defines the fundamental limit of lossless compression.

### 2.2 Minimum Description Length (Two-Part Code)
For a model with $k$ parameters evaluated on $n$ samples with mean squared error $\text{MSE}$:
$$\text{MDL} \approx \frac{k}{2} \log_2(n) + \frac{n}{2} \log_2(\text{MSE})$$
Notice that as $k$ increases, the first term (model cost) grows logarithmically with $n$, while the second term (data cost given the model) decreases.

## 3. Practical Exercises

### Exercise 1: Computing Entropy of Text and Noise
1. Open notebook `notebooks/01-compression/01_entropy.ipynb`.
2. Generate three sequences:
   - Structured repetitive text (e.g. `ABABABAB...`).
   - Natural English text sample.
   - Uniformly distributed random bytes.
3. Compute the empirical entropy of each sequence and compare against the file size after gzip/zlib compression.

### Exercise 2: Polynomial Regression and Overfitting
1. Generate synthetic noisy data:
   $$y = \sin(2\pi x) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, 0.15^2)$$
   with $n = 30$ samples in $[0, 1]$.
2. Fit polynomials of degree $d \in [1, 15]$.
3. Observe how higher-degree polynomials achieve zero training MSE but wildly oscillate between data points.

### Exercise 3: Automated Model Selection via MDL / BIC
1. Calculate the MDL score for each polynomial degree $d \in [1, 15]$.
2. Identify the minimum of the MDL curve.
3. Verify that the model selected by MDL corresponds to the degree that maximizes test generalization.
