# SAGA Algorithm: An approch For Experimental Validation

This repository provides a comprehensive implementation and analysis of the **SAGA algorithm**, a state-of-the-art incremental gradient method with variance reduction for optimization introduced by $\textit{Defazio, Bach et Lacoste\-Julien}$.

---

## Problem Setting

We consider two optimization problems commonly used in machine learning: 

### 1. **Ridge Regression**

The objective is to solve the following strongly convex optimization problem:
$$F_{\text{ridge}}(w) = \frac{1}{2n} \parallel Xw - y\parallel^2 + \frac{\lambda}{2} \parallel w\parallel^2,$$
where:
- $X \in \mathbb{R}^{n \times d}$ is the feature matrix,
- $y \in \mathbb{R}^n$ is the target vector,
- $w \in \mathbb{R}^d$ is the weight vector to be optimized,
- $\lambda > 0$ is the regularization parameter (set to $1 \times 10^{-5}$ in our experiments).

The $L_2$ regularization term ensures strong convexity, which guarantees a unique global minimum.

---

### 2. **Lasso**
The objective is to solve the following convex (but not strongly convex) optimization problem:
$$F_{\text{lasso}}(w) = \frac{1}{2n} \parallel Xw - y\parallel^2 + \lambda \parallel w\parallel_1,$$

where:
- $\parallel w\parallel_1$ is the $L_1$ norm of the weight vector,
- $\lambda > 0$ is the regularization parameter (set to $1.0$ in our experiments).

The $L_1$ regularization term promotes sparsity in the solution, making Lasso particularly useful for feature selection.

---

## Optimization Methods

We compare the following optimization algorithms for Ridge Regression and Lasso:

### 1. **SAGA (Stochastic Average Gradient Accelerated)**
- **Key Idea**: SAGA uses a memory structure to store past gradients, enabling efficient variance reduction and faster convergence.
- **Advantages**:
  - Unbiased gradient estimator.
  - Supports composite objectives (e.g., non-smooth regularizers like $L_1$).
  - Linear convergence for strongly convex objectives and $\mathcal{O}(1/t)$ convergence for convex objectives.
- **Memory Cost**: Requires storing a table of gradients for all $n$ samples.

### 2. **SAG (Stochastic Average Gradient)**
- **Key Idea**: SAG also uses a memory structure but employs a biased gradient estimator.
- **Advantages**:
  - Linear convergence for strongly convex objectives.
- **Limitations**:
  - Slower initial convergence due to biased gradient updates.
  - No support for non-smooth regularizers without modifications.

### 3. **SVRG (Stochastic Variance Reduced Gradient)**
- **Key Idea**: SVRG periodically computes the full gradient to reduce the variance of stochastic gradient estimates.
- **Advantages**:
  - Linear convergence for strongly convex objectives.
  - Memory-efficient (does not require storing a table of gradients).
- **Limitations**:
  - Requires periodic full gradient computations, which can be expensive for large datasets.

### 4. **SGD (Stochastic Gradient Descent)**
- **Key Idea**: SGD uses a single stochastic gradient per iteration, making it computationally efficient but prone to high variance.
- **Advantages**:
  - Low per-iteration cost.
- **Limitations**:
  - Slow convergence due to high variance.
  - Requires careful tuning of the learning rate.

---

## Experiments

We conduct numerical experiments on the **Diabetes dataset** from scikit-learn, which contains 442 samples and 10 features. The experiments are designed to validate the theoretical convergence rates of SAGA and compare its performance to SAG, SVRG, and SGD.

### 1. **Ridge Regression**
![Ridge Regression Convergence](RidgeTrain.jpg)

### 2. **Lasso**
![Lasso Convergence](LassoTrain.jpg)


Please refer to the original paper, `defazio_2014.pdf`, available in this repository, for a complete and rigorous analysis. Each step of the theoretical proofs and algorithmic details is meticulously documented in the paper.

## References
- Defazio, A., Bach, F., & Lacoste-Julien, S. (2014). **SAGA: A Fast Incremental Gradient Method with Support for Non-Strongly Convex Composite Objectives**. *Advances in Neural Information Processing Systems (NeurIPS)*.
- Scikit-learn: Machine Learning in Python. [https://scikit-learn.org](https://scikit-learn.org)
