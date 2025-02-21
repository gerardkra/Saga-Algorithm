# SAGA Algorithm: Theory, Proofs, and Experimental Validation

This repository provides a comprehensive implementation and analysis of the **SAGA algorithm**, a state-of-the-art incremental gradient method with variance reduction for optimization. The project includes:
- **Theoretical foundations** of SAGA, including detailed convergence proofs.
- **Experimental validation** on real datasets, comparing SAGA to other optimization methods (SAG, SVRG, and SGD).
- **Jupyter Notebook** (`SAGA.ipynb`) containing all experiments and visualizations.

---

## Problem Setting

We consider two optimization problems commonly used in machine learning:

### 1. **Ridge Regression**
The objective is to solve the following strongly convex optimization problem:
\[
F_{\text{ridge}}(w) = \frac{1}{2n} \|Xw - y\|^2 + \frac{\lambda}{2} \|w\|^2,
\]
where:
- \(X \in \mathbb{R}^{n \times d}\) is the feature matrix,
- \(y \in \mathbb{R}^n\) is the target vector,
- \(w \in \mathbb{R}^d\) is the weight vector to be optimized,
- \(\lambda > 0\) is the regularization parameter (set to \(1 \times 10^{-5}\) in our experiments).

The \(L_2\) regularization term ensures strong convexity, which guarantees a unique global minimum.

---

### 2. **Lasso**
The objective is to solve the following convex (but not strongly convex) optimization problem:
\[
F_{\text{lasso}}(w) = \frac{1}{2n} \|Xw - y\|^2 + \lambda \|w\|_1,
\]
where:
- \(\|w\|_1\) is the \(L_1\) norm of the weight vector,
- \(\lambda > 0\) is the regularization parameter (set to \(1.0\) in our experiments).

The \(L_1\) regularization term promotes sparsity in the solution, making Lasso particularly useful for feature selection.

---

## Optimization Methods

We compare the following optimization algorithms for Ridge Regression and Lasso:

### 1. **SAGA (Stochastic Average Gradient Accelerated)**
- **Key Idea**: SAGA uses a memory structure to store past gradients, enabling efficient variance reduction and faster convergence.
- **Advantages**:
  - Unbiased gradient estimator.
  - Supports composite objectives (e.g., non-smooth regularizers like \(L_1\)).
  - Linear convergence for strongly convex objectives and \(\mathcal{O}(1/t)\) convergence for convex objectives.
- **Memory Cost**: Requires storing a table of gradients for all \(n\) samples.

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
- **Objective**: Minimize \(F_{\text{ridge}}(w)\).
- **Methods**: Standard gradient methods (SGD, SAG, SVRG, SAGA).
- **Results**: SAGA and SVRG achieve linear convergence, outperforming SGD and SAG in terms of convergence speed.

### 2. **Lasso**
- **Objective**: Minimize \(F_{\text{lasso}}(w)\).
- **Methods**: Proximal variants (Prox-SGD, Prox-SAG, Prox-SAGA, Prox-SVRG) to handle the \(L_1\) regularization term.
- **Results**: SAGA and SVRG achieve faster convergence compared to SGD and SAG, demonstrating the effectiveness of variance reduction for non-smooth objectives.

---

## Repository Structure

- **Notebook**:
  - `SAGA.ipynb`: Jupyter Notebook containing all experiments, implementations, and visualizations.
- **Data**:
  - The Diabetes dataset is loaded directly from scikit-learn in the notebook.
- **Results**:
  - Convergence plots for Ridge Regression and Lasso are generated and displayed within the notebook.

---

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/saga-optimization.git
   cd saga-optimization
