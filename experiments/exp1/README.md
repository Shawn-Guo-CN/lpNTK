# EXP 1: Analyse the Gradient Matrices

This experiment is to analyse the gradient matrices of the models on different data sets, and the aim is to figure out which form to use to represent the gradient matrices.
Possible options:

1. F-norm: if the on-diagonal elements are significantly larger than the off-diagonal elements. However, the `exp1.1` shows that the sum of on-digonal elements is **not** significantly larger than the sum of off-diagonal elements.
2. pNTK proposed in [this paper](https://arxiv.org/pdf/2206.12543.pdf).
3. Labelled element: suppose one sample is labelled as `i`, the other is labelled as `j`, then the element `(i,j)` in the gradient matrix if it is also the largest element in the matrix. However, the `exp1.1` shows that the labelled element is **not** the maximal in most cases (>99.9%).

Each option is tested in the corresponding sub-directory.

## Prerequisites

Exp 0