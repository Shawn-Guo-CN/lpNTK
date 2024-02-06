# EXP 4: Pruning

This experiment is to prune the samples from various data sets, given the clustering results we got from Exp 3.
The following pruning algorithms are implemented:

1. Uniform random baseline in `exp4.1`
2. Bottom subtree pruning in `exp4.1` (given a hierarchical clustering tree)
3. Farthest cluster pruning in `exp4.2` (given a farthest point clustering list)
4. Average similarity pruning in `exp4.3` (given outputs from `exp3.3`)

In the meantime, a Gaussian-based demo is implemented in `exp4.0`, in order to show that I.I.D is not a *necessary* condition for good generalisation.

## Prerequisites

1. Exp 0
2. Exp 2
3. Exp 3