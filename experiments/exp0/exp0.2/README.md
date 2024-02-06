# EXP 0.2: Get the GradMatrix with Pre-trained Checkpoints

This experiment is to calculate the gradient matrix between all pairs of data.
This matrix serve as the core for the following analysis, of which the essence
is the outer product between the first-order derivatives of the output w.r.t.
data (since the parameters are assumed to be fixed).

## Prerequisite

Exp 0.1

## Inputs

1. Config file: `scripts/exp0/exp0.2/config.toml`
2. Checkpoint file specified in the config, e.g. `scripts/exp0/exp0.2/checkpoints/MNIST/LeNet_best.pt`

## Outputs

For a dataset and a corresponding model, the following files are generated:

1. a series of files that contain the gradient matrix between all pairs of data:
`data/{dataset}/GradMatrix/{i}.npy`.
> The first element of the numpy array is the gradient matrix between $x_i$ and $x_i$, the second element is the gradient matrix between $x_i$ and $x_{i+1}$, and so on.
