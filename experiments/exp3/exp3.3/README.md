# EXP 3.3: Average Distance to Samples in the Same Class

This experiment is to calculate the average distance between samples in the same class, and sort them in decent order.
This is to use as a heuristic to prune the samples that are highly similar to other samples in the same class.

The steps to complete this experiment are:

1. sort the distance/similarity between samples from the same class,
2. remove the samples whose most similar sample is not itself.


## Inputs

1. Sorted kernel files generated in Exp 2: `data/{dataset}/{kernel_name}Kernel/sorted/*.npy`
2. The index of the class to be clustered: `int cls_idx`

## Outputs

1. A list of the indices of samples in the same class `cls_idx`: `data/{dataset}/{kernel_name}_avg_dis/cls_idx_{cls_idx}.pkl`