# Exp 3.1: Hierarchical Clustering

This experiment is to use the hierarchical clustering algorithm to cluster the samples in various datasets.
There are two steps to complete this experiment:

1. Preprocessing: sort the distance/similarity between samples from the same class.

2. Clustering: hierarchically merge the samples following the sorted distance/similarity obtained in the previous step.

## Inputs

1. Sorted kernel files generated in Exp 2: `data/{dataset}/{kernel_name}Kernel_sorted/*.npy`
2. The index of the class to be clustered: `int cls_idx`

## Outputs

1. A list of the indices of samples in the class `cls_idx`: `data/{dataset}/{kernel_name}_hier_cluster/cls_idx_{cls_idx}.pkl`
2. An np array storing the merge order of samples in the class `cls_idx`: `data/{dataset}/{kernel_name}_hier_cluster/union_tracker_cls{cls_idx}.npy`