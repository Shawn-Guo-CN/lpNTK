# EXP 3.2: Farthest Point Clustering

This experiment is to use the farthest point clustering algorithm to cluster the samples in various datasets.
There are two steps to complete this experiment:

1. Preprocessing: 
    1. sort the distance/similarity between samples from the same class,
    2. remove the samples whose most similar sample is not itself.

2. Clustering: farthest point clustering algorithm.

## Inputs

1. Sorted kernel files generated in Exp 2: `data/{dataset}/{kernel_name}Kernel/sorted/*.npy`
2. The index of the class to be clustered: `int cls_idx`

## Outputs

1. A list of the indices of samples in the class `cls_idx`: `data/{dataset}/{kernel_name}_far_cluster/cls_idx_{cls_idx}.pkl`
2. A list of clusters along with the samples in that cluster: `data/{dataset}/{kernel_name}_far_cluster/clusters_{cls_idx}.pkl`