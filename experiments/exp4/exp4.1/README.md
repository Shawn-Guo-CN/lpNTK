# Exp 4.1: Prune Low-level Subtrees Hierarchical Clustering

This experiment is to use the hierarchical clustering algorithm to cluster the samples in various datasets.
There are two steps to complete this experiment:

1. Preprocessing: sort the distance/similarity between samples from the same class.

2. Clustering: hierarchically merge the samples following the sorted distance/similarity obtained in the previous step.

## Inputs

1. Lists of the indices of samples: `data/{dataset}/{kernel_name}_hier_cluster/cls_idx_{cls_idx}.pkl`
2. NP arrays storing the merge order of samples: `data/{dataset}/{kernel_name}_hier_cluster/union_tracker_cls{cls_idx}.npy`
3. Ratio to be kept `float ratio_kept`

## Outputs

1. Numpy file storing the pruned dataset: `data/{dataset}/pruned/{kernel_name}_hier_{ratio_kept}.pkl`