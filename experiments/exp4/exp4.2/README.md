# Exp 4.2: Prune Samples with Farthest Point Clustering

This experiment is to use the farthest point clustering algorithm to cluster the samples in various datasets.
There are two steps to complete this experiment:

1. Preprocessing: sort the distance/similarity between samples from the same class.

2. Clustering: hierarchically merge the samples following the sorted distance/similarity obtained in the previous step.

## Inputs

1. Ratio of samples to be kept `float ratio`
2. The pickle file storing the farthest point clustering results: `data/{dataset}/{kernel_name}Kernel/far_cluster_{ratio}/cluster_list_{class_id}.pkl`

## Outputs

1. Numpy file storing the pruned dataset: `data/{dataset}/pruned/{kernel_name}_hier_{ratio_kept}.pkl`