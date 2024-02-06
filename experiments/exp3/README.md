# EXP 3: Clustering

This experiment is to cluster the samples from various data sets, given the kernel we got from Exp 2. 
The following clustering algorithms are implemented:

1. Hierarchical clustering in `exp3.1`
2. Farthest point clustering in `exp3.2`

Beyond the clustering algorithms, we also implement the following heuristics:

3. Average distance to other samples in the same class in `exp3.3` 
(note that the "class" here is the label of the sample)

All of the above algorithms, however, depend on the sorted distance/similarity between all pairs of data. 
Therefore, we need to the experiment below prior to running the above experiments:

- Sorting in `exp3.0`: sort the distance/similarity between a specific sample and all the others (not necessarily from the same class!).

## Prerequisites

1. Exp 2