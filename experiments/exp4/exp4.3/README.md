# Exp 4.3: Prune Samples with Average lpNTK

This experiment is to use the average lpNTK values of one sample to others in the same class to prune samples in various datasets.


## Inputs

1. Ratio of samples to be kept `float ratio`
2. The `.npz` file storing the lpNTK values between samples in classes: `data/{dataset}/{kernel_name}Kernel/class_merged/{class_id}.npz`

## Outputs

1. Numpy file storing the pruned dataset: `data/{dataset}/pruned/{kernel_name}_avglp_{ratio_kept}.npy`