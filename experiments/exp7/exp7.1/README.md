# EXP 7.1: Learning Difficulty on Various Dataset Sizes

This experiment is to verify that the learning difficulty of samples on the universal dataset would become less correlated with smaller datasets.
To do so, we will:

1. measure the learning difficulty of samples on datasets of which each class/category contains [4096, 1024, 256, 64, 8, 2, 1] samples;
2. given a pair of sizes, calculate the correlation between the learning difficulty of samples on datasets of those sizes.

## Input

1. config file that describes the benchmarks (e.g. MNIST or CIFAR10) and the parameters of the experiment (e.g. the number of samples in each class/category): `scripts/exp7/exp7.1/config.toml`

## Output

1. a csv file that contains the correlation between the learning difficulty of samples on datasets of different sizes: `results/exp7/exp7.1/ld_corr_{benchmark}.csv`
2. a plot that shows the correlation between the learning difficulty of samples on datasets of different sizes: `results/exp7/exp7.1/ld_corr_{benchmark}.png`
