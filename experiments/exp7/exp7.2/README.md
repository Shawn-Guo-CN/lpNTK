# EXP 7.2: Control Learning Difficulty

This experiment is to show that the learning difficulty of samples can be manipulated by controlling the interaction between samples.
To do so, given a specific sample, we will:

1. add more interchangeable samples to it, and show that the sample becomes easier to learn;
2. add more contradictory samples to it, and show that the sample becomes harder to learn;
3. (not sure) add more unrelated samples to it, and show that the learning difficulty of the sample remains unchanged. (Note that it is quite possible that there is no such unrelated samples in the benchmarks.)

## Input

1. config file that describes the benchmarks (e.g. MNIST or CIFAR10) and the parameters of the experiment (e.g. the number of samples in each class/category): `scripts/exp7/exp7.2/config.toml`
2. results from EXP2 and EXP3 to determine the similarity between samples

## Output

1. curves that show the learning difficulty of a specific samples with more interchangeable/unrelated/contradictory samples: `results/exp7/exp7.2/{benchmark}_{sample_id}.pdf`