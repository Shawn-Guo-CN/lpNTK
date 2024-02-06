# EXP 0.1: Pretrain Models and Save Checkpoints

This experiment is to pre-train the models on different data sets, and save the checkpoints of iterations/epochs for the analysis in the following experiments.

## Inputs

1. Config file: `scripts/exp0/exp0.1/config.toml`

## Outputs

For each dataset and the corresponding model, the following files are generated:

1. checkpoint of the initial ($0$-th step) parameters of the model: `checkpoints/{dataset}/{model_name}_init.pt`;
2. checkpoint of the parameters after $1-10$-th iterations/epochs: `checkpoints//{dataset}/{iter_or_epoch}/{model_name}_{time_step}.pt`;
3. checkpoint of the parameters at the final ($T$-th step) iteration/epoch: `checkpoints/{dataset}/{model_name}_final.pt`;

