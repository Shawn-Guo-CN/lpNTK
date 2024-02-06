# lpNTK: Sample Relationship from Learning Dynamics Matters for Generalisation

This is the official git repo for the paper [Sample Relationship from Learning Dynamics Matters for Generalisation](https://arxiv.org/abs/2401.08808).

## Installation

The configuration of the running environment of this project is given in the `environment.yml` file. You can create a new conda environment with the following command:

```bash
conda env create -f environment.yml
```

## Running the experiments

All experiments are organised in the `experiments` folder. Each experiment is a separate python file, and a README file can be found to explain the purpose of the experiment and how to run it.

I hereby take experiment 0.1, pretrain models and save checkpoints, as an example.
The README file of this experiment is `experiments/exp0/exp0.1/README.md`.
The configuration file of this experiment is `experiments/exp0/exp0.1/config.toml`.
To run it on the CIFAR10 dataset, you can use the following command:

```bash
python ./scripts/exp0/exp0.1/run.py --cifar10
```

## Citation

If you find this work useful, please consider citing it:

```bibtex
@inproceedings{guo2024lpntk,
title={Sample Relationship from Learning Dynamics Matters for Generalisation},
author={Guo, Shangmin and Ren, Yi and Albrecht, Stefano V and Smith, Kenny},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=8Ju0VmvMCW}
}
```