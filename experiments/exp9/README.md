The purpose of this experiment is to do empirical comparison between our data pruning methods v.s. the existing works.
We compare our methods with the following methods:

1. Random Pruning: randomly remove part of the samples in the original training set.
2. Average NTK Norm: the average Frobenius norm of NTK($x_s$, $x_i$) where $x_s$ is the sample to remove (or not), and $x_i$ is a sample in the same class to $x_s$.
3. Forget Score: 
```
@inproceedings{
toneva2018forget,
title={An Empirical Study of Example Forgetting during Deep Neural Network Learning},
author={Mariya Toneva and Alessandro Sordoni and Remi Tachet des Combes and Adam Trischler and Yoshua Bengio and Geoffrey J. Gordon},
booktitle={International Conference on Learning Representations},
year={2019},
url={https://openreview.net/forum?id=BJlxm30cKm},
}
```
4. GraNd and EL2N:
```
@inproceedings{Mansheej2021diet,
 author = {Paul, Mansheej and Ganguli, Surya and Dziugaite, Gintare Karolina},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {M. Ranzato and A. Beygelzimer and Y. Dauphin and P.S. Liang and J. Wortman Vaughan},
 pages = {20596--20607},
 publisher = {Curran Associates, Inc.},
 title = {Deep Learning on a Data Diet: Finding Important Examples Early in Training},
 url = {https://proceedings.neurips.cc/paper/2021/file/ac56f8fe9eea3e4a365f29f0f1957c55-Paper.pdf},
 volume = {34},
 year = {2021}
}
```