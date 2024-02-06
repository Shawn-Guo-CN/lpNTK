import os, sys
import argparse
from munch import Munch, munchify
from tqdm import tqdm

import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

PROJ_DIR = os.path.expanduser("~/GitWS/Transmisstion-Phase")
DATA_DIR = os.path.join(PROJ_DIR, "data")
SRC_DIR = os.path.join(PROJ_DIR, "src")
LOGS_DIR = os.path.join(PROJ_DIR, "logs", "exp1")
SCRIPTS_DIR = os.path.join(PROJ_DIR, "scripts")
CHECKPOINTS_DIR = os.path.join(PROJ_DIR, "checkpoints")
RESULTS_DIR = os.path.join(PROJ_DIR, "results")

sys.path.append(PROJ_DIR)
from utils import create_dir_for_file


def plot_hist(data_list, var_name, args, dataset) -> None:
    plt.rcParams["figure.figsize"] = (20,10)
    plt.hist(np.asarray(data_list), density=False, bins=30)  # density=False would     make counts
    plt.ylabel('Frequency')
    plt.xlabel(var_name)
    plt.grid()

    out_path = os.path.join(
                   RESULTS_DIR, 'exp1.1', dataset,
                   f"{args.start_idx}_{args.end_idx}_{var_name}.pdf"
               )
    create_dir_for_file(out_path)
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.clf()


def analyse_matrices(args, dataset) -> None:
    dataloader = eval('datasets.'+dataset)(DATA_DIR, train=True, download=True, 
                      transform=transforms.ToTensor()
                     )

    on_off_diag_ratio_list = []
    max_on_diag_ratio_list = []
    max_over_sum_list = []
    max_over_second_list = []
    i_is_max_list = []
    j_is_max_list = []
    ij_is_max_list = []

    for base_idx in tqdm(range(args.start_idx, args.end_idx)):
        npz_fname = os.path.join(DATA_DIR, 
                                 dataset, 
                                 "GradMatrix", 
                                 f"{base_idx}.npz"
                                )
        grad_matrix_list = np.load(npz_fname)['data'].astype(np.float64)

        assert grad_matrix_list.shape[1] == grad_matrix_list.shape[2]
        dim = grad_matrix_list.shape[1]
        dia = np.diag_indices(dim)
        src_label = dataloader[base_idx][1]

        for shift_idx in range(grad_matrix_list.shape[0]):
            tgt_label = dataloader[base_idx + shift_idx][1]
            grad_matrix = grad_matrix_list[shift_idx]

            dia_sum = np.sum(np.abs(grad_matrix[dia]))
            off_sum = np.sum(np.abs(grad_matrix)) - dia_sum
            on_off_diag_ratio_list.append(dia_sum / off_sum)

            flat_matrix = grad_matrix.flatten()
            flat_matrix.sort()
            second_max = flat_matrix[-2]
            max_over_second_list.append(
                    min(grad_matrix.max() / second_max, 100.)
                )
            max_over_sum_list.append(grad_matrix.max() / grad_matrix.sum())

            i = grad_matrix.argmax() // dim
            j = grad_matrix.argmax() %  dim

            if i == j:
                max_on_diag_ratio_list.append(1)
            else:
                max_on_diag_ratio_list.append(0)
            
            if i == src_label:
                i_is_max_list.append(1)
            else:
                i_is_max_list.append(0)

            if j == tgt_label:
                j_is_max_list.append(1)
            else:
                j_is_max_list.append(0)

            if i == src_label and j == tgt_label:
                ij_is_max_list.append(1)
            else:
                ij_is_max_list.append(0)
    
    plot_hist(on_off_diag_ratio_list, 'sum_on-off_diagonal', args, dataset)
    plot_hist(max_on_diag_ratio_list, 'max_on_diagonal', args, dataset)
    plot_hist(max_over_sum_list, 'max_over_sum', args, dataset)
    plot_hist(max_over_second_list, 'max_over_second', args, dataset)
    plot_hist(i_is_max_list, 'label_src_is_max', args, dataset)
    plot_hist(j_is_max_list, 'label_tgt_is_max', args, dataset)
    plot_hist(ij_is_max_list, 'label_is_max', args, dataset)


def run(args) -> None:
    if args.mnist:
        analyse_matrices(args, 'MNIST')
    
    if args.cifar10:
        analyse_matrices(args, 'CIFAR10')

    if args.cifar100:
        analyse_matrices(args, 'CIFAR100')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, 
                        default="On-diagonal/Off-diagonal"
                       )
    parser.add_argument("--mnist", action="store_true", default=False, 
        help="run mnist experiment")
    parser.add_argument("--cifar10", action="store_true", default=False, 
        help="run cifar10 experiment")
    parser.add_argument("--cifar100", action="store_true", default=False, 
        help="run cifar100 experiment")
    parser.add_argument("--start-idx", type=int, default=0,
        help="start index of the GradMatrix calculation")
    parser.add_argument("--end-idx", type=int, default=None,
        help="end index of the GradMatrix calculation")
    parser.add_argument("--notes",   default=None)
    args, unknown = parser.parse_known_args()

    run(args)