import os, sys
import toml
import argparse
from munch import Munch, munchify

from tqdm import tqdm
import numpy as np
from numpy.typing import ArrayLike
from torchvision import datasets, transforms


PROJ_DIR = os.path.expanduser("~/GitWS/Transmisstion-Phase")
DATA_DIR = os.path.join(PROJ_DIR, "data")
SRC_DIR = os.path.join(PROJ_DIR, "src")
LOGS_DIR = os.path.join(PROJ_DIR, "logs", "exp2.1")
SCRIPTS_DIR = os.path.join(PROJ_DIR, "scripts")
CHECKPOINTS_DIR = os.path.join(PROJ_DIR, "checkpoints")
RESULTS_DIR = os.path.join(PROJ_DIR, "results")


sys.path.append(PROJ_DIR)
from utils import create_dir_for_file


def process_sample_npz(base_idx:int, dataset:str, dataloader) -> None:
    npz_fname = os.path.join(DATA_DIR, 
                             dataset, 
                             "GradMatrix", 
                             f"{base_idx}.npz"
                            )
    grad_matrix_list = np.load(npz_fname)['data'].astype(np.float32)

    assert grad_matrix_list.shape[1] == grad_matrix_list.shape[2]
    dim = grad_matrix_list.shape[1]
    src_label = dataloader[base_idx][1]
    src_mask = -1. * np.ones(dim)
    src_mask[src_label] = 1.

    sum_list = []
    mask_sum_list = []
    for shift_idx in range(grad_matrix_list.shape[0]):
        sum_list.append(np.sum(grad_matrix_list[shift_idx], dtype=np.float64))

        tgt_label = dataloader[base_idx + shift_idx][1]
        tgt_mask = -1. * np.ones(dim)
        tgt_mask[tgt_label] = 1.

        tmp_mask = np.outer(src_mask, tgt_mask)
        tmp_grad_matrix = np.multiply(grad_matrix_list[shift_idx], tmp_mask)
        mask_sum_list.append(np.sum(tmp_grad_matrix, dtype=np.float64))
    
    out_path = os.path.join(
                   DATA_DIR, dataset, 'SumKernel', 'raw', f"{base_idx}.npz"
               )
    create_dir_for_file(out_path)
    np.savez_compressed(out_path, data=np.asarray(sum_list))

    out_path = os.path.join(
                   DATA_DIR, dataset, 'MaskSumKernel', 'raw', f"{base_idx}.npz"
               )
    create_dir_for_file(out_path)
    np.savez_compressed(out_path, data=np.asarray(mask_sum_list))


def get_kernel(args:argparse, dataset:str) -> None:
    dataloader = eval('datasets.'+dataset)(DATA_DIR, train=True, download=True, 
                      transform=transforms.ToTensor()
                     )

    for base_idx in tqdm(range(args.start_idx, args.end_idx)):
        process_sample_npz(base_idx, dataset, dataloader)


def run(args:argparse) -> None:
    if args.mnist:
        get_kernel(args, "MNIST")

    if args.cifar10:
        get_kernel(args, "CIFAR10")

    if args.cifar100:
        get_kernel(args, "CIFAR100")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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