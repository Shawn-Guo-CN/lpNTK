import os, sys
import toml
import argparse
from munch import Munch, munchify
from typing import List

from tqdm import tqdm, trange
import numpy as np
from numpy.typing import ArrayLike
from torchvision import datasets, transforms


PROJ_DIR = os.path.expanduser("~/GitWS/Transmisstion-Phase")
DATA_DIR = os.path.join(PROJ_DIR, "data")
SRC_DIR = os.path.join(PROJ_DIR, "src")
LOGS_DIR = os.path.join(PROJ_DIR, "logs", "exp3")
SCRIPTS_DIR = os.path.join(PROJ_DIR, "scripts")
CHECKPOINTS_DIR = os.path.join(PROJ_DIR, "checkpoints")
RESULTS_DIR = os.path.join(PROJ_DIR, "results")

sys.path.append(PROJ_DIR)
from utils import create_dir_for_file


def concat_two_sorted_norms(norm1:ArrayLike, 
                            norm2:ArrayLike,
                            idx2cls:List,
                            cls:int,
                           ) -> ArrayLike:
    if norm1.shape[1] == 0: return norm2
    if norm2.shape[1] == 0: return norm1
    
    out_vec = []
    cur1 = 0
    cur2 = 0
    
    while not (cur1 == norm1.shape[1] and cur2 == norm2.shape[1]):
        if (cur1 < norm1.shape[1] and cur2 == norm2.shape[1]) \
            or (cur1 < norm1.shape[1] and norm1[0, cur1] >= norm2[0, cur2]):
            if idx2cls[int(norm1[1, cur1])] == cls \
               and idx2cls[int(norm1[2, cur1])] == cls:
                out_vec.append(norm1[:, cur1])
            cur1 += 1
            continue
        elif (cur2 < norm2.shape[1] and cur1 == norm1.shape[1]) \
            or (cur2 < norm2.shape[1] and norm2[0, cur2] > norm1[0, cur1]):
            if idx2cls[int(norm2[1, cur2])] == cls \
               and idx2cls[int(norm2[2, cur2])] == cls:
                out_vec.append(norm2[:, cur2])
            cur2 += 1
            continue
        else:
            print('Unanticipated case.')
            
    assert not len(out_vec) == 0
    return np.asarray(out_vec).transpose()


def _test_concat_function():
    vec1 = np.asarray([
        [9., 7., 5.],
        [1., 3., 4.],
        [2., 2., 2.]
    ])
    vec2 = np.asarray([
        [10., 8., 6., 4., 2.],
        [2.,  7., 8., 9., 10.],
        [2.,  2., 2., 2., 2.]
    ])
    idx2cls = {1:0, 2:0, 3:0, 4:0, 6:1, 7:1, 8:1, 9:1, 10:1}
    cls_id = 0
    
    out_vec = concat_two_sorted_norms(vec1, vec2, idx2cls, cls_id)
    expect_out = np.asarray([
        [10., 9., 7., 5.],
        [2.,  1., 3., 4.],
        [2.,  2., 2., 2.]
    ])
    assert np.allclose(out_vec, expect_out)
    print('passed first test')
    
    vec_list = [vec1, vec2]
    sorted_norm = np.array([]).reshape(3,0)
    for idx in trange(len(vec_list)):
        dis_vec = vec_list[idx]
        sorted_norm = concat_two_sorted_norms(sorted_norm, dis_vec, 
                                                  idx2cls, args.class_id
                                                 )
    assert np.allclose(out_vec, expect_out)
    print('passed second test')

def _filter_intraclass_norm(dis_vec:ArrayLike, 
                            idx2cls:List,
                            cls:int,
                           ) -> ArrayLike:
    out_vec = []
    for i in range(dis_vec.shape[1]):
        if idx2cls[int(dis_vec[2, i])] == cls:
            out_vec.append(dis_vec[:, i])
    return np.asarray(out_vec).transpose()
    
    
def get_intraclass_norm(args:argparse, dataset:set) -> None:
    trainset = eval('datasets.'+dataset)(DATA_DIR, train=True, download=True, 
                      transform=transforms.ToTensor()
                     )
    idx2cls = [x[1] for x in trainset]
    
    _range = trange(len(trainset)) if args.verbose else range(len(trainset))
    for idx in _range:
        base_cls = idx2cls[idx]
        npz_fname = os.path.join(DATA_DIR, 
                                     dataset, 
                                     f"{args.kernel}Kernel",
                                     "sorted",
                                     f"{idx}.npz"
                                    )
        dis_vec = np.load(npz_fname)['data'].astype(np.float32)
        dis_vec = _filter_intraclass_norm(dis_vec, idx2cls, base_cls)
        
        out_path = os.path.join(DATA_DIR,
                            dataset, 
                            f"{args.kernel}Kernel",
                            "intraclass_merged",
                            f"{idx}.npz"
                           )
        create_dir_for_file(out_path)
        np.savez_compressed(out_path, data=dis_vec)


def merge_class_kernel(args:argparse, dataset:str) -> None:
    trainset = eval('datasets.'+dataset)(DATA_DIR, train=True, download=True, 
                      transform=transforms.ToTensor()
                     )
    idx2cls = [x[1] for x in trainset]

    sorted_norm = np.array([]).reshape(3,0)
    for idx in trange(len(trainset)):
        if not trainset[idx][1] == args.class_id: 
            continue
        else:
            npz_fname = os.path.join(DATA_DIR, 
                                     dataset, 
                                     f"{args.kernel}Kernel",
                                     "intraclass_merged",
                                     f"{idx}.npz"
                                    )
            dis_vec = np.load(npz_fname)['data'].astype(np.float32)
            if np.isnan(dis_vec).any() or np.isinf(dis_vec).any():
                raise ValueError(f"{npz_fname} contains NaN or Inf.")
            sorted_norm = concat_two_sorted_norms(sorted_norm, dis_vec, 
                                                  idx2cls, args.class_id
                                                 )

    out_path = os.path.join(DATA_DIR,
                            dataset, 
                            f"{args.kernel}Kernel",
                            "class_merged",
                            f"{args.class_id}.npz"
                           )
    create_dir_for_file(out_path)
    np.savez_compressed(out_path, data=sorted_norm)


def run(args:argparse) -> None:
    if args.mnist:
        merge_class_kernel(args, "MNIST")

    if args.cifar10:
        merge_class_kernel(args, "CIFAR10")

    if args.cifar100:
        merge_class_kernel(args, "CIFAR100")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist", action="store_true", default=False, 
    help="run mnist experiment")
    parser.add_argument("--cifar10", action="store_true", default=False, 
    help="run cifar10 experiment")
    parser.add_argument("--cifar100", action="store_true", default=False, 
    help="run cifar100 experiment")
    parser.add_argument("--kernel", type=str, default="Sum",
        help="kernel to sort")
    parser.add_argument("--class-id", type=int, default=0,
        help="class index to preprocess")
    parser.add_argument("--verbose", action="store_true", default=False,
        help="print verbose information")
    parser.add_argument("--notes",   default=None)
    args, unknown = parser.parse_known_args()

    run(args)