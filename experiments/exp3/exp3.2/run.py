import os, sys
import argparse
from collections import defaultdict, OrderedDict
from typing import List

from tqdm import tqdm, trange
import numpy as np
from numpy.typing import ArrayLike
from torchvision import datasets, transforms
import pickle


PROJ_DIR = os.path.expanduser("~/GitWS/Transmisstion-Phase")
DATA_DIR = os.path.join(PROJ_DIR, "data")
SRC_DIR = os.path.join(PROJ_DIR, "src")
LOGS_DIR = os.path.join(PROJ_DIR, "logs", "exp3")
SCRIPTS_DIR = os.path.join(PROJ_DIR, "scripts")
CHECKPOINTS_DIR = os.path.join(PROJ_DIR, "checkpoints")
RESULTS_DIR = os.path.join(PROJ_DIR, "results")


sys.path.append(PROJ_DIR)
from utils import create_dir_for_file
from analysis.farthest_clustering import Cluster


def farthest_clustering(num_cluster:int, 
                        idx2sortnorm_dict:dict,
                        sorted_norm:ArrayLike,
                        verbose:bool=False,
                       ) -> List[Cluster]:
    cluster_list = []

    _head = int(sorted_norm[1, 0])
    _sample_dict = idx2sortnorm_dict[_head]
    cluster_list.append(Cluster(_head, _sample_dict))
    num_clusters = 1

    pbar = tqdm(total=num_cluster) if verbose else None
    while num_clusters < num_cluster:
        # 1. find the new centre in this iteration
        min_sim = float('inf')
        min_cluster_idx = -1
        for cluster_idx, cluster in enumerate(cluster_list):
            cluster_min = list(cluster.sample_od.items())[-1][1]
            cluster_minidx = list(cluster.sample_od.items())[-1][0]
            if cluster_min < min_sim and not cluster.head == cluster_minidx:
                min_sim = cluster_min
                min_cluster_idx = cluster_idx
        
        # 2. create the new cluster
        new_centre_idx = \
            list(cluster_list[min_cluster_idx].sample_od.items())[-1][0]
        cluster_list[min_cluster_idx].remove(new_centre_idx)
        new_cluster = Cluster(new_centre_idx, defaultdict(OrderedDict))
        new_cluster.sample_od[new_centre_idx] = \
            idx2sortnorm_dict[new_centre_idx][new_centre_idx]

        # 3. move the elements from the old clusters to the new cluster
        for cluster in cluster_list:
            tmp_keys = reversed(list(cluster.sample_od.keys()).copy())
            for _sample_idx in tmp_keys:
                # if the sample is more similar to the new centre
                if cluster.sample_od[_sample_idx] < \
                       idx2sortnorm_dict[new_centre_idx][_sample_idx] \
                   and _sample_idx != cluster.head:
                    new_cluster.sample_od[_sample_idx] = \
                        idx2sortnorm_dict[new_centre_idx][_sample_idx]
                    cluster.remove(_sample_idx)

        # 4. sort the samples in the new cluster
        new_cluster.sample_od = OrderedDict(
                                    sorted(new_cluster.sample_od.items(), 
                                           key=lambda x: x[1], 
                                           reverse=True)
                                )
        
        # 5. append the new cluster to the cluster list
        cluster_list.append(new_cluster)
        num_clusters += 1
        if verbose: pbar.update(1)

    return cluster_list


def _test_farthest_clustering() -> None:
    sorted_norm = np.asarray([
        [30., 25., 24., 23., 22., 18., 15., 15., 12., 10.],
        [  0,   0,   3,   2,   1,   2,   0,   0,   1,   1],
        [  0,   1,   3,   2,   1,   3,   2,   3,   2,   3]
    ])

    idx2sortnorm_dict = defaultdict(OrderedDict)
    for i in range(sorted_norm.shape[1]):
        src_idx = int(sorted_norm[1, i])
        tgt_idx = int(sorted_norm[2, i])
        idx2sortnorm_dict[src_idx][tgt_idx] = sorted_norm[0, i]
        idx2sortnorm_dict[tgt_idx][src_idx] = sorted_norm[0, i]

    num_cluster = 2
    cluster_list = farthest_clustering(num_cluster,
                                       idx2sortnorm_dict,
                                       sorted_norm,
                                       verbose=False
                                      )
    print(f'num_cluster: {len(cluster_list)} / {num_cluster}')
    print(f'1st cluster head: {cluster_list[0].head} / 0')
    print(f'1st cluster samples: {cluster_list[0].get_sample_list()}')
    print(f'2nd cluster head: {cluster_list[1].head} / 3')
    print(f'2nd cluster samples: {cluster_list[1].get_sample_list()}')
    Cluster.save(cluster_list, 'tmp_test.pkl')


def farthest_clustering_class(args: argparse, dataset:str) -> None:
    trainset = eval('datasets.'+dataset)(DATA_DIR, train=True, download=True, 
                      transform=transforms.ToTensor()
                     )

    npz_name = os.path.join(DATA_DIR,
                            dataset,
                            f"{args.kernel}Kernel",
                            "class_merged",
                            f"{args.class_id}.npz"
                           )
    sorted_norm = np.load(npz_name)['data'].astype(np.float32)
    
    cls2selfmax_fname = os.path.join(DATA_DIR,
                                     dataset,
                                     f"{args.kernel}Kernel",
                                     "cls_to_selfmax.pkl"
                           )
    selfmax_list = pickle.load(open(cls2selfmax_fname, "rb"))[args.class_id]

    cls_idx_list = []
    num_cls_samples = 0
    idx2sortnorm_dict = defaultdict(OrderedDict)

    for i, (_, label) in enumerate(trainset):
        if label == args.class_id:
            num_cls_samples += 1
            if i in selfmax_list: cls_idx_list.append(i)
    if args.verbose: \
        print(f'num of samples in class {args.class_id}: {num_cls_samples}')
    if args.verbose: print(f'num of selfmax samples: {len(cls_idx_list)}')

    """
    if args.verbose: print("removing contradictory samples...", end='')
    tmp_range = trange(sorted_norm.shape[1]) if args.verbose \
                else range(sorted_norm.shape[1])
    for i in tmp_range:
        if sorted_norm[0, -1-i] < 0:
            try:
                cls_idx_list.remove(int(sorted_norm[1, -1-i]))
            except ValueError:
                pass
            try:
                cls_idx_list.remove(int(sorted_norm[2, -1-i]))
            except ValueError:
                pass
        else:
            break
    if args.verbose: print("done")
    """

    if args.verbose: print("establishing sample norm list...")
    tmp_range = trange(sorted_norm.shape[1]) if args.verbose \
                else range(sorted_norm.shape[1])
    for i in tmp_range:
        src_idx = int(sorted_norm[1, i])
        tgt_idx = int(sorted_norm[2, i])
        idx2sortnorm_dict[src_idx][tgt_idx] = sorted_norm[0, i]
        idx2sortnorm_dict[tgt_idx][src_idx] = sorted_norm[0, i]
    if args.verbose: print("done")
    
    if args.verbose: print("farthest clustering...")
    if len(cls_idx_list) > int(num_cls_samples * args.ratio):
        cluster_list = farthest_clustering(int(num_cls_samples * args.ratio),
                                           idx2sortnorm_dict,
                                           sorted_norm,
                                           args.verbose
                                          )
    else:
        cluster_list = [Cluster(head=i, sample_od={i:'inf'}) 
                        for i in cls_idx_list]
    if args.verbose: print("done")

    if args.verbose: print("saving results...", end='')
    out_fname = os.path.join(DATA_DIR,
                             dataset,
                             f"{args.kernel}Kernel",
                             f"far_cluster_{args.ratio}",
                             f"cluster_list_{args.class_id}.pkl"
                            )
    create_dir_for_file(out_fname)
    Cluster.save(cluster_list, out_fname)


def run(args: argparse) -> None:
    if args.mnist:
        farthest_clustering_class(args, "MNIST")
    
    if args.cifar10:
        farthest_clustering_class(args, "CIFAR10")

    if args.cifar100:
        farthest_clustering_class(args, "CIFAR100")


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
    parser.add_argument("--ratio", type=float, default=0.5,
        help="ratio of data to keep")
    parser.add_argument("--verbose", action="store_true", default=False,
        help="verbose mode")
    parser.add_argument("--notes",   default=None)
    args, unknown = parser.parse_known_args()

    run(args)
    # _test_farthest_clustering()