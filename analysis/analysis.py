import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import StepLR
from functorch import make_functional, vmap, vjp, jvp, jacrev

from tqdm import tqdm
import numpy as np
import pickle

from utils import get_args, set_seed
from hierarchical_clustering import UnionTracker, Cluster


def get_avg_norm_same_diff() -> None:
    args = get_args()
    set_seed(args.seed)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    train_set = datasets.MNIST('./data', train=True, transform=transform)
    dataset = train_set

    num_sample = len(dataset)
    ntk_norm_list = []

    for i in range(num_sample):
        ntk_norm = np.load(args.ntk_norm_path + f'{i}.npy')
        ntk_norm_list.append(ntk_norm)

    avgnorm_same = []
    avgnorm_diff = []
    idx_list = []
    for i in range(args.ntk_start_idx, args.ntk_end_idx):
        avgnorm_in_list = []
        avgnorm_out_list = []
        for j in range(num_sample):
            if dataset[i][1] == dataset[j][1]:
                if i <= j:
                    avgnorm_in_list.append(ntk_norm_list[i][j-i])
                else:
                    avgnorm_in_list.append(ntk_norm_list[j][i-j])
            else:
                if i <= j:
                    avgnorm_out_list.append(ntk_norm_list[i][j-j])
                else:
                    avgnorm_out_list.append(ntk_norm_list[j][i-j])
        idx_list.append(i)
        avgnorm_same.append(np.mean(avgnorm_in_list))
        avgnorm_diff.append(np.mean(avgnorm_out_list))
    
    np.save(args.out_path + f'{args.ntk_start_idx}_{args.ntk_end_idx}.npy', 
            np.stack([idx_list, avgnorm_same, avgnorm_diff]))


def get_ntk_norm_sorted() -> None:
    args = get_args()
    set_seed(args.seed)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    train_set = datasets.MNIST('./data', train=True, transform=transform)
    dataset = train_set

    num_sample = len(dataset)
    sorted_norm = np.array([]).reshape(3,0)

    for i in range(num_sample):
        if not dataset[i][1] == args.class_id:
            continue
        else:
            ntk_norm = np.load(f'{args.ntk_norm_path}{i}.npy')
            i_idx = i * np.ones(ntk_norm.shape[0], dtype=int)
            j_idx = i + np.arange(ntk_norm.shape[0])
            sorted_norm = np.concatenate((sorted_norm, np.stack([ntk_norm, i_idx, j_idx])), axis=1)
            sorted_norm = sorted_norm[:, np.argsort(sorted_norm[0])]
    
    print(sorted_norm.shape)
    np.save(f'{args.out_path}sorted_class_{args.class_id}.npy', sorted_norm)


def get_hierarchical_clusters() -> None:
    args = get_args()
    set_seed(args.seed)

    sorted_norm = np.load(f'{args.in_path}sorted_class_{args.class_id}.npy')
    sorted_norm = np.flip(sorted_norm, axis=1)
    train_set = datasets.MNIST('./data', train=True, transform=None)
    dataset = train_set

    union_tracker = UnionTracker()
    cluster_list = []
    cls_idx_list = []
    idx2cluster_dict = {}

    for i, (img, label) in enumerate(dataset):
        if label == args.class_id:
            cls_idx_list.append(i)
            cluster_list.append(Cluster(**{'type': 'leaf', 'sample_idx': i}))
            idx2cluster_dict[i] = cluster_list[-1].idx

    j = 0
    for i in range(sorted_norm.shape[1]):
        cur_norm = sorted_norm[0, i]
        src_idx = int(sorted_norm[1, i])
        tgt_idx = int(sorted_norm[2, i])

        if not tgt_idx in cls_idx_list:
            continue

        src_cluster_idx = idx2cluster_dict[src_idx]
        tgt_cluster_idx = idx2cluster_dict[tgt_idx]

        if src_cluster_idx == tgt_cluster_idx:
            continue

        l_sample_list = cluster_list[src_cluster_idx].sample_idx_list
        r_sample_list = cluster_list[tgt_cluster_idx].sample_idx_list
        if set(l_sample_list).issubset(set(r_sample_list)) or set(r_sample_list).issubset(set(l_sample_list)):
            continue

        cluster_list.append(Cluster(**{'type': 'internal',
                                       'lchild': src_cluster_idx,
                                       'rchild': tgt_cluster_idx,
                                       'norm': cur_norm,
                                       'lchild_sample_idx': l_sample_list,
                                       'rchild_sample_idx': r_sample_list
        }))
        new_cluster_idx = cluster_list[-1].idx
        assert new_cluster_idx == Cluster.count - 1

        for sample_idx in cluster_list[-1].sample_idx_list:
            idx2cluster_dict[sample_idx] = new_cluster_idx

        idx2cluster_dict[src_idx] = new_cluster_idx
        idx2cluster_dict[tgt_idx] = new_cluster_idx
        union_tracker.add(src_cluster_idx, tgt_cluster_idx, cur_norm, new_cluster_idx)

        j += 1
        if j >= len(cls_idx_list)-1:
            break
        del l_sample_list
        del r_sample_list

    union_tracker.save(f'{args.out_path}union_tracker_class_{args.class_id}.npy')
    with open(f'{args.out_path}sample_idx_list_class_{args.class_id}.pkl', 'wb') as f:
        pickle.dump(cls_idx_list, f)


if __name__ == '__main__':
    get_avg_norm_same_diff()
    get_ntk_norm_sorted()
    get_hierarchical_clusters()