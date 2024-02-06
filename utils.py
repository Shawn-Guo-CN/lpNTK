import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
from pathlib import Path
import numpy as np
from typing import Dict
import argparse
import time
import fcntl
from typing import List
from tqdm import trange
from torchvision import datasets, transforms
from munch import Munch


SAMPLE_SHAPE = {"MNIST": (784,),
                "CIFAR10": (3072,),
                "CIFAR100": (3072,),
               }
PROJ_DIR = os.path.expanduser("~/GitWS/Transmisstion-Phase")
DATA_DIR = os.path.join(PROJ_DIR, "data")


def get_args() -> argparse:
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    parser.add_argument('--hidden-size', type=int, default=1024, metavar='N',
                        help='size of hidden layer in MLP (default: 1024')

    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    
    # for splitting NTK calculation
    parser.add_argument('--ntk-start-idx', type=int, default=0, metavar='N',
                        help='start index of NTK calculation')
    parser.add_argument('--ntk-end-idx', type=int, default=-1, metavar='N',
                        help='end index of NTK calculation')
    parser.add_argument('--ntk-norm-path', type=str, default='./data/mnist_ntk/', metavar='N',
                        help='path to NTK norm')
    parser.add_argument('--class-id', type=int, default=0, metavar='N',
                        help='class to analyse')
    parser.add_argument('--out-path', type=str, default='./data/mnist_norm_class/', metavar='N',
                        help='path to save results')
    parser.add_argument('--in-path', type=str, default='./data/mnist_norm_class/', metavar='N',
                        help='path to inputs')


    args = parser.parse_args()

    return args


def set_seed(seed:int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def create_dir(dir_path:str) -> None:
    if not os.path.exists(dir_path):
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def create_dir_for_file(file_path:str) -> None:
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def update_config(unknown, config) -> Dict:
    for arg in unknown:
        if arg.startswith(("-", "--")):
            k, v = arg.split('=')
            k = k.replace("--", "")
            k = k.replace("-", "_")
            assert k in config['default'], f"unknown arg: {k=}"
            v_new = type(config['default'][k])(eval(v))
            print(f"Overwriting hps.{k} from {config['default'][k]} to {v_new}")
            config['default'][k] = v_new

    return config


def update_munch_config(config1:Munch, config2:Munch) -> Munch:
    for key in config2.keys():
        config1[key] = config2[key]
    return config1


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_l2distance(hid, y, num_cls:int=10) -> torch.Tensor:
    pre_prob = nn.Softmax(1)(hid)
    true_prob = F.one_hot(y, num_classes=num_cls)
    L2_dist = torch.norm(pre_prob-true_prob,dim=1)
    return L2_dist.detach().cpu().numpy()


def acquire_flock(lock_file, time_out=10.0, open_mode=os.O_RDWR | os.O_CREAT):
    fd = os.open(lock_file, open_mode)

    lock_file_fd = None
    
    start_time = time.time()
    while time.time() - start_time < time_out:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (IOError, OSError):
            pass
        else:
            lock_file_fd = fd
            break
        time.sleep(0.5)
    
    if lock_file_fd is None:
        os.close(fd)
        
    return lock_file_fd


def release_flock(lock_file_fd):
    fcntl.flock(lock_file_fd, fcntl.LOCK_UN)
    os.close(lock_file_fd)
    return


def rebuild_datasets_with_index_list(
    dataset:str, 
    keep_idx_list:List, 
    out_fpath:str,
    verbose:bool=False
) -> None:
    pruned_images = np.empty(SAMPLE_SHAPE[dataset])
    pruned_labels = []
    
    trainset = eval('datasets.'+dataset)(DATA_DIR, train=True, download=True, 
                      transform=transforms.ToTensor()
                     )
    
    if verbose: print("Building pruned dataset...")
    tmp_range = trange(len(keep_idx_list)) if verbose else \
                    range(len(keep_idx_list))
    for idx in tmp_range:
        pruned_images = \
            np.vstack((pruned_images, 
                       trainset[keep_idx_list[idx]][0].numpy().flatten()
                      ))
        pruned_labels.append(trainset.targets[keep_idx_list[idx]])
    pruned_labels = np.array(pruned_labels)
    if verbose: print("Done.")
    
    ratio = round(pruned_images.shape[0] / len(trainset), 2)
    if verbose: print(f"Ratio: {ratio}")
    
    create_dir_for_file(out_fpath)
    np.save(out_fpath, (pruned_images, pruned_labels))