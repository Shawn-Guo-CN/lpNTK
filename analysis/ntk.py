import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import StepLR
from functorch import make_functional, vmap, vjp, jvp, jacrev
from torch.func import functional_call, vmap, vjp, jvp, jacrev

import wandb
from tqdm import tqdm
import numpy as np
from munch import Munch
import pickle

from utils import set_seed, create_dir_for_file
from models import LeNet, ResNet18, ResNet50


def empirical_ntk(fnet_single, params, x1, x2):
    # Compute J(x1)
    jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1)
    jac1 = jac1.values()
    jac1 = [j.flatten(2) for j in jac1]

    # Compute J(x2)
    jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2)
    jac2 = jac2.values()
    jac2 = [j.flatten(2) for j in jac2]

    # Compute J(x1) @ J(x2).T
    result = torch.stack([torch.einsum('Naf,Mbf->NMab', j1, j2) 
                                               for j1, j2 in zip(jac1, jac2)])
    result = result.sum(0)
    return result

def get_ntk(config:Munch, dataset:str) -> None:
    args = config.default
    dataset_config = config[dataset]
    args.update(dataset_config)
    args.update({'experiment':config.wandb.experiment})

    use_cuda = args.use_cuda and torch.cuda.is_available()
    set_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    transform = eval(dataset_config.transform)
    dataset1 = eval('datasets.'+dataset)(args.data_dir, 
                                         train=True, 
                                         download=True, 
                                         transform=transform
                                        )
    kwargs = {'batch_size': args.batch_size,
              'shuffle': False,
              'num_workers': 1,
              'pin_memory': True
             }
    num_sample = len(dataset1)

    model = eval(args.model)().to(device)
    model.eval()
    model.load_state_dict(torch.load(args.pt_file))
    params = {k: v.detach() for k, v in model.named_parameters()}
    def fnet_single(params, x):
        return functional_call(model, params, (x.unsqueeze(0),)).squeeze(0)

    # problem_idx_fname = f'{args.results_dir}/{dataset}/problem_grad_idx.pkl'
    # problem_idx_list = pickle.load(open(problem_idx_fname, 'rb'))
    
    for i in range(args.start_idx, args.end_idx):
        # if not i in problem_idx_list: continue
        subset = Subset(dataset1, range(i, num_sample))
        dataloader = DataLoader(subset, **kwargs)
        ntk_list = []
        for _, (batch_data, _) in enumerate(dataloader):
            ntk = empirical_ntk(fnet_single, 
                                params, 
                                torch.unsqueeze(dataset1[i][0], 0).to(device),
                                batch_data.to(device)
                               ).squeeze(dim=0).detach()
            ntk_list.append(ntk)
        ntk_list = torch.cat(ntk_list, dim=0).cpu().type(torch.FloatTensor)
        out_file = f'{args.results_dir}/{dataset}/GradMatrix/{i}.npz'
        create_dir_for_file(out_file)
        np.savez_compressed(out_file, data=ntk_list.numpy())
