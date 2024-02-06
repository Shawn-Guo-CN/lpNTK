import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from utils import get_args, set_seed
from modules import MLP, LeNet
from pruned_datasets import PrunedMNIST
from train_models import train, test

def train_model(data:str, seed:int):
    args = get_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    set_seed(seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    dataset1 = None
    if data == 'mnist':
        dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    else:
        dataset1 = PrunedMNIST(f'./data/MNIST/{data}', transform=transform)
    
    dataset2 = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # model = MLP(hid_size=args.hidden_size).to(device)
    model = LeNet().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=2, gamma=args.gamma)
    test_acc_list = []
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test_acc = test(model, device, test_loader)
        test_acc_list.append(test_acc)
        scheduler.step()

    return test_acc_list


def main():
    seed_list = [1, 12, 123, 1234, 12345, 5, 54 , 543, 5432, 54321]
    data_list = [
        # fill here the comparison you want to make
    ]
    
    for data in data_list:
        testacc_log = []
        for seed in seed_list:
            testacc_log.append(train_model(data, seed))
        np.save(f'./results/mnist/{data}_testacc_log.npy', np.array(testacc_log))

    for data in data_list:
        testacc_log = np.load(f'./results/mnist/{data}_testacc_log.npy')
        print(f'{data} averaged test accuracy: {np.mean(testacc_log, axis=0)}')

if __name__ == '__main__':
    main()
