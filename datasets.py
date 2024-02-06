from torch.utils.data import Dataset
from torchvision import datasets, transforms

class CIFAR10(Dataset):
    def __init__(self, **kwargs):
        self.cifar10 = datasets.CIFAR10(**kwargs)
        
    def __getitem__(self, index):
        data, target = self.cifar10[index]
        
        return data, target, index

    def __len__(self):
        return len(self.cifar10)
    
    
class MNIST(Dataset):
    def __init__(self, **kwargs):
        self.mnist = datasets.MNIST(**kwargs)
        
    def __getitem__(self, index):
        data, target = self.mnist[index]
        
        return data, target, index

    def __len__(self):
        return len(self.mnist)