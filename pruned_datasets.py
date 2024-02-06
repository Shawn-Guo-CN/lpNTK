from typing import Tuple
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.utils.data import Dataset


class PrunedDataset(Dataset, ABC):
    def __init__(self, file, transform=None) -> None:
        super().__init__()
        self.file = file
        self.transform = transform
        
        self.samples, self.labels = np.load(self.file, allow_pickle=True)
        self.samples = self.samples[1:]
        assert self.samples.shape[0] == self.labels.shape[0]
        
    def __len__(self):
        return self.samples.shape[0]
    
    @abstractmethod
    def __getitem__(self, index:int) -> Tuple[torch.Tensor, int]:
        return super().__getitem__(index)

class PrunedMNIST(PrunedDataset):
    def __init__(self, file, transform=None):
        super().__init__(file, transform)
    
    def __getitem__(self, index:int) -> Tuple[torch.Tensor, int]:
        sample = np.float32(self.samples[index].reshape(28, 28))
        if self.transform:
            sample = self.transform(sample)
        label = self.labels[index]
        return sample, label

class PrunedCIFAR10(PrunedDataset):
    def __init__(self, file, transform=None) -> None:
        super().__init__(file, transform)
        
    def __getitem__(self, index:int) -> Tuple[torch.Tensor, int]:
        sample = np.float32(self.samples[index].reshape(3, 32, 32))
        sample = sample.transpose(1, 2, 0)
        if self.transform:
            sample = self.transform(sample)
        label = self.labels[index]
        return sample, label