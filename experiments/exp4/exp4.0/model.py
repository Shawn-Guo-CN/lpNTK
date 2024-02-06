import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim:int=784, hid_size:int=128, out_dim:int=10) \
    -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hid_size = hid_size
        self.out_dim = out_dim
        self.fc1 = nn.Linear(self.in_dim, self.hid_size)
        self.fc2 = nn.Linear(self.hid_size, self.hid_size)
        self.fc3 = nn.Linear(self.hid_size, self.out_dim)
        self.act = nn.ReLU(True)
        self._weight_init()

    def forward(self, x):
        h1 = self.act(self.fc1(x))
        h2 = self.act(self.fc2(h1))
        out = self.fc3(h2)
        return out
    
    def _weight_init(self):
        nn.init.normal_(self.fc1.weight, 0, 1)
        nn.init.normal_(self.fc1.bias, 0, 1)
        nn.init.normal_(self.fc2.weight, 0, 1./self.hid_size)
        nn.init.normal_(self.fc2.bias, 0, 1./self.hid_size)
        nn.init.normal_(self.fc3.weight, 0, 1./self.hid_size)
        nn.init.normal_(self.fc3.bias, 0, 1./self.hid_size)