"""
File: models.py
------------------
This file holds the torch.nn model classes (i.e., the neural network architectures). 
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pdb


# TODO: add more models here.

# probably don't need much here since we will use torchvision resnet model. Might add simple MLP as baseline though. 


# specify num_classes and input_shape 
class MLP(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        input_dim = 1
        for dim in input_shape:
            input_dim *= dim
        self.fc1 = nn.Linear(input_dim, 124)
        self.fc2 = nn.Linear(124, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
