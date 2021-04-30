
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import sys

# Test output size of Conv2d

inputs = torch.rand(1,1,28,28)
# mod = nn.Conv2d(1,32,(3,3))
mod = nn.Sequential(
    nn.Conv2d(1, 32, (3, 3)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2)),
    # nn.Conv2d(32, 64, (3, 3)),
    # nn.ReLU(),
    # nn.MaxPool2d((2, 2))
)
out = mod(inputs)
print(out.shape)

