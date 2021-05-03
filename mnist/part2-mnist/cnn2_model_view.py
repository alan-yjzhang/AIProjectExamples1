#! /usr/bin/env python
import cv2
import _pickle as c_pickle, gzip
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch import tensor
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms.functional as tvisionF
import sys

sys.path.append("..")
import utils
from utils import *
from train_utils import batchify_data, run_epoch, train_model, Flatten

class CNN(nn.Module):

    def __init__(self, input_dimension):
        super(CNN, self).__init__()
        # 1 input image channel, 32 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.dropout = nn.Dropout(0.5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # 5*5 from image dimension
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x1 = self.conv1(x)
        x10 = F.max_pool2d(F.relu(x1), (2,2))
        x2 = self.conv2(x10)
        x20 = F.max_pool2d(F.relu(x2), (2,2))
        x3 = x20.view(-1, 64*5*5)
        # or  x3 = x20.view(-1, self.num_flat_features(x20))
        x4 = F.relu(self.fc1(x3))
        x5 = self.dropout(x4)
        out = self.fc2(x5)
        return out


model = torch.load("mnist_model_cnn2.pt")
img = cv2.imread('../Datasets/digit_8.png',0)
img2 = cv2.resize(img, (28,28))
# plt.imshow(img)

# img3 = np.reshape(img2, (28,28,1)) # expected image format:  ( x_shape, y_shape, n_channels,)
tensorImg = tvisionF.to_tensor(img2)
tensorImg = tensorImg.unsqueeze(0)

outputs= []
def hook(module, input, output):
    outputs.append(output)

model.conv1.register_forward_hook(hook)

out = model(tensorImg)
# print(outputs)

tempOut = outputs[0].squeeze()
if tempOut.ndim > 1:
    plot_intermediate_out(tempOut) # 2d data
else:
    tempOut = tempOut.detach().numpy()
    plt.plot(tempOut) # 1d data
print("finished")
