#! /usr/bin/env python
import cv2
import _pickle as c_pickle, gzip
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import sys
import torchvision.transforms.functional as tvisionF
from torch import tensor
sys.path.append("..")
import utils
from utils import *
from train_utils import batchify_data, run_epoch, train_model, Flatten


# Note: please generate named model
# https://stackoverflow.com/questions/66152766/how-to-assign-a-name-for-a-pytorch-layer
# the following model is an expected result from nnet_cnn.py (named model)
#
model = torch.load("mnist_model_cnn.pt")
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


#
# layer_name = 'Conv1'
# intermediate_layer_model = nn.Model(inputs=model.input,
#                                           outputs=model.get_layer(layer_name).output)
# intermediate_output = intermediate_layer_model.predict(img)
#
#
# temp = intermediate_output.reshape(800,64,2) # 2 feature
# plt.imshow(temp[:,:,2],cmap='gray')
# note that output should be reshape in 3 dimension


