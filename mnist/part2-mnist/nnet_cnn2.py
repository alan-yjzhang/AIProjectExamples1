#! /usr/bin/env python

import _pickle as c_pickle, gzip
import numpy as np
from tqdm import tqdm
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
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

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def main():
    # Load the dataset
    num_classes = 10
    X_train, y_train, X_test, y_test = get_MNIST_data()

    # We need to rehape the data back into a 1x28x28 image
    X_train = np.reshape(X_train, (X_train.shape[0], 1, 28, 28))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, 28, 28))

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = y_train[dev_split_index:]
    X_train = X_train[:dev_split_index]
    y_train = y_train[:dev_split_index]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [y_train[i] for i in permutation]

    # Split dataset into batches
    batch_size = 32
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    #################################
    ## Model specification TODO
    img_rows = 28
    img_cols = 28
    input_dimension = img_rows * img_cols
    model = CNN(input_dimension)
    ##################################

    train_model(train_batches, dev_batches, model, nesterov=True)

    torch.save(model, 'mnist_model_cnn2.pt')

    ## Evaluate the model on test data
    loss, accuracy = run_epoch(test_batches, model.eval(), None)

    print ("Loss on test set:"  + str(loss) + " Accuracy on test set: " + str(accuracy))

# Epoch 10:
# Train loss: 0.020385 | Train accuracy: 0.993405
# Val loss:   0.037970 | Val accuracy:   0.990475
# Loss on test set:0.030 Accuracy on test set: 0.990

if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)
    main()
