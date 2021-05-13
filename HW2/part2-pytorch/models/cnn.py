import torch
import torch.nn as nn

class VanillaCNN(nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()
        #############################################################################
        # TODO: Initialize the Vanilla CNN                                          #
        #       Conv: 7x7 kernel, stride 1 and padding 0                            #
        #       Max Pooling: 2x2 kernel, stride 2                                   #
        #############################################################################
        self.conv1 = nn.Conv2d(3, 32, 7)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.l1 = nn.Linear(32 * 13 * 13, 10)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################


    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        relu = nn.ReLU()
        x = self.pool1(relu(self.conv1(x)))
        x = x.view(-1, 32 * 13 * 13)
        outs = self.l1(x)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return outs