import torch
import torch.nn as nn

class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        '''
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        '''
        super(TwoLayerNet, self).__init__()
        #############################################################################
        # TODO: Initialize the TwoLayerNet, use sigmoid activation between layers   #
        #############################################################################
        self.l1 = nn.Linear(input_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, num_classes)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        out = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        dim_count = 1
        for i in range(x.ndim):
            if i != 0:
                dim_count *= x.shape[i]
        x = x.reshape((x.shape[0], dim_count))
        sig = nn.Sigmoid()
        x = sig(self.l1(x))
        out = self.l2(x)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out