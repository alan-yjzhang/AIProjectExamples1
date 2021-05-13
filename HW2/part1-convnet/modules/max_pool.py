import numpy as np

class MaxPooling:
    '''
    Max Pooling of input
    '''
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None
        self.mask = None

    def forward(self, x):
        '''
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        '''
        out = None
        #############################################################################
        # TODO: Implement the max pooling forward pass.                             #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #############################################################################
        N = x.shape[0]
        in_channels = x.shape[1]
        H_out = int((x.shape[2] - self.kernel_size)/self.stride + 1)
        W_out = int((x.shape[3] - self.kernel_size)/self.stride + 1)
        out = np.zeros((N, in_channels, H_out, W_out))

        for k in range(N):
            for i in range(in_channels):
                for j in range(H_out):
                    for l in range(W_out):
                        out[k,:,j,l] = np.amax(x[k,:,j*self.stride:j*self.stride+self.kernel_size,l*self.stride:l*self.stride+self.kernel_size], axis = (1,2))

        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = (x, H_out, W_out)
        return out

    def backward(self, dout):
        '''
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return:
        '''
        x, H_out, W_out = self.cache
        #############################################################################
        # TODO: Implement the max pooling backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                     #
        #       2) You may find np.unravel_index useful                             #
        #############################################################################
        self.dx = np.zeros(x.shape)
        N = x.shape[0]
        in_channels = x.shape[1]
        H_out = int((x.shape[2] - self.kernel_size)/self.stride + 1)
        W_out = int((x.shape[3] - self.kernel_size)/self.stride + 1)

        for k in range(N):
            for i in range(in_channels):
                for j in range(H_out):
                    for l in range(W_out):
                        res = np.amax(x[k,i,j*self.stride:j*self.stride+self.kernel_size,l*self.stride:l*self.stride+self.kernel_size], axis = (0,1))
                        self.dx[k,i,j*self.stride:j*self.stride+self.kernel_size,l*self.stride:l*self.stride+self.kernel_size] += np.multiply((x[k,i,j*self.stride:j*self.stride+self.kernel_size,l*self.stride:l*self.stride+self.kernel_size] >= res),  dout[k, i, j, l])

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
