import numpy as np

class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        '''
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        '''
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels,  self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        '''
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        '''
        out = None
        #############################################################################
        # TODO: Implement the convolution forward pass.                             #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################
        N = x.shape[0]
        out_channels = self.weight.shape[0]
        padded = np.pad(x, ((0, 0), (0, 0), (self.padding,self.padding), (self.padding,self.padding)), 'constant')
        H_prime = int((padded.shape[2] - self.kernel_size)/self.stride + 1)
        W_prime = int((padded.shape[3] - self.kernel_size)/self.stride + 1)
        out = np.zeros((N, out_channels, H_prime, W_prime))

        for k in range(N):
            for i in range(out_channels):
                for j in range(H_prime):
                    for l in range(W_prime):
                        out[k,i,j,l] = np.sum(np.multiply(padded[k,:,j*self.stride:j*self.stride+self.kernel_size,l*self.stride:l*self.stride+self.kernel_size], self.weight[i])) + self.bias[i]
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x
        return out

    def backward(self, dout):
        '''
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        '''
        x = self.cache
        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the convolution with loops                     #
        #       2) don't forget padding when computing dx                           #
        #############################################################################
        # self.dx = dout
        padded = np.pad(x, ((0, 0), (0, 0), (self.padding,self.padding), (self.padding,self.padding)), 'constant')
        self.dx = np.zeros(padded.shape)
        self.dw = np.zeros((self.out_channels, self.in_channels,  self.kernel_size, self.kernel_size))
        self.db = np.zeros(self.out_channels)
        N = x.shape[0]
        out_channels = self.weight.shape[0]
        
        H_prime = int((padded.shape[2] - self.kernel_size)/self.stride + 1)
        W_prime = int((padded.shape[3] - self.kernel_size)/self.stride + 1)

        for k in range(N):
            for i in range(out_channels):
                self.db[i] += np.sum(dout[k,i,:,:])
                for j in range(H_prime):
                    for l in range(W_prime):
                        self.dx[k,:,j*self.stride:j*self.stride+self.kernel_size,l*self.stride:l*self.stride+self.kernel_size] += self.weight[i] * dout[k, i, j, l]
                        self.dw[i] += padded[k,:,j*self.stride:j*self.stride+self.kernel_size,l*self.stride:l*self.stride+self.kernel_size] * dout[k, i, j, l]
                        
        self.dx = self.dx[:,:,self.padding:self.dx.shape[2]-self.padding,self.padding:self.dx.shape[3]-self.padding]
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################