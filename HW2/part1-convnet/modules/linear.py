import numpy as np

class Linear:
    '''
    A linear layer with weight W and bias b. Output is computed by y = Wx + b
    '''
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.in_dim, self.out_dim)
        np.random.seed(1024)
        self.bias = np.zeros(self.out_dim)

        self.dx = None
        self.dw = None
        self.db = None


    def forward(self, x):
        '''
        Forward pass of linear layer
        :param x: input data, (N, d1, d2, ..., dn) where the product of d1, d2, ..., dn is equal to self.in_dim
        :return: The output computed by Wx+b. Save necessary variables in cache for backward
        '''
        out = None
        #############################################################################
        # TODO: Implement the forward pass.                                         #
        #    HINT: You may want to flatten the input first                          #
        #############################################################################
        dim_sum = 1
        for i in range(x.ndim):
            if i != 0:
                dim_sum *= x.shape[i]

        reshaped = x.reshape(x.shape[0], dim_sum)
        out = reshaped @ self.weight + self.bias
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x
        return out

    def backward(self, dout):
        '''
        Computes the backward pass of linear layer
        :param dout: Upstream gradients, (N, self.out_dim)
        :return: nothing but dx, dw, and db of self should be updated
        '''
        x = self.cache
        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        #############################################################################
        dim_sum = 1
        for i in range(x.ndim):
            if i != 0:
                dim_sum *= x.shape[i]

        reshaped = x.reshape(x.shape[0], dim_sum)

        self.dx = (dout @ self.weight.T).reshape(x.shape)
        self.dw = reshaped.T @ dout
        self.db = np.sum(dout, axis = 0)

        # self.gradients['W2'] = np.dot(np.transpose(o1), output)
        # self.gradients['b2'] = np.sum(output, axis = 0)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
