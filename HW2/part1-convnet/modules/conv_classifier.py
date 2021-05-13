from .softmax_ce import SoftmaxCrossEntropy
from .relu import ReLU
from .max_pool import MaxPooling
from .convolution import Conv2D
from .linear import Linear


class ConvNet:
    '''
    Max Pooling of input
    '''
    def __init__(self, modules, criterion):
        self.modules = []
        for m in modules:
            if m['type'] == 'Conv2D':
                self.modules.append(
                    Conv2D(m['in_channels'],
                           m['out_channels'],
                           m['kernel_size'],
                           m['stride'],
                           m['padding'])
                )
            elif m['type'] == 'ReLU':
                self.modules.append(
                    ReLU()
                )
            elif m['type'] == 'MaxPooling':
                self.modules.append(
                    MaxPooling(m['kernel_size'],
                               m['stride'])
                )
            elif m['type'] == 'Linear':
                self.modules.append(
                    Linear(m['in_dim'],
                           m['out_dim'])
                )
        if criterion['type'] == 'SoftmaxCrossEntropy':
            self.criterion = SoftmaxCrossEntropy()
        else:
            raise ValueError("Wrong Criterion Passed")

    def forward(self, x, y):
        '''
        The forward pass of the model
        :param x: input data: (N, C, H, W)
        :param y: input label: (N, )
        :return:
          probs: the probabilities of all classes: (N, num_classes)
          loss: the cross entropy loss
        '''
        probs = None
        loss = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement forward pass of the model                                 #
        #############################################################################
        # for m in self.modules:
        #     print(m["type"])
        # c1 = self.modules.forward(x)
        # r1 = ReLU().forward(c1)
        # m1 = MaxPooling().forward(r1)
        # o1 = Linear().forward(m1)
        # probs, loss = SoftmaxCrossEntropy().forward(o1, y)
        next_in = x
        for m in self.modules:
            next_in = m.forward(next_in)
        probs, loss = self.criterion.forward(next_in, y)
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return probs, loss

    def backward(self):
        '''
        The backward pass of the model
        :return: nothing but dx, dw, and db of all modules are updated
        '''
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement backward pass of the model                                #
        #############################################################################
        self.criterion.backward()
        next_in = self.criterion.dx

        # print(next_in)
        for m in reversed(self.modules):
            m.backward(next_in)
            next_in = m.dx
            # print(next_in)
            

        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################