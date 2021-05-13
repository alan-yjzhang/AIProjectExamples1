import numpy as np
import torch
import torch.nn as nn

class LSTM(nn.Module):
    # An implementation of naive LSTM using Pytorch Linear layers and activations
    # You will need to complete the class init function, forward function and weight initialization


    def __init__(self, input_size, hidden_size):
        """ Init function for VanillaRNN class
            Args:
                input_size (int): the number of features in the inputs.
                hidden_size (int): the size of the hidden layer
            Returns: 
                None
        """
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        ################################################################################
        # TODO:                                                                        #
        #   Declare LSTM weights and attributes as you wish here.                      #
        #   You should include weights and biases regarding using nn.Parameter:        #
        #       1) i_t: input gate                                                     #
        #       2) f_t: forget gate                                                    #
        #       3) g_t: cell gate, or the tilded cell state                            #
        #       4) o_t: output gate                                                    #
        #   You also need to include correct activation functions                      #
        #   Initialize the gates in the order above!                                   #
        #   Initialize parameters in the order they appear in the equation!            #                                                              #
        ################################################################################
        concat_size = input_size + hidden_size
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # i_t: input gate
        self.W_ii = nn.Parameter(torch.zeros(input_size, hidden_size))
        self.W_ih = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.b_ii = nn.Parameter(torch.zeros(hidden_size))
        self.b_ih = nn.Parameter(torch.zeros(hidden_size))
        
        
        # f_t: the forget gate
        self.W_fi = nn.Parameter(torch.zeros(input_size, hidden_size))
        self.W_fh = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.b_fi = nn.Parameter(torch.zeros(hidden_size))
        self.b_fh = nn.Parameter(torch.zeros(hidden_size))
        
        # g_t: the cell gate
        self.W_gi = nn.Parameter(torch.zeros(input_size, hidden_size))
        self.W_gh = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.b_gi = nn.Parameter(torch.zeros(hidden_size))
        self.b_gh = nn.Parameter(torch.zeros(hidden_size))
        
        # o_t: the output gate
        self.W_oi = nn.Parameter(torch.zeros(input_size, hidden_size))
        self.W_oh = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.b_oi = nn.Parameter(torch.zeros(hidden_size))
        self.b_oh = nn.Parameter(torch.zeros(hidden_size))




        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        self.init_hidden()

    def init_hidden(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
         
    def forward(self, x: torch.Tensor, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        
        ################################################################################
        # TODO:                                                                        #
        #   Implement the forward pass of LSTM. Please refer to the equations in the   #
        #   corresponding section of jupyter notebook. Iterate through all the time    #
        #   steps and return only the hidden and cell state, h_t and c_t.              # 
        #   Note that this time you are also iterating over all of the time steps.     #
        ################################################################################
        x_sizes = list(x.size())
        batch_size = x_sizes[0]
        seq_size = x_sizes[1]
        
        h_t, c_t = None, None
        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size)
            c_t = torch.zeros(batch_size, self.hidden_size)
        else:
            h_t, c_t = init_states
            
        for seq_num in range(seq_size):
            x_i = x[:, seq_num, :]

            i_t = self.sigmoid(x_i @ self.W_ii + self.b_ii + h_t @ self.W_ih + self.b_ih)
            f_t = self.sigmoid(x_i @ self.W_fi + self.b_fi + h_t @ self.W_fh + self.b_fh)
            g_t = self.tanh(x_i @ self.W_gi + self.b_gi + h_t @ self.W_gh + self.b_gh)
            o_t = self.sigmoid(x_i @ self.W_oi + self.b_oi + h_t @ self.W_oh + self.b_oh)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * self.tanh(c_t)
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        return (h_t, c_t)

