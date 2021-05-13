import torch
import torch.nn as nn

class TotalVariationLoss(nn.Module):
    def forward(self, img, tv_weight):
        """
            Compute total variation loss.

            Inputs:
            - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
            - tv_weight: Scalar giving the weight w_t to use for the TV loss.

            Returns:
            - loss: PyTorch Variable holding a scalar giving the total variation loss
              for img weighted by tv_weight.
            """

        ##############################################################################
        # TODO: Implement total variation loss function                              #
        # Please pay attention to use torch tensor math function to finish it.       #
        # Otherwise, you may run into the issues later that dynamic graph is broken  #
        # and gradient can not be derived.                                           #
        ##############################################################################
        shifted_h = img[:,:,1:,:]
        excluded_h = img[:,:,:-1,:]
        shifted_w = img[:,:,:,1:]
        excluded_w = img[:,:,:,:-1]
        h_square = torch.square(shifted_h - excluded_h)
        w_square = torch.square(shifted_w - excluded_w)
        loss = tv_weight * torch.sum(h_square.sum() + w_square.sum()) 
        return loss
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################