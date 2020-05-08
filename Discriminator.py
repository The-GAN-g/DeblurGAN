import torch.nn as nn
import numpy as np
import functools

# The Discriminator is inspired by the DeblurGAN Paper(https://github.com/KupynOrest/DeblurGAN)
class Discriminator(nn.Module):   
    """
    Defines a PatchGAN discriminator
    """
    def __init__(self, input_nc = 3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        """
         Parameters:
            input_nc (int)  -- the number of channels in input images. For color images this is 3.
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer (can be nn.BatchNorm2d or nn.InstanceNorm2d)
            use_sigmoid     -- use sigmoid or not
        """

        super(Discriminator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
            print("Log (Discriminator): Used Bias for Instance Normalization")
        else:
            print("Log (Discriminator): Used Bias for Batch Normalization")
            use_bias = norm_layer == nn.InstanceNorm2d

        #use_bias = True
        
        kw = 4  # kernel size
        padw = int(np.ceil((kw - 1) / 2))  # 2

        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        
        nf_mult = 1
        nf_mult_prev = 1

        # runs for 2 iterations
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)  # n_layers = 3
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size = kw, stride = 2, padding = padw, bias = use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        # output image till here is 32 * 32 * 256
        
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8) 
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size = kw, stride = 1, padding = padw, bias = use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)


    def forward(self, input):
        """
        Forward propagation of the network
        """
        return self.model(input)