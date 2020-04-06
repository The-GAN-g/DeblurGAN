import torch.nn as nn 
import torch
#import torch.nn.functional as F
import numpy as np

class Discriminator(nn.Module):   
    """Defines a PatchGAN discriminator"""
    def __init__(self, input_nc = 3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[],
                 use_parallel=True):  
        super(Discriminator, self).__init__()
        
        """
         Parameters:
            input_nc (int)  -- the number of channels in input images. For color images this is 3.
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer (can be nn.BatchNorm2d or nn.InstanceNorm2d)
        """
        
#         ndf = 64
#         output_nc = 3
#         input_shape_discriminator = (256, 256, output_nc)    
#         n_layers = 3
#         use_sigmoid = False
#         gpu_ids = []
#         norm_layer = nn.BatchNorm2d
#         ndf = 64
#         input_nc = 3
#         use_parallel = True

        kw = 4 # kernel size
        padw = int(np.ceil((kw - 1) / 2)) # 2

        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers): # runs for 2 iterations
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8) # n_layers = 3
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw), # default bias value is True
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        # output image till here is 32 * 32 * 256
        
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8) 
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)
        
    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)        

if __name__ == '__main__':
    d = Discriminator()
    print(d)

