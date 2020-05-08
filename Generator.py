import torch.nn as nn 
import torch
import functools

# The Generator and Resnet blocks are inspired by the DeblurGAN Paper(https://github.com/KupynOrest/DeblurGAN)
class ResnetBlock(nn.Module):
    """
    RESNET Generator

    Parameters:
    Instantiate a PyTorch Resnet Block using Sequential API.
        input: Input tensor
        filters: Number of filters to use
        kernel_size: Shape of the kernel for the convolution
        strides: Shape of the strides for the convolution
        use_dropout: Boolean value to determine the use of dropout
        return: Pytorch Model
    """
    def __init__(self, dim, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        
        blocks = [ 
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True)
        ]
        
        if use_dropout:
            blocks += [
                nn.Dropout(0.5)
            ]
        
        blocks += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias),
            norm_layer(dim)
        ]
            
        self.conv_block = nn.Sequential(*blocks)
    
    def forward(self, x):
        # Two convolution layers followed by a direct connection between input and output
        out = x + self.conv_block(x)
        return out
        
        
class Generator(nn.Module):   
    """
    Defines Generator with 9 RESNET blocks
    """
    def __init__(self, input_nc = 3, ngf = 64, n_layers=3, norm_layer=nn.BatchNorm2d, use_dropout=True,
                 n_blocks = 9, learn_residual=True):

        assert (n_blocks >= 0)
        super(Generator, self).__init__()
        self.learn_residual = learn_residual

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
            print("Log (Generator): Used Bias for Instance Normalization")
        else:
            print("Log (Generator): Used Bias for Batch Normalization")
            use_bias = norm_layer == nn.InstanceNorm2d
        
        #use_bias = True

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        ]
        
        # Increase filter number
        model += [
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(256),
            nn.ReLU(True)
        ]
        
        # Apply 9 ResNet blocks
        for i in range(n_blocks):
            model += [
                ResnetBlock(256, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
            ]
          
        # Decrease filter number to 3 (RGB)
        model += [
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(64),
            nn.ReLU(True),
        ]
        
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels=3 , kernel_size=7, padding=0),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
        
    def forward(self, input):
        """
        Forward propagation of the network
        """
        output = self.model(input)
        # Add direct connection from input to output and re-center to [-1, 1]
        if self.learn_residual:
            output = torch.clamp(input + output, min=-1, max=1)
        return output