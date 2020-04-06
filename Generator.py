import torch.nn as nn 
import torch
#import torch.nn.functional as F

class ResnetBlock(nn.Module):
    """RESNET Block"""
    def __init__(self, dim, padding_type, norm_layer, use_dropout):
        super(ResnetBlock, self).__init__()
        
        blocks = [ 
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            norm_layer(dim),
            nn.ReLU(True)
        ]
        
        if use_dropout:
            blocks += [
                nn.Dropout(0.5)
            ]
        
        blocks += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            norm_layer(dim)
        ]
            
        self.conv_block = nn.Sequential(*blocks)
    
    def forward(self, x):
            out = x + self.conv_block(x)
            return out
        
class Generator(nn.Module):   
    """Defines a RESNET Generator"""
    def __init__(self, input_nc = 3, ngf = 64, n_layers=3, norm_layer=nn.BatchNorm2d, use_dropout=True, 
                 n_blocks = 9, gpu_ids=[], learn_residual = True,
                 use_parallel=True, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(Generator, self).__init__()
        
        use_gpu = len(gpu_ids) > 0
        
        if use_gpu:
            assert (torch.cuda.is_available())
        
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            norm_layer(ngf),
            nn.ReLU(True)
        ]
        
        # Increase filter number
        model += [
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            norm_layer(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            norm_layer(256),
            nn.ReLU(True)
        ]
        
        # Apply 9 ResNet blocks
        for i in range(n_blocks):
            model += [
                ResnetBlock(256, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout)
            ]
          
        # Decrease filter number to 3 (RGB)
        model += [
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            norm_layer(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            norm_layer(64),
            nn.ReLU(True),
        ]
        
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels = 3 , kernel_size=7, padding=0),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
        
    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
            output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            output = self.model(input)
        
        # Add direct connection from input to output and recenter to [-1, 1]
        if self.learn_residual:
            # output = input + output
            output = torch.clamp(input + output, min=-1, max=1)
        return output
    
if __name__ == '__main__':
    g = Generator()
    print(g)