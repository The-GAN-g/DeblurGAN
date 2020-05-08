import torchvision.models as models
import torch.nn as nn 
import torch
from torch.autograd import Variable

def mse_loss(y_true, y_pred):
    criterion = torch.nn.MSELoss()
    # calculate loss
    loss = criterion(y_pred,y_true)
    return loss


# Simplified version of the DeblurGAN paper(https://github.com/KupynOrest/DeblurGAN)
def perceptual_loss(y_true, y_pred):
    conv_3_3_layer = 14
    cnn = models.vgg19(pretrained=True).features
    cnn = cnn.cuda()
    model = nn.Sequential()
    model = model.cuda()
    for i, layer in enumerate(list(cnn)):
        model.add_module(str(i), layer)
        if i == 'conv_3_3_layer':
            break
    criterion = nn.MSELoss()
    fake = model.forward(y_pred)
    real = model.forward(y_true)
    f_real = real.detach()
    loss = criterion(fake, f_real)
    return loss

# Function is taken from the DeblurGAN paper(https://github.com/KupynOrest/DeblurGAN)
def calc_gradient_penalty(D, real_data, fake_data):
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    LAMBDA = 10
    interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)
    
    disc_interpolates = D.forward(interpolates)

    gradients = torch.autograd.grad(
    outputs=disc_interpolates, inputs=interpolates, grad_outputs=torch.ones(disc_interpolates.size()).cuda(), create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty