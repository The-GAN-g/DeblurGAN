import CreateDataLoader as cd
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from collections import OrderedDict
from constants import Constants as c

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def tensor2im(image_tensor, imtype=np.uint8):
    """
    Convert tensor into an image
    """
    image_numpy = image_tensor[0].cpu().float().detach().numpy()
    # image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def get_visuals(real_A, fake_B, real_B):
    real_A = tensor2im(real_A)
    fake_B = tensor2im(fake_B)
    real_B = tensor2im(real_B)
    return OrderedDict([('Blurred_Train', real_A), ('Restored_Train', fake_B), ('Sharp_Train', real_B)])


def get_errors(gan_loss, content_loss, d_loss):
    return OrderedDict([('G_GAN', gan_loss),
                        ('G_L1', content_loss),
                        ('D_real+fake', d_loss)
                        ])


def print_errors(epoch, i, errors, t):
    message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
    for k, v in errors.items():
        message += '%s: %.3f ' % (k, v)
    print(message)
    with open("log_epochs.txt", "a") as log_file:
        log_file.write('%s\n' % message)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print('Total number of parameters: %d' % num_params)


def checkDataloader():
    """
    to check if data is loaded
    """
    data_loader = cd.CreateDataLoader(batchSize=1)
    dataset = data_loader.load_data()

    def imshow(img):
       npimg = img.numpy()
       plt.imshow(np.transpose(npimg, (1, 2, 0)))

    for i, data in enumerate(dataset):
    # show images
       if i==0:
           fig = plt.figure(figsize=(20, 15))
           imshow(torchvision.utils.make_grid(data['A']))
           imshow(torchvision.utils.make_grid(data['B']))
       else:
           break

def plotter(res_d, res_g):
    ep = list(range(1, c.n_epoch+1))
    plt.plot(ep, res_g, label="G Loss")
    plt.plot(ep, res_d, label="D Loss")
    plt.title('Losses vs Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.xlabel('Epoch')
    plt.savefig("LvE.png")