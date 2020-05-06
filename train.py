import time
import torch
import metrics
import numpy as np
import utils as util
from Generator import Generator
from Losses import perceptual_loss
from torch.autograd import Variable
from constants import Constants as c
from Discriminator import Discriminator
from Losses import calc_gradient_penalty
from CreateDataLoader import CreateDataLoader
# import sys
# from Losses import real_mse_loss
# from Losses import fake_mse_loss


def train(D, G, curr_lr, lr, n_epoch, beta1, beta2, bs):
    data_loader = CreateDataLoader(batchSize=bs)
    dataset = data_loader.load_data()

    # Create optimizers for the generators and discriminators
    optimizer_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))

    res_d = []
    res_g = []
    total_steps = 0
    for epoch in range(1, n_epoch + 1):
        print("Running epoch:", epoch)
        start_time_epoch = time.time()
        sum_d = 0
        sum_g = 0
        for i, data in enumerate(dataset):
            total_steps += c.batchSize

            images_X = data['A']
            images_Y = data['B']

            # move images to GPU if available (otherwise stay on CPU)
            # train discriminator on real
            real_A = Variable(images_X.to(device))
            fake_B = G.forward(real_A)
            real_B = Variable(images_Y.to(device))

            # =======================Train the discriminator=======================#
            for iter in range(5):
                optimizer_D.zero_grad()

                # Real images
                D_real = D.forward(real_B)
                # Fake images
                D_fake = D.forward(fake_B.detach())
                # Gradient penalty
                gradient_penalty = calc_gradient_penalty(D, real_B.data, fake_B.data)
                d_loss = D_fake.mean() - D_real.mean() + gradient_penalty
                d_loss.backward(retain_graph=True)

                optimizer_D.step()
                if iter == 4:
                    sum_d += d_loss.item()

            #========================Train the generator===========================#
            optimizer_G.zero_grad()

            fake_B = G.forward(real_A)
            D_fake = D.forward(fake_B)
            g_loss = -D_fake.mean()
            g_contentloss = perceptual_loss(fake_B, real_B) * 100
            g_total_loss = g_loss + g_contentloss
            g_total_loss.backward()

            optimizer_G.step()

            # printing pnsr & SSIM metrics at certain frequency
            if total_steps % c.display_freq == 4:
                image_res = util.get_visuals(real_A, fake_B, real_B)
                psnr = metrics.PSNR(image_res['Restored_Train'], image_res['Sharp_Train'])
                print('PSNR on Train (at epoch {0}) = {1}'.format(epoch, psnr))
                ssim = metrics.SSIM_my(image_res['Restored_Train'], image_res['Sharp_Train'])
                print('SSIM_my on Train (at epoch {0}) = {1}'.format(epoch, ssim))

            # print losses & errors
            # if total_steps % c.print_freq == 0:
            #     err = util.get_errors(g_loss, g_contentloss, d_loss)
            #     t = (time.time() - start_time_epoch) / c.batchSize
            #     util.print_errors(epoch, i, err, t)

            # sum the loss over all the image
            sum_g += g_total_loss.item()

        # decaying learning rate
        if epoch > 150:
            lrd = 0.0001 / 150
            new_lr = curr_lr - lrd

            for param_group in optimizer_D.param_groups:
                param_group['lr'] = new_lr
            for param_group in optimizer_G.param_groups:
                param_group['lr'] = new_lr
            print('Update learning rate: %f -> %f' % (curr_lr, new_lr))
            curr_lr = new_lr

        # saving model after every 50 epochs
        if epoch % c.save_freq == 0:
            torch.save(G.state_dict(), 'model_G_' + str(epoch) + '.pt')
            torch.save(D.state_dict(), 'model_D_' + str(epoch) + '.pt')
        res_d.append(np.mean(sum_d))
        res_g.append(np.mean(sum_g))
        end_time_epoch = time.time()

        print("Time for epoch {0}: {1} | Disc loss: {2}  | Gen loss: {3}".format(epoch, (end_time_epoch - start_time_epoch), res_d[epoch-1], res_g[epoch-1]))

    torch.save(G.state_dict(), 'model_G_last.pt')
    torch.save(D.state_dict(), 'model_D_last.pt')
    print("Model Saved!")
    util.plotter(res_d, res_g)
        # print losses after every 50 epochs
        # losses = []
        # if epoch % c.print_every == 0:
        #     # append real and fake discriminator losses and the generator loss
        #     losses.append((d_loss.data.item(), g_total_loss.data.item()))
        #     print('Epoch [{:5d}/{:5d}] | Disc loss: {:6.4f}  | Gen loss: {:6.4f} | Images processed: {}'.format(
        #         epoch, n_epoch, d_loss.item(), g_total_loss.item(), i + 1))


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define Discriminator and Generator
    D = Discriminator()
    G = Generator()

    # initialize the weights
    G.apply(util.weights_init)
    D.apply(util.weights_init)

    print('---------- Networks initialized -------------')
    util.print_network(G)
    if c.isTrain:
        util.print_network(D)
    print('---------------------------------------------')

    G.to(device)
    D.to(device)

    bs = c.batchSize

    print("Hold your horses! The GAN is being trained...")
    start_time_gan = time.time()
    train(D, G, c.curr_lr, c.lr, c.n_epoch, c.beta1, c.beta2, bs)
    end_time_gan = time.time()
    print("Total time to train the model:", end_time_gan - start_time_gan)
    print("There...it is trained!")