import numpy as np
from CreateDataLoader import CreateDataLoader
from Generator import Generator
import torch
from PIL import Image
import metrics


def deprocess_image(img):
    img = img * 127.5 + 127.5
    return img.astype('uint8')


def test(batch_size):
    avgPSNR = 0.0
    avgSSIM = 0.0
    counter = 0
    data_loader = CreateDataLoader(batch_size)
    dataset = data_loader.load_data()
    for i, data in enumerate(dataset):
        counter += 1
        if i > 1:
            break
        images_X = data['A']
        images_Y = data['B']
        G.eval()
        images_X = images_X.to(device)
        generated = G(images_X)
        # generated[generated < 0] = 0

        generated = generated.cpu().detach().numpy()
        x_test = images_X.cpu().float().numpy()
        y_test = images_Y.cpu().float().numpy()

        # generated = output.cpu().detach().numpy()
        # images_X = images_X.cpu().detach().numpy()
        # images_Y = images_Y.detach().numpy()
        for j in range(generated.shape[0]):
            y = y_test[j, :, :, :] # original sharp
            x = x_test[j, :, :, :] # blurred
            img = generated[j, :, :, :] # generated

            avgPSNR += metrics.PSNR(img, y)
            avgSSIM += metrics.SSIM_my(img, y)

            out = np.concatenate((y, x, img), axis=1)
            img = (np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0
            img = img.astype(np.uint8)
            z = i*10 + j
            im = Image.fromarray(img)
            im.save("results{}.png".format(z))
    return avgPSNR, avgSSIM, counter


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    G = Generator(use_dropout=False).to(device)

    G.load_state_dict(torch.load('model_G_last.pt'))

    psnr, ssim, count = test(1)
    psnr /= count
    ssim /= count
    print("psnr: ", psnr)
    print("ssim: ", ssim)
    met = "Average psnr: " + str(psnr) + ", Average SSIM(my): " + str(ssim)
    f = open("metric_result.txt", "w")
    f.write(met)
    f.close()
