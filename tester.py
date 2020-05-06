from CreateDataLoader import CreateDataLoader
import sys
import torch
from torch.autograd import Variable

data_loader = CreateDataLoader(batchSize=1)
dataset = data_loader.load_data()

x = list(range(1, 11))
print(x)

for i, data in enumerate(dataset):
    if i < 1:
        images_X = data['A']
        images_Y = data['B']
        #print(type(images_X))
        #print(images_X.data)
        #print("***********************")
        #print(images_X)
        # move images to GPU if available (otherwise stay on CPU)
        # train discriminator on real
        #real_A = Variable(images_X)
        #print(type(real_A.data))
        # fake_B = G.forward(real_A)
        # real_B = Variable(images_Y.to(device))
    else:
        sys.exit()